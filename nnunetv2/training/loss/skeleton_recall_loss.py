"""
Soft Skeleton Recall Loss for tubular structures.
Ported from MIC-DKFZ/Skeleton-Recall (ECCV 2024, Kirchhoff et al.)
https://github.com/MIC-DKFZ/Skeleton-Recall/blob/master/nnunetv2/training/loss/dice.py
"""
from typing import Callable

import torch
from torch import nn

from nnunetv2.utilities.ddp_allgather import AllGatherGrad
from nnunetv2.utilities.helpers import softmax_helper_dim1


class SoftSkeletonRecallLoss(nn.Module):
    """Soft recall loss computed only on skeleton voxels of the ground truth.

    Recall = sum(softmax(x) * skel_gt) / (sum(skel_gt) + smooth)

    The loss returns the negative recall (minimized during training).
    Background channel is always skipped (do_bg must be False).

    Based on MIC-DKFZ/Skeleton-Recall (ECCV 2024, Kirchhoff et al.)
    """

    def __init__(
        self,
        apply_nonlin: Callable = None,
        batch_dice: bool = False,
        do_bg: bool = False,
        smooth: float = 1.0,
        ddp: bool = True,
    ):
        super().__init__()
        if do_bg:
            raise ValueError(
                "SoftSkeletonRecallLoss does not support do_bg=True. "
                "Skeleton recall on background is meaningless."
            )
        self.apply_nonlin = apply_nonlin
        self.batch_dice = batch_dice
        self.do_bg = do_bg
        self.smooth = smooth
        self.ddp = ddp

    def forward(self, x: torch.Tensor, y: torch.Tensor, loss_mask=None) -> torch.Tensor:
        """
        Args:
            x: network output logits, shape (B, C, Z, Y, X)
            y: skeleton ground truth labels, shape (B, 1, Z, Y, X) with integer class values
               (same label scheme as segmentation; 0 = background / no skeleton)
            loss_mask: optional mask, shape (B, 1, Z, Y, X), 1 where valid

        Returns:
            Scalar loss (negative recall, minimised)
        """
        shp_x = x.shape  # (B, C, ...)

        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        # Skip background channel: work only on foreground classes
        # x[:, 0] = background → skip; x[:, 1:] = foreground
        x_fg = x[:, 1:]  # (B, C-1, ...)

        # Convert skeleton labels to one-hot for foreground classes only
        # y has shape (B, 1, ...) with integer labels 0..C-1
        # We want y_onehot of shape (B, C-1, ...) for classes 1..C-1
        with torch.no_grad():
            if x_fg.ndim != y.ndim:
                y = y.view((y.shape[0], 1, *y.shape[1:]))

            num_fg_classes = x_fg.shape[1]
            # Build one-hot for all classes [0..C-1], then drop background
            y_onehot_full = torch.zeros(
                (y.shape[0], shp_x[1], *y.shape[2:]),
                device=x.device,
                dtype=torch.float32,
            )
            y_onehot_full.scatter_(1, y.long(), 1)
            # Keep only foreground channels (drop channel 0 = background)
            y_onehot = y_onehot_full[:, 1:].float()  # (B, C-1, ...)

        if self.batch_dice:
            axes = [0] + list(range(2, x_fg.ndim))
        else:
            axes = list(range(2, x_fg.ndim))

        # Numerator: predicted probability at skeleton voxels
        if loss_mask is not None:
            intersect = (x_fg * y_onehot * loss_mask).sum(dim=axes, dtype=torch.float32)
            sum_gt = (y_onehot * loss_mask).sum(dim=axes, dtype=torch.float32)
        else:
            intersect = (x_fg * y_onehot).sum(dim=axes, dtype=torch.float32)
            sum_gt = y_onehot.sum(dim=axes, dtype=torch.float32)

        if self.ddp and self.batch_dice:
            intersect = AllGatherGrad.apply(intersect).sum(0, dtype=torch.float32)
            sum_gt = AllGatherGrad.apply(sum_gt).sum(0, dtype=torch.float32)

        if self.batch_dice:
            intersect = intersect.sum(0, dtype=torch.float32)
            sum_gt = sum_gt.sum(0, dtype=torch.float32)

        recall = (intersect + self.smooth) / (sum_gt + self.smooth).clamp_min(1e-8)
        recall = recall.mean()

        return -recall
