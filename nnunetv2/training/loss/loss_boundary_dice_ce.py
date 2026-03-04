# nnunetv2/training/loss/loss_boundary_dice_ce.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.training.loss.robust_ce_loss import RobustCrossEntropyLoss
from nnunetv2.utilities.helpers import softmax_helper_dim1


class BoundaryLoss(nn.Module):
    """
    Numerically stable boundary loss (simplified distance-weighted version).

    Computes: mean over batch of sum_c(softmax(logits)_c * boundary_c)

    boundary should be a float mask (B, C, ...) where high values indicate
    boundary regions for each class. If not provided, falls back to a morphological
    boundary derived from the one-hot target.

    AMP note: the caller is responsible for ensuring this runs in FP32.
    """

    def __init__(self, smooth: float = 1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        boundary: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Always run in FP32 to avoid NaN/overflow under AMP
        logits = logits.float()

        probs = F.softmax(logits, dim=1)  # (B, C, ...)

        if boundary is not None:
            bnd = boundary.float()
            # Clamp to avoid degenerate values
            bnd = torch.clamp(bnd, 0.0, 1.0)
        else:
            bnd = self._boundary_from_target(target, num_classes=probs.shape[1])

        # Weighted sum: (B,) then mean
        loss = (probs * bnd).sum(dim=list(range(1, probs.ndim))).mean()
        return loss

    @staticmethod
    def _boundary_from_target(target: torch.Tensor, num_classes: int) -> torch.Tensor:
        """Morphological boundary mask derived from one-hot encoded target."""
        target_long = target.long()
        # Handle shape (B, 1, ...) or (B, ...)
        target_squeezed = target_long[:, 0] if (target_long.ndim >= 2 and target_long.shape[1] == 1) else target_long
        oh = F.one_hot(target_squeezed, num_classes=num_classes)  # (B, ..., C)
        # Move class dim to position 1
        perm = [0, oh.ndim - 1] + list(range(1, oh.ndim - 1))
        oh = oh.permute(*perm).float()  # (B, C, ...)

        # Morphological boundary via max-pool
        kernel_size, padding = 3, 1
        dilated = F.max_pool3d(oh, kernel_size=kernel_size, stride=1, padding=padding)
        eroded = 1.0 - F.max_pool3d(1.0 - oh, kernel_size=kernel_size, stride=1, padding=padding)
        boundary = torch.clamp(dilated - eroded, 0.0, 1.0)
        return boundary


class BoundaryDiceCELoss(nn.Module):
    """
    Combined loss: w_dice * Dice + w_ce * CE + w_boundary * Boundary

    The boundary weight can be set dynamically via the ``weight_boundary``
    attribute (updated by the trainer each epoch for the warmup/ramp schedule).
    """

    def __init__(
        self,
        weight_boundary: float = 0.2,
        weight_dice: float = 0.5,
        weight_ce: float = 0.5,
        ignore_label: int | None = None,
        batch_dice: bool = True,
        ddp: bool = False,
    ):
        super().__init__()
        self.weight_boundary = weight_boundary
        self.weight_dice = weight_dice
        self.weight_ce = weight_ce

        self.boundary_loss = BoundaryLoss()
        self.dice_loss = MemoryEfficientSoftDiceLoss(
            apply_nonlin=softmax_helper_dim1,
            smooth=1e-5,
            batch_dice=batch_dice,
            do_bg=False,
            ddp=ddp,
        )
        ce_kwargs: dict = {}
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label
        self.ce_loss = RobustCrossEntropyLoss(**ce_kwargs)
        self.ignore_label = ignore_label

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        boundary: torch.Tensor | None = None,
        return_components: bool = False,
    ) -> torch.Tensor | dict:
        if self.ignore_label is not None:
            assert target.shape[1] == 1
            mask = target != self.ignore_label
            target_dice = torch.where(mask, target, torch.zeros_like(target))
            num_fg = mask.sum()
        else:
            target_dice = target
            mask = None
            num_fg = None

        loss_dice = self.dice_loss(logits, target_dice, loss_mask=mask) if self.weight_dice != 0 else 0.0
        loss_ce = (
            self.ce_loss(logits, target[:, 0])
            if self.weight_ce != 0 and (num_fg is None or num_fg > 0)
            else 0.0
        )

        if self.weight_boundary != 0:
            loss_boundary = self.boundary_loss(logits, target, boundary)
        else:
            loss_boundary = torch.tensor(0.0, device=logits.device, dtype=torch.float32)

        total = (
            self.weight_dice * loss_dice
            + self.weight_ce * loss_ce
            + self.weight_boundary * loss_boundary
        )

        if return_components:
            return {
                'total': total,
                'dice': loss_dice if isinstance(loss_dice, torch.Tensor) else torch.tensor(loss_dice, device=logits.device, dtype=torch.float32),
                'ce': loss_ce if isinstance(loss_ce, torch.Tensor) else torch.tensor(loss_ce, device=logits.device, dtype=torch.float32),
                'boundary': loss_boundary,
            }

        return total

