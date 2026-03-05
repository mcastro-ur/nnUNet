"""
Compound loss: Dice + Skeleton Recall + CrossEntropy.
Based on MIC-DKFZ/Skeleton-Recall compound_losses.py
https://github.com/MIC-DKFZ/Skeleton-Recall/blob/master/nnunetv2/training/loss/compound_losses.py
"""
import torch
from torch import nn

from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.training.loss.robust_ce_loss import RobustCrossEntropyLoss
from nnunetv2.training.loss.skeleton_recall_loss import SoftSkeletonRecallLoss
from nnunetv2.utilities.helpers import softmax_helper_dim1


class DC_SkelREC_and_CE_loss(nn.Module):
    """Dice + Skeleton Recall + CrossEntropy combined loss.

    Loss = weight_ce * CE(net_output, target)
         + weight_dice * Dice(net_output, target)
         + weight_srec * SkeletonRecall(net_output, skel)

    Based on MIC-DKFZ/Skeleton-Recall (ECCV 2024, Kirchhoff et al.)
    """

    def __init__(
        self,
        soft_dice_kwargs: dict,
        soft_skelrec_kwargs: dict,
        ce_kwargs: dict,
        weight_ce: float = 1.0,
        weight_dice: float = 1.0,
        weight_srec: float = 1.0,
        ignore_label: int = None,
        dice_class=MemoryEfficientSoftDiceLoss,
    ):
        super().__init__()

        if ignore_label is not None:
            ce_kwargs["ignore_index"] = ignore_label

        self.weight_ce = weight_ce
        self.weight_dice = weight_dice
        self.weight_srec = weight_srec
        self.ignore_label = ignore_label

        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)
        self.skel_rec = SoftSkeletonRecallLoss(
            apply_nonlin=softmax_helper_dim1, **soft_skelrec_kwargs
        )

    def forward(
        self,
        net_output: torch.Tensor,
        target: torch.Tensor,
        skel: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            net_output: network logits, shape (B, C, Z, Y, X)
            target: segmentation ground truth, shape (B, 1, Z, Y, X) with integer labels
            skel: skeleton ground truth, shape (B, 1, Z, Y, X) with integer labels
                  (IOU label only; 0 everywhere else)

        Returns:
            Scalar combined loss
        """
        if self.ignore_label is not None:
            assert target.shape[1] == 1, (
                "ignore_label is not implemented for one-hot encoded targets "
                "(DC_SkelREC_and_CE_loss)"
            )
            mask = target != self.ignore_label
            target_dice = torch.where(mask, target, torch.zeros_like(target))
            num_fg = mask.sum()
        else:
            target_dice = target
            mask = None

        dc_loss = (
            self.dc(net_output, target_dice, loss_mask=mask)
            if self.weight_dice != 0
            else 0
        )
        ce_loss = (
            self.ce(net_output, target[:, 0])
            if self.weight_ce != 0
            and (self.ignore_label is None or num_fg > 0)
            else 0
        )
        srec_loss = (
            self.skel_rec(net_output, skel)
            if self.weight_srec != 0
            else 0
        )

        return (
            self.weight_ce * ce_loss
            + self.weight_dice * dc_loss
            + self.weight_srec * srec_loss
        )
