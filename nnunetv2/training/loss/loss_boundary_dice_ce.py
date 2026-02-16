# -*- coding: utf-8 -*-
"""
Created on Wed Feb 11 14:33:57 2026

@author: Miguel Castro USER
"""

# nnunetv2/training/loss/loss_boundary_dice_ce.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.training.loss.robust_ce_loss import RobustCrossEntropyLoss

class BoundaryLoss(nn.Module):
    """
    Boundary loss (Kervadec et al.) version simple pour segmentation multi-classe.
    inputs: logits (B, C, ...), target: (B, ...), dist_maps: (B, C, ...)
    dist_maps = signed distance map de chaque classe (GT)
    """
    def __init__(self):
        super().__init__()

    def forward(self, logits, target, dist_maps):
        # logits -> probas softmax
        probs = F.softmax(logits, dim=1)  # (B, C, ...)
        # one-hot du GT
        num_classes = probs.shape[1]
        target_oh = F.one_hot(target.long(), num_classes=num_classes)  # (B, ..., C)
        target_oh = target_oh.permute(0, -1, *range(1, target_oh.ndim-1)).float()  # (B, C, ...)

        # boundary loss = somme_c ? P_c * D_c
        # ici: moyenne sur batch + voxels
        bl = (probs * dist_maps).sum(dim=(1, 2, 3, 4))  # pour 3D; adapte dim si 2D
        return bl.mean()


class BoundaryDiceCELoss(nn.Module):
    """
    0.3 * BoundaryLoss + 0.5 * DiceLoss + 0.2 * CE
    """
    def __init__(self,
                 weight_boundary=0.3,
                 weight_dice=0.5,
                 weight_ce=0.2,
                 ignore_label=None):
        super().__init__()
        self.weight_boundary = weight_boundary
        self.weight_dice = weight_dice
        self.weight_ce = weight_ce

        self.boundary_loss = BoundaryLoss()
        self.dice_loss = MemoryEfficientSoftDiceLoss(
            apply_nonlin=lambda x: F.softmax(x, dim=1),
            smooth=1.0,
            batch_dice=True,
            do_bg=True
        )
        self.ce_loss = RobustCrossEntropyLoss(ignore_label=ignore_label)

    def forward(self, logits, target, dist_maps):
        """
        logits: (B, C, ...)
        target: (B, ...) labels entiers
        dist_maps: (B, C, ...) distance map par classe
        """
        # Dice + CE utilisent (logits, target)
        loss_dice = self.dice_loss(logits, target)
        loss_ce = self.ce_loss(logits, target)

        # BoundaryLoss utilise aussi dist_maps
        loss_boundary = self.boundary_loss(logits, target, dist_maps)

        loss = (self.weight_boundary * loss_boundary +
                self.weight_dice * loss_dice +
                self.weight_ce * loss_ce)
        return loss
