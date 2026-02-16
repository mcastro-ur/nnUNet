# -*- coding: utf-8 -*-
"""
Created on Wed Feb 11 14:33:57 2026

@author: Miguel Castro USER
"""

import torch
import torch.nn.functional as F
import numpy as np
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.training.loss.robust_ce_loss import RobustCrossEntropyLoss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.utilities.helpers import softmax_helper_dim1

from boundary_dataset_wrapper import BoundaryDatasetWrapper
from torch.cuda.amp import autocast, GradScaler

class BoundaryLoss(torch.nn.Module):
    # """Simple Boundary Loss (distance-weighted) pour nnU-Net"""
    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, target, precomputed_boundary=None):
        probs = F.softmax(logits, 1)
        if precomputed_boundary is not None:
            boundary = precomputed_boundary  # déjà au bon shape, mêmes classes
        else:
            boundary = self._get_boundary_mask_from_target(target)  # fallback
        # (optionnel) nuller les grosses classes ici
        return (probs * boundary).sum(dim=(1,2,3,4)).mean()    

    def _get_boundary_mask(self, target_oh):
        # Cr�e masque fronti�re simple (1 aux bordures, d�cro�t vers centre)
        B, C, *spatial = target_oh.shape
        
        # Dilatation simple (kernel 3x3x3)
        dilated = torch.nn.functional.max_pool3d(target_oh, kernel_size=3, stride=1, padding=1)
        eroded = torch.nn.functional.max_pool3d(1 - target_oh, kernel_size=3, stride=1, padding=1)
        
        # Boundary = dilated - eroded (plus fort aux fronti�res)
        boundary = dilated - eroded
        boundary = torch.clamp(boundary, 0, 1)
        
        return boundary

class BoundaryDiceCELoss(torch.nn.Module):
    def __init__(self, weight_boundary=0.3, weight_dice=0.5, weight_ce=0.2):
        super().__init__()
        self.boundary_loss = BoundaryLoss()
        self.dice_loss = MemoryEfficientSoftDiceLoss(smooth=1e-5, batch_dice=True, do_bg=True)
        # ? SUPPRIME ignore_label !
        self.ce_loss = RobustCrossEntropyLoss()  # ? SANS ignore_label
        self.weight_boundary = weight_boundary
        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
    
    def forward(self, logits, target):
        l_dice = self.dice_loss(logits, target)
        l_ce = self.ce_loss(logits, target)
        l_boundary = self.boundary_loss(logits, target)
        return self.weight_boundary * l_boundary + self.weight_dice * l_dice + self.weight_ce * l_ce

class nnUNetTrainerV2_BoundaryDiceCE(nnUNetTrainer):
    #"""Trainer avec Boundary+Dice+CE (0.3/0.5/0.2)"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scaler = GradScaler()  # si pas déjà géré par le parent

    def _build_loss(self):
        loss_core = BoundaryDiceCELoss(
            weight_boundary=0.3, weight_dice=0.5, weight_ce=0.2
        )
        if self._do_i_compile():
            loss_core = torch.compile(loss_core, mode="max-autotune")  # ou "reduce-overhead"

        if self.enable_deep_supervision:
            weights = self._compute_ds_weights()  # tel que dans ton code
            loss = DeepSupervisionWrapper(loss_core, weights)
        else:
            loss = loss_core
        return loss
    
   
    def run_iteration(self, data, do_backprop=True):
        data_input = data['data']
        target = data['seg']
        boundary = data.get('boundary', None)

        with autocast(dtype=torch.float16):  # AMP ON
            logits = self.network(data_input)
            # >>> LIGNE IMPORTANTE : on passe boundary à la loss <<<
            loss = self.loss(logits, target, boundary=boundary)

        if do_backprop:
            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.scaler.step(self.optimizer)
            self.scaler.update()

        return loss.detach().cpu().numpy()
