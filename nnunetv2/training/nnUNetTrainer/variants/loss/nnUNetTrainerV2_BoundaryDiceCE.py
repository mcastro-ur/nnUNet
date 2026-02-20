# -*- coding: utf-8 -*-
"""
Created on Wed Feb 11 14:33:57 2026

@author: Miguel Castro USER
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Union, List
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.training.loss.robust_ce_loss import RobustCrossEntropyLoss
from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler

from nnunetv2.training.nnUNetTrainer.variants.loss.MergeBoundaryIntoSegBGv2 import MergeBoundaryIntoSegBGv2
from nnunetv2.training.nnUNetTrainer.variants.loss.SplitBoundaryFromSegBGv2 import SplitBoundaryFromSegBGv2

# Classes petites structures
SMALL_CLASSES = [2, 3, 5, 6, 7, 8, 9, 10]


class BoundaryLoss(torch.nn.Module):
    """Boundary Loss ultra-légère avec kernel Sobel - focus sur petites structures"""
    def __init__(self, smooth=1e-5, focus_small_classes=True, class_weights=None):
        super().__init__()
        self.smooth = smooth
        self.focus_small_classes = focus_small_classes
        self.small_class_ids = SMALL_CLASSES
        self.class_weights = class_weights
        self.register_buffer('sobel_kernel', self._create_sobel_kernel())
    
    def _create_sobel_kernel(self):
        """Kernel Sobel 3D pour détection de contours"""
        kernel = torch.tensor([
            [[-1, -2, -1], [-2, -4, -2], [-1, -2, -1]],
            [[-2, -4, -2], [-4, 24, -4], [-2, -4, -2]],
            [[-1, -2, -1], [-2, -4, -2], [-1, -2, -1]]
        ], dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 24.0
        return kernel
    
    def forward(self, logits, target, precomputed_boundary=None):
        """Calcul boundary loss avec class weights"""
        pred_classes = torch.argmax(logits, dim=1, keepdim=True).float()
        
        if target.dim() == 5 and target.shape[1] == 1:
            target = target.float()
        else:
            target = target.unsqueeze(1).float()
        
        # ✅ CORRECTION : Déplacer le kernel sur le bon device ET dtype
        sobel_kernel = self.sobel_kernel.to(device=target.device, dtype=target.dtype)
        
        # Détection contours
        target_boundary = F.conv3d(target, sobel_kernel, padding=1).abs()
        pred_boundary = F.conv3d(pred_classes, sobel_kernel, padding=1).abs()
        
        # Appliquer class weights
        if self.class_weights is not None:
            weight_map = torch.ones_like(target)
            for cls_id, weight in self.class_weights.items():
                weight_map[target == cls_id] = weight
            loss = F.mse_loss(pred_boundary * weight_map, target_boundary * weight_map)
        elif self.focus_small_classes:
            small_mask = torch.zeros_like(target, dtype=torch.bool)
            for cls_id in self.small_class_ids:
                small_mask = small_mask | (target == cls_id)
            weight_map = torch.ones_like(target)
            weight_map[small_mask] = 3.0
            loss = F.mse_loss(pred_boundary * weight_map, target_boundary * weight_map)
        else:
            loss = F.mse_loss(pred_boundary, target_boundary)
        
        return loss


class WeightedCrossEntropyLoss(torch.nn.Module):
    """CrossEntropy avec class weights"""
    def __init__(self, class_weights=None):
        super().__init__()
        self.class_weights = class_weights
        self.ce = RobustCrossEntropyLoss()
    
    def forward(self, logits, target):
        if self.class_weights is not None:
            num_classes = logits.shape[1]
            weight_tensor = torch.ones(num_classes, device=logits.device)
            for cls_id, weight in self.class_weights.items():
                if int(cls_id) < num_classes:
                    weight_tensor[int(cls_id)] = weight
            
            if target.dim() == 5 and target.shape[1] == 1:
                target = target.squeeze(1)
            return F.cross_entropy(logits, target.long(), weight=weight_tensor)
        else:
            return self.ce(logits, target)


class BoundaryDiceCELoss(torch.nn.Module):
    """Loss combinée qui LIT les poids du JSON"""
    def __init__(self, weight_boundary=0.3, weight_dice=0.5, weight_ce=0.2, 
                 class_weights=None, ddp=False):
        super().__init__()
        self.boundary_loss = BoundaryLoss(focus_small_classes=True, class_weights=class_weights)
        self.dice_loss = MemoryEfficientSoftDiceLoss(
            smooth=1e-5, batch_dice=True, do_bg=False, ddp=ddp
        )
        self.ce_loss = WeightedCrossEntropyLoss(class_weights=class_weights)
        self.weight_boundary = weight_boundary
        self.weight_dice = weight_dice
        self.weight_ce = weight_ce

    def forward(self, logits, target):
        l_dice = self.dice_loss(logits, target)
        l_ce = self.ce_loss(logits, target)
        l_boundary = self.boundary_loss(logits, target)
        
        return (self.weight_boundary * l_boundary + 
                self.weight_dice * l_dice + 
                self.weight_ce * l_ce)


class nnUNetTrainerV2_BoundaryDiceCE(nnUNetTrainer):
    """Trainer qui LIT les paramètres custom du nnUNetPlans.json"""
    
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        
        # IMPORTANT : Lire la config AVANT d'appeler super().__init__
        # car super().__init__ utilise certains de ces paramètres
        config = plans['configurations'][configuration]
        
        # Lire small_structure_params pour oversampling
        self._custom_oversample = 0.70  # default
        if 'small_structure_params' in config:
            ssp = config['small_structure_params']
            oversampling = ssp.get('oversampling_factor', {})
            if isinstance(oversampling, dict):
                max_oversample = max([v for v in oversampling.values() if isinstance(v, (int, float))])
                self._custom_oversample = min(0.95, max_oversample / 10.0)
        
        # Lire loss params
        self._loss_weights = {'boundary': 0.4, 'dice': 0.4, 'ce': 0.2}
        self._class_weights = None
        if 'loss' in config:
            loss_cfg = config['loss']
            if 'weights' in loss_cfg and len(loss_cfg['weights']) == 3:
                self._loss_weights = {
                    'dice': float(loss_cfg['weights'][0]),
                    'ce': float(loss_cfg['weights'][1]),
                    'boundary': float(loss_cfg['weights'][2])
                }
            if 'class_weights' in loss_cfg:
                self._class_weights = {int(k): float(v) for k, v in loss_cfg['class_weights'].items()}
        
        # Lire training params
        self._custom_num_epochs = config.get('training', {}).get('num_epochs', 1500)
        
        # Lire network params
        self._custom_weight_decay = config.get('network', {}).get('weight_decay', 3e-5)
        
        # Stocker augmentation params
        self._custom_augmentation = config.get('augmentation', {})
        
        # Maintenant appeler super().__init__
        super().__init__(plans, configuration, fold, dataset_json, device)
        
        # Override des params après init
        self.oversample_foreground_percent = self._custom_oversample
        self.num_epochs = self._custom_num_epochs
        self.weight_decay = self._custom_weight_decay
        
        # Log des params custom
        self.print_to_log_file(f"Custom oversample: {self.oversample_foreground_percent}")
        self.print_to_log_file(f"Custom epochs: {self.num_epochs}")
        self.print_to_log_file(f"Custom loss weights: {self._loss_weights}")
        if self._class_weights:
            self.print_to_log_file(f"Custom class weights: {self._class_weights}")
    
    def configure_optimizers(self):
        """Utiliser weight_decay du JSON"""
        optimizer = torch.optim.SGD(
            self.network.parameters(), 
            self.initial_lr, 
            weight_decay=self.weight_decay,
            momentum=0.99, 
            nesterov=True
        )
        lr_scheduler = PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs)
        return optimizer, lr_scheduler
    
    def _build_loss(self):
        """Loss qui utilise les poids du JSON"""
        from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
        import numpy as np
        
        loss_core = BoundaryDiceCELoss(
            weight_boundary=self._loss_weights['boundary'],
            weight_dice=self._loss_weights['dice'],
            weight_ce=self._loss_weights['ce'],
            class_weights=self._class_weights,
            ddp=self.is_ddp
        )
        
        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            
            if self.is_ddp and not self._do_i_compile():
                weights[-1] = 1e-6
            else:
                weights[-1] = 0
            
            weights = weights / weights.sum()
            loss_core = DeepSupervisionWrapper(loss_core, weights)
        
        return loss_core
    
    def get_training_transforms(self, patch_size, rotation_for_DA, deep_supervision_scales,
                               mirror_axes, do_dummy_2d_data_aug, use_mask_for_norm=None,
                               is_cascaded=False, foreground_labels=None, regions=None,
                               ignore_label=None):
        """Augmentations qui LISENT le JSON"""
        
        from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform
        from batchgeneratorsv2.transforms.utils.compose import ComposeTransforms
        from batchgeneratorsv2.transforms.spatial.spatial import SpatialTransform
        from batchgeneratorsv2.transforms.utils.random import RandomTransform
        from batchgeneratorsv2.transforms.intensity.gaussian_noise import GaussianNoiseTransform
        from batchgeneratorsv2.transforms.noise.gaussian_blur import GaussianBlurTransform
        from batchgeneratorsv2.transforms.intensity.brightness import MultiplicativeBrightnessTransform
        from batchgeneratorsv2.transforms.intensity.contrast import ContrastTransform, BGContrast
        from batchgeneratorsv2.transforms.intensity.gamma import GammaTransform
        from batchgeneratorsv2.transforms.spatial.mirroring import MirrorTransform
        from batchgeneratorsv2.transforms.spatial.low_resolution import SimulateLowResolutionTransform
        from batchgeneratorsv2.transforms.utils.deep_supervision_downsampling import DownsampleSegForDSTransform
        from batchgeneratorsv2.transforms.utils.remove_label import RemoveLabelTansform
        from batchgeneratorsv2.transforms.utils.pseudo2d import Convert3DTo2DTransform, Convert2DTo3DTransform
        
        from nnunetv2.training.nnUNetTrainer.variants.loss.merge_boundary_into_seg import MergeBoundaryIntoSeg
        from nnunetv2.training.nnUNetTrainer.variants.loss.split_boundary_from_seg import SplitBoundaryFromSeg
 
        # Lire params custom du JSON
        aug_cfg = self._custom_augmentation
        
        # Rotation
        rot_degrees = aug_cfg.get('rotation_degrees', [-30, 30])
        rotation_custom = tuple(np.array(rot_degrees) / 360 * 2 * np.pi)
        
        # Scaling
        scale_range = tuple(aug_cfg.get('scale_range', [0.7, 1.4]))
        
        # Elastic deform
        elastic_cfg = aug_cfg.get('elastic_deform', {})
        elastic_alpha = tuple(elastic_cfg.get('alpha', [0, 900]))
        elastic_sigma = (elastic_cfg.get('sigma', 9), elastic_cfg.get('sigma', 9) + 4)
        
        # Gamma
        gamma_range = tuple(aug_cfg.get('gamma_range', [0.7, 1.5]))
        
        transforms = []
        
        if do_dummy_2d_data_aug:
            ignore_axes = (0,)
            transforms.append(Convert3DTo2DTransform())
            patch_size_spatial = patch_size[1:]
        else:
            patch_size_spatial = patch_size
            ignore_axes = None
        
        # Merge boundary
        transforms.append(MergeBoundaryIntoSeg(boundary_key='boundary', seg_key='segmentation'))
        
        # SpatialTransform avec params du JSON
#        transforms.append(
#            SpatialTransform(
#                patch_size_spatial,
#                patch_center_dist_from_border=0,
#                random_crop=False,
#                p_elastic_deform=0.3,
#                elastic_deform_scale=elastic_alpha,
#                elastic_deform_magnitude=elastic_sigma,
#                p_rotation=0.3,
#                rotation=rotation_custom,
#                p_scaling=0.3,
#                scaling=scale_range,
#                p_synchronize_scaling_across_axes=1,
#                bg_style_seg_sampling=False,
#                mode_seg='nearest',
#                border_mode_seg='constant',
#                center_deformation=True,
#                padding_mode_image='zeros'
#            )
#        )
        # SpatialTransform - VERSION MINIMALISTE SANS BORDER_MODE
        transforms.append(
            SpatialTransform(
                patch_size_spatial,
                patch_center_dist_from_border=0,
                random_crop=False,
                p_rotation=0.2,
                rotation=rotation_custom,
                p_scaling=0.2,
                scaling=scale_range,
                p_synchronize_scaling_across_axes=1,
                bg_style_seg_sampling=False
            )
        )
        # Split boundary
        transforms.append(SplitBoundaryFromSeg(boundary_key='boundary', seg_key='segmentation'))
        
        if do_dummy_2d_data_aug:
            transforms.append(Convert2DTo3DTransform())
        
        # Gaussian noise
        if aug_cfg.get('additive_gaussian_noise', {}).get('enabled', True):
            noise_std = aug_cfg.get('additive_gaussian_noise', {}).get('std', 0.1)
            transforms.append(RandomTransform(
                GaussianNoiseTransform(noise_variance=(0, noise_std), p_per_channel=1, synchronize_channels=True),
                apply_probability=0.1
            ))
        
        transforms.append(RandomTransform(
            GaussianBlurTransform(blur_sigma=(0.5, 1.), synchronize_channels=False, 
                                 synchronize_axes=False, p_per_channel=0.5, benchmark=True),
            apply_probability=0.2
        ))
        
        transforms.append(RandomTransform(
            MultiplicativeBrightnessTransform(multiplier_range=BGContrast((0.75, 1.25)), 
                                             synchronize_channels=False, p_per_channel=1),
            apply_probability=0.15
        ))
        
        transforms.append(RandomTransform(
            ContrastTransform(contrast_range=BGContrast((0.75, 1.25)), preserve_range=True, 
                            synchronize_channels=False, p_per_channel=1),
            apply_probability=0.15
        ))
        
        # Simulate low res
        if aug_cfg.get('simulate_low_res', {}).get('enabled', True):
            zoom_range = tuple(aug_cfg.get('simulate_low_res', {}).get('zoom_range', [0.5, 1]))
            transforms.append(RandomTransform(
                SimulateLowResolutionTransform(scale=zoom_range, synchronize_channels=False, 
                                              synchronize_axes=True, ignore_axes=ignore_axes, 
                                              allowed_channels=None, p_per_channel=0.5),
                apply_probability=0.25
            ))
        
        # Gamma avec params du JSON
        transforms.append(RandomTransform(
            GammaTransform(gamma=BGContrast(gamma_range), p_invert_image=1, 
                          synchronize_channels=False, p_per_channel=1, p_retain_stats=1),
            apply_probability=0.1
        ))
        
        transforms.append(RandomTransform(
            GammaTransform(gamma=BGContrast(gamma_range), p_invert_image=0, 
                          synchronize_channels=False, p_per_channel=1, p_retain_stats=1),
            apply_probability=0.3
        ))
        
        if mirror_axes is not None and len(mirror_axes) > 0:
            transforms.append(MirrorTransform(allowed_axes=mirror_axes))
        
        transforms.append(RemoveLabelTansform(-1, 0))
        
        if deep_supervision_scales is not None:
            transforms.append(DownsampleSegForDSTransform(ds_scales=deep_supervision_scales))
        
        return ComposeTransforms(transforms)