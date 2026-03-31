"""
nnUNetTrainerV2_BoundaryDiceCE_1500ep  — v2
============================================
Sous-classe optimisée pour 1500 epochs + petites structures pelviennes (IOU, etc.)

Changements vs v1 :
  - Schedule sigmoid au lieu de linéaire (hérité du base trainer v2)
  - warmup_end: 150 → 50  (activation plus tôt, cohérent avec réseau plus stable)
  - ramp_end:   600 → 300 (plateau atteint à mi-entraînement)
  - clip_grad:  5.0 → 1.0 (hérité du base trainer v2) [FIX-4]
  - LR initial: 5e-4 → 1e-3 (le base trainer v2 gère déjà 1e-3, on laisse faire)

Schedule effectif (1500 epochs) :
  Epochs   0..49   → w_boundary = 0.000  (warmup Dice+CE uniquement)
  Epochs  50..299  → sigmoid ramp 0 → 0.2
  Epochs 300..1500 → w_boundary = 0.200  (plateau constant)

Loss au régime permanent :
  L = 0.5*Dice + 0.5*CE + 0.2*BoundaryLoss(EDT pondérée par classe)
"""

import torch

from nnunetv2.training.nnUNetTrainer.variants.loss.nnUNetTrainerV2_BoundaryDiceCE import (
    nnUNetTrainerV2_BoundaryDiceCE,
)


class nnUNetTrainerV2_BoundaryDiceCE_1500ep(nnUNetTrainerV2_BoundaryDiceCE):

    # ── Schedule adapté aux 1500 epochs ─────────────────────────────────────
    w_boundary_max:        float = 0.2
    w_boundary_warmup_end: int   = 50    # v1 était 150 — trop tardif pour 1500ep
    w_boundary_ramp_end:   int   = 300   # v1 était 600 — plateau à 20% de l'entraînement
    w_boundary_schedule:   str   = 'sigmoid'

    # ── Poids boundary par classe pour Dataset092_Prostate26 ─────────────────
    # Surcharge les poids du base trainer pour coller à tes 11 classes exactes
    # Ajuste selon ton dataset.json (labels 0-10)
    boundary_class_weights = {
        0:  0.0,   # background
        1:  0.5,   # ex: structure 1
        2:  0.5,   # ex: structure 2
        3:  1.5,   # ex: structure 3 (petite)
        4:  0.3,   # ex: structure 4 (os)
        5:  0.3,   # ex: structure 5 (os)
        6:  1.5,   # ex: Prostate
        7:  2.0,   # ex: IOU — supervision boundary maximale
        8:  1.2,   # ex: Vesicules séminales
        9:  0.8,
        10: 1.0,
    }

    def __init__(self, plans: dict, configuration: str, fold: int,
                 dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)

        self.num_epochs = 1500

        # LR : le base trainer v2 met déjà 1e-3, on garde
        # clip_grad : le base trainer v2 met déjà 1.0, on garde
        # Pas besoin de re-définir ici sauf pour override explicite
