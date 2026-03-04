import torch

from nnunetv2.training.nnUNetTrainer.variants.loss.nnUNetTrainerV2_BoundaryDiceCE import nnUNetTrainerV2_BoundaryDiceCE


class nnUNetTrainerV2_BoundaryDiceCE_1500ep(nnUNetTrainerV2_BoundaryDiceCE):
    """
    Optimized for 1500 epochs + small urinary OAR structures.

    Schedule:
      Epochs   0..149  → w_boundary = 0.0  (warmup, Dice+CE only)
      Epochs 150..599  → linear ramp  0.0 → 0.2
      Epochs 600..1500 → w_boundary = 0.2  (constant)

    Loss effective at full regime:
      L = 0.5*Dice + 0.5*CE + 0.2*Boundary
    """

    w_boundary_max: float = 0.2
    w_boundary_warmup_end: int = 150
    w_boundary_ramp_end: int = 600

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 1500
        # Even more conservative LR for 1500-epoch training
        import os
        if not os.environ.get("NNUNET_INITIAL_LR", "").strip():
            self.initial_lr = 5e-4
        if not os.environ.get("NNUNET_CLIP_GRAD", "").strip():
            self.clip_grad = 5.0
