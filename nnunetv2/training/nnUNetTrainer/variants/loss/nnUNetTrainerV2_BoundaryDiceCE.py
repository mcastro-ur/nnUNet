# nnunetv2/training/nnUNetTrainer/variants/loss/nnUNetTrainerV2_BoundaryDiceCE.py
"""
nnUNetTrainerV2_BoundaryDiceCE
==============================
Trainer that adds a boundary-weighted loss on top of the standard Dice + CE loss.

Loss = w_dice * Dice + w_ce * CE + w_boundary * BoundaryLoss

Boundary weight schedule (applied per-epoch, updated in on_train_epoch_start):
  - epochs 0..49  : w_boundary = 0   (warmup)
  - epochs 50..99 : linear ramp 0 → w_boundary_max
  - epochs >= 100 : w_boundary = w_boundary_max (constant)

Design choices (see problem statement):
  - torch.compile disabled (_do_i_compile returns False)
  - BoundaryLoss computed in FP32 even under AMP (autocast disabled locally)
  - NaN/Inf detection: skip optimizer step, raise only after 3 consecutive bad steps
  - Deep supervision: boundary loss applied only to the full-resolution output
  - No modifications to plans/dataset
"""

import os

import numpy as np
import torch
from torch import autocast

from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.loss_boundary_dice_ce import BoundaryDiceCELoss
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.helpers import dummy_context


class _BoundaryDiceCEWithDS(torch.nn.Module):
    """
    Wraps BoundaryDiceCELoss so that it can be used with deep supervision.

    For deep-supervised outputs (list/tuple), the boundary loss is applied
    ONLY to the first (full-resolution) output; the remaining outputs use
    the plain Dice + CE component of the loss (w_boundary forced to 0).

    The ``weight_factors`` follow the same exponential-decay scheme as in the
    standard nnUNet DeepSupervisionWrapper.
    """

    def __init__(self, loss_fn: BoundaryDiceCELoss, weight_factors: tuple):
        super().__init__()
        self.loss_fn = loss_fn
        self.weight_factors = weight_factors

    def forward(self, net_output, target, boundary=None):
        if isinstance(net_output, (list, tuple)):
            # Deep-supervised: multiple outputs + multiple targets
            assert isinstance(target, (list, tuple))
            total = torch.tensor(0.0, device=net_output[0].device, dtype=net_output[0].dtype)
            for i, (out_i, tgt_i, w) in enumerate(zip(net_output, target, self.weight_factors)):
                if w == 0:
                    continue
                # Boundary only for the full-res (first) output
                bnd_i = boundary if i == 0 else None
                total = total + w * self.loss_fn(out_i, tgt_i, bnd_i)
            return total
        else:
            return self.loss_fn(net_output, target, boundary)


class nnUNetTrainerV2_BoundaryDiceCE(nnUNetTrainer):
    """
    nnUNet trainer with Boundary + Dice + CE loss.

    Extra hyperparameters:
      w_boundary_max : float  – peak boundary weight once fully ramped (default 0.2)
      w_boundary_warmup_end : int  – epoch at which ramp starts (default 50)
      w_boundary_ramp_end   : int  – epoch at which boundary weight reaches max (default 100)
    """

    # ------------------------------------------------------------------ #
    # Class-level defaults (can be overridden in subclasses)              #
    # ------------------------------------------------------------------ #
    w_boundary_max: float = 0.2
    w_boundary_warmup_end: int = 50   # epochs 0..(warmup_end-1) → w_boundary = 0
    w_boundary_ramp_end: int = 100    # epochs warmup_end..(ramp_end-1) → linear ramp

    # ------------------------------------------------------------------ #
    # Init: lower LR + tighter clip, apply env-var ramp overrides         #
    # ------------------------------------------------------------------ #
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)

        # Lower defaults for BoundaryDiceCE (more conservative than base trainer)
        # Only override if the user has not already set via NNUNET_INITIAL_LR env var
        if not os.environ.get("NNUNET_INITIAL_LR", "").strip():
            self.initial_lr = 1e-3
        if not os.environ.get("NNUNET_CLIP_GRAD", "").strip():
            self.clip_grad = 5.0

        # Apply env-var overrides for ramp schedule
        _warmup = os.environ.get("NNUNET_BOUNDARY_WARMUP_END", "").strip()
        if _warmup:
            self.w_boundary_warmup_end = int(_warmup)

        _ramp = os.environ.get("NNUNET_BOUNDARY_RAMP_END", "").strip()
        if _ramp:
            self.w_boundary_ramp_end = int(_ramp)

        _bmax = os.environ.get("NNUNET_BOUNDARY_MAX", "").strip()
        if _bmax:
            self.w_boundary_max = float(_bmax)

        # Counter for consecutive NaN/Inf steps (used for tolerance logic)
        self._nan_step_count = 0

    # ------------------------------------------------------------------ #
    # torch.compile disabled for this trainer                             #
    # ------------------------------------------------------------------ #
    def _do_i_compile(self) -> bool:
        return False

    # ------------------------------------------------------------------ #
    # Loss construction                                                   #
    # ------------------------------------------------------------------ #
    def _build_loss(self):
        ignore_label = self.label_manager.ignore_label

        loss_fn = BoundaryDiceCELoss(
            weight_boundary=0.0,                       # starts at 0; updated per epoch
            weight_dice=0.5,
            weight_ce=0.5,
            ignore_label=ignore_label,
            batch_dice=self.configuration_manager.batch_dice,
            ddp=self.is_ddp,
        )

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            if self.is_ddp:
                weights[-1] = 1e-6
            else:
                weights[-1] = 0
            weights = weights / weights.sum()
            loss = _BoundaryDiceCEWithDS(loss_fn, tuple(weights))
        else:
            loss = loss_fn  # single output

        # Keep a direct reference to loss_fn for weight updates
        self._boundary_loss_fn = loss_fn
        return loss

    # ------------------------------------------------------------------ #
    # Boundary weight schedule                                            #
    # ------------------------------------------------------------------ #
    def _get_boundary_weight(self, epoch: int) -> float:
        if epoch < self.w_boundary_warmup_end:
            return 0.0
        if epoch >= self.w_boundary_ramp_end:
            return self.w_boundary_max
        ramp_len = self.w_boundary_ramp_end - self.w_boundary_warmup_end
        if ramp_len <= 0:
            return self.w_boundary_max
        progress = (epoch - self.w_boundary_warmup_end) / ramp_len
        return float(progress * self.w_boundary_max)

    def on_train_epoch_start(self):
        super().on_train_epoch_start()
        w = self._get_boundary_weight(self.current_epoch)
        self._boundary_loss_fn.weight_boundary = w
        if self.local_rank == 0:
            self.print_to_log_file(f"  [BoundaryDiceCE] w_boundary = {w:.4f}")

    def on_train_epoch_end(self, train_outputs):
        """Override to use nanmean so skipped NaN steps don't corrupt epoch loss."""
        from nnunetv2.utilities.collate_outputs import collate_outputs
        import torch.distributed as dist
        outputs = collate_outputs(train_outputs)

        if self.is_ddp:
            losses_tr = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(losses_tr, outputs['loss'])
            loss_here = np.nanmean(np.vstack(losses_tr))
        else:
            loss_here = np.nanmean(outputs['loss'])

        self.logger.log('train_losses', loss_here, self.current_epoch)

        if self.verbose_train and 'grad_norm' in outputs and self.local_rank == 0:
            avg_grad_norm = np.nanmean(outputs['grad_norm'])
            self.print_to_log_file(f"  [verbose] avg_grad_norm = {avg_grad_norm:.4f}")

    # ------------------------------------------------------------------ #
    # Training step: pass boundary to loss, compute boundary in FP32     #
    # ------------------------------------------------------------------ #
    def train_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']
        # Optional pre-computed boundary masks (B, C_boundary, Z, Y, X)
        boundary = batch.get('boundary', None)

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [t.to(self.device, non_blocking=True) for t in target]
        else:
            target = target.to(self.device, non_blocking=True)

        if boundary is not None:
            boundary = boundary.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)

        with autocast(self.device.type, enabled=self.use_amp) if self.device.type == 'cuda' else dummy_context():
            output = self.network(data)

        # Compute loss in FP32 to avoid NaN/overflow in boundary term
        with torch.autocast(self.device.type, enabled=False) if self.device.type == 'cuda' else dummy_context():
            # Cast output back to float32 if needed
            if isinstance(output, (list, tuple)):
                output_fp32 = [o.float() for o in output]
                target_fp32 = [t.float() if t.is_floating_point() else t for t in target] if isinstance(target, (list, tuple)) else target
            else:
                output_fp32 = output.float()
                target_fp32 = target

            l = self.loss(output_fp32, target_fp32, boundary)

        # NaN/Inf tolerance: skip optimizer step, raise only after >3 consecutive bad steps
        if not torch.isfinite(l):
            self._nan_step_count += 1
            if self.local_rank == 0:
                self.print_to_log_file(
                    f"WARNING: non-finite loss at epoch {self.current_epoch} "
                    f"(consecutive={self._nan_step_count}): {l.item()}"
                )
            if self._nan_step_count > 3:
                raise RuntimeError(
                    f"Non-finite loss persists for {self._nan_step_count} consecutive steps "
                    f"at epoch {self.current_epoch}: {l.item()}"
                )
            self.optimizer.zero_grad(set_to_none=True)
            return {'loss': np.nan}

        self._nan_step_count = 0

        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.clip_grad)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.clip_grad)
            self.optimizer.step()

        ret = {'loss': l.detach().cpu().numpy()}

        if self.verbose_train:
            ret['grad_norm'] = float(grad_norm)

        return ret

