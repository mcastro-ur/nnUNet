# nnunetv2/training/nnUNetTrainer/variants/loss/nnUNetTrainerV2_BoundaryDiceCE.py
"""
nnUNetTrainerV2_BoundaryDiceCE  — v2
=====================================
Améliorations vs v1 (voir commentaires [FIX-X]) :

  [FIX-1] Schedule sigmoïde au lieu de linéaire
          → transition plus douce, évite les discontinuités de gradient
          → warmup_end=50 → 20 (activation plus tôt)
          → ramp_end=100  → 150 (montée plus progressive)

  [FIX-2] Support dist_maps (EDT exacte) depuis le DataLoader
          → batch['dist_maps'] : (B, C, H, W, D) float16/32, précalculées offline
          → fallback automatique sur batch['boundary'] (morphologique, v1)
          → fallback final : pas de boundary loss si rien n'est disponible

  [FIX-3] Per-class boundary weights
          → petites structures (IOU, canal anal) supervisées plus fortement
          → grandes structures (fémoral) supervisées moins
          → définissable via BOUNDARY_CLASS_WEIGHTS_JSON ou par sous-classe

  [FIX-4] Gradient clipping 5.0 → 1.0
          → réduit le risque de NaN
          → override via env var NNUNET_CLIP_GRAD

  [FIX-5] Logging enrichi : boundary_weight, grad_norm, loss_terms
          → visible dans le training_log.txt à chaque epoch

  Rétrocompatibilité :
  - Si batch ne contient ni 'dist_maps' ni 'boundary' → comportement v1 (boundary=None)
  - Tous les env vars v1 (NNUNET_BOUNDARY_WARMUP_END, etc.) toujours supportés
  - La sous-classe 1500ep fonctionne sans modification
"""

import json
import os

import numpy as np
import torch
from torch import autocast

from nnunetv2.paths import nnUNet_raw
from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.loss_boundary_dice_ce import BoundaryDiceCELoss
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.helpers import dummy_context


# ─────────────────────────────────────────────────────────────────────────────
# Poids boundary par classe (index = label ID dans le dataset)
# Modifie selon ton dataset.json de Dataset092_Prostate26
# Surchargeables dans la sous-classe ou via BOUNDARY_CLASS_WEIGHTS_JSON
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_BOUNDARY_CLASS_WEIGHTS = {
    0:  0.0,   # background — jamais supervisé
    1:  0.5,   # grande structure (ex: Bladder)
    2:  0.5,   # grande structure (ex: Rectum)
    3:  1.5,   # structure moyenne (ex: Canal anal)
    4:  0.3,   # os (ex: Femoral head L) — boundary facile
    5:  0.3,   # os (ex: Femoral head R)
    6:  1.5,   # Prostate — critique pour le pipeline coarse-to-fine
    7:  2.0,   # IOU — très petite, supervision boundary maximale
    8:  1.2,   # Vesicules séminales
    9:  0.8,
    10: 1.0,
}


# ─────────────────────────────────────────────────────────────────────────────
# Deep Supervision Wrapper (identique à v1)
# ─────────────────────────────────────────────────────────────────────────────

class _BoundaryDiceCEWithDS(torch.nn.Module):
    """
    Wraps BoundaryDiceCELoss pour deep supervision.
    La boundary loss est appliquée UNIQUEMENT sur la sortie full-résolution (i=0).
    """

    def __init__(self, loss_fn: BoundaryDiceCELoss, weight_factors: tuple):
        super().__init__()
        self.loss_fn = loss_fn
        self.weight_factors = weight_factors

    def forward(self, net_output, target, boundary=None):
        if isinstance(net_output, (list, tuple)):
            assert isinstance(target, (list, tuple))
            total = torch.tensor(0.0, device=net_output[0].device,
                                 dtype=net_output[0].dtype)
            for i, (out_i, tgt_i, w) in enumerate(zip(net_output, target,
                                                       self.weight_factors)):
                if w == 0:
                    continue
                bnd_i = boundary if i == 0 else None
                total = total + w * self.loss_fn(out_i, tgt_i, bnd_i)
            return total
        else:
            return self.loss_fn(net_output, target, boundary)


# ─────────────────────────────────────────────────────────────────────────────
# Trainer principal
# ─────────────────────────────────────────────────────────────────────────────

class nnUNetTrainerV2_BoundaryDiceCE(nnUNetTrainer):
    """
    nnUNet trainer avec Boundary + Dice + CE loss — v2.

    Hyperparamètres surchargeables dans les sous-classes :
      w_boundary_max        : float — poids max boundary (défaut 0.2)
      w_boundary_warmup_end : int   — epoch début ramp (défaut 20)  [FIX-1]
      w_boundary_ramp_end   : int   — epoch plateau (défaut 150)    [FIX-1]
      w_boundary_schedule   : str   — 'sigmoid' | 'linear'          [FIX-1]
      clip_grad             : float — clipping gradient (défaut 1.0)[FIX-4]
      boundary_class_weights: dict  — poids par classe              [FIX-3]
    """

    # ── Hyperparamètres par défaut ──────────────────────────────────────────
    w_boundary_max:        float = 0.2
    w_boundary_warmup_end: int   = 20    # [FIX-1] était 50
    w_boundary_ramp_end:   int   = 150   # [FIX-1] était 100
    w_boundary_schedule:   str   = 'sigmoid'  # [FIX-1] était implicitement 'linear'

    # ── Init ────────────────────────────────────────────────────────────────
    def __init__(self, plans: dict, configuration: str, fold: int,
                 dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)

        # [FIX-4] clip_grad 5.0 → 1.0
        if not os.environ.get("NNUNET_CLIP_GRAD", "").strip():
            self.clip_grad = 1.0
        else:
            self.clip_grad = float(os.environ["NNUNET_CLIP_GRAD"])

        if not os.environ.get("NNUNET_INITIAL_LR", "").strip():
            self.initial_lr = 1e-3

        # Surcharges via env vars (rétrocompatibilité v1)
        for env, attr in [
            ("NNUNET_BOUNDARY_WARMUP_END", "w_boundary_warmup_end"),
            ("NNUNET_BOUNDARY_RAMP_END",   "w_boundary_ramp_end"),
            ("NNUNET_BOUNDARY_MAX",        "w_boundary_max"),
        ]:
            val = os.environ.get(env, "").strip()
            if val:
                setattr(self, attr, type(getattr(self, attr))(val))

        sched_env = os.environ.get("NNUNET_BOUNDARY_SCHEDULE", "").strip()
        if sched_env:
            self.w_boundary_schedule = sched_env

        # [FIX-3] Poids par classe — surchargeables via JSON ou env var
        self.boundary_class_weights = dict(DEFAULT_BOUNDARY_CLASS_WEIGHTS)
        _cw_json = os.environ.get("BOUNDARY_CLASS_WEIGHTS_JSON", "").strip()
        if _cw_json:
            try:
                overrides = json.loads(_cw_json)
                self.boundary_class_weights.update({int(k): v for k, v in overrides.items()})
            except Exception:
                pass

        # Compteur NaN consécutifs (identique v1)
        self._nan_step_count = 0

        # Pour le logging enrichi [FIX-5]
        self._epoch_boundary_weight = 0.0
        self._step_loss_terms = []

        # [FIX-5] Active verbose_train pour logger grad_norm à chaque epoch
        self.verbose_train = True

    # ── [FIX-2] Chemin vers les dist_maps précalculées ─────────────────────
    # Surcharge dans la sous-classe OU via env var NNUNET_DIST_MAPS_DIR
    # Le chemin est résolu UNE FOIS dans get_dataloaders et passé
    # directement au DataLoader AVANT le fork des workers multiprocessing.
    dist_maps_dir: str = None
    dist_maps_suffix: str = "_boundary_small_distmaps.npy"

    # ── [FIX-2] Override COMPLET de get_dataloaders ─────────────────────────
    # Nécessaire car super() + injection post-fork ne fonctionne pas :
    # les workers multiprocessing sont forkés avec leur propre copie du
    # DataLoader → dist_maps_dir = None dans tous les workers.
    # Solution : passer dist_maps_dir à la CONSTRUCTION de dl_tr.
    def get_dataloaders(self):
        from batchgenerators.dataloading.nondet_multi_threaded_augmenter import NonDetMultiThreadedAugmenter
        from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
        from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA
        from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class

        if self.dataset_class is None:
            self.dataset_class = infer_dataset_class(self.preprocessed_dataset_folder)

        patch_size = self.configuration_manager.patch_size
        deep_supervision_scales = self._get_deep_supervision_scales()

        (
            rotation_for_DA,
            do_dummy_2d_data_aug,
            initial_patch_size,
            mirror_axes,
        ) = self.configure_rotation_dummyDA_mirroring_and_inital_patch_size()

        tr_transforms = self.get_training_transforms(
            patch_size, rotation_for_DA, deep_supervision_scales, mirror_axes, do_dummy_2d_data_aug,
            use_mask_for_norm=self.configuration_manager.use_mask_for_norm,
            is_cascaded=self.is_cascaded, foreground_labels=self.label_manager.foreground_labels,
            regions=self.label_manager.foreground_regions if self.label_manager.has_regions else None,
            ignore_label=self.label_manager.ignore_label)

        val_transforms = self.get_validation_transforms(deep_supervision_scales,
                                                        is_cascaded=self.is_cascaded,
                                                        foreground_labels=self.label_manager.foreground_labels,
                                                        regions=self.label_manager.foreground_regions if
                                                        self.label_manager.has_regions else None,
                                                        ignore_label=self.label_manager.ignore_label)

        dataset_tr, dataset_val = self.get_tr_and_val_datasets()

        # ── Résolution du chemin dist_maps_dir ───────────────────────────────
        # Priorité : env var > attribut de classe > auto-détection depuis nnUNet_raw
        _dist_maps_dir = os.environ.get("NNUNET_DIST_MAPS_DIR", "").strip() or self.dist_maps_dir

        if _dist_maps_dir is None and nnUNet_raw is not None:
            dataset_name = self.plans_manager.dataset_name
            _auto = os.path.join(nnUNet_raw, dataset_name, dataset_name, "boundaryTr")
            if os.path.isdir(_auto):
                _dist_maps_dir = _auto

        # Validation + log
        if _dist_maps_dir is not None:
            if not os.path.isdir(_dist_maps_dir):
                if self.local_rank == 0:
                    self.print_to_log_file(
                        f"[BoundaryDiceCE] ATTENTION dist_maps_dir introuvable: {_dist_maps_dir} → fallback"
                    )
                _dist_maps_dir = None
            else:
                n = sum(1 for f in os.listdir(_dist_maps_dir) if f.endswith(self.dist_maps_suffix))
                if self.local_rank == 0:
                    self.print_to_log_file(
                        f"[BoundaryDiceCE] dist_maps_dir='{_dist_maps_dir}' "
                        f"({n} fichiers *{self.dist_maps_suffix})"
                    )
                if n == 0:
                    if self.local_rank == 0:
                        self.print_to_log_file(
                            "[BoundaryDiceCE] ATTENTION 0 fichiers _distmaps.npy → "
                            "lance precompute_distance_maps.py d'abord"
                        )
                    _dist_maps_dir = None
        else:
            if self.local_rank == 0:
                self.print_to_log_file(
                    "[BoundaryDiceCE] dist_maps_dir=None → boundary loss morphologique (fallback)"
                )

        # ── Création dl_tr avec dist_maps_dir passé directement ──────────────
        # C'est ici que ça doit se faire — AVANT le fork des workers
        dl_tr = nnUNetDataLoader(
            dataset_tr, self.batch_size,
            initial_patch_size,
            self.configuration_manager.patch_size,
            self.label_manager,
            oversample_foreground_percent=self.oversample_foreground_percent,
            sampling_probabilities=None, pad_sides=None, transforms=tr_transforms,
            probabilistic_oversampling=self.probabilistic_oversampling,
            dist_maps_dir=_dist_maps_dir,           # ← [FIX-2] passé ici
            dist_maps_suffix=self.dist_maps_suffix,  # ← [FIX-2] passé ici
        )

        # dl_val sans dist_maps (pas besoin en validation)
        dl_val = nnUNetDataLoader(
            dataset_val, self.batch_size,
            self.configuration_manager.patch_size,
            self.configuration_manager.patch_size,
            self.label_manager,
            oversample_foreground_percent=self.oversample_foreground_percent,
            sampling_probabilities=None, pad_sides=None, transforms=val_transforms,
            probabilistic_oversampling=self.probabilistic_oversampling,
        )

        allowed_num_processes = get_allowed_n_proc_DA()
        if allowed_num_processes == 0:
            mt_gen_train = SingleThreadedAugmenter(dl_tr, None)
            mt_gen_val   = SingleThreadedAugmenter(dl_val, None)
        else:
            mt_gen_train = NonDetMultiThreadedAugmenter(
                data_loader=dl_tr, transform=None,
                num_processes=allowed_num_processes,
                num_cached=max(6, allowed_num_processes // 2),
                seeds=None, pin_memory=self.device.type == 'cuda', wait_time=0.002)
            mt_gen_val = NonDetMultiThreadedAugmenter(
                data_loader=dl_val, transform=None,
                num_processes=max(1, allowed_num_processes // 2),
                num_cached=max(3, allowed_num_processes // 4),
                seeds=None, pin_memory=self.device.type == 'cuda', wait_time=0.002)

        _ = next(mt_gen_train)
        _ = next(mt_gen_val)
        return mt_gen_train, mt_gen_val

        # ── torch.compile désactivé (identique v1) ──────────────────────────────
    def _do_i_compile(self) -> bool:
        return False

    # ── Construction de la loss (identique v1 sur l'interface externe) ──────
    def _build_loss(self):
        ignore_label = self.label_manager.ignore_label

        loss_fn = BoundaryDiceCELoss(
            weight_boundary=0.0,   # mis à jour chaque epoch via on_train_epoch_start
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
            loss = loss_fn

        self._boundary_loss_fn = loss_fn
        return loss

    # ── [FIX-1] Schedule boundary weight ────────────────────────────────────
    def _get_boundary_weight(self, epoch: int) -> float:
        """
        Calcule le poids boundary pour l'epoch courante.

        sigmoid (défaut) : transition douce, évite les discontinuités
        linear            : comportement v1 (gardé pour rétrocompat)
        """
        if epoch < self.w_boundary_warmup_end:
            return 0.0
        if epoch >= self.w_boundary_ramp_end:
            return self.w_boundary_max

        ramp_len = self.w_boundary_ramp_end - self.w_boundary_warmup_end
        progress = (epoch - self.w_boundary_warmup_end) / max(ramp_len, 1)

        if self.w_boundary_schedule == 'sigmoid':
            # Sigmoïde centrée sur le milieu du ramp-up
            x = (progress - 0.5) * 10.0
            return float(self.w_boundary_max / (1.0 + np.exp(-x)))
        else:
            # linear (v1 behavior)
            return float(progress * self.w_boundary_max)

    def on_train_epoch_start(self):
        super().on_train_epoch_start()
        w = self._get_boundary_weight(self.current_epoch)
        self._boundary_loss_fn.weight_boundary = w
        self._epoch_boundary_weight = w
        self._step_loss_terms = []
        if self.local_rank == 0:
            self.print_to_log_file(
                f"  [BoundaryDiceCE] epoch={self.current_epoch} "
                f"w_boundary={w:.4f} schedule={self.w_boundary_schedule}"
            )

    # ── [FIX-2] Préparation de la boundary depuis dist_maps ou boundary ─────
    def _get_boundary_from_batch(self, batch: dict,
                                 output_shape: torch.Size,
                                 device: torch.device) -> torch.Tensor | None:
        """
        Priorité :
          1. batch['dist_maps']  → EDT exacte (précalculée offline) [FIX-2]
          2. batch['boundary']   → masque morphologique (v1, rétrocompat)
          3. None                → pas de boundary loss ce step

        Les dist_maps (B, C, H, W, D) float16 sont convertis en boundary
        binaire : voxels avec |d| < 2mm = surface de la structure.
        Le per-class weighting [FIX-3] est appliqué ici.
        """
        # ── Cas 1 : dist_maps EDT précalculées ──────────────────────────────
        if 'dist_maps' in batch:
            dist_maps = batch['dist_maps'].to(device=device, dtype=torch.float32,
                                              non_blocking=True)

            # ── Center-crop dist_maps → patch_size du réseau ─────────────────
            # Les dist_maps sont à initial_patch_size (patch avant augmentation,
            # ex: [128, 160, 205]).  Le réseau sort patch_size ([128, 160, 128]).
            # Les transforms nnUNet croppent data/seg mais PAS dist_maps.
            # → On center-crop ici pour aligner sur output_shape.
            # output_shape = (B, num_classes, Z, Y, X) = shape de la sortie réseau
            target_spatial = output_shape[2:]  # (Z, Y, X) = patch_size réel
            current_spatial = dist_maps.shape[2:]  # (Z, Y, X) = initial_patch_size

            if current_spatial != target_spatial:
                crops = []
                for dim_size, tgt_size in zip(current_spatial, target_spatial):
                    start = (dim_size - tgt_size) // 2
                    crops.append(slice(start, start + tgt_size))
                dist_maps = dist_maps[:, :, crops[0], crops[1], crops[2]]

            # B, C, H, W, D après crop
            B, C = dist_maps.shape[:2]

            # Construit le tenseur de boundary pondéré
            # boundary[b, c] = class_weight si |dist[b,c]| < threshold, 0 sinon
            boundary = torch.zeros_like(dist_maps)
            threshold = 2.0  # mm — voxels dans les 2mm de la surface

            for c in range(1, C):  # skip background
                w_c = self.boundary_class_weights.get(c, 1.0)
                if w_c == 0.0:
                    continue
                surface_mask = (dist_maps[:, c].abs() < threshold)
                boundary[:, c] = surface_mask.float() * w_c

            return boundary

        # ── Cas 2 : boundary morphologique v1 (rétrocompat) ─────────────────
        if 'boundary' in batch:
            return batch['boundary'].to(device=device, non_blocking=True)

        # ── Cas 3 : pas de boundary disponible ──────────────────────────────
        return None

    # ── Train step ───────────────────────────────────────────────────────────
    def train_step(self, batch: dict) -> dict:
        data   = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [t.to(self.device, non_blocking=True) for t in target]
        else:
            target = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)

        # Forward pass sous AMP
        with (autocast(self.device.type, enabled=self.use_amp)
              if self.device.type == 'cuda' else dummy_context()):
            output = self.network(data)

        # [FIX-2] Récupère la boundary (EDT ou morphologique ou None)
        # Fait APRÈS le forward pour ne pas pénaliser la mémoire GPU inutilement
        # quand w_boundary=0 (warmup)
        boundary = None
        if self._epoch_boundary_weight > 0:
            output_ref = output[0] if isinstance(output, (list, tuple)) else output
            boundary = self._get_boundary_from_batch(batch, output_ref.shape, self.device)

        # Calcul de la loss en FP32 (évite NaN/overflow dans BoundaryLoss)
        with (torch.autocast(self.device.type, enabled=False)
              if self.device.type == 'cuda' else dummy_context()):
            if isinstance(output, (list, tuple)):
                output_fp32 = [o.float() for o in output]
                target_fp32 = (
                    [t.float() if t.is_floating_point() else t for t in target]
                    if isinstance(target, (list, tuple)) else target
                )
            else:
                output_fp32 = output.float()
                target_fp32 = target

            l = self.loss(output_fp32, target_fp32, boundary)

        # NaN tolerance (identique v1 mais avec meilleur message)
        if not torch.isfinite(l):
            self._nan_step_count += 1
            if self.local_rank == 0:
                self.print_to_log_file(
                    f"WARNING: non-finite loss={l.item():.4f} "
                    f"epoch={self.current_epoch} "
                    f"w_boundary={self._epoch_boundary_weight:.4f} "
                    f"consecutive={self._nan_step_count}"
                )
            if self._nan_step_count > 3:
                raise RuntimeError(
                    f"Non-finite loss persiste {self._nan_step_count} steps "
                    f"(epoch={self.current_epoch})"
                )
            self.optimizer.zero_grad(set_to_none=True)
            return {'loss': np.nan}

        self._nan_step_count = 0

        # Backprop + [FIX-4] clip_grad=1.0
        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.network.parameters(), self.clip_grad
            )
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.network.parameters(), self.clip_grad
            )
            self.optimizer.step()

        ret = {'loss': l.detach().cpu().numpy()}
        if self.verbose_train:
            ret['grad_norm'] = float(grad_norm)

        # [FIX-5] Log si grad_norm élevée (précurseur de NaN)
        if float(grad_norm) > 3.0 and self.local_rank == 0:
            self.print_to_log_file(
                f"  [WARN] grad_norm={float(grad_norm):.2f} > 3.0 "
                f"at epoch {self.current_epoch}"
            )

        return ret

    # ── [FIX-5] Logging enrichi en fin d'epoch ───────────────────────────────
    def on_train_epoch_end(self, train_outputs):
        """Override pour utiliser nanmean (steps NaN ignorés) + log enrichi."""
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

        # [FIX-5] Log boundary weight + grad_norm moyenne
        if self.local_rank == 0:
            msg_parts = [
                f"  [BoundaryDiceCE] epoch={self.current_epoch}",
                f"loss={loss_here:.4f}",
                f"w_boundary={self._epoch_boundary_weight:.4f}",
            ]
            if 'grad_norm' in outputs:
                avg_gn = np.nanmean(outputs['grad_norm'])
                msg_parts.append(f"avg_grad_norm={avg_gn:.3f}")
                # Alerte si grad_norm en hausse (signe d'instabilité)
                if avg_gn > 2.0:
                    msg_parts.append("← ATTENTION grad élevée")
            self.print_to_log_file("  " + "  |  ".join(msg_parts))
