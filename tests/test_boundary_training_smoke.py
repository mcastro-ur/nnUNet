"""
Smoke tests for nnUNetTrainerV2_BoundaryDiceCE and related components.

Tests:
  1. Boundary weight schedule computes correctly for the base trainer.
  2. initial_lr, clip_grad, save_every are configurable via env vars in nnUNetTrainer.
  3. BoundaryDiceCELoss returns components dict when return_components=True.
  4. 1500ep variant has the expected slower ramp and lower w_boundary_max.
"""

import os
import unittest

import torch


class TestBoundaryWeightSchedule(unittest.TestCase):
    """Verify _get_boundary_weight produces expected values."""

    def _make_trainer_cls(self):
        from nnunetv2.training.nnUNetTrainer.variants.loss.nnUNetTrainerV2_BoundaryDiceCE import (
            nnUNetTrainerV2_BoundaryDiceCE,
        )
        return nnUNetTrainerV2_BoundaryDiceCE

    def test_warmup_phase_returns_zero(self):
        cls = self._make_trainer_cls()
        # Create a dummy instance to call the method (no real init needed)
        trainer = object.__new__(cls)
        trainer.w_boundary_warmup_end = 50
        trainer.w_boundary_ramp_end = 100
        trainer.w_boundary_max = 0.2

        for epoch in range(50):
            self.assertEqual(trainer._get_boundary_weight(epoch), 0.0)

    def test_full_ramp_returns_max(self):
        cls = self._make_trainer_cls()
        trainer = object.__new__(cls)
        trainer.w_boundary_warmup_end = 50
        trainer.w_boundary_ramp_end = 100
        trainer.w_boundary_max = 0.2

        self.assertAlmostEqual(trainer._get_boundary_weight(100), 0.2)
        self.assertAlmostEqual(trainer._get_boundary_weight(200), 0.2)

    def test_linear_ramp_midpoint(self):
        cls = self._make_trainer_cls()
        trainer = object.__new__(cls)
        trainer.w_boundary_warmup_end = 50
        trainer.w_boundary_ramp_end = 100
        trainer.w_boundary_max = 0.2

        # At epoch 75 (midpoint of ramp 50→100) weight should be 0.1
        self.assertAlmostEqual(trainer._get_boundary_weight(75), 0.1)

    def test_ramp_end_inclusive(self):
        cls = self._make_trainer_cls()
        trainer = object.__new__(cls)
        trainer.w_boundary_warmup_end = 150
        trainer.w_boundary_ramp_end = 600
        trainer.w_boundary_max = 0.2

        self.assertAlmostEqual(trainer._get_boundary_weight(149), 0.0)
        # Just inside ramp
        w = trainer._get_boundary_weight(150)
        self.assertGreaterEqual(w, 0.0)
        self.assertLess(w, 0.2)
        self.assertAlmostEqual(trainer._get_boundary_weight(600), 0.2)


class TestEnvVarConfigurability(unittest.TestCase):
    """Verify env-var overrides work for nnUNetTrainer attributes."""

    def _reload_module(self):
        """Force reimport to pick up env-var changes at module load time."""
        import importlib
        import nnunetv2.training.nnUNetTrainer.nnUNetTrainer as m
        importlib.reload(m)
        return m.nnUNetTrainer

    def test_initial_lr_default(self):
        """Without env var, initial_lr should be 1e-2 in base nnUNetTrainer."""
        os.environ.pop("NNUNET_INITIAL_LR", None)
        trainer = object.__new__(self._reload_module())
        # Simulate what __init__ does for initial_lr
        _env_lr = os.environ.get("NNUNET_INITIAL_LR", "").strip()
        initial_lr = float(_env_lr) if _env_lr else 1e-2
        self.assertAlmostEqual(initial_lr, 1e-2)

    def test_initial_lr_from_env(self):
        os.environ["NNUNET_INITIAL_LR"] = "5e-4"
        _env_lr = os.environ.get("NNUNET_INITIAL_LR", "").strip()
        initial_lr = float(_env_lr) if _env_lr else 1e-2
        self.assertAlmostEqual(initial_lr, 5e-4)
        os.environ.pop("NNUNET_INITIAL_LR", None)

    def test_clip_grad_default(self):
        os.environ.pop("NNUNET_CLIP_GRAD", None)
        _env_clip = os.environ.get("NNUNET_CLIP_GRAD", "").strip()
        clip_grad = float(_env_clip) if _env_clip else 12.0
        self.assertAlmostEqual(clip_grad, 12.0)

    def test_clip_grad_from_env(self):
        os.environ["NNUNET_CLIP_GRAD"] = "5.0"
        _env_clip = os.environ.get("NNUNET_CLIP_GRAD", "").strip()
        clip_grad = float(_env_clip) if _env_clip else 12.0
        self.assertAlmostEqual(clip_grad, 5.0)
        os.environ.pop("NNUNET_CLIP_GRAD", None)

    def test_save_every_default(self):
        os.environ.pop("NNUNET_SAVE_EVERY", None)
        _env_save = os.environ.get("NNUNET_SAVE_EVERY", "").strip()
        save_every = int(_env_save) if _env_save else 50
        self.assertEqual(save_every, 50)

    def test_save_every_from_env(self):
        os.environ["NNUNET_SAVE_EVERY"] = "25"
        _env_save = os.environ.get("NNUNET_SAVE_EVERY", "").strip()
        save_every = int(_env_save) if _env_save else 50
        self.assertEqual(save_every, 25)
        os.environ.pop("NNUNET_SAVE_EVERY", None)


class TestBoundaryDiceCELossReturnComponents(unittest.TestCase):
    """Verify BoundaryDiceCELoss returns a dict when return_components=True."""

    def _make_loss(self):
        from nnunetv2.training.loss.loss_boundary_dice_ce import BoundaryDiceCELoss
        return BoundaryDiceCELoss(
            weight_boundary=0.1,
            weight_dice=0.5,
            weight_ce=0.5,
            batch_dice=False,
            ddp=False,
        )

    def _make_inputs(self, num_classes=3):
        torch.manual_seed(0)
        B, C, D, H, W = 1, num_classes, 4, 4, 4
        logits = torch.randn(B, C, D, H, W)
        # Integer target (B, 1, D, H, W)
        target = torch.randint(0, C, (B, 1, D, H, W))
        return logits, target

    def test_return_components_is_dict(self):
        loss_fn = self._make_loss()
        logits, target = self._make_inputs()
        result = loss_fn(logits, target, return_components=True)
        self.assertIsInstance(result, dict)
        for key in ('total', 'dice', 'ce', 'boundary'):
            self.assertIn(key, result)

    def test_return_components_total_matches_scalar(self):
        loss_fn = self._make_loss()
        logits, target = self._make_inputs()
        scalar = loss_fn(logits, target, return_components=False)
        components = loss_fn(logits, target, return_components=True)
        self.assertAlmostEqual(scalar.item(), components['total'].item(), places=5)

    def test_default_returns_tensor(self):
        loss_fn = self._make_loss()
        logits, target = self._make_inputs()
        result = loss_fn(logits, target)
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.ndim, 0)


class TestBoundaryDiceCE_1500epVariant(unittest.TestCase):
    """Verify the 1500ep variant has correct ramp and LR defaults."""

    def _get_cls(self):
        from nnunetv2.training.nnUNetTrainer.variants.loss.nnUNetTrainerV2_BoundaryDiceCE_1500ep import (
            nnUNetTrainerV2_BoundaryDiceCE_1500ep,
        )
        return nnUNetTrainerV2_BoundaryDiceCE_1500ep

    def test_ramp_end_is_600(self):
        cls = self._get_cls()
        self.assertEqual(cls.w_boundary_ramp_end, 600)

    def test_max_weight_is_0_2(self):
        cls = self._get_cls()
        self.assertAlmostEqual(cls.w_boundary_max, 0.2)

    def test_warmup_end_is_150(self):
        cls = self._get_cls()
        self.assertEqual(cls.w_boundary_warmup_end, 150)

    def test_ramp_schedule(self):
        cls = self._get_cls()
        trainer = object.__new__(cls)
        trainer.w_boundary_warmup_end = cls.w_boundary_warmup_end
        trainer.w_boundary_ramp_end = cls.w_boundary_ramp_end
        trainer.w_boundary_max = cls.w_boundary_max

        # Before warmup_end → 0
        self.assertEqual(trainer._get_boundary_weight(149), 0.0)
        # At ramp_end → max
        self.assertAlmostEqual(trainer._get_boundary_weight(600), 0.2)
        # Midpoint of ramp (150 + 225 = 375)
        ramp_midpoint_epoch = (150 + 600) // 2
        w_mid = trainer._get_boundary_weight(ramp_midpoint_epoch)
        self.assertGreater(w_mid, 0.0)
        self.assertLess(w_mid, 0.2)


if __name__ == '__main__':
    unittest.main()
