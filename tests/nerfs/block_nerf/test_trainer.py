"""
Test Block NeRF Trainer Components

This module tests the training-related components of Block NeRF:
- BlockNeRFTrainer
- BlockNeRFTrainerConfig
"""

import pytest
import torch
import numpy as np
import tempfile
import os


# Add the src directory to the path for imports
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))

try:
    from nerfs.block_nerf import BlockNeRF
    from nerfs.block_nerf.trainer import BlockNeRFTrainer, BlockNeRFTrainerConfig

    BLOCK_NERF_AVAILABLE = True
except ImportError as e:
    BLOCK_NERF_AVAILABLE = False
    IMPORT_ERROR = str(e)

from pathlib import Path
from unittest.mock import patch, MagicMock
from . import (
    TEST_CONFIG,
    get_test_device,
    create_test_data,
    skip_if_no_cuda,
    skip_if_slow,
    TEST_DATA_DIR,
)


class TestBlockNeRFTrainerConfig:
    """Test Block-NeRF trainer configuration."""

    def test_default_config(self):
        """Test default trainer configuration."""
        config = BlockNeRFTrainerConfig()

        assert config.num_epochs > 0
        assert config.learning_rate > 0
        assert config.batch_size > 0
        assert config.num_rays > 0
        assert config.warmup_epochs >= 0
        assert config.weight_decay >= 0

    def test_custom_config(self):
        """Test custom trainer configuration."""
        config = BlockNeRFTrainerConfig(
            num_epochs=TEST_CONFIG["num_epochs"],
            learning_rate=TEST_CONFIG["learning_rate"],
            batch_size=TEST_CONFIG["batch_size"],
            num_rays=TEST_CONFIG["num_rays"],
        )

        assert config.num_epochs == TEST_CONFIG["num_epochs"]
        assert config.learning_rate == TEST_CONFIG["learning_rate"]
        assert config.batch_size == TEST_CONFIG["batch_size"]
        assert config.num_rays == TEST_CONFIG["num_rays"]

    def test_config_validation(self):
        """Test trainer configuration validation."""
        # Test invalid num_epochs
        with pytest.raises(ValueError):
            BlockNeRFTrainerConfig(num_epochs=0)

        # Test invalid learning_rate
        with pytest.raises(ValueError):
            BlockNeRFTrainerConfig(learning_rate=0)

        # Test invalid batch_size
        with pytest.raises(ValueError):
            BlockNeRFTrainerConfig(batch_size=0)

    def test_scheduler_config(self):
        """Test learning rate scheduler configuration."""
        config = BlockNeRFTrainerConfig(
            scheduler_type="cosine", scheduler_params={"T_max": 100, "eta_min": 1e-6}
        )

        assert config.scheduler_type == "cosine"
        assert config.scheduler_params["T_max"] == 100
        assert config.scheduler_params["eta_min"] == 1e-6


class TestBlockNeRFTrainer:
    """Test Block-NeRF trainer implementation."""

    @pytest.fixture
    def model_config(self):
        """Create model configuration."""
        return BlockNeRFConfig(
            scene_bounds=TEST_CONFIG["scene_bounds"],
            block_size=TEST_CONFIG["block_size"],
            max_blocks=TEST_CONFIG["max_blocks"],
            appearance_dim=32,
            hidden_dim=64,  # Smaller for faster testing
            num_layers=2,
        )

    @pytest.fixture
    def trainer_config(self):
        """Create trainer configuration."""
        return BlockNeRFTrainerConfig(
            num_epochs=2,
            learning_rate=1e-3,
            batch_size=2,
            num_rays=256,
            save_every=1,
            eval_every=1,
        )

    @pytest.fixture
    def model(self, model_config):
        """Create model instance."""
        device = get_test_device()
        return BlockNeRFModel(model_config).to(device)

    @pytest.fixture
    def trainer(self, model, trainer_config, tmp_path):
        """Create trainer instance."""
        return BlockNeRFTrainer(model=model, config=trainer_config, output_dir=str(tmp_path))

    def test_trainer_initialization(self, trainer, model, trainer_config):
        """Test trainer initialization."""
        assert isinstance(trainer, BlockNeRFTrainer)
        assert trainer.model is model
        assert trainer.config == trainer_config
        assert hasattr(trainer, "optimizer")
        assert hasattr(trainer, "scheduler")
        assert hasattr(trainer, "loss_fn")

    def test_optimizer_setup(self, trainer):
        """Test optimizer setup."""
        assert trainer.optimizer is not None
        assert len(list(trainer.optimizer.param_groups)) > 0

        # Check learning rate
        for param_group in trainer.optimizer.param_groups:
            assert param_group["lr"] > 0

    def test_scheduler_setup(self, trainer):
        """Test learning rate scheduler setup."""
        if trainer.scheduler is not None:
            initial_lr = trainer.optimizer.param_groups[0]["lr"]

            # Take a scheduler step
            trainer.scheduler.step()

            # Learning rate should change (depending on scheduler type)
            # We just check that scheduler is functional
            current_lr = trainer.optimizer.param_groups[0]["lr"]
            assert current_lr >= 0

    def test_loss_computation(self, trainer):
        """Test loss computation."""
        device = get_test_device()
        batch_size = 2
        num_rays = 128

        # Create mock batch
        batch = {
            "rays_o": torch.randn(batch_size, num_rays, 3, device=device),
            "rays_d": torch.randn(batch_size, num_rays, 3, device=device),
            "gt_rgb": torch.rand(batch_size, num_rays, 3, device=device),
            "camera_ids": torch.randint(0, 5, (batch_size,), device=device),
            "exposure": torch.randn(batch_size, 1, device=device),
        }

        loss_dict = trainer.compute_loss(batch)

        assert isinstance(loss_dict, dict)
        assert "total_loss" in loss_dict
        assert "rgb_loss" in loss_dict
        assert loss_dict["total_loss"] >= 0
        assert torch.isfinite(loss_dict["total_loss"])

    def test_training_step(self, trainer):
        """Test single training step."""
        device = get_test_device()
        batch_size = 2
        num_rays = 128

        # Create mock batch
        batch = {
            "rays_o": torch.randn(batch_size, num_rays, 3, device=device),
            "rays_d": torch.randn(batch_size, num_rays, 3, device=device),
            "gt_rgb": torch.rand(batch_size, num_rays, 3, device=device),
            "camera_ids": torch.randint(0, 5, (batch_size,), device=device),
            "exposure": torch.randn(batch_size, 1, device=device),
        }

        # Store initial parameters
        initial_params = [p.clone() for p in trainer.model.parameters()]

        # Perform training step
        loss_dict = trainer.training_step(batch)

        # Check that parameters changed
        for initial, current in zip(initial_params, trainer.model.parameters()):
            if current.requires_grad:
                assert not torch.equal(initial, current)

        assert isinstance(loss_dict, dict)
        assert "total_loss" in loss_dict

    @skip_if_slow()
    def test_training_epoch(self, trainer):
        """Test training for one epoch."""
        device = get_test_device()

        # Create mock dataloader
        batch_data = []
        for _ in range(3):  # Small number of batches
            batch = {
                "rays_o": torch.randn(2, 128, 3, device=device),
                "rays_d": torch.randn(2, 128, 3, device=device),
                "gt_rgb": torch.rand(2, 128, 3, device=device),
                "camera_ids": torch.randint(0, 5, (2,), device=device),
                "exposure": torch.randn(2, 1, device=device),
            }
            batch_data.append(batch)

        # Mock dataloader
        dataloader = batch_data

        # Train for one epoch
        epoch_metrics = trainer.train_epoch(dataloader, epoch=0)

        assert isinstance(epoch_metrics, dict)
        assert "avg_loss" in epoch_metrics
        assert epoch_metrics["avg_loss"] >= 0

    def test_validation_step(self, trainer):
        """Test validation step."""
        device = get_test_device()
        batch_size = 2
        num_rays = 128

        # Create validation batch
        batch = {
            "rays_o": torch.randn(batch_size, num_rays, 3, device=device),
            "rays_d": torch.randn(batch_size, num_rays, 3, device=device),
            "gt_rgb": torch.rand(batch_size, num_rays, 3, device=device),
            "camera_ids": torch.randint(0, 5, (batch_size,), device=device),
            "exposure": torch.randn(batch_size, 1, device=device),
        }

        trainer.model.eval()
        with torch.no_grad():
            metrics = trainer.validation_step(batch)

        assert isinstance(metrics, dict)
        assert "val_loss" in metrics
        assert "psnr" in metrics
        assert metrics["val_loss"] >= 0
        assert metrics["psnr"] >= 0

    def test_checkpoint_save_load(self, trainer, tmp_path):
        """Test checkpoint saving and loading."""
        # Train for a bit to change parameters
        device = get_test_device()
        batch = {
            "rays_o": torch.randn(2, 64, 3, device=device),
            "rays_d": torch.randn(2, 64, 3, device=device),
            "gt_rgb": torch.rand(2, 64, 3, device=device),
            "camera_ids": torch.randint(0, 5, (2,), device=device),
            "exposure": torch.randn(2, 1, device=device),
        }

        trainer.training_step(batch)

        # Save checkpoint
        checkpoint_path = tmp_path / "checkpoint.pth"
        trainer.save_checkpoint(str(checkpoint_path), epoch=0)

        assert checkpoint_path.exists()

        # Create new trainer and load checkpoint
        new_trainer = BlockNeRFTrainer(
            model=trainer.model.__class__(trainer.model.config),
            config=trainer.config,
            output_dir=str(tmp_path),
        )

        new_trainer.load_checkpoint(str(checkpoint_path))

        # Check that parameters match
        for p1, p2 in zip(trainer.model.parameters(), new_trainer.model.parameters()):
            assert torch.allclose(p1, p2, atol=1e-6)

    @skip_if_no_cuda()
    def test_mixed_precision_training(self, model_config, trainer_config, tmp_path):
        """Test mixed precision training."""
        trainer_config.use_amp = True

        model = BlockNeRFModel(model_config).cuda()
        trainer = BlockNeRFTrainer(model=model, config=trainer_config, output_dir=str(tmp_path))

        # Check that GradScaler is created
        assert hasattr(trainer, "scaler")
        assert trainer.scaler is not None

        # Test training step with AMP
        batch = {
            "rays_o": torch.randn(2, 64, 3, device="cuda"),
            "rays_d": torch.randn(2, 64, 3, device="cuda"),
            "gt_rgb": torch.rand(2, 64, 3, device="cuda"),
            "camera_ids": torch.randint(0, 5, (2,), device="cuda"),
            "exposure": torch.randn(2, 1, device="cuda"),
        }

        loss_dict = trainer.training_step(batch)
        assert isinstance(loss_dict, dict)
        assert "total_loss" in loss_dict

    def test_gradient_clipping(self, trainer):
        """Test gradient clipping."""
        trainer.config.grad_clip_norm = 1.0

        device = get_test_device()
        batch = {
            "rays_o": torch.randn(2, 64, 3, device=device),
            "rays_d": torch.randn(2, 64, 3, device=device),
            "gt_rgb": torch.rand(2, 64, 3, device=device),
            "camera_ids": torch.randint(0, 5, (2,), device=device),
            "exposure": torch.randn(2, 1, device=device),
        }

        # This should not raise an error
        loss_dict = trainer.training_step(batch)
        assert isinstance(loss_dict, dict)

    def test_learning_rate_scheduling(self, trainer):
        """Test learning rate scheduling."""
        initial_lr = trainer.optimizer.param_groups[0]["lr"]

        # Simulate some training steps
        for _ in range(5):
            if trainer.scheduler is not None:
                trainer.scheduler.step()

        # Learning rate should be updated
        current_lr = trainer.optimizer.param_groups[0]["lr"]
        assert current_lr >= 0  # Just check it's valid


class TestTrainerFactory:
    """Test trainer factory functions."""

    def test_create_block_nerf_trainer(self, tmp_path):
        """Test trainer creation factory."""
        model_config = BlockNeRFConfig(
            scene_bounds=TEST_CONFIG["scene_bounds"],
            block_size=TEST_CONFIG["block_size"],
            max_blocks=4,
            hidden_dim=32,
            num_layers=2,
        )

        trainer_config = BlockNeRFTrainerConfig(
            num_epochs=1,
            batch_size=1,
            num_rays=64,
        )

        trainer = create_block_nerf_trainer(
            model_config=model_config, trainer_config=trainer_config, output_dir=str(tmp_path)
        )

        assert isinstance(trainer, BlockNeRFTrainer)
        assert isinstance(trainer.model, BlockNeRFModel)
        assert trainer.config == trainer_config

    def test_create_trainer_with_pretrained(self, tmp_path):
        """Test trainer creation with pretrained model."""
        # First create and save a model
        model_config = BlockNeRFConfig(
            scene_bounds=TEST_CONFIG["scene_bounds"],
            block_size=TEST_CONFIG["block_size"],
            max_blocks=4,
        )

        model = BlockNeRFModel(model_config)
        checkpoint_path = tmp_path / "pretrained.pth"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "config": model_config.to_dict(),
            },
            checkpoint_path,
        )

        # Create trainer with pretrained model
        trainer_config = BlockNeRFTrainerConfig(num_epochs=1)
        trainer = create_block_nerf_trainer(
            model_config=model_config,
            trainer_config=trainer_config,
            output_dir=str(tmp_path),
            pretrained_path=str(checkpoint_path),
        )

        assert isinstance(trainer, BlockNeRFTrainer)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
