"""
Test Mega-NeRF Trainer Module

This module tests the Mega-NeRF trainer components including:
- MegaNeRFTrainerConfig
- MegaNeRFTrainer
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import tempfile
import shutil

from src.nerfs.mega_nerf.trainer import MegaNeRFTrainerConfig, MegaNeRFTrainer


class TestMegaNeRFTrainerConfig:
    """Test MegaNeRFTrainerConfig class."""

    def test_default_config(self):
        """Test default configuration creation."""
        config = MegaNeRFTrainerConfig()

        assert config.num_epochs == 100
        assert config.batch_size == 1024
        assert config.learning_rate == 5e-4
        assert config.log_interval == 100
        assert config.val_interval == 500
        assert config.save_interval == 1000
        assert config.checkpoint_dir == "checkpoints"
        assert config.use_mixed_precision is True
        assert config.gradient_clip_val is None

    def test_custom_config(self):
        """Test custom configuration creation."""
        config = MegaNeRFTrainerConfig(
            num_epochs=50,
            batch_size=512,
            learning_rate=1e-3,
            log_interval=50,
            val_interval=200,
            save_interval=500,
            checkpoint_dir="custom_checkpoints",
            use_mixed_precision=False,
            gradient_clip_val=1.0,
        )

        assert config.num_epochs == 50
        assert config.batch_size == 512
        assert config.learning_rate == 1e-3
        assert config.log_interval == 50
        assert config.val_interval == 200
        assert config.save_interval == 500
        assert config.checkpoint_dir == "custom_checkpoints"
        assert config.use_mixed_precision is False
        assert config.gradient_clip_val == 1.0

    def test_validation(self):
        """Test configuration validation."""
        # Test invalid num_epochs
        with pytest.raises(ValueError, match="num_epochs must be positive"):
            MegaNeRFTrainerConfig(num_epochs=0)

        # Test invalid batch_size
        with pytest.raises(ValueError, match="batch_size must be positive"):
            MegaNeRFTrainerConfig(batch_size=0)

        # Test invalid learning_rate
        with pytest.raises(ValueError, match="learning_rate must be positive"):
            MegaNeRFTrainerConfig(learning_rate=0)

        # Test invalid gradient_clip_val
        with pytest.raises(ValueError, match="gradient_clip_val must be positive"):
            MegaNeRFTrainerConfig(gradient_clip_val=0)


class TestMegaNeRFTrainer:
    """Test MegaNeRFTrainer class."""

    def test_initialization(self, model, trainer_config, device):
        """Test trainer initialization."""
        trainer = MegaNeRFTrainer(model, trainer_config)
        trainer = trainer.to(device)

        assert trainer.model == model
        assert trainer.config == trainer_config
        assert trainer.device == device
        assert hasattr(trainer, "optimizer")
        assert hasattr(trainer, "scheduler")

    def test_optimizer_initialization(self, model, trainer_config, device):
        """Test optimizer initialization."""
        trainer = MegaNeRFTrainer(model, trainer_config)
        trainer = trainer.to(device)

        assert trainer.optimizer is not None
        assert len(trainer.optimizer.param_groups) > 0

        # Check learning rate
        for param_group in trainer.optimizer.param_groups:
            assert param_group["lr"] == trainer_config.learning_rate

    def test_scheduler_initialization(self, model, trainer_config, device):
        """Test scheduler initialization."""
        trainer = MegaNeRFTrainer(model, trainer_config)
        trainer = trainer.to(device)

        assert trainer.scheduler is not None

    def test_mixed_precision_setup(self, model, trainer_config, device):
        """Test mixed precision setup."""
        config = MegaNeRFTrainerConfig(
            num_epochs=10, batch_size=128, learning_rate=1e-3, use_mixed_precision=True
        )

        trainer = MegaNeRFTrainer(model, config)
        trainer = trainer.to(device)

        assert hasattr(trainer, "scaler")
        assert trainer.scaler is not None

    def test_training_step(self, model, trainer_config, device, sample_rays):
        """Test single training step."""
        trainer = MegaNeRFTrainer(model, trainer_config)
        trainer = trainer.to(device)

        rays_o, rays_d = sample_rays
        rays_o = rays_o.to(device)
        rays_d = rays_d.to(device)

        # Mock target colors
        target_colors = torch.rand_like(rays_o)

        loss = trainer.training_step(rays_o, rays_d, target_colors)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # Scalar
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)
        assert loss > 0  # Loss should be positive

    def test_validation_step(self, model, trainer_config, device, sample_rays):
        """Test single validation step."""
        trainer = MegaNeRFTrainer(model, trainer_config)
        trainer = trainer.to(device)

        rays_o, rays_d = sample_rays
        rays_o = rays_o.to(device)
        rays_d = rays_d.to(device)

        # Mock target colors
        target_colors = torch.rand_like(rays_o)

        metrics = trainer.validation_step(rays_o, rays_d, target_colors)

        assert isinstance(metrics, dict)
        assert "val_loss" in metrics
        assert "val_psnr" in metrics
        assert "val_ssim" in metrics

        for value in metrics.values():
            assert not torch.isnan(value)
            assert not torch.isinf(value)

    def test_checkpoint_saving(self, model, trainer_config, temp_dir, device):
        """Test checkpoint saving and loading."""
        config = MegaNeRFTrainerConfig(
            num_epochs=10, batch_size=128, learning_rate=1e-3, checkpoint_dir=str(temp_dir)
        )

        trainer = MegaNeRFTrainer(model, config)
        trainer = trainer.to(device)

        # Save checkpoint
        checkpoint_path = Path(temp_dir) / "test_checkpoint.pth"
        trainer.save_checkpoint(checkpoint_path)

        assert checkpoint_path.exists()

        # Load checkpoint
        new_trainer = MegaNeRFTrainer(model, config)
        new_trainer = new_trainer.to(device)

        new_trainer.load_checkpoint(checkpoint_path)

        # Check that optimizer states are loaded
        assert len(trainer.optimizer.param_groups) == len(new_trainer.optimizer.param_groups)

    def test_gradient_clipping(self, model, trainer_config, device, sample_rays):
        """Test gradient clipping."""
        config = MegaNeRFTrainerConfig(
            num_epochs=10, batch_size=128, learning_rate=1e-3, gradient_clip_val=1.0
        )

        trainer = MegaNeRFTrainer(model, config)
        trainer = trainer.to(device)

        rays_o, rays_d = sample_rays
        rays_o = rays_o.to(device)
        rays_d = rays_d.to(device)
        target_colors = torch.rand_like(rays_o)

        # This should not raise an error
        loss = trainer.training_step(rays_o, rays_d, target_colors)
        assert not torch.isnan(loss)

    def test_early_stopping(self, model, trainer_config, device):
        """Test early stopping functionality."""
        config = MegaNeRFTrainerConfig(
            num_epochs=10, batch_size=128, learning_rate=1e-3, early_stopping_patience=3
        )

        trainer = MegaNeRFTrainer(model, config)
        trainer = trainer.to(device)

        # Simulate decreasing validation loss
        for i in range(5):
            trainer.early_stopping(1.0 - i * 0.1)

        # Should not stop early with decreasing loss
        assert not trainer.early_stopping.should_stop

        # Simulate increasing validation loss
        for i in range(5):
            trainer.early_stopping(1.0 + i * 0.1)

        # Should stop early with increasing loss
        assert trainer.early_stopping.should_stop

    def test_evaluation_mode(self, model, trainer_config, device):
        """Test evaluation mode switching."""
        trainer = MegaNeRFTrainer(model, trainer_config)
        trainer = trainer.to(device)

        # Switch to evaluation mode
        trainer.eval()

        assert not trainer.model.training

        # Switch back to training mode
        trainer.train()

        assert trainer.model.training

    def test_learning_rate_scheduling(self, model, trainer_config, device):
        """Test learning rate scheduling."""
        trainer = MegaNeRFTrainer(model, trainer_config)
        trainer = trainer.to(device)

        initial_lr = trainer.optimizer.param_groups[0]["lr"]

        # Step the scheduler
        trainer.scheduler.step()

        new_lr = trainer.optimizer.param_groups[0]["lr"]

        # Learning rate should change
        assert new_lr != initial_lr

    def test_model_parameter_count(self, model, trainer_config, device):
        """Test model parameter counting."""
        trainer = MegaNeRFTrainer(model, trainer_config)
        trainer = trainer.to(device)

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        assert total_params > 0
        assert trainable_params > 0
        assert trainable_params <= total_params

    def test_loss_computation(self, model, trainer_config, device, sample_rays):
        """Test loss computation with different loss types."""
        trainer = MegaNeRFTrainer(model, trainer_config)
        trainer = trainer.to(device)

        rays_o, rays_d = sample_rays
        rays_o = rays_o.to(device)
        rays_d = rays_d.to(device)
        target_colors = torch.rand_like(rays_o)

        # Test MSE loss
        loss = trainer.compute_loss(rays_o, rays_d, target_colors, loss_type="mse")
        assert isinstance(loss, torch.Tensor)
        assert loss > 0

        # Test L1 loss
        loss = trainer.compute_loss(rays_o, rays_d, target_colors, loss_type="l1")
        assert isinstance(loss, torch.Tensor)
        assert loss > 0

        # Test invalid loss type
        with pytest.raises(ValueError, match="Unsupported loss type"):
            trainer.compute_loss(rays_o, rays_d, target_colors, loss_type="invalid")
