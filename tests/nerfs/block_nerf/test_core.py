"""
Test Block NeRF Core Components

This module tests the core components of Block NeRF:
- BlockNeRF model
- BlockNeRFConfig
- BlockNeRFLoss
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
    from nerfs.block_nerf.config import BlockNeRFConfig
    from nerfs.block_nerf.loss import BlockNeRFLoss

    BLOCK_NERF_AVAILABLE = True
except ImportError as e:
    BLOCK_NERF_AVAILABLE = False
    IMPORT_ERROR = str(e)

from unittest.mock import patch, MagicMock

from src.nerfs.block_nerf.core import (
    BlockNeRFConfig,
    BlockNeRFModel,
    BlockNeRFLoss,
    check_compatibility,
    get_device_info,
)
from . import (
    TEST_CONFIG,
    get_test_device,
    create_test_data,
    create_test_camera,
    skip_if_no_cuda,
)


class TestBlockNeRFConfig:
    """Test Block-NeRF configuration."""

    def test_default_config(self):
        """Test default configuration creation."""
        config = BlockNeRFConfig()

        assert config.scene_bounds is not None
        assert config.block_size > 0
        assert config.max_blocks > 0
        assert config.appearance_dim > 0
        assert config.num_encoding_levels > 0

    def test_custom_config(self):
        """Test custom configuration creation."""
        config = BlockNeRFConfig(
            scene_bounds=TEST_CONFIG["scene_bounds"],
            block_size=TEST_CONFIG["block_size"],
            max_blocks=TEST_CONFIG["max_blocks"],
            appearance_dim=TEST_CONFIG["appearance_dim"],
        )

        assert config.scene_bounds == TEST_CONFIG["scene_bounds"]
        assert config.block_size == TEST_CONFIG["block_size"]
        assert config.max_blocks == TEST_CONFIG["max_blocks"]
        assert config.appearance_dim == TEST_CONFIG["appearance_dim"]

    def test_config_validation(self):
        """Test configuration validation."""
        # Test invalid scene bounds
        with pytest.raises(ValueError):
            BlockNeRFConfig(scene_bounds=(0, 0, 0))  # Wrong length

        # Test invalid block size
        with pytest.raises(ValueError):
            BlockNeRFConfig(block_size=0)

        # Test invalid max blocks
        with pytest.raises(ValueError):
            BlockNeRFConfig(max_blocks=0)

    def test_config_serialization(self):
        """Test configuration serialization and deserialization."""
        config = BlockNeRFConfig(
            scene_bounds=TEST_CONFIG["scene_bounds"], block_size=TEST_CONFIG["block_size"]
        )

        # Test to_dict
        config_dict = config.to_dict()
        assert isinstance(config_dict, dict)
        assert "scene_bounds" in config_dict
        assert "block_size" in config_dict

        # Test from_dict
        config_restored = BlockNeRFConfig.from_dict(config_dict)
        assert config_restored.scene_bounds == config.scene_bounds
        assert config_restored.block_size == config.block_size


class TestBlockNeRFModel:
    """Test Block-NeRF model implementation."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return BlockNeRFConfig(
            scene_bounds=TEST_CONFIG["scene_bounds"],
            block_size=TEST_CONFIG["block_size"],
            max_blocks=TEST_CONFIG["max_blocks"],
            appearance_dim=TEST_CONFIG["appearance_dim"],
            hidden_dim=128,
            num_layers=4,
        )

    @pytest.fixture
    def model(self, config):
        """Create test model."""
        device = get_test_device()
        return BlockNeRFModel(config).to(device)

    def test_model_initialization(self, model, config):
        """Test model initialization."""
        assert isinstance(model, BlockNeRFModel)
        assert model.config == config
        assert hasattr(model, "network")
        assert hasattr(model, "appearance_embedding")
        assert hasattr(model, "exposure_encoder")

    def test_model_parameters(self, model):
        """Test model parameter count and requirements."""
        params = list(model.parameters())
        assert len(params) > 0

        total_params = sum(p.numel() for p in params)
        assert total_params > 0

        # Check gradient requirements
        trainable_params = sum(p.numel() for p in params if p.requires_grad)
        assert trainable_params > 0

    def test_model_forward(self, model):
        """Test model forward pass."""
        device = get_test_device()
        batch_size = TEST_CONFIG["batch_size"]

        # Create test inputs
        positions = torch.randn(batch_size, 1024, 3, device=device)
        directions = torch.randn(batch_size, 1024, 3, device=device)
        camera_ids = torch.randint(0, 10, (batch_size,), device=device)
        exposure = torch.randn(batch_size, 1, device=device)

        # Forward pass
        model.eval()
        with torch.no_grad():
            outputs = model(
                positions=positions, directions=directions, camera_ids=camera_ids, exposure=exposure
            )

        # Check outputs
        assert "density" in outputs
        assert "color" in outputs
        assert "features" in outputs

        assert outputs["density"].shape == (batch_size, 1024, 1)
        assert outputs["color"].shape == (batch_size, 1024, 3)
        assert torch.isfinite(outputs["density"]).all()
        assert torch.isfinite(outputs["color"]).all()
        assert (outputs["color"] >= 0).all() and (outputs["color"] <= 1).all()

    def test_model_training_mode(self, model):
        """Test model training mode."""
        device = get_test_device()
        batch_size = 2

        positions = torch.randn(batch_size, 512, 3, device=device)
        directions = torch.randn(batch_size, 512, 3, device=device)
        camera_ids = torch.randint(0, 5, (batch_size,), device=device)
        exposure = torch.randn(batch_size, 1, device=device)

        model.train()
        outputs = model(
            positions=positions, directions=directions, camera_ids=camera_ids, exposure=exposure
        )

        # Check gradients can be computed
        loss = outputs["density"].mean() + outputs["color"].mean()
        loss.backward()

        # Check that gradients exist
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None

    @skip_if_no_cuda()
    def test_model_cuda_compatibility(self, config):
        """Test model CUDA compatibility."""
        model = BlockNeRFModel(config).cuda()

        batch_size = 2
        positions = torch.randn(batch_size, 256, 3, device="cuda")
        directions = torch.randn(batch_size, 256, 3, device="cuda")
        camera_ids = torch.randint(0, 5, (batch_size,), device="cuda")
        exposure = torch.randn(batch_size, 1, device="cuda")

        with torch.no_grad():
            outputs = model(
                positions=positions, directions=directions, camera_ids=camera_ids, exposure=exposure
            )

        assert all(v.device.type == "cuda" for v in outputs.values())

    def test_block_assignment(self, model):
        """Test block assignment functionality."""
        device = get_test_device()

        # Test points in different regions
        positions = torch.tensor(
            [
                [0.0, 0.0, 0.0],  # Center
                [5.0, 5.0, 0.0],  # Corner
                [-5.0, -5.0, 0.0],  # Opposite corner
            ],
            device=device,
        ).unsqueeze(0)

        block_ids = model.get_block_ids(positions)
        assert block_ids.shape == (1, 3)
        assert (block_ids >= 0).all()
        assert (block_ids < model.config.max_blocks).all()


class TestBlockNeRFLoss:
    """Test Block-NeRF loss functions."""

    @pytest.fixture
    def loss_fn(self):
        """Create loss function."""
        return BlockNeRFLoss(
            rgb_weight=1.0,
            depth_weight=0.1,
            distortion_weight=0.01,
            interlevel_weight=0.1,
        )

    def test_rgb_loss(self, loss_fn):
        """Test RGB loss computation."""
        device = get_test_device()
        batch_size = TEST_CONFIG["batch_size"]
        num_rays = 512

        pred_rgb = torch.rand(batch_size, num_rays, 3, device=device)
        gt_rgb = torch.rand(batch_size, num_rays, 3, device=device)

        loss = loss_fn.rgb_loss(pred_rgb, gt_rgb)

        assert loss.dim() == 0  # Scalar loss
        assert loss >= 0
        assert torch.isfinite(loss)

    def test_depth_loss(self, loss_fn):
        """Test depth loss computation."""
        device = get_test_device()
        batch_size = TEST_CONFIG["batch_size"]
        num_rays = 512

        pred_depth = torch.rand(batch_size, num_rays, 1, device=device) * 10
        gt_depth = torch.rand(batch_size, num_rays, 1, device=device) * 10
        depth_mask = torch.ones(batch_size, num_rays, 1, device=device)

        loss = loss_fn.depth_loss(pred_depth, gt_depth, depth_mask)

        assert loss.dim() == 0
        assert loss >= 0
        assert torch.isfinite(loss)

    def test_distortion_loss(self, loss_fn):
        """Test distortion loss computation."""
        device = get_test_device()
        batch_size = 2
        num_rays = 256
        num_samples = 64

        # Create ray samples
        t_vals = torch.linspace(0, 1, num_samples, device=device)
        t_vals = t_vals.expand(batch_size, num_rays, num_samples)
        weights = torch.rand(batch_size, num_rays, num_samples, device=device)
        weights = weights / weights.sum(dim=-1, keepdim=True)

        loss = loss_fn.distortion_loss(t_vals, weights)

        assert loss.dim() == 0
        assert loss >= 0
        assert torch.isfinite(loss)

    def test_total_loss(self, loss_fn):
        """Test total loss computation."""
        device = get_test_device()
        batch_size = 2
        num_rays = 256

        # Create mock predictions and targets
        predictions = {
            "rgb": torch.rand(batch_size, num_rays, 3, device=device),
            "depth": torch.rand(batch_size, num_rays, 1, device=device) * 10,
            "weights": torch.rand(batch_size, num_rays, 64, device=device),
            "t_vals": torch.linspace(0, 1, 64, device=device).expand(batch_size, num_rays, 64),
        }

        targets = {
            "rgb": torch.rand(batch_size, num_rays, 3, device=device),
            "depth": torch.rand(batch_size, num_rays, 1, device=device) * 10,
            "depth_mask": torch.ones(batch_size, num_rays, 1, device=device),
        }

        loss_dict = loss_fn(predictions, targets)

        assert isinstance(loss_dict, dict)
        assert "total_loss" in loss_dict
        assert "rgb_loss" in loss_dict
        assert loss_dict["total_loss"] >= 0
        assert torch.isfinite(loss_dict["total_loss"])


class TestUtilityFunctions:
    """Test utility functions."""

    def test_check_compatibility(self):
        """Test compatibility checking."""
        result = check_compatibility()

        assert isinstance(result, dict)
        assert "torch_version" in result
        assert "cuda_available" in result
        assert "device_count" in result

    def test_get_device_info(self):
        """Test device information retrieval."""
        info = get_device_info()

        assert isinstance(info, dict)
        assert "device_type" in info
        assert "device_count" in info

        if torch.cuda.is_available():
            assert "cuda_version" in info
            assert "gpu_memory" in info

    @skip_if_no_cuda()
    def test_gpu_memory_info(self):
        """Test GPU memory information."""
        info = get_device_info()

        assert "gpu_memory" in info
        assert "free_memory" in info
        assert info["gpu_memory"] > 0
        assert info["free_memory"] >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
