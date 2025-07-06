"""
Test suite for Instant NGP implementation.

This module provides comprehensive tests for all components of the Instant NGP
implementation including hash encoding, model architecture, training, and rendering.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import tempfile
import json
import os
from pathlib import Path
from PIL import Image

import sys
import os

# Add the src directory to Python path for importing modules
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from nerfs.instant_ngp.core import (
    InstantNGPConfig,
    InstantNGPModel,
    HashEncoder,
    SHEncoder,
    InstantNGPLoss,
    InstantNGPRenderer,
)
from nerfs.instant_ngp.trainer import InstantNGPTrainer
from nerfs.instant_ngp.dataset import InstantNGPDataset
from nerfs.instant_ngp.utils import (
    contract_to_unisphere,
    morton_encode_3d,
    compute_tv_loss,
    adaptive_sampling,
    estimate_normals,
)


class TestHashEncoder:
    """Test cases for hash encoding."""

    def test_hash_encoder_init(self):
        """Test hash encoder initialization."""
        config = InstantNGPConfig()
        encoder = HashEncoder(config)

        assert encoder.num_levels == config.num_levels
        assert encoder.level_dim == config.level_dim
        assert len(encoder.embeddings) == config.num_levels
        assert encoder.output_dim == config.num_levels * config.level_dim

    def test_hash_encoder_forward(self):
        """Test hash encoder forward pass."""
        config = InstantNGPConfig(num_levels=4, level_dim=2)
        encoder = HashEncoder(config)

        # Test with batch of positions
        positions = torch.randn(100, 3) * 0.5  # Keep in [-1, 1] range
        encoded = encoder(positions)

        assert encoded.shape == (100, config.num_levels * config.level_dim)
        assert not torch.isnan(encoded).any()
        assert not torch.isinf(encoded).any()

    def test_hash_encoder_interpolation(self):
        """Test trilinear interpolation."""
        config = InstantNGPConfig()
        encoder = HashEncoder(config)

        # Test interpolation function
        features = torch.randn(10, 8, 4)  # [batch, 8_corners, feature_dim]
        weights = torch.rand(10, 3)  # [batch, xyz_weights]

        result = encoder.trilinear_interpolation(features, weights)

        assert result.shape == (10, 4)
        assert not torch.isnan(result).any()

    def test_hash_encoder_reproducibility(self):
        """Test that encoding is reproducible."""
        config = InstantNGPConfig()
        encoder = HashEncoder(config)

        positions = torch.randn(50, 3) * 0.5

        # Encode twice
        encoded1 = encoder(positions)
        encoded2 = encoder(positions)

        assert torch.allclose(encoded1, encoded2, atol=1e-6)


class TestSHEncoder:
    """Test cases for spherical harmonics encoding."""

    def test_sh_encoder_init(self):
        """Test SH encoder initialization."""
        encoder = SHEncoder(degree=4)
        assert encoder.degree == 4
        assert encoder.output_dim == 16  # (degree + 1) ** 2

    def test_sh_encoder_forward(self):
        """Test SH encoder forward pass."""
        encoder = SHEncoder(degree=3)

        # Test with normalized directions
        directions = torch.randn(100, 3)
        directions = directions / torch.norm(directions, dim=-1, keepdim=True)

        encoded = encoder(directions)

        assert encoded.shape == (100, 16)  # degree=3 -> (3+1)^2 = 16
        assert not torch.isnan(encoded).any()

    def test_sh_encoder_normalization(self):
        """Test that encoder handles non-normalized directions."""
        encoder = SHEncoder(degree=2)

        # Non-normalized directions
        directions = torch.randn(50, 3) * 5
        encoded = encoder(directions)

        assert encoded.shape == (50, 9)  # degree=2 -> 9 coefficients
        assert not torch.isnan(encoded).any()


class TestInstantNGPModel:
    """Test cases for main InstantNGP model."""

    def test_model_init(self):
        """Test model initialization."""
        config = InstantNGPConfig()
        model = InstantNGPModel(config)

        assert isinstance(model.position_encoder, HashEncoder)
        assert isinstance(model.direction_encoder, SHEncoder)
        assert model.sigma_net is not None
        assert model.color_net is not None

    def test_model_forward(self):
        """Test model forward pass."""
        config = InstantNGPConfig()
        model = InstantNGPModel(config)

        positions = torch.randn(100, 3) * 0.5
        directions = torch.randn(100, 3)
        directions = directions / torch.norm(directions, dim=-1, keepdim=True)

        # Test with positions only
        density = model.get_density(positions)
        assert density.shape == (100, 1)
        assert (density >= 0).all()  # After activation

        # Test full forward pass
        rgb, density = model(positions, directions)
        assert rgb.shape == (100, 3)
        assert density.shape == (100, 1)
        assert (rgb >= 0).all() and (rgb <= 1).all()  # After sigmoid
        assert (density >= 0).all()

    def test_model_no_directions(self):
        """Test model forward without directions."""
        config = InstantNGPConfig()
        model = InstantNGPModel(config)

        positions = torch.randn(50, 3) * 0.5

        # Should return only density
        result = model(positions)
        assert isinstance(result, tuple)
        assert len(result) == 2

        rgb, density = result
        assert rgb is None or rgb.shape == (50, 3)
        assert density.shape == (50, 1)

    def test_model_gradient_flow(self):
        """Test that gradients flow through the model."""
        config = InstantNGPConfig()
        model = InstantNGPModel(config)

        positions = torch.randn(10, 3, requires_grad=True)
        directions = torch.randn(10, 3)
        directions = directions / torch.norm(directions, dim=-1, keepdim=True)

        rgb, density = model(positions, directions)
        loss = rgb.sum() + density.sum()
        loss.backward()

        # Check gradients exist
        assert positions.grad is not None
        assert not torch.isnan(positions.grad).any()


class TestInstantNGPLoss:
    """Test cases for loss function."""

    def test_loss_init(self):
        """Test loss initialization."""
        config = InstantNGPConfig()
        loss_fn = InstantNGPLoss(config)

        assert loss_fn.lambda_entropy == config.lambda_entropy
        assert loss_fn.lambda_tv == config.lambda_tv

    def test_loss_forward(self):
        """Test loss computation."""
        config = InstantNGPConfig()
        loss_fn = InstantNGPLoss(config)

        pred_rgb = torch.rand(100, 3)
        target_rgb = torch.rand(100, 3)
        pred_density = torch.rand(100, 1)
        positions = torch.rand(100, 3)

        loss = loss_fn(pred_rgb, target_rgb, pred_density, positions)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # Scalar
        assert loss >= 0

    def test_loss_components(self):
        """Test individual loss components."""
        config = InstantNGPConfig()
        loss_fn = InstantNGPLoss(config)

        pred_rgb = torch.rand(50, 3)
        target_rgb = torch.rand(50, 3)

        # Test RGB loss only
        loss = loss_fn(pred_rgb, target_rgb)
        assert loss >= 0

        # Test with density
        pred_density = torch.rand(50, 1)
        loss_with_density = loss_fn(pred_rgb, target_rgb, pred_density)
        assert loss_with_density >= loss  # Should be at least as large


class TestInstantNGPRenderer:
    """Test cases for renderer."""

    def test_renderer_init(self):
        """Test renderer initialization."""
        config = InstantNGPConfig()
        renderer = InstantNGPRenderer(config)

        assert renderer.config == config

    def test_ray_sampling(self):
        """Test ray sampling."""
        config = InstantNGPConfig()
        renderer = InstantNGPRenderer(config)

        rays_o = torch.zeros(10, 3)
        rays_d = torch.tensor([[0, 0, -1]]).expand(10, 3).float()
        near = torch.ones(10, 1) * 0.1
        far = torch.ones(10, 1) * 5.0

        z_vals = renderer.sample_rays(rays_o, rays_d, near, far, num_samples=64)

        assert z_vals.shape == (10, 64)
        assert (z_vals >= 0.1).all()
        assert (z_vals <= 5.0).all()

        # Check sorting
        assert (z_vals[:, 1:] >= z_vals[:, :-1]).all()

    def test_volume_rendering(self):
        """Test volume rendering."""
        config = InstantNGPConfig()
        renderer = InstantNGPRenderer(config)

        rgb = torch.rand(10, 64, 3)
        density = torch.rand(10, 64, 1)
        z_vals = torch.linspace(0.1, 5.0, 64).expand(10, 64)
        rays_d = torch.tensor([[0, 0, -1]]).expand(10, 3).float()

        rgb_map, depth_map, acc_map, weights = renderer.volume_render(rgb, density, z_vals, rays_d)

        assert rgb_map.shape == (10, 3)
        assert depth_map.shape == (10,)
        assert acc_map.shape == (10,)
        assert weights.shape == (10, 64)

        # Check valid ranges
        assert (rgb_map >= 0).all() and (rgb_map <= 1).all()
        assert (depth_map >= 0.1).all() and (depth_map <= 5.0).all()
        assert (acc_map >= 0).all() and (acc_map <= 1).all()
        assert (weights >= 0).all()

    def test_render_rays_integration(self):
        """Test full ray rendering pipeline."""
        config = InstantNGPConfig()
        model = InstantNGPModel(config)
        renderer = InstantNGPRenderer(config)

        rays_o = torch.zeros(5, 3)
        rays_d = torch.tensor([[0, 0, -1]]).expand(5, 3).float()
        near = torch.ones(5) * 0.1
        far = torch.ones(5) * 2.0

        with torch.no_grad():
            results = renderer.render_rays(model, rays_o, rays_d, near, far, num_samples=32)

        assert "rgb" in results
        assert "depth" in results
        assert "acc" in results

        assert results["rgb"].shape == (5, 3)
        assert results["depth"].shape == (5,)
        assert results["acc"].shape == (5,)


class TestDataset:
    """Test cases for dataset."""

    def create_dummy_dataset(self, tmp_dir):
        """Create a dummy NeRF dataset for testing."""
        # Create dummy transforms.json
        transforms = {"camera_angle_x": 0.6911112070083618, "frames": []}

        # Create dummy images and poses
        for i in range(5):
            # Simple pose (camera looking down -z axis)
            pose = np.eye(4)
            pose[2, 3] = -4  # Move camera back
            pose[0, 3] = i * 0.1  # Slight x offset

            transforms["frames"].append(
                {"file_path": f"./image_{i:03d}", "transform_matrix": pose.tolist()}
            )

            # Create dummy image
            img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
            Image.fromarray(img).save(tmp_dir / f"image_{i:03d}.png")

        # Save transforms
        with open(tmp_dir / "transforms_train.json", "w") as f:
            json.dump(transforms, f)

        with open(tmp_dir / "transforms_val.json", "w") as f:
            json.dump({"camera_angle_x": 0.6911112070083618, "frames": transforms["frames"][:1]}, f)

    def test_dataset_init(self):
        """Test dataset initialization."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            self.create_dummy_dataset(tmp_path)

            dataset = InstantNGPDataset(data_root=str(tmp_path), split="train", img_wh=(32, 32))

            assert len(dataset) > 0
            assert dataset.img_wh == (32, 32)
            assert hasattr(dataset, "poses")
            assert hasattr(dataset, "images")

    def test_dataset_getitem(self):
        """Test dataset item access."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            self.create_dummy_dataset(tmp_path)

            # Test training dataset (returns rays)
            train_dataset = InstantNGPDataset(
                data_root=str(tmp_path), split="train", img_wh=(16, 16)
            )

            item = train_dataset[0]
            assert "rays_o" in item
            assert "rays_d" in item
            assert "rgbs" in item

            # Test validation dataset (returns full images)
            val_dataset = InstantNGPDataset(data_root=str(tmp_path), split="val", img_wh=(16, 16))

            item = val_dataset[0]
            assert "rays_o" in item
            assert "rays_d" in item
            assert "rgbs" in item
            assert "pose" in item

    def test_dataloader_creation(self):
        """Test dataloader creation."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            self.create_dummy_dataset(tmp_path)

            dataloader = create_instant_ngp_dataloader(
                data_root=str(
                    tmp_path,
                )
            )

            batch = next(iter(dataloader))
            assert "rays_o" in batch
            assert "rays_d" in batch
            assert "rgbs" in batch

            assert batch["rays_o"].shape[0] == 64
            assert batch["rays_d"].shape == (64, 3)
            assert batch["rgbs"].shape == (64, 3)


class TestUtils:
    """Test cases for utility functions."""

    def test_contract_to_unisphere(self):
        """Test coordinate contraction."""
        positions = torch.tensor(
            [
                [0.5, 0.5, 0.5],  # Inside unit sphere
                [2.0, 0.0, 0.0],  # Outside unit sphere
                [0.0, 3.0, 4.0],  # Outside unit sphere
            ]
        )

        contracted = contract_to_unisphere(positions)

        # Check that all points are now within [-2, 2] range
        assert (torch.abs(contracted) <= 2.0).all()

        # Points inside unit sphere should be unchanged
        assert torch.allclose(contracted[0], positions[0])

        # Points outside should be contracted
        assert torch.norm(contracted[1]) < torch.norm(positions[1])
        assert torch.norm(contracted[2]) < torch.norm(positions[2])

    def test_morton_encode_3d(self):
        """Test Morton encoding."""
        coords = torch.tensor(
            [
                [0, 0, 0],
                [1, 1, 1],
                [2, 3, 4],
            ]
        )

        encoded = morton_encode_3d(coords)

        assert encoded.shape == (3,)
        assert encoded.dtype == torch.long
        assert (encoded >= 0).all()

    def test_compute_tv_loss(self):
        """Test total variation loss."""
        # Create a simple 3D grid
        grid = torch.randn(1, 8, 4, 4, 4)  # [batch, features, D, H, W]

        tv_loss = compute_tv_loss(grid)

        assert isinstance(tv_loss, torch.Tensor)
        assert tv_loss.dim() == 0  # Scalar
        assert tv_loss >= 0

    def test_adaptive_sampling(self):
        """Test adaptive sampling."""
        z_vals = torch.linspace(0, 1, 64).expand(10, 64)
        weights = torch.rand(10, 64)
        weights = weights / weights.sum(dim=1, keepdim=True)  # Normalize

        new_z_vals = adaptive_sampling(z_vals, weights, num_samples=32)

        assert new_z_vals.shape == (10, 32)
        assert (new_z_vals >= 0).all()
        assert (new_z_vals <= 1).all()

        # Check sorting
        assert (new_z_vals[:, 1:] >= new_z_vals[:, :-1]).all()

    def test_estimate_normals(self):
        """Test normal estimation."""
        # Create simple positions
        positions = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.1, 0.0, 0.0],
                [0.0, 0.1, 0.0],
                [0.0, 0.0, 0.1],
            ]
        )

        # Mock density function (simple sphere)
        def density_fn(pos):
            dist = torch.norm(pos, dim=-1, keepdim=True)
            return torch.exp(-dist * 5)  # Exponential falloff

        normals = estimate_normals(positions, density_fn)

        assert normals.shape == (4, 3)

        # Normals should be unit vectors
        norms = torch.norm(normals, dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


class TestTrainer:
    """Test cases for trainer."""

    def test_trainer_init(self):
        """Test trainer initialization."""
        config = InstantNGPConfig()
        trainer = InstantNGPTrainer(config)

        assert trainer.config == config
        assert isinstance(trainer.model, InstantNGPModel)
        assert isinstance(trainer.loss_fn, InstantNGPLoss)
        assert isinstance(trainer.renderer, InstantNGPRenderer)

    def test_trainer_step(self):
        """Test single training step."""
        config = InstantNGPConfig()
        trainer = InstantNGPTrainer(config)

        # Create dummy batch
        batch = {
            "rays_o": torch.randn(64, 3),
            "rays_d": torch.randn(64, 3),
            "rgbs": torch.rand(64, 3),
        }

        # Normalize ray directions
        batch["rays_d"] = batch["rays_d"] / torch.norm(batch["rays_d"], dim=-1, keepdim=True)

        # Run single step
        loss = trainer.train_step(batch)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert loss >= 0


def run_basic_tests():
    """Run basic functionality tests."""
    print("Running Instant NGP basic tests...")

    # Test model creation
    config = InstantNGPConfig()
    model = InstantNGPModel(config)
    print("✓ Model creation successful")

    # Test forward pass
    positions = torch.randn(10, 3) * 0.5
    directions = torch.randn(10, 3)
    directions = directions / torch.norm(directions, dim=-1, keepdim=True)

    with torch.no_grad():
        rgb, density = model(positions, directions)

    print(f"✓ Forward pass successful: RGB {rgb.shape}, Density {density.shape}")

    # Test trainer
    trainer = InstantNGPTrainer(config)
    print("✓ Trainer creation successful")

    print("All basic tests passed!")


if __name__ == "__main__":
    # Run basic tests if called directly
    run_basic_tests()
