"""
Test Block NeRF Renderer Components

This module tests the rendering-related components of Block Ne RF:
- BlockNeRFRenderer
- BlockNeRFRendererConfig
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
    from nerfs.block_nerf.renderer import BlockNeRFRenderer, BlockNeRFRendererConfig

    BLOCK_NERF_AVAILABLE = True
except ImportError as e:
    BLOCK_NERF_AVAILABLE = False
    IMPORT_ERROR = str(e)

from pathlib import Path
from unittest.mock import patch, MagicMock

from src.nerfs.block_nerf.renderer import (
    BlockNeRFRenderer,
    BlockNeRFRendererConfig,
    create_block_nerf_renderer,
)
from src.nerfs.block_nerf.core import BlockNeRFConfig, BlockNeRFModel
from . import (
    TEST_CONFIG,
    get_test_device,
    create_test_camera,
    skip_if_no_cuda,
    skip_if_slow,
)


class TestBlockNeRFRendererConfig:
    """Test Block-NeRF renderer configuration."""

    def test_default_config(self):
        """Test default renderer configuration."""
        config = BlockNeRFRendererConfig()

        assert config.chunk_size > 0
        assert config.num_samples > 0
        assert config.num_importance_samples >= 0
        assert config.perturb >= 0
        assert config.raw_noise_std >= 0
        assert config.white_bkgd in [True, False]

    def test_custom_config(self):
        """Test custom renderer configuration."""
        config = BlockNeRFRendererConfig(
            chunk_size=2048,
            num_samples=128,
            num_importance_samples=64,
            perturb=1.0,
            white_bkgd=True,
        )

        assert config.chunk_size == 2048
        assert config.num_samples == 128
        assert config.num_importance_samples == 64
        assert config.perturb == 1.0
        assert config.white_bkgd == True

    def test_config_validation(self):
        """Test renderer configuration validation."""
        # Test invalid chunk_size
        with pytest.raises(ValueError):
            BlockNeRFRendererConfig(chunk_size=0)

        # Test invalid num_samples
        with pytest.raises(ValueError):
            BlockNeRFRendererConfig(num_samples=0)

        # Test invalid perturb
        with pytest.raises(ValueError):
            BlockNeRFRendererConfig(perturb=-1.0)

    def test_hierarchical_sampling_config(self):
        """Test hierarchical sampling configuration."""
        config = BlockNeRFRendererConfig(
            use_hierarchical_sampling=True,
            num_importance_samples=64,
            importance_sample_bias=0.1,
        )

        assert config.use_hierarchical_sampling == True
        assert config.num_importance_samples == 64
        assert config.importance_sample_bias == 0.1


class TestBlockNeRFRenderer:
    """Test Block-NeRF renderer implementation."""

    @pytest.fixture
    def model_config(self):
        """Create model configuration."""
        return BlockNeRFConfig(
            scene_bounds=TEST_CONFIG["scene_bounds"],
            block_size=TEST_CONFIG["block_size"],
            max_blocks=TEST_CONFIG["max_blocks"],
            appearance_dim=32,
            hidden_dim=64,
            num_layers=2,
        )

    @pytest.fixture
    def renderer_config(self):
        """Create renderer configuration."""
        return BlockNeRFRendererConfig(
            chunk_size=512,
            num_samples=32,
            num_importance_samples=16,
            perturb=0.0,  # Deterministic for testing
            white_bkgd=False,
        )

    @pytest.fixture
    def model(self, model_config):
        """Create trained model."""
        device = get_test_device()
        model = BlockNeRFModel(model_config).to(device)
        model.eval()
        return model

    @pytest.fixture
    def renderer(self, model, renderer_config):
        """Create renderer instance."""
        return BlockNeRFRenderer(model=model, config=renderer_config)

    def test_renderer_initialization(self, renderer, model, renderer_config):
        """Test renderer initialization."""
        assert isinstance(renderer, BlockNeRFRenderer)
        assert renderer.model is model
        assert renderer.config == renderer_config
        assert hasattr(renderer, "block_rasterizer")

    def test_ray_generation(self, renderer):
        """Test camera ray generation."""
        device = get_test_device()

        # Camera parameters
        intrinsics = torch.tensor(
            [[800.0, 0.0, 400.0], [0.0, 800.0, 300.0], [0.0, 0.0, 1.0]], device=device
        )

        pose = torch.eye(4, device=device)
        image_size = (600, 800)  # H, W

        # Generate rays
        rays_o, rays_d = renderer.generate_rays(intrinsics, pose, image_size)

        assert rays_o.shape == (600, 800, 3)
        assert rays_d.shape == (600, 800, 3)
        assert torch.isfinite(rays_o).all()
        assert torch.isfinite(rays_d).all()

        # Check ray directions are normalized
        ray_norms = torch.norm(rays_d, dim=-1)
        assert torch.allclose(ray_norms, torch.ones_like(ray_norms), atol=1e-6)

    def test_ray_sampling(self, renderer):
        """Test ray sampling along rays."""
        device = get_test_device()
        batch_size = 4
        num_rays = 256

        # Create rays
        rays_o = torch.randn(batch_size, num_rays, 3, device=device)
        rays_d = torch.randn(batch_size, num_rays, 3, device=device)
        rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)

        # Sample points along rays
        t_vals, pts = renderer.sample_rays(
            rays_o, rays_d, near=0.1, far=10.0, num_samples=renderer.config.num_samples
        )

        assert t_vals.shape == (batch_size, num_rays, renderer.config.num_samples)
        assert pts.shape == (batch_size, num_rays, renderer.config.num_samples, 3)
        assert torch.isfinite(t_vals).all()
        assert torch.isfinite(pts).all()

        # Check t_vals are sorted
        assert (t_vals.diff(dim=-1) >= 0).all()

    def test_volume_rendering(self, renderer):
        """Test volume rendering computation."""
        device = get_test_device()
        batch_size = 2
        num_rays = 64
        num_samples = 32

        # Create sample data
        densities = torch.rand(batch_size, num_rays, num_samples, 1, device=device)
        colors = torch.rand(batch_size, num_rays, num_samples, 3, device=device)
        t_vals = torch.linspace(0.1, 10.0, num_samples, device=device)
        t_vals = t_vals.expand(batch_size, num_rays, num_samples)

        # Compute volume rendering
        rgb, depth, weights, alpha = renderer.volume_render(densities, colors, t_vals)

        assert rgb.shape == (batch_size, num_rays, 3)
        assert depth.shape == (batch_size, num_rays, 1)
        assert weights.shape == (batch_size, num_rays, num_samples)
        assert alpha.shape == (batch_size, num_rays, 1)

        assert torch.isfinite(rgb).all()
        assert torch.isfinite(depth).all()
        assert torch.isfinite(weights).all()
        assert torch.isfinite(alpha).all()

        # Check constraints
        assert (rgb >= 0).all() and (rgb <= 1).all()
        assert (depth >= 0).all()
        assert (weights >= 0).all()
        assert (alpha >= 0).all() and (alpha <= 1).all()

        # Check weights sum to approximately 1
        weight_sums = weights.sum(dim=-1)
        assert torch.allclose(weight_sums, alpha.squeeze(-1), atol=1e-5)

    def test_hierarchical_sampling(self, renderer):
        """Test hierarchical sampling."""
        device = get_test_device()
        batch_size = 2
        num_rays = 32
        num_coarse = 16
        num_fine = 8

        # Create coarse sampling data
        t_vals_coarse = torch.linspace(0.1, 10.0, num_coarse, device=device)
        t_vals_coarse = t_vals_coarse.expand(batch_size, num_rays, num_coarse)
        weights_coarse = torch.rand(batch_size, num_rays, num_coarse, device=device)
        weights_coarse = weights_coarse / weights_coarse.sum(dim=-1, keepdim=True)

        # Hierarchical sampling
        t_vals_fine = renderer.hierarchical_sample(t_vals_coarse, weights_coarse, num_fine)

        assert t_vals_fine.shape == (batch_size, num_rays, num_fine)
        assert torch.isfinite(t_vals_fine).all()

        # Check that fine samples are within coarse bounds
        assert (t_vals_fine >= t_vals_coarse.min(dim=-1, keepdim=True)[0]).all()
        assert (t_vals_fine <= t_vals_coarse.max(dim=-1, keepdim=True)[0]).all()

    def test_render_rays(self, renderer):
        """Test full ray rendering pipeline."""
        device = get_test_device()
        batch_size = 2
        num_rays = 64

        # Create rays
        rays_o = torch.randn(batch_size, num_rays, 3, device=device)
        rays_d = torch.randn(batch_size, num_rays, 3, device=device)
        rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)

        # Additional inputs
        camera_ids = torch.randint(0, 5, (batch_size,), device=device)
        exposure = torch.randn(batch_size, 1, device=device)

        # Render rays
        with torch.no_grad():
            outputs = renderer.render_rays(
                rays_o=rays_o,
                rays_d=rays_d,
                camera_ids=camera_ids,
                exposure=exposure,
                near=0.1,
                far=10.0,
            )

        assert "rgb" in outputs
        assert "depth" in outputs
        assert "weights" in outputs
        assert "alpha" in outputs

        assert outputs["rgb"].shape == (batch_size, num_rays, 3)
        assert outputs["depth"].shape == (batch_size, num_rays, 1)
        assert torch.isfinite(outputs["rgb"]).all()
        assert torch.isfinite(outputs["depth"]).all()
        assert (outputs["rgb"] >= 0).all() and (outputs["rgb"] <= 1).all()

    def test_render_image(self, renderer):
        """Test full image rendering."""
        device = get_test_device()

        # Camera parameters
        camera_data = create_test_camera()
        intrinsics = camera_data["intrinsics"]
        pose = camera_data["pose"]
        image_size = (64, 80)  # Small for fast testing

        # Additional inputs
        camera_id = torch.tensor(0, device=device)
        exposure = torch.tensor([0.5], device=device)

        # Render image
        with torch.no_grad():
            rendered = renderer.render_image(
                intrinsics=intrinsics,
                pose=pose,
                image_size=image_size,
                camera_id=camera_id,
                exposure=exposure,
                chunk_size=1024,  # Process in chunks
            )

        assert "rgb" in rendered
        assert "depth" in rendered

        assert rendered["rgb"].shape == (64, 80, 3)
        assert rendered["depth"].shape == (64, 80, 1)
        assert torch.isfinite(rendered["rgb"]).all()
        assert torch.isfinite(rendered["depth"]).all()
        assert (rendered["rgb"] >= 0).all() and (rendered["rgb"] <= 1).all()

    @skip_if_slow()
    def test_render_video_frames(self, renderer):
        """Test video frame rendering."""
        device = get_test_device()

        # Create camera trajectory
        num_frames = 4
        poses = []
        for i in range(num_frames):
            angle = 2 * np.pi * i / num_frames
            pose = torch.eye(4, device=device)
            pose[0, 3] = 2.0 * np.cos(angle)
            pose[2, 3] = 2.0 * np.sin(angle)
            poses.append(pose)

        camera_data = create_test_camera()
        intrinsics = camera_data["intrinsics"]
        image_size = (32, 40)  # Very small for testing

        # Render frames
        frames = []
        with torch.no_grad():
            for pose in poses:
                rendered = renderer.render_image(
                    intrinsics=intrinsics,
                    pose=pose,
                    image_size=image_size,
                    camera_id=torch.tensor(0, device=device),
                    exposure=torch.tensor([0.5], device=device),
                    chunk_size=512,
                )
                frames.append(rendered["rgb"])

        assert len(frames) == num_frames
        for frame in frames:
            assert frame.shape == (32, 40, 3)
            assert torch.isfinite(frame).all()

    def test_block_composition(self, renderer):
        """Test block composition for seamless rendering."""
        device = get_test_device()

        # Create rays that span multiple blocks
        rays_o = torch.tensor(
            [
                [0.0, 0.0, 0.0],  # Block center
                [4.0, 4.0, 0.0],  # Block boundary
                [-4.0, -4.0, 0.0],  # Another block
            ],
            device=device,
        ).unsqueeze(0)

        rays_d = torch.tensor(
            [
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
            ],
            device=device,
        ).unsqueeze(0)

        # Render with block composition
        with torch.no_grad():
            outputs = renderer.render_rays(
                rays_o=rays_o,
                rays_d=rays_d,
                camera_ids=torch.tensor([0], device=device),
                exposure=torch.tensor([[0.5]], device=device),
                near=0.1,
                far=10.0,
                use_block_composition=True,
            )

        assert outputs["rgb"].shape == (1, 3, 3)
        assert torch.isfinite(outputs["rgb"]).all()

    @skip_if_no_cuda()
    def test_cuda_rendering(self, model_config, renderer_config):
        """Test CUDA rendering performance."""
        model = BlockNeRFModel(model_config).cuda().eval()
        renderer = BlockNeRFRenderer(model=model, config=renderer_config)

        # Create larger batch for GPU testing
        batch_size = 4
        num_rays = 1024

        rays_o = torch.randn(batch_size, num_rays, 3, device="cuda")
        rays_d = torch.randn(batch_size, num_rays, 3, device="cuda")
        rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)

        camera_ids = torch.randint(0, 5, (batch_size,), device="cuda")
        exposure = torch.randn(batch_size, 1, device="cuda")

        # Render with CUDA
        with torch.no_grad():
            outputs = renderer.render_rays(
                rays_o=rays_o,
                rays_d=rays_d,
                camera_ids=camera_ids,
                exposure=exposure,
                near=0.1,
                far=10.0,
            )

        assert all(v.device.type == "cuda" for v in outputs.values())

    def test_rendering_precision(self, renderer):
        """Test different precision modes."""
        device = get_test_device()

        rays_o = torch.randn(1, 64, 3, device=device)
        rays_d = torch.randn(1, 64, 3, device=device)
        rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)

        camera_ids = torch.tensor([0], device=device)
        exposure = torch.tensor([[0.5]], device=device)

        # Test with different precision
        with torch.no_grad():
            # Float32
            outputs_fp32 = renderer.render_rays(
                rays_o=rays_o.float(),
                rays_d=rays_d.float(),
                camera_ids=camera_ids,
                exposure=exposure.float(),
                near=0.1,
                far=10.0,
            )

            # Check outputs are finite
            assert torch.isfinite(outputs_fp32["rgb"]).all()


class TestRendererFactory:
    """Test renderer factory functions."""

    def test_create_block_nerf_renderer(self):
        """Test renderer creation factory."""
        model_config = BlockNeRFConfig(
            scene_bounds=TEST_CONFIG["scene_bounds"],
            block_size=TEST_CONFIG["block_size"],
            max_blocks=4,
            hidden_dim=32,
            num_layers=2,
        )

        renderer_config = BlockNeRFRendererConfig(
            chunk_size=256,
            num_samples=16,
        )

        renderer = create_block_nerf_renderer(
            model_config=model_config, renderer_config=renderer_config
        )

        assert isinstance(renderer, BlockNeRFRenderer)
        assert isinstance(renderer.model, BlockNeRFModel)
        assert renderer.config == renderer_config

    def test_create_renderer_with_checkpoint(self, tmp_path):
        """Test renderer creation with model checkpoint."""
        # Create and save model
        model_config = BlockNeRFConfig(
            scene_bounds=TEST_CONFIG["scene_bounds"],
            block_size=TEST_CONFIG["block_size"],
            max_blocks=4,
        )

        model = BlockNeRFModel(model_config)
        checkpoint_path = tmp_path / "model.pth"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "config": model_config.to_dict(),
            },
            checkpoint_path,
        )

        # Create renderer from checkpoint
        renderer_config = BlockNeRFRendererConfig()
        renderer = create_block_nerf_renderer(
            model_config=model_config,
            renderer_config=renderer_config,
            checkpoint_path=str(checkpoint_path),
        )

        assert isinstance(renderer, BlockNeRFRenderer)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
