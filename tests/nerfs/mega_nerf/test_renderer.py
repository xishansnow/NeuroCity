"""
Test Mega-NeRF Renderer Module

This module tests the Mega-NeRF renderer components including:
- MegaNeRFRendererConfig
- MegaNeRFRenderer
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import tempfile
import shutil

from src.nerfs.mega_nerf.renderer import MegaNeRFRendererConfig, MegaNeRFRenderer


class TestMegaNeRFRendererConfig:
    """Test MegaNeRFRendererConfig class."""

    def test_default_config(self):
        """Test default configuration creation."""
        config = MegaNeRFRendererConfig()

        assert config.image_width == 800
        assert config.image_height == 600
        assert config.render_batch_size == 4096
        assert config.render_chunk_size == 1024
        assert config.num_coarse_samples == 64
        assert config.num_fine_samples == 128
        assert config.near == 0.1
        assert config.far == 10.0
        assert config.output_dir == "renders"

    def test_custom_config(self):
        """Test custom configuration creation."""
        config = MegaNeRFRendererConfig(
            image_width=512,
            image_height=512,
            render_batch_size=2048,
            render_chunk_size=512,
            num_coarse_samples=32,
            num_fine_samples=64,
            near=0.5,
            far=20.0,
            output_dir="custom_renders",
        )

        assert config.image_width == 512
        assert config.image_height == 512
        assert config.render_batch_size == 2048
        assert config.render_chunk_size == 512
        assert config.num_coarse_samples == 32
        assert config.num_fine_samples == 64
        assert config.near == 0.5
        assert config.far == 20.0
        assert config.output_dir == "custom_renders"

    def test_validation(self):
        """Test configuration validation."""
        # Test invalid image dimensions
        with pytest.raises(ValueError, match="image_width must be positive"):
            MegaNeRFRendererConfig(image_width=0)

        with pytest.raises(ValueError, match="image_height must be positive"):
            MegaNeRFRendererConfig(image_height=0)

        # Test invalid batch sizes
        with pytest.raises(ValueError, match="render_batch_size must be positive"):
            MegaNeRFRendererConfig(render_batch_size=0)

        with pytest.raises(ValueError, match="render_chunk_size must be positive"):
            MegaNeRFRendererConfig(render_chunk_size=0)

        # Test invalid sample counts
        with pytest.raises(ValueError, match="num_coarse_samples must be positive"):
            MegaNeRFRendererConfig(num_coarse_samples=0)

        with pytest.raises(ValueError, match="num_fine_samples must be positive"):
            MegaNeRFRendererConfig(num_fine_samples=0)

        # Test invalid near/far values
        with pytest.raises(ValueError, match="near must be positive"):
            MegaNeRFRendererConfig(near=0)

        with pytest.raises(ValueError, match="far must be greater than near"):
            MegaNeRFRendererConfig(near=10.0, far=5.0)


class TestMegaNeRFRenderer:
    """Test MegaNeRFRenderer class."""

    def test_initialization(self, model, renderer_config, device):
        """Test renderer initialization."""
        renderer = MegaNeRFRenderer(model, renderer_config)
        renderer = renderer.to(device)

        assert renderer.model == model
        assert renderer.config == renderer_config
        assert renderer.device == device

    def test_ray_generation(
        self, model, renderer_config, device, sample_camera_pose, sample_intrinsics
    ):
        """Test ray generation from camera pose and intrinsics."""
        renderer = MegaNeRFRenderer(model, renderer_config)
        renderer = renderer.to(device)

        camera_pose = sample_camera_pose.to(device)
        intrinsics = sample_intrinsics.to(device)

        rays_o, rays_d = renderer.generate_rays(camera_pose, intrinsics)

        expected_height, expected_width = renderer_config.image_height, renderer_config.image_width
        expected_rays = expected_height * expected_width

        assert rays_o.shape == (expected_rays, 3)
        assert rays_d.shape == (expected_rays, 3)
        assert not torch.isnan(rays_o).any()
        assert not torch.isnan(rays_d).any()

        # Ray directions should be normalized
        ray_norms = torch.norm(rays_d, dim=-1)
        assert torch.allclose(ray_norms, torch.ones_like(ray_norms), atol=1e-6)

    def test_single_image_rendering(
        self, model, renderer_config, device, sample_camera_pose, sample_intrinsics
    ):
        """Test single image rendering."""
        renderer = MegaNeRFRenderer(model, renderer_config)
        renderer = renderer.to(device)

        camera_pose = sample_camera_pose.to(device)
        intrinsics = sample_intrinsics.to(device)

        image = renderer.render_image(camera_pose, intrinsics)

        expected_height = renderer_config.image_height
        expected_width = renderer_config.image_width

        assert image.shape == (expected_height, expected_width, 3)
        assert not torch.isnan(image).any()
        assert torch.all(image >= 0) and torch.all(image <= 1)  # Colors in [0, 1]

    def test_batch_rendering(self, model, renderer_config, device, synthetic_cameras):
        """Test batch rendering of multiple images."""
        renderer = MegaNeRFRenderer(model, renderer_config)
        renderer = renderer.to(device)

        # Convert synthetic cameras to poses and intrinsics
        poses = []
        intrinsics = []

        for camera in synthetic_cameras[:3]:  # Use first 3 cameras
            pose = torch.tensor(camera.transform_matrix, device=device)
            intrinsic = torch.tensor(camera.intrinsics, device=device)
            poses.append(pose)
            intrinsics.append(intrinsic)

        images = renderer.render_batch(poses, intrinsics)

        assert len(images) == len(poses)
        for image in images:
            expected_height = renderer_config.image_height
            expected_width = renderer_config.image_width
            assert image.shape == (expected_height, expected_width, 3)
            assert not torch.isnan(image).any()

    def test_video_rendering(
        self, model, renderer_config, device, sample_camera_pose, sample_intrinsics
    ):
        """Test video rendering with camera trajectory."""
        renderer = MegaNeRFRenderer(model, renderer_config)
        renderer = renderer.to(device)

        # Create a simple circular trajectory
        num_frames = 10
        radius = 3.0
        height = 1.0

        poses = []
        intrinsics = []

        for i in range(num_frames):
            angle = 2 * np.pi * i / num_frames

            # Create camera pose
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            z = height

            pose = torch.eye(4, device=device)
            pose[:3, 3] = torch.tensor([x, y, z], device=device)

            poses.append(pose)
            intrinsics.append(sample_intrinsics.to(device))

        video = renderer.render_video(poses, intrinsics)

        assert video.shape == (
            num_frames,
            renderer_config.image_height,
            renderer_config.image_width,
            3,
        )
        assert not torch.isnan(video).any()

    def test_spiral_trajectory(self, model, renderer_config, device, sample_intrinsics):
        """Test spiral camera trajectory generation."""
        renderer = MegaNeRFRenderer(model, renderer_config)
        renderer = renderer.to(device)

        num_frames = 20
        radius = 3.0
        height_range = (0.5, 2.0)

        poses = renderer.generate_spiral_trajectory(
            num_frames=num_frames,
            radius=radius,
            height_range=height_range,
            center=torch.tensor([0.0, 0.0, 0.0], device=device),
        )

        assert len(poses) == num_frames
        for pose in poses:
            assert pose.shape == (4, 4)
            assert not torch.isnan(pose).any()

            # Check that camera positions are within expected range
            pos = pose[:3, 3]
            distance = torch.norm(pos[:2])  # Distance from center in XY plane
            assert distance <= radius * 1.1  # Allow some tolerance
            assert height_range[0] <= pos[2] <= height_range[1]

    def test_save_renders(
        self, model, renderer_config, temp_dir, device, sample_camera_pose, sample_intrinsics
    ):
        """Test saving rendered images."""
        config = MegaNeRFRendererConfig(
            image_width=256,
            image_height=256,
            render_batch_size=1024,
            render_chunk_size=512,
            output_dir=str(temp_dir),
        )

        renderer = MegaNeRFRenderer(model, config)
        renderer = renderer.to(device)

        camera_pose = sample_camera_pose.to(device)
        intrinsics = sample_intrinsics.to(device)

        # Render and save image
        image = renderer.render_image(camera_pose, intrinsics)
        output_path = Path(temp_dir) / "test_render.png"

        renderer.save_image(image, output_path)

        assert output_path.exists()
        assert output_path.stat().st_size > 0

    def test_memory_usage(
        self, model, renderer_config, device, sample_camera_pose, sample_intrinsics
    ):
        """Test memory usage during rendering."""
        renderer = MegaNeRFRenderer(model, renderer_config)
        renderer = renderer.to(device)

        camera_pose = sample_camera_pose.to(device)
        intrinsics = sample_intrinsics.to(device)

        if device == "cuda":
            torch.cuda.empty_cache()
            initial_memory = torch.cuda.memory_allocated()

        # Render image
        image = renderer.render_image(camera_pose, intrinsics)

        if device == "cuda":
            final_memory = torch.cuda.memory_allocated()
            # Memory usage should be reasonable (not excessive)
            memory_increase = final_memory - initial_memory
            assert memory_increase < 1024 * 1024 * 1024  # Less than 1GB increase

        assert not torch.isnan(image).any()

    def test_error_handling(self, model, renderer_config, device):
        """Test error handling for invalid inputs."""
        renderer = MegaNeRFRenderer(model, renderer_config)
        renderer = renderer.to(device)

        # Test invalid camera pose
        invalid_pose = torch.randn(3, 4, device=device)  # Wrong shape
        intrinsics = torch.eye(3, device=device)

        with pytest.raises(ValueError, match="Camera pose must be 4x4"):
            renderer.render_image(invalid_pose, intrinsics)

        # Test invalid intrinsics
        valid_pose = torch.eye(4, device=device)
        invalid_intrinsics = torch.randn(2, 2, device=device)  # Wrong shape

        with pytest.raises(ValueError, match="Camera intrinsics must be 3x3"):
            renderer.render_image(valid_pose, invalid_intrinsics)

    def test_rendering_quality(
        self, model, renderer_config, device, sample_camera_pose, sample_intrinsics
    ):
        """Test rendering quality metrics."""
        renderer = MegaNeRFRenderer(model, renderer_config)
        renderer = renderer.to(device)

        camera_pose = sample_camera_pose.to(device)
        intrinsics = sample_intrinsics.to(device)

        image = renderer.render_image(camera_pose, intrinsics)

        # Check image statistics
        assert image.min() >= 0.0
        assert image.max() <= 1.0

        # Check that image is not all black or all white
        assert image.mean() > 0.01  # Not all black
        assert image.mean() < 0.99  # Not all white

        # Check that image has some variation
        assert image.std() > 0.01  # Some variation in pixel values

    def test_chunked_rendering(
        self, model, renderer_config, device, sample_camera_pose, sample_intrinsics
    ):
        """Test chunked rendering for large images."""
        # Use smaller chunk size for testing
        config = MegaNeRFRendererConfig(
            image_width=512,
            image_height=512,
            render_batch_size=1024,
            render_chunk_size=256,  # Small chunk size
            num_coarse_samples=32,
            num_fine_samples=64,
        )

        renderer = MegaNeRFRenderer(model, config)
        renderer = renderer.to(device)

        camera_pose = sample_camera_pose.to(device)
        intrinsics = sample_intrinsics.to(device)

        image = renderer.render_image(camera_pose, intrinsics)

        assert image.shape == (512, 512, 3)
        assert not torch.isnan(image).any()
