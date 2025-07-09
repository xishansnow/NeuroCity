"""
Mega-NeRF Test Configuration

This module contains pytest fixtures and configuration for Mega-NeRF tests.
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import tempfile
import shutil

from src.nerfs.mega_nerf import (
    MegaNeRF,
    MegaNeRFConfig,
    MegaNeRFTrainerConfig,
    MegaNeRFRendererConfig,
    MegaNeRFDatasetConfig,
    CameraInfo,
)


@pytest.fixture
def device():
    """Get the device for testing."""
    if torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"


@pytest.fixture
def mega_nerf_config():
    """Create a basic Mega-NeRF configuration for testing."""
    return MegaNeRFConfig(
        num_submodules=4,
        grid_size=(2, 2),
        hidden_dim=64,
        num_layers=4,
        batch_size=128,
        learning_rate=1e-3,
        max_iterations=1000,
        num_coarse=64,
        num_fine=128,
        near=0.1,
        far=10.0,
        use_appearance_embedding=False,
        scene_bounds=(-5, -5, -2, 5, 5, 2),
        foreground_ratio=0.8,
    )


@pytest.fixture
def trainer_config():
    """Create a basic trainer configuration for testing."""
    return MegaNeRFTrainerConfig(
        num_epochs=10,
        batch_size=128,
        learning_rate=1e-3,
        log_interval=50,
        val_interval=100,
        save_interval=200,
        checkpoint_dir="test_checkpoints",
        use_mixed_precision=False,
        gradient_clip_val=None,
    )


@pytest.fixture
def renderer_config():
    """Create a basic renderer configuration for testing."""
    return MegaNeRFRendererConfig(
        image_width=256,
        image_height=256,
        render_batch_size=1024,
        render_chunk_size=512,
        num_coarse_samples=64,
        num_fine_samples=128,
        near=0.1,
        far=10.0,
        output_dir="test_outputs",
    )


@pytest.fixture
def dataset_config():
    """Create a basic dataset configuration for testing."""
    return MegaNeRFDatasetConfig(
        data_root="test_data",
        split="train",
        image_scale=0.5,
        load_images=False,
        use_cache=False,
        ray_batch_size=512,
        num_rays_per_image=256,
        num_partitions=4,
    )


@pytest.fixture
def model(mega_nerf_config, device):
    """Create a Mega-NeRF model for testing."""
    model = MegaNeRF(mega_nerf_config)
    return model.to(device)


@pytest.fixture
def synthetic_cameras():
    """Create synthetic camera data for testing."""
    cameras = []

    # Create a simple circular camera trajectory
    radius = 3.0
    height = 1.0
    num_cameras = 8

    for i in range(num_cameras):
        angle = 2 * np.pi * i / num_cameras

        # Camera position
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        z = height

        # Camera orientation (looking at origin)
        position = np.array([x, y, z])
        center = np.array([0, 0, 0])

        forward = center - position
        forward = forward / np.linalg.norm(forward)

        right = np.cross(forward, np.array([0, 0, 1]))
        right = right / np.linalg.norm(right)

        up = np.cross(right, forward)

        # Transform matrix
        transform_matrix = np.eye(4)
        transform_matrix[:3, 0] = right
        transform_matrix[:3, 1] = up
        transform_matrix[:3, 2] = forward
        transform_matrix[:3, 3] = position

        # Intrinsics
        intrinsics = np.array([[400, 0, 128], [0, 400, 128], [0, 0, 1]])

        camera = CameraInfo(
            transform_matrix=transform_matrix,
            intrinsics=intrinsics,
            image_path=f"test_image_{i:04d}.png",
            image_id=i,
            width=256,
            height=256,
        )

        cameras.append(camera)

    return cameras


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def test_data_dir(temp_dir):
    """Create a test data directory with synthetic data."""
    data_dir = Path(temp_dir) / "test_data"
    data_dir.mkdir(parents=True, exist_ok=True)

    # Create transforms.json for NeRF format
    transforms = {
        "w": 256,
        "h": 256,
        "fl_x": 400.0,
        "fl_y": 400.0,
        "cx": 128.0,
        "cy": 128.0,
        "frames": [],
    }

    # Add synthetic camera frames
    radius = 3.0
    height = 1.0
    num_cameras = 8

    for i in range(num_cameras):
        angle = 2 * np.pi * i / num_cameras

        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        z = height

        position = np.array([x, y, z])
        center = np.array([0, 0, 0])

        forward = center - position
        forward = forward / np.linalg.norm(forward)

        right = np.cross(forward, np.array([0, 0, 1]))
        right = right / np.linalg.norm(right)

        up = np.cross(right, forward)

        transform_matrix = np.eye(4)
        transform_matrix[:3, 0] = right
        transform_matrix[:3, 1] = up
        transform_matrix[:3, 2] = forward
        transform_matrix[:3, 3] = position

        frame = {
            "file_path": f"images/frame_{i:04d}",
            "transform_matrix": transform_matrix.tolist(),
        }
        transforms["frames"].append(frame)

    # Save transforms.json
    import json

    with open(data_dir / "transforms.json", "w") as f:
        json.dump(transforms, f, indent=2)

    # Create images directory
    images_dir = data_dir / "images"
    images_dir.mkdir(exist_ok=True)

    return str(data_dir)


@pytest.fixture
def sample_rays():
    """Create sample rays for testing."""
    batch_size = 100
    rays_o = torch.randn(batch_size, 3)
    rays_d = torch.randn(batch_size, 3)
    rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)

    return rays_o, rays_d


@pytest.fixture
def sample_points():
    """Create sample 3D points for testing."""
    batch_size = 100
    points = torch.randn(batch_size, 3)
    return points


@pytest.fixture
def sample_viewdirs():
    """Create sample view directions for testing."""
    batch_size = 100
    viewdirs = torch.randn(batch_size, 3)
    viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
    return viewdirs


@pytest.fixture
def sample_camera_pose():
    """Create a sample camera pose for testing."""
    pose = torch.eye(4)
    pose[:3, 3] = torch.tensor([3.0, 0.0, 1.0])
    return pose


@pytest.fixture
def sample_intrinsics():
    """Create sample camera intrinsics for testing."""
    intrinsics = torch.tensor([[400.0, 0.0, 128.0], [0.0, 400.0, 128.0], [0.0, 0.0, 1.0]])
    return intrinsics
