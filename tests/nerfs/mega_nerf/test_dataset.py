"""
Test Mega-NeRF Dataset Module

This module tests the Mega-NeRF dataset components including:
- MegaNeRFDatasetConfig
- MegaNeRFDataset
- CameraInfo
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import tempfile
import shutil
import json

from src.nerfs.mega_nerf.dataset import MegaNeRFDatasetConfig, MegaNeRFDataset, CameraInfo


class TestMegaNeRFDatasetConfig:
    """Test MegaNeRFDatasetConfig class."""

    def test_default_config(self):
        """Test default configuration creation."""
        config = MegaNeRFDatasetConfig()

        assert config.data_root == "data"
        assert config.split == "train"
        assert config.image_scale == 1.0
        assert config.load_images is True
        assert config.use_cache is True
        assert config.ray_batch_size == 1024
        assert config.num_rays_per_image == 1024
        assert config.num_partitions == 8

    def test_custom_config(self):
        """Test custom configuration creation."""
        config = MegaNeRFDatasetConfig(
            data_root="custom_data",
            split="val",
            image_scale=0.5,
            load_images=False,
            use_cache=False,
            ray_batch_size=512,
            num_rays_per_image=512,
            num_partitions=4,
        )

        assert config.data_root == "custom_data"
        assert config.split == "val"
        assert config.image_scale == 0.5
        assert config.load_images is False
        assert config.use_cache is False
        assert config.ray_batch_size == 512
        assert config.num_rays_per_image == 512
        assert config.num_partitions == 4

    def test_validation(self):
        """Test configuration validation."""
        # Test invalid image_scale
        with pytest.raises(ValueError, match="image_scale must be positive"):
            MegaNeRFDatasetConfig(image_scale=0)

        with pytest.raises(ValueError, match="image_scale must be positive"):
            MegaNeRFDatasetConfig(image_scale=-0.1)

        # Test invalid batch sizes
        with pytest.raises(ValueError, match="ray_batch_size must be positive"):
            MegaNeRFDatasetConfig(ray_batch_size=0)

        with pytest.raises(ValueError, match="num_rays_per_image must be positive"):
            MegaNeRFDatasetConfig(num_rays_per_image=0)

        # Test invalid split
        with pytest.raises(ValueError, match="split must be one of"):
            MegaNeRFDatasetConfig(split="invalid")


class TestCameraInfo:
    """Test CameraInfo class."""

    def test_initialization(self):
        """Test CameraInfo initialization."""
        transform_matrix = np.eye(4)
        intrinsics = np.array([[400, 0, 128], [0, 400, 128], [0, 0, 1]])

        camera = CameraInfo(
            transform_matrix=transform_matrix,
            intrinsics=intrinsics,
            image_path="test.png",
            image_id=0,
            width=256,
            height=256,
        )

        assert np.array_equal(camera.transform_matrix, transform_matrix)
        assert np.array_equal(camera.intrinsics, intrinsics)
        assert camera.image_path == "test.png"
        assert camera.image_id == 0
        assert camera.width == 256
        assert camera.height == 256

    def test_validation(self):
        """Test CameraInfo validation."""
        # Test invalid transform matrix
        with pytest.raises(ValueError, match="transform_matrix must be 4x4"):
            CameraInfo(
                transform_matrix=np.eye(3),
                intrinsics=np.eye(3),
                image_path="test.png",
                image_id=0,
                width=256,
                height=256,
            )

        # Test invalid intrinsics
        with pytest.raises(ValueError, match="intrinsics must be 3x3"):
            CameraInfo(
                transform_matrix=np.eye(4),
                intrinsics=np.eye(2),
                image_path="test.png",
                image_id=0,
                width=256,
                height=256,
            )

        # Test invalid dimensions
        with pytest.raises(ValueError, match="width and height must be positive"):
            CameraInfo(
                transform_matrix=np.eye(4),
                intrinsics=np.eye(3),
                image_path="test.png",
                image_id=0,
                width=0,
                height=256,
            )


class TestMegaNeRFDataset:
    """Test MegaNeRFDataset class."""

    def test_initialization(self, test_data_dir, dataset_config):
        """Test dataset initialization."""
        config = MegaNeRFDatasetConfig(
            data_root=test_data_dir, split="train", load_images=False, use_cache=False
        )

        dataset = MegaNeRFDataset(config)

        assert dataset.config == config
        assert dataset.data_root == test_data_dir
        assert len(dataset.cameras) > 0

    def test_load_transforms(self, test_data_dir):
        """Test loading transforms.json."""
        config = MegaNeRFDatasetConfig(
            data_root=test_data_dir, split="train", load_images=False, use_cache=False
        )

        dataset = MegaNeRFDataset(config)

        # Check that cameras were loaded
        assert len(dataset.cameras) > 0

        # Check camera properties
        for camera in dataset.cameras:
            assert camera.transform_matrix.shape == (4, 4)
            assert camera.intrinsics.shape == (3, 3)
            assert camera.width > 0
            assert camera.height > 0
            assert camera.image_id >= 0

    def test_preprocess_cameras(self, test_data_dir):
        """Test camera preprocessing."""
        config = MegaNeRFDatasetConfig(
            data_root=test_data_dir, split="train", load_images=False, use_cache=False
        )

        dataset = MegaNeRFDataset(config)

        # Check that cameras are properly preprocessed
        for camera in dataset.cameras:
            # Check that transform matrix is valid (orthogonal rotation part)
            rotation = camera.transform_matrix[:3, :3]
            identity = np.eye(3)
            assert np.allclose(rotation @ rotation.T, identity, atol=1e-6)

            # Check that intrinsics are valid
            assert camera.intrinsics[2, 2] == 1.0  # Last element should be 1
            assert camera.intrinsics[0, 1] == 0.0  # No skew
            assert camera.intrinsics[1, 0] == 0.0  # No skew

    def test_get_camera_matrix(self, test_data_dir):
        """Test getting camera matrix."""
        config = MegaNeRFDatasetConfig(
            data_root=test_data_dir, split="train", load_images=False, use_cache=False
        )

        dataset = MegaNeRFDataset(config)

        for i, camera in enumerate(dataset.cameras):
            camera_matrix = dataset.get_camera_matrix(i)

            assert camera_matrix.shape == (3, 4)
            assert not np.isnan(camera_matrix).any()

    def test_generate_rays(self, test_data_dir):
        """Test ray generation."""
        config = MegaNeRFDatasetConfig(
            data_root=test_data_dir, split="train", load_images=False, use_cache=False
        )

        dataset = MegaNeRFDataset(config)

        for i, camera in enumerate(dataset.cameras):
            rays_o, rays_d = dataset.generate_rays(i)

            expected_rays = camera.width * camera.height
            assert rays_o.shape == (expected_rays, 3)
            assert rays_d.shape == (expected_rays, 3)
            assert not np.isnan(rays_o).any()
            assert not np.isnan(rays_d).any()

            # Ray directions should be normalized
            ray_norms = np.linalg.norm(rays_d, axis=-1)
            assert np.allclose(ray_norms, np.ones_like(ray_norms), atol=1e-6)

    def test_data_augmentation(self, test_data_dir):
        """Test data augmentation."""
        config = MegaNeRFDatasetConfig(
            data_root=test_data_dir,
            split="train",
            load_images=False,
            use_cache=False,
            use_data_augmentation=True,
        )

        dataset = MegaNeRFDataset(config)

        # Test that augmentation is applied
        for i in range(min(3, len(dataset.cameras))):
            rays_o, rays_d = dataset.generate_rays(i)

            # With augmentation, rays should be slightly perturbed
            # but still valid
            assert not np.isnan(rays_o).any()
            assert not np.isnan(rays_d).any()

    def test_train_val_split(self, test_data_dir):
        """Test train/validation split."""
        # Test training split
        train_config = MegaNeRFDatasetConfig(
            data_root=test_data_dir, split="train", load_images=False, use_cache=False
        )
        train_dataset = MegaNeRFDataset(train_config)

        # Test validation split
        val_config = MegaNeRFDatasetConfig(
            data_root=test_data_dir, split="val", load_images=False, use_cache=False
        )
        val_dataset = MegaNeRFDataset(val_config)

        # Both should have cameras
        assert len(train_dataset.cameras) > 0
        assert len(val_dataset.cameras) > 0

    def test_error_handling(self, temp_dir):
        """Test error handling for invalid data."""
        # Test non-existent data root
        config = MegaNeRFDatasetConfig(
            data_root="non_existent_path", split="train", load_images=False, use_cache=False
        )

        with pytest.raises(FileNotFoundError, match="Data root does not exist"):
            MegaNeRFDataset(config)

        # Test missing transforms.json
        empty_dir = Path(temp_dir) / "empty_data"
        empty_dir.mkdir(exist_ok=True)

        config = MegaNeRFDatasetConfig(
            data_root=str(empty_dir), split="train", load_images=False, use_cache=False
        )

        with pytest.raises(FileNotFoundError, match="transforms.json not found"):
            MegaNeRFDataset(config)

    def test_dataset_statistics(self, test_data_dir):
        """Test dataset statistics computation."""
        config = MegaNeRFDatasetConfig(
            data_root=test_data_dir, split="train", load_images=False, use_cache=False
        )

        dataset = MegaNeRFDataset(config)

        stats = dataset.get_statistics()

        assert "num_cameras" in stats
        assert "image_width" in stats
        assert "image_height" in stats
        assert "scene_bounds" in stats
        assert "camera_bounds" in stats

        assert stats["num_cameras"] == len(dataset.cameras)
        assert stats["image_width"] > 0
        assert stats["image_height"] > 0
        assert len(stats["scene_bounds"]) == 6
        assert len(stats["camera_bounds"]) == 6

    def test_dataset_visualization(self, test_data_dir, temp_dir):
        """Test dataset visualization."""
        config = MegaNeRFDatasetConfig(
            data_root=test_data_dir, split="train", load_images=False, use_cache=False
        )

        dataset = MegaNeRFDataset(config)

        # Test camera trajectory visualization
        output_path = Path(temp_dir) / "camera_trajectory.png"
        dataset.visualize_camera_trajectory(output_path)

        # Note: This might not create a file if matplotlib is not available
        # but should not raise an error

    def test_dataset_serialization(self, test_data_dir, temp_dir):
        """Test dataset serialization."""
        config = MegaNeRFDatasetConfig(
            data_root=test_data_dir, split="train", load_images=False, use_cache=False
        )

        dataset = MegaNeRFDataset(config)

        # Save dataset info
        output_path = Path(temp_dir) / "dataset_info.json"
        dataset.save_info(output_path)

        assert output_path.exists()

        # Load dataset info
        loaded_info = dataset.load_info(output_path)

        assert "num_cameras" in loaded_info
        assert "config" in loaded_info
        assert loaded_info["num_cameras"] == len(dataset.cameras)

    def test_ray_batching(self, test_data_dir):
        """Test ray batching functionality."""
        config = MegaNeRFDatasetConfig(
            data_root=test_data_dir,
            split="train",
            load_images=False,
            use_cache=False,
            ray_batch_size=100,
            num_rays_per_image=200,
        )

        dataset = MegaNeRFDataset(config)

        # Test getting a batch of rays
        if len(dataset.cameras) > 0:
            batch = dataset.get_ray_batch()

            assert "rays_o" in batch
            assert "rays_d" in batch
            assert batch["rays_o"].shape[0] <= config.ray_batch_size
            assert batch["rays_d"].shape[0] <= config.ray_batch_size

    def test_camera_filtering(self, test_data_dir):
        """Test camera filtering functionality."""
        config = MegaNeRFDatasetConfig(
            data_root=test_data_dir, split="train", load_images=False, use_cache=False
        )

        dataset = MegaNeRFDataset(config)

        # Test filtering cameras by position
        if len(dataset.cameras) > 0:
            filtered_cameras = dataset.filter_cameras_by_position(
                center=np.array([0.0, 0.0, 0.0]), max_distance=10.0
            )

            assert len(filtered_cameras) <= len(dataset.cameras)
            assert all(isinstance(cam, CameraInfo) for cam in filtered_cameras)
