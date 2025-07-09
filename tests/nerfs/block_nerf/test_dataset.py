"""
Test Block NeRF Dataset Components

This module tests the dataset-related components of Block NeRF:
- BlockNeRFDataset
- BlockNeRFDatasetConfig
"""

import pytest
import torch
import numpy as np
import tempfile
import os
import json


# Add the src directory to the path for imports
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))

try:
    from nerfs.block_nerf import BlockNeRF
    from nerfs.block_nerf.dataset import BlockNeRFDataset, BlockNeRFDatasetConfig

    BLOCK_NERF_AVAILABLE = True
except ImportError as e:
    BLOCK_NERF_AVAILABLE = False
    IMPORT_ERROR = str(e)

from unittest.mock import patch, MagicMock

from . import (
    TEST_CONFIG,
    get_test_device,
    TEST_DATA_DIR,
    skip_if_slow,
)


class TestBlockNeRFDatasetConfig:
    """Test Block-NeRF dataset configuration."""

    def test_default_config(self):
        """Test default dataset configuration."""
        config = BlockNeRFDatasetConfig()

        assert config.data_dir is not None
        assert config.scene_bounds is not None
        assert config.block_size > 0
        assert config.image_downscale > 0
        assert config.num_rays > 0
        assert config.batch_size > 0

    def test_custom_config(self):
        """Test custom dataset configuration."""
        config = BlockNeRFDatasetConfig(
            data_dir="/path/to/data",
            scene_bounds=TEST_CONFIG["scene_bounds"],
            block_size=TEST_CONFIG["block_size"],
            image_downscale=2,
            num_rays=512,
            batch_size=8,
        )

        assert config.data_dir == "/path/to/data"
        assert config.scene_bounds == TEST_CONFIG["scene_bounds"]
        assert config.block_size == TEST_CONFIG["block_size"]
        assert config.image_downscale == 2
        assert config.num_rays == 512
        assert config.batch_size == 8

    def test_config_validation(self):
        """Test dataset configuration validation."""
        # Test invalid image_downscale
        with pytest.raises(ValueError):
            BlockNeRFDatasetConfig(image_downscale=0)

        # Test invalid num_rays
        with pytest.raises(ValueError):
            BlockNeRFDatasetConfig(num_rays=0)

        # Test invalid batch_size
        with pytest.raises(ValueError):
            BlockNeRFDatasetConfig(batch_size=0)

    def test_data_augmentation_config(self):
        """Test data augmentation configuration."""
        config = BlockNeRFDatasetConfig(
            use_data_augmentation=True,
            color_jitter_strength=0.1,
            exposure_range=(-1.0, 1.0),
            pose_noise_std=0.01,
        )

        assert config.use_data_augmentation == True
        assert config.color_jitter_strength == 0.1
        assert config.exposure_range == (-1.0, 1.0)
        assert config.pose_noise_std == 0.01


class TestBlockNeRFDataset:
    """Test Block-NeRF dataset implementation."""

    @pytest.fixture
    def mock_data_dir(self, tmp_path):
        """Create mock dataset directory."""
        data_dir = tmp_path / "test_dataset"
        data_dir.mkdir()

        # Create transforms.json
        transforms = {
            "frames": [
                {
                    "file_path": "./images/frame_000.png",
                    "transform_matrix": np.eye(4).tolist(),
                    "camera_angle_x": 0.8,
                    "camera_id": 0,
                    "exposure": 0.5,
                },
                {
                    "file_path": "./images/frame_001.png",
                    "transform_matrix": np.eye(4).tolist(),
                    "camera_angle_x": 0.8,
                    "camera_id": 1,
                    "exposure": 0.6,
                },
            ],
            "camera_angle_x": 0.8,
            "scene_bounds": TEST_CONFIG["scene_bounds"],
        }

        with open(data_dir / "transforms.json", "w") as f:
            json.dump(transforms, f)

        # Create images directory
        images_dir = data_dir / "images"
        images_dir.mkdir()

        # Create dummy images (just touch files for testing)
        for i in range(2):
            (images_dir / f"frame_{i:03d}.png").touch()

        return data_dir

    @pytest.fixture
    def config(self, mock_data_dir):
        """Create dataset configuration."""
        return BlockNeRFDatasetConfig(
            data_dir=str(mock_data_dir),
            scene_bounds=TEST_CONFIG["scene_bounds"],
            block_size=TEST_CONFIG["block_size"],
            image_downscale=4,  # Small images for testing
            num_rays=64,
            batch_size=2,
        )

    @pytest.fixture
    def dataset(self, config):
        """Create dataset instance."""
        with patch("PIL.Image.open") as mock_open:
            # Mock image loading
            mock_image = MagicMock()
            mock_image.size = (800, 600)
            mock_image.convert.return_value = mock_image
            mock_open.return_value = mock_image

            # Mock numpy array conversion
            with patch("numpy.array") as mock_array:
                mock_array.return_value = np.random.rand(600, 800, 3)

                dataset = BlockNeRFDataset(config, split="train")
                return dataset

    def test_dataset_initialization(self, dataset, config):
        """Test dataset initialization."""
        assert isinstance(dataset, BlockNeRFDataset)
        assert dataset.config == config
        assert hasattr(dataset, "frames")
        assert hasattr(dataset, "intrinsics")
        assert hasattr(dataset, "block_assignments")

    def test_dataset_length(self, dataset):
        """Test dataset length."""
        assert len(dataset) > 0
        assert len(dataset) == len(dataset.frames)

    def test_load_transforms(self, config, mock_data_dir):
        """Test transforms loading."""
        with patch("PIL.Image.open") as mock_open:
            mock_image = MagicMock()
            mock_image.size = (800, 600)
            mock_image.convert.return_value = mock_image
            mock_open.return_value = mock_image

            with patch("numpy.array") as mock_array:
                mock_array.return_value = np.random.rand(600, 800, 3)

                dataset = BlockNeRFDataset(config, split="train")

                assert len(dataset.frames) == 2
                assert "transform_matrix" in dataset.frames[0]
                assert "camera_id" in dataset.frames[0]
                assert "exposure" in dataset.frames[0]

    def test_intrinsics_computation(self, dataset):
        """Test camera intrinsics computation."""
        assert dataset.intrinsics is not None
        assert dataset.intrinsics.shape == (3, 3)
        assert torch.isfinite(dataset.intrinsics).all()

        # Check that focal length is positive
        assert dataset.intrinsics[0, 0] > 0
        assert dataset.intrinsics[1, 1] > 0

    def test_block_assignment(self, dataset):
        """Test block assignment to frames."""
        assert hasattr(dataset, "block_assignments")
        assert len(dataset.block_assignments) == len(dataset.frames)

        for block_id in dataset.block_assignments:
            assert isinstance(block_id, int)
            assert 0 <= block_id < dataset.config.max_blocks

    def test_get_item(self, dataset):
        """Test dataset item retrieval."""
        item = dataset[0]

        assert isinstance(item, dict)
        assert "rays_o" in item
        assert "rays_d" in item
        assert "gt_rgb" in item
        assert "camera_id" in item
        assert "exposure" in item
        assert "block_id" in item

        # Check tensor shapes
        assert item["rays_o"].shape == (dataset.config.num_rays, 3)
        assert item["rays_d"].shape == (dataset.config.num_rays, 3)
        assert item["gt_rgb"].shape == (dataset.config.num_rays, 3)
        assert item["camera_id"].shape == ()
        assert item["exposure"].shape == (1,)
        assert item["block_id"].shape == ()

    def test_ray_sampling(self, dataset):
        """Test ray sampling from images."""
        # Test different sampling strategies
        for strategy in ["random", "uniform", "importance"]:
            dataset.config.ray_sampling_strategy = strategy

            item = dataset[0]

            assert item["rays_o"].shape[0] == dataset.config.num_rays
            assert item["rays_d"].shape[0] == dataset.config.num_rays
            assert torch.isfinite(item["rays_o"]).all()
            assert torch.isfinite(item["rays_d"]).all()

            # Check ray directions are normalized
            ray_norms = torch.norm(item["rays_d"], dim=-1)
            assert torch.allclose(ray_norms, torch.ones_like(ray_norms), atol=1e-6)

    def test_data_augmentation(self, config, mock_data_dir):
        """Test data augmentation."""
        config.use_data_augmentation = True
        config.color_jitter_strength = 0.1
        config.exposure_range = (-0.5, 0.5)
        config.pose_noise_std = 0.01

        with patch("PIL.Image.open") as mock_open:
            mock_image = MagicMock()
            mock_image.size = (800, 600)
            mock_image.convert.return_value = mock_image
            mock_open.return_value = mock_image

            with patch("numpy.array") as mock_array:
                mock_array.return_value = np.random.rand(600, 800, 3)

                dataset = BlockNeRFDataset(config, split="train")

                # Get multiple samples from same frame
                item1 = dataset[0]
                item2 = dataset[0]

                # With augmentation, samples should be different
                assert not torch.equal(item1["gt_rgb"], item2["gt_rgb"])

    def test_validation_split(self, config, mock_data_dir):
        """Test validation split."""
        config.val_split = 0.5

        with patch("PIL.Image.open") as mock_open:
            mock_image = MagicMock()
            mock_image.size = (800, 600)
            mock_image.convert.return_value = mock_image
            mock_open.return_value = mock_image

            with patch("numpy.array") as mock_array:
                mock_array.return_value = np.random.rand(600, 800, 3)

                train_dataset = BlockNeRFDataset(config, split="train")
                val_dataset = BlockNeRFDataset(config, split="val")

                # Check that splits are different
                assert len(train_dataset) != len(val_dataset)
                assert len(train_dataset) + len(val_dataset) == 2

    def test_exposure_conditioning(self, dataset):
        """Test exposure conditioning."""
        item = dataset[0]

        assert "exposure" in item
        assert item["exposure"].shape == (1,)
        assert torch.isfinite(item["exposure"]).all()

    def test_camera_id_mapping(self, dataset):
        """Test camera ID mapping."""
        item = dataset[0]

        assert "camera_id" in item
        assert isinstance(item["camera_id"].item(), int)
        assert 0 <= item["camera_id"].item() < dataset.config.max_cameras

    def test_batch_processing(self, dataset):
        """Test batch processing."""
        batch_size = 4
        indices = list(range(min(batch_size, len(dataset))))

        batch = [dataset[i] for i in indices]

        # Check that all items have consistent keys
        keys = set(batch[0].keys())
        for item in batch[1:]:
            assert set(item.keys()) == keys

    @skip_if_slow()
    def test_large_dataset_loading(self, tmp_path):
        """Test loading larger dataset."""
        # Create larger mock dataset
        data_dir = tmp_path / "large_dataset"
        data_dir.mkdir()

        # Create more frames
        frames = []
        for i in range(100):
            frame = {
                "file_path": f"./images/frame_{i:03d}.png",
                "transform_matrix": np.eye(4).tolist(),
                "camera_angle_x": 0.8,
                "camera_id": i % 10,
                "exposure": 0.5 + 0.1 * np.sin(i * 0.1),
            }
            frames.append(frame)

        transforms = {
            "frames": frames,
            "camera_angle_x": 0.8,
            "scene_bounds": TEST_CONFIG["scene_bounds"],
        }

        with open(data_dir / "transforms.json", "w") as f:
            json.dump(transforms, f)

        images_dir = data_dir / "images"
        images_dir.mkdir()

        for i in range(100):
            (images_dir / f"frame_{i:03d}.png").touch()

        # Create dataset
        config = BlockNeRFDatasetConfig(
            data_dir=str(data_dir),
            scene_bounds=TEST_CONFIG["scene_bounds"],
            block_size=TEST_CONFIG["block_size"],
            num_rays=1024,
            batch_size=8,
        )

        with patch("PIL.Image.open") as mock_open:
            mock_image = MagicMock()
            mock_image.size = (800, 600)
            mock_image.convert.return_value = mock_image
            mock_open.return_value = mock_image

            with patch("numpy.array") as mock_array:
                mock_array.return_value = np.random.rand(600, 800, 3)

                dataset = BlockNeRFDataset(config, split="train")

                assert len(dataset) == 100

                # Test random access
                item = dataset[50]
                assert isinstance(item, dict)


class TestDataLoader:
    """Test data loader functionality."""

    @pytest.fixture
    def mock_dataset(self):
        """Create mock dataset."""
        dataset = MagicMock()
        dataset.__len__ = MagicMock(return_value=10)
        dataset.__getitem__ = MagicMock(
            side_effect=lambda i: {
                "rays_o": torch.randn(64, 3),
                "rays_d": torch.randn(64, 3),
                "gt_rgb": torch.rand(64, 3),
                "camera_id": torch.tensor(i % 5),
                "exposure": torch.tensor([0.5]),
                "block_id": torch.tensor(i % 4),
            }
        )
        return dataset

    def test_dataloader_creation(self, mock_dataset):
        """Test dataloader creation."""
        config = BlockNeRFDatasetConfig(
            batch_size=4,
            num_workers=0,  # No multiprocessing for testing
            shuffle=True,
        )

        dataloader = create_block_nerf_dataloader(
            dataset=mock_dataset, config=config, split="train"
        )

        assert dataloader is not None
        assert dataloader.batch_size == config.batch_size

    def test_dataloader_iteration(self, mock_dataset):
        """Test dataloader iteration."""
        config = BlockNeRFDatasetConfig(
            batch_size=2,
            num_workers=0,
        )

        dataloader = create_block_nerf_dataloader(
            dataset=mock_dataset, config=config, split="train"
        )

        for batch in dataloader:
            assert isinstance(batch, dict)
            assert "rays_o" in batch
            assert "rays_d" in batch
            assert "gt_rgb" in batch

            # Check batch dimensions
            assert batch["rays_o"].shape[0] == config.batch_size
            assert batch["rays_d"].shape[0] == config.batch_size
            assert batch["gt_rgb"].shape[0] == config.batch_size

            break  # Test only first batch

    def test_collate_function(self, mock_dataset):
        """Test custom collate function."""
        config = BlockNeRFDatasetConfig(batch_size=3, num_workers=0)

        dataloader = create_block_nerf_dataloader(
            dataset=mock_dataset, config=config, split="train"
        )

        batch = next(iter(dataloader))

        # Check that tensors are properly batched
        assert batch["rays_o"].shape == (3, 64, 3)
        assert batch["rays_d"].shape == (3, 64, 3)
        assert batch["gt_rgb"].shape == (3, 64, 3)
        assert batch["camera_id"].shape == (3,)
        assert batch["exposure"].shape == (3, 1)
        assert batch["block_id"].shape == (3,)


class TestDatasetFactory:
    """Test dataset factory functions."""

    def test_create_block_nerf_dataset(self, tmp_path):
        """Test dataset creation factory."""
        # Create minimal mock data
        data_dir = tmp_path / "test_data"
        data_dir.mkdir()

        transforms = {
            "frames": [
                {
                    "file_path": "./images/frame_000.png",
                    "transform_matrix": np.eye(4).tolist(),
                    "camera_angle_x": 0.8,
                    "camera_id": 0,
                    "exposure": 0.5,
                }
            ],
            "camera_angle_x": 0.8,
            "scene_bounds": TEST_CONFIG["scene_bounds"],
        }

        with open(data_dir / "transforms.json", "w") as f:
            json.dump(transforms, f)

        images_dir = data_dir / "images"
        images_dir.mkdir()
        (images_dir / "frame_000.png").touch()

        # Create dataset
        config = BlockNeRFDatasetConfig(
            data_dir=str(data_dir),
            scene_bounds=TEST_CONFIG["scene_bounds"],
            block_size=TEST_CONFIG["block_size"],
        )

        with patch("PIL.Image.open") as mock_open:
            mock_image = MagicMock()
            mock_image.size = (800, 600)
            mock_image.convert.return_value = mock_image
            mock_open.return_value = mock_image

            with patch("numpy.array") as mock_array:
                mock_array.return_value = np.random.rand(600, 800, 3)

                dataset = create_block_nerf_dataset(config, split="train")

                assert isinstance(dataset, BlockNeRFDataset)
                assert len(dataset) == 1

    def test_create_train_val_datasets(self, tmp_path):
        """Test train/val dataset creation."""
        # Create mock data with multiple frames
        data_dir = tmp_path / "test_data"
        data_dir.mkdir()

        frames = []
        for i in range(4):
            frame = {
                "file_path": f"./images/frame_{i:03d}.png",
                "transform_matrix": np.eye(4).tolist(),
                "camera_angle_x": 0.8,
                "camera_id": i,
                "exposure": 0.5,
            }
            frames.append(frame)

        transforms = {
            "frames": frames,
            "camera_angle_x": 0.8,
            "scene_bounds": TEST_CONFIG["scene_bounds"],
        }

        with open(data_dir / "transforms.json", "w") as f:
            json.dump(transforms, f)

        images_dir = data_dir / "images"
        images_dir.mkdir()
        for i in range(4):
            (images_dir / f"frame_{i:03d}.png").touch()

        # Create datasets
        config = BlockNeRFDatasetConfig(
            data_dir=str(data_dir),
            scene_bounds=TEST_CONFIG["scene_bounds"],
            block_size=TEST_CONFIG["block_size"],
            val_split=0.5,
        )

        with patch("PIL.Image.open") as mock_open:
            mock_image = MagicMock()
            mock_image.size = (800, 600)
            mock_image.convert.return_value = mock_image
            mock_open.return_value = mock_image

            with patch("numpy.array") as mock_array:
                mock_array.return_value = np.random.rand(600, 800, 3)

                train_dataset = create_block_nerf_dataset(config, split="train")
                val_dataset = create_block_nerf_dataset(config, split="val")

                assert len(train_dataset) == 2
                assert len(val_dataset) == 2
                assert len(train_dataset) + len(val_dataset) == 4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
