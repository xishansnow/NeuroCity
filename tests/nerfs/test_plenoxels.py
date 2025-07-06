"""
Tests for Plenoxels implementation.

This test suite provides comprehensive testing for the Plenoxels implementation,
including core functionality, training, dataset handling, and utilities.
"""

import unittest
import torch
import torch.nn.functional as F
import numpy as np
import os
import sys
import tempfile
import shutil
from pathlib import Path
from torch.utils.data import DataLoader
import json
import cv2

# Add src to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.nerfs.plenoxels.core import (
    PlenoxelConfig,
    VoxelGrid,
    SphericalHarmonics,
    PlenoxelModel,
    PlenoxelLoss,
    VolumetricRenderer,
)
from src.nerfs.plenoxels.trainer import PlenoxelTrainer, PlenoxelTrainerConfig
from src.nerfs.plenoxels.dataset import (
    PlenoxelDataset,
    PlenoxelDatasetConfig,
    load_blender_data,
    load_colmap_data,
    create_plenoxel_dataset,
    create_plenoxel_dataloader,
)
from src.nerfs.plenoxels.utils.rendering_utils import (
    generate_rays,
    sample_points_along_rays,
    volume_render,
    compute_ray_aabb_intersection,
    hierarchical_sampling,
)
from src.nerfs.plenoxels.utils.voxel_utils import (
    compute_voxel_bounds,
    world_to_voxel_coords,
    voxel_to_world_coords,
)
from src.nerfs.plenoxels.utils.metrics_utils import (
    compute_psnr,
    compute_ssim,
    compute_lpips,
)


class TestPlenoxelCore(unittest.TestCase):
    """Test core Plenoxels functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = PlenoxelConfig(
            grid_resolution=(32, 32, 32),
            scene_bounds=torch.tensor([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0], device=self.device),
            sh_degree=4,
            num_samples=128,
            batch_size=4096,
            learning_rate=0.01,
            device=self.device,
        )

    def test_voxel_grid_initialization(self):
        """Test VoxelGrid initialization and basic operations."""
        resolution = (32, 32, 32)
        scene_bounds = torch.tensor([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0], device=self.device)
        grid = VoxelGrid(resolution, scene_bounds, device=self.device)

        # Test grid shape
        self.assertEqual(grid.density.shape, resolution)
        self.assertEqual(grid.sh_coeffs.shape[:-2], resolution)  # [D, H, W, C, 3]

        # Test device placement
        self.assertEqual(grid.density.device.type, self.device.type)
        self.assertEqual(grid.sh_coeffs.device.type, self.device.type)

        # Test coordinate conversion
        points = torch.rand(100, 3, device=self.device) * 2 - 1  # [-1, 1]
        voxel_coords = grid.world_to_voxel_coords(points)
        world_coords = grid.voxel_to_world_coords(voxel_coords)
        self.assertTrue(torch.allclose(points, world_coords, atol=1e-5))

    def test_spherical_harmonics(self):
        """Test spherical harmonics computation."""
        # SphericalHarmonics is a static class, no need to instantiate
        dirs = torch.randn(100, 3, device=self.device)
        dirs = F.normalize(dirs, dim=-1)

        # Compute basis for degree 4
        basis = SphericalHarmonics.eval_sh_basis(4, dirs)
        num_coeffs = SphericalHarmonics.get_num_coeffs(4)

        # Test shape and values
        self.assertEqual(basis.shape, (100, num_coeffs))
        self.assertTrue(torch.all(torch.isfinite(basis)))

        # Test basis properties
        # 1. First basis function (Y00) should be constant
        y00 = 1.0 / np.sqrt(4 * np.pi)
        self.assertTrue(
            torch.allclose(basis[:, 0], torch.full((100,), y00, device=self.device), atol=1e-3)
        )

        # 2. Test orthogonality of basis functions
        # Compute Gram matrix
        gram = torch.mm(basis.T, basis) / basis.shape[0]  # Normalize by number of samples
        # Diagonal should be close to 1
        self.assertTrue(
            torch.allclose(torch.diag(gram), torch.ones_like(torch.diag(gram)), atol=1e-2)
        )
        # Off-diagonal elements should be close to 0
        off_diag = gram - torch.diag(torch.diag(gram))
        self.assertTrue(torch.all(torch.abs(off_diag) < 1e-1))

    def test_volumetric_renderer(self):
        """Test volumetric rendering."""
        resolution = (32, 32, 32)
        scene_bounds = torch.tensor([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0], device=self.device)
        grid = VoxelGrid(resolution, scene_bounds, device=self.device)
        renderer = VolumetricRenderer(self.config)

        # Generate test rays
        rays_o = torch.rand(64, 3, device=self.device) * 2 - 1
        rays_d = torch.randn(64, 3, device=self.device)
        rays_d = F.normalize(rays_d, dim=-1)

        # Test rendering
        with torch.no_grad():
            output = renderer.render_rays(
                grid,
                rays_o,
                rays_d,
                num_samples=64,
                near=0.1,
                far=10.0,
            )

        # Check outputs
        self.assertIn("rgb", output)
        self.assertIn("depth", output)
        self.assertIn("weights", output)
        self.assertEqual(output["rgb"].shape, (64, 3))
        self.assertEqual(output["depth"].shape, (64, 1))  # Depth is [N, 1]
        self.assertTrue(torch.all(output["rgb"] >= 0) and torch.all(output["rgb"] <= 1))

    def test_model_forward(self):
        """Test full model forward pass."""
        model = PlenoxelModel(self.config)

        # Generate test data
        rays_o = torch.rand(32, 3, device=self.device) * 2 - 1
        rays_d = torch.randn(32, 3, device=self.device)
        rays_d = F.normalize(rays_d, dim=-1)

        # Test forward pass
        with torch.no_grad():
            output = model(rays_o, rays_d)

        # Check outputs
        self.assertIn("rgb", output)
        self.assertIn("depth", output)
        self.assertEqual(output["rgb"].shape, (32, 3))
        self.assertEqual(output["depth"].shape, (32, 1))

    def test_loss_computation(self):
        """Test loss computation."""
        loss_fn = PlenoxelLoss(self.config)
        model = PlenoxelModel(self.config)

        # Generate test data
        outputs = {
            "rgb": torch.rand(32, 3, device=self.device),
            "depth": torch.rand(32, device=self.device),
            "weights": torch.rand(32, 64, device=self.device),
        }
        batch = {
            "rgb": torch.rand(32, 3, device=self.device),
            "depth": torch.rand(32, device=self.device),
        }

        # Compute loss
        loss_dict = loss_fn(outputs, batch, model, global_step=0)

        # Check outputs
        self.assertIn("total_loss", loss_dict)
        self.assertIn("rgb_loss", loss_dict)
        self.assertTrue(all(torch.isfinite(v) for v in loss_dict.values()))


class TestPlenoxelDataset(unittest.TestCase):
    """Test dataset functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = PlenoxelDatasetConfig(
            data_dir=self.temp_dir,
            dataset_type="blender",
            downsample_factor=1,
            batch_size=4096,
            num_rays_train=1024,
            precrop_fraction=0.5,
            precrop_iterations=500,
        )

    def tearDown(self):
        """Clean up temporary files."""
        shutil.rmtree(self.temp_dir)

    def test_blender_data_loading(self):
        """Test Blender data loading."""
        # Create dummy data
        os.makedirs(os.path.join(self.temp_dir, "train"))

        # Create transforms.json for all splits
        for split in ["train", "val", "test"]:
            transforms = {
                "camera_angle_x": 0.8,
                "frames": [
                    {"file_path": f"train/r_{i}.png", "transform_matrix": np.eye(4).tolist()}
                    for i in range(3)
                ],
            }
            with open(os.path.join(self.temp_dir, f"transforms_{split}.json"), "w") as f:
                json.dump(transforms, f)

        # Create images
        for i in range(3):
            img = (np.random.rand(100, 100, 4) * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(self.temp_dir, "train", f"r_{i}.png"), img)

        # Test data loading
        data = load_blender_data(self.temp_dir)
        self.assertIn("images", data)
        self.assertIn("poses", data)
        self.assertEqual(len(data["images"]), 9)  # 3 images per split

    def test_colmap_data_loading(self):
        """Test COLMAP data loading."""
        # Skip if test data not available
        if not os.path.exists("test_demo_scene"):
            self.skipTest("COLMAP test data not available")

        data = load_colmap_data("test_demo_scene")
        if data:  # Only test if data loading succeeded
            self.assertIn("images", data)
            self.assertIn("poses", data)

    def test_dataset_creation(self):
        """Test dataset creation and configuration."""
        config = create_plenoxel_dataset(
            data_dir="test_demo_scene",
            dataset_type="blender",
            downsample_factor=2,
        )
        self.assertIsInstance(config, PlenoxelDatasetConfig)
        self.assertEqual(config.data_dir, "test_demo_scene")
        self.assertEqual(config.dataset_type, "blender")
        self.assertEqual(config.downsample_factor, 2)


class TestPlenoxelTraining(unittest.TestCase):
    """Test training functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.temp_dir = tempfile.mkdtemp()
        self.model_config = PlenoxelConfig(
            grid_resolution=(32, 32, 32),
            scene_bounds=torch.tensor([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0]),
            sh_degree=2,
            num_samples=64,
        )
        self.trainer_config = PlenoxelTrainerConfig(
            max_epochs=10,
            learning_rate=0.1,
            coarse_to_fine=True,
            resolution_schedule=[(16, 16, 16), (32, 32, 32)],
            resolution_epochs=[5, 10],
            output_dir=self.temp_dir,
            experiment_name="test_experiment",
        )
        self.dataset_config = PlenoxelDatasetConfig(
            data_dir="test_demo_scene",
            dataset_type="blender",
            batch_size=1024,
            num_rays_train=512,
        )

    def tearDown(self):
        """Clean up temporary files."""
        shutil.rmtree(self.temp_dir)

    def test_trainer_initialization(self):
        """Test trainer initialization."""
        trainer = PlenoxelTrainer(
            model_config=self.model_config,
            trainer_config=self.trainer_config,
            dataset_config=self.dataset_config,
        )

        self.assertIsNotNone(trainer.model)
        self.assertIsNotNone(trainer.optimizer)
        self.assertIsNotNone(trainer.loss_fn)

    def test_training_step(self):
        """Test single training step."""
        trainer = PlenoxelTrainer(
            model_config=self.model_config,
            trainer_config=self.trainer_config,
            dataset_config=self.dataset_config,
        )

        # Run training step
        metrics = trainer._train_epoch()
        self.assertIn("total_loss", metrics)
        self.assertTrue(isinstance(metrics["total_loss"], float))
        self.assertTrue(np.isfinite(metrics["total_loss"]))

    def test_validation_step(self):
        """Test validation step."""
        trainer = PlenoxelTrainer(
            model_config=self.model_config,
            trainer_config=self.trainer_config,
            dataset_config=self.dataset_config,
        )

        # Create dummy validation data
        N = 32
        rays_o = torch.rand(N, 3, device=self.device)
        rays_d = F.normalize(torch.randn(N, 3, device=self.device), dim=-1)
        rgb = torch.rand(N, 3, device=self.device)

        # Run validation
        with torch.no_grad():
            outputs = trainer.model(rays_o, rays_d)

        # Check outputs
        self.assertIn("rgb", outputs)
        self.assertIn("depth", outputs)
        self.assertEqual(outputs["rgb"].shape, (N, 3))
        self.assertEqual(outputs["depth"].shape, (N, 1))
        self.assertTrue(torch.all(outputs["rgb"] >= 0) and torch.all(outputs["rgb"] <= 1))

    def test_checkpoint_saving_loading(self):
        """Test checkpoint management."""
        trainer = PlenoxelTrainer(
            model_config=self.model_config,
            trainer_config=self.trainer_config,
            dataset_config=self.dataset_config,
        )

        # Save checkpoint
        checkpoint_path = os.path.join(self.temp_dir, "checkpoint.pt")
        trainer.save_checkpoint(checkpoint_path)

        # Load checkpoint
        trainer.load_checkpoint(checkpoint_path)

        self.assertTrue(os.path.exists(checkpoint_path))


class TestPlenoxelUtils(unittest.TestCase):
    """Test utility functions."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_ray_generation(self):
        """Test ray generation utilities."""
        H, W = 100, 100
        focal = 50.0
        poses = torch.eye(4, device=self.device).unsqueeze(0)  # [1, 4, 4]

        rays_o, rays_d = generate_rays(poses, focal, H, W)

        self.assertEqual(rays_o.shape, (1, H, W, 3))
        self.assertEqual(rays_d.shape, (1, H, W, 3))

        # Check ray directions are normalized
        rays_d_norm = torch.norm(rays_d, dim=-1)
        self.assertTrue(torch.allclose(rays_d_norm, torch.ones_like(rays_d_norm), atol=1e-3))

        # Check ray origins are at camera center
        self.assertTrue(
            torch.allclose(rays_o[0], poses[0, :3, 3].view(1, 1, 3).expand(H, W, 3), atol=1e-3)
        )

        # Check central ray direction (should point along -z axis)
        center_ray = rays_d[0, H // 2, W // 2]
        expected_dir = F.normalize(torch.tensor([0, 0, -1], device=self.device))
        self.assertTrue(torch.allclose(center_ray, expected_dir, atol=1e-3))

    def test_ray_bounds(self):
        """Test ray bounds computation."""
        rays_o = torch.rand(100, 3, device=self.device)
        rays_d = F.normalize(torch.randn(100, 3, device=self.device), dim=-1)

        # Create AABB bounds
        aabb_min = torch.tensor([-1.0, -1.0, -1.0], device=self.device)
        aabb_max = torch.tensor([1.0, 1.0, 1.0], device=self.device)

        near, far = compute_ray_aabb_intersection(rays_o, rays_d, aabb_min, aabb_max)

        self.assertEqual(near.shape, (100,))
        self.assertEqual(far.shape, (100,))
        self.assertTrue(torch.all(near < far))

    def test_metrics(self):
        """Test metric computations."""
        pred = torch.rand(32, 3, device=self.device)
        target = torch.rand(32, 3, device=self.device)

        # Test PSNR
        psnr = compute_psnr(pred, target)
        self.assertTrue(isinstance(psnr, float))
        self.assertTrue(np.isfinite(psnr))

        # Test SSIM
        H, W = 32, 32  # Make image large enough for SSIM window
        pred_img = pred.view(1, 3, H, W).permute(0, 2, 3, 1).cpu().numpy()  # [1, H, W, C]
        target_img = target.view(1, 3, H, W).permute(0, 2, 3, 1).cpu().numpy()
        ssim = compute_ssim(pred_img[0], target_img[0])
        self.assertTrue(isinstance(ssim, float))
        self.assertTrue(np.isfinite(ssim))

        # Skip LPIPS if not available
        try:
            lpips = compute_lpips(pred_img, target_img)
            self.assertTrue(isinstance(lpips, float))
            self.assertTrue(np.isfinite(lpips))
        except ImportError:
            pass


if __name__ == "__main__":
    unittest.main()
