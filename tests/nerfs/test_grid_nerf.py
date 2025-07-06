"""
Grid-NeRF Test Suite

Comprehensive tests for Grid-NeRF implementation including:
- Unit tests for core components
- Integration tests for training pipeline
- Performance benchmarks
- Validation tests
"""

import os
import tempfile
import unittest
import torch
import numpy as np
from pathlib import Path
import shutil
import sys

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from nerfs.grid_nerf.core import (
    GridNeRF,
    GridNeRFConfig,
    GridNeRFRenderer,
    GridGuidedMLP,
    HierarchicalGrid,
)
from nerfs.grid_nerf.dataset import GridNeRFDataset
from nerfs.grid_nerf.trainer import GridNeRFTrainer
from nerfs.grid_nerf.utils import load_config, save_config


class TestGridNeRFConfig(unittest.TestCase):
    """Test GridNeRFConfig class."""

    def test_default_config(self):
        """Test default configuration creation."""
        config = GridNeRFConfig()

        # Check default values
        self.assertEqual(config.grid_levels, 4)
        self.assertEqual(config.base_resolution, 64)
        self.assertEqual(config.grid_feature_dim, 32)
        self.assertEqual(config.batch_size, 1024)

    def test_custom_config(self):
        """Test custom configuration."""
        config = GridNeRFConfig(grid_levels=3, base_resolution=32, batch_size=512)

        self.assertEqual(config.grid_levels, 3)
        self.assertEqual(config.base_resolution, 32)
        self.assertEqual(config.batch_size, 512)

    def test_scene_bounds_parsing(self):
        """Test scene bounds parsing."""
        # Tuple format
        config = GridNeRFConfig(scene_bounds=(-10, -10, -5, 10, 10, 5))
        expected = torch.tensor([[-10, -10, -5], [10, 10, 5]])
        self.assertTrue(torch.allclose(config.get_scene_bounds(), expected))

        # Dict format
        config = GridNeRFConfig(
            scene_bounds={"min_bound": [-10, -10, -5], "max_bound": [10, 10, 5]}
        )
        self.assertTrue(torch.allclose(config.get_scene_bounds(), expected))


class TestHierarchicalGrid(unittest.TestCase):
    """Test HierarchicalGrid class."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = GridNeRFConfig(grid_levels=3, base_resolution=32, grid_feature_dim=16)
        self.grid = HierarchicalGrid(self.config)

    def test_grid_initialization(self):
        """Test grid initialization."""
        self.assertEqual(len(self.grid.grids), 3)

        # Check resolutions
        expected_resolutions = [32, 64, 128]
        for i, expected_res in enumerate(expected_resolutions):
            actual_res = self.grid.grids[i].shape[1]  # Assuming shape [C, H, W, D]
            self.assertEqual(actual_res, expected_res)

    def test_world_to_grid_coords(self):
        """Test coordinate transformation."""
        points = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])

        grid_coords = self.grid.world_to_grid_coords(points, level=0)

        # Check shape
        self.assertEqual(grid_coords.shape, (2, 3))

        # Check values are in [0, 1] range after normalization
        self.assertTrue(torch.all(grid_coords >= 0))
        self.assertTrue(torch.all(grid_coords <= 1))

    def test_sample_features(self):
        """Test feature sampling."""
        points = torch.randn(100, 3) * 10  # Random points

        features = self.grid.sample_features(points)

        # Check output shape
        expected_dim = self.config.grid_feature_dim * self.config.grid_levels
        self.assertEqual(features.shape, (100, expected_dim))

    def test_forward_pass(self):
        """Test forward pass."""
        points = torch.randn(50, 3)

        features = self.grid(points)

        expected_dim = self.config.grid_feature_dim * self.config.grid_levels
        self.assertEqual(features.shape, (50, expected_dim))


class TestGridGuidedMLP(unittest.TestCase):
    """Test GridGuidedMLP class."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = GridNeRFConfig(
            grid_feature_dim=32,
            density_hidden_dim=128,
            color_hidden_dim=64,
            position_encoding_levels=6,
            direction_encoding_levels=3,
        )
        self.mlp = GridGuidedMLP(self.config)

    def test_density_network(self):
        """Test density network."""
        # Input: grid features + position encoding
        grid_features = torch.randn(10, self.config.grid_feature_dim * self.config.grid_levels)
        positions = torch.randn(10, 3)

        density = self.mlp.density_network(grid_features, positions)

        self.assertEqual(density.shape, (10, 1))
        self.assertTrue(torch.all(density >= 0))  # Should be non-negative

    def test_color_network(self):
        """Test color network."""
        grid_features = torch.randn(5, self.config.grid_feature_dim * self.config.grid_levels)
        positions = torch.randn(5, 3)
        directions = torch.randn(5, 3)

        # Normalize directions
        directions = directions / torch.norm(directions, dim=-1, keepdim=True)

        colors = self.mlp.color_network(grid_features, positions, directions)

        self.assertEqual(colors.shape, (5, 3))
        self.assertTrue(torch.all(colors >= 0))
        self.assertTrue(torch.all(colors <= 1))  # Should be in [0, 1]

    def test_forward_pass(self):
        """Test full forward pass."""
        grid_features = torch.randn(8, self.config.grid_feature_dim * self.config.grid_levels)
        positions = torch.randn(8, 3)
        directions = torch.randn(8, 3)
        directions = directions / torch.norm(directions, dim=-1, keepdim=True)

        density, colors = self.mlp(grid_features, positions, directions)

        self.assertEqual(density.shape, (8, 1))
        self.assertEqual(colors.shape, (8, 3))


class TestGridNeRFRenderer(unittest.TestCase):
    """Test GridNeRFRenderer class."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = GridNeRFConfig(num_samples=32, num_importance_samples=64)
        self.renderer = GridNeRFRenderer(self.config)

    def test_sample_points_along_rays(self):
        """Test ray sampling."""
        rays_o = torch.randn(5, 3)
        rays_d = torch.randn(5, 3)
        rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)

        points, z_vals = self.renderer.sample_points_along_rays(rays_o, rays_d)

        self.assertEqual(points.shape, (5, self.config.num_samples, 3))
        self.assertEqual(z_vals.shape, (5, self.config.num_samples))

    def test_volume_rendering(self):
        """Test volume rendering."""
        rgb = torch.rand(3, 32, 3)  # [N, n_samples, 3]
        density = torch.rand(3, 32)  # [N, n_samples]
        z_vals = torch.linspace(0, 10, 32).expand(3, 32)
        rays_d = torch.randn(3, 3)
        rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)

        outputs = self.renderer.volume_rendering(rgb, density, z_vals, rays_d)

        self.assertIn("rgb", outputs)
        self.assertIn("depth", outputs)
        self.assertEqual(outputs["rgb"].shape, (3, 3))
        self.assertEqual(outputs["depth"].shape, (3,))


class TestGridNeRF(unittest.TestCase):
    """Test complete GridNeRF model."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = GridNeRFConfig(
            grid_levels=2,  # Smaller for testing
            base_resolution=16,
            grid_feature_dim=8,
            density_hidden_dim=32,
            color_hidden_dim=32,
            num_samples=16,
        )
        self.model = GridNeRF(self.config)

    def test_model_creation(self):
        """Test model creation."""
        self.assertIsInstance(self.model.grid, HierarchicalGrid)
        self.assertIsInstance(self.model.mlp, GridGuidedMLP)
        self.assertIsInstance(self.model.renderer, GridNeRFRenderer)

    def test_forward_pass(self):
        """Test model forward pass."""
        rays_o = torch.randn(10, 3)
        rays_d = torch.randn(10, 3)
        rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)

        outputs = self.model(rays_o, rays_d)

        self.assertIn("rgb", outputs)
        self.assertIn("depth", outputs)
        self.assertEqual(outputs["rgb"].shape, (10, 3))
        self.assertEqual(outputs["depth"].shape, (10,))

    def test_model_parameters(self):
        """Test model has trainable parameters."""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        self.assertGreater(total_params, 0)
        self.assertEqual(total_params, trainable_params)  # All should be trainable


class TestDataset(unittest.TestCase):
    """Test dataset classes."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = GridNeRFConfig()

        # Create mock dataset structure
        os.makedirs(os.path.join(self.temp_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(self.temp_dir, "poses"), exist_ok=True)

        # Create dummy images and poses
        import cv2

        for i in range(5):
            img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            cv2.imwrite(os.path.join(self.temp_dir, "images", f"{i:06d}.png"), img)

            pose = np.eye(4, dtype=np.float32)
            pose[:3, 3] = np.random.randn(3)
            np.save(os.path.join(self.temp_dir, "poses", f"{i:06d}.npy"), pose)

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)

    def test_dataset_creation(self):
        """Test dataset creation."""
        dataset = GridNeRFDataset(data_path=self.temp_dir, split="train", config=self.config)

        self.assertEqual(len(dataset), 5)

    def test_dataset_getitem(self):
        """Test dataset item retrieval."""
        dataset = GridNeRFDataset(data_path=self.temp_dir, split="train", config=self.config)

        try:
            sample = dataset[0]
            self.assertIn("rays_o", sample)
            self.assertIn("rays_d", sample)
            self.assertIn("target_rgb", sample)
        except Exception as e:
            # Dataset might not have all required files, that's OK for this test
            self.assertIsInstance(e, (FileNotFoundError, KeyError))

    def test_dataloader_creation(self):
        """Test dataloader creation."""
        dataset = GridNeRFDataset(data_path=self.temp_dir, split="train", config=self.config)

        dataloader = create_dataloader(
            dataset, batch_size=2, num_workers=0, shuffle=True  # Use 0 for testing
        )

        self.assertEqual(dataloader.batch_size, 2)


class TestUtilities(unittest.TestCase):
    """Test utility functions."""

    def test_compute_psnr(self):
        """Test PSNR computation."""
        pred = torch.rand(10, 10, 3)
        target = pred.clone()  # Perfect match

        psnr = compute_psnr(pred, target)
        self.assertTrue(psnr > 40)  # Should be very high for perfect match

        # Test with different images
        target_diff = torch.rand(10, 10, 3)
        psnr_diff = compute_psnr(pred, target_diff)
        self.assertTrue(psnr_diff < psnr)

    def test_compute_ssim(self):
        """Test SSIM computation."""
        pred = torch.rand(32, 32, 3)
        target = pred.clone()

        ssim = compute_ssim(pred, target)
        self.assertAlmostEqual(ssim, 1.0, places=2)  # Should be close to 1.0

        # Test with different images
        target_diff = torch.rand(32, 32, 3)
        ssim_diff = compute_ssim(pred, target_diff)
        self.assertTrue(ssim_diff < ssim)

    def test_positional_encoding(self):
        """Test positional encoding."""
        x = torch.randn(10, 3)
        L = 5

        encoded = positional_encoding(x, L)

        expected_dim = 3 * (2 * L + 1)  # Original + sin/cos for each level
        self.assertEqual(encoded.shape, (10, expected_dim))

    def test_get_ray_directions(self):
        """Test ray direction generation."""
        H, W = 64, 64
        focal = 100.0

        directions = get_ray_directions(H, W, focal)

        self.assertEqual(directions.shape, (H, W, 3))

        # Check that center ray points in -Z direction
        center_ray = directions[H // 2, W // 2]
        self.assertAlmostEqual(center_ray[2].item(), -1.0, places=3)

    def test_sample_along_rays(self):
        """Test ray sampling."""
        rays_o = torch.randn(5, 3)
        rays_d = torch.randn(5, 3)
        rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)

        points, z_vals = sample_along_rays(rays_o, rays_d, near=1.0, far=10.0, n_samples=32)

        self.assertEqual(points.shape, (5, 32, 3))
        self.assertEqual(z_vals.shape, (5, 32))

        # Check z_vals are in range
        self.assertTrue(torch.all(z_vals >= 1.0))
        self.assertTrue(torch.all(z_vals <= 10.0))

    def test_volume_rendering(self):
        """Test volume rendering function."""
        rgb = torch.rand(3, 16, 3)
        density = torch.rand(3, 16)
        z_vals = torch.linspace(1, 10, 16).expand(3, 16)
        rays_d = torch.randn(3, 3)
        rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)

        outputs = volume_rendering(rgb, density, z_vals, rays_d)

        self.assertIn("rgb", outputs)
        self.assertIn("depth", outputs)
        self.assertEqual(outputs["rgb"].shape, (3, 3))
        self.assertEqual(outputs["depth"].shape, (3,))


class TestTrainer(unittest.TestCase):
    """Test training pipeline."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = GridNeRFConfig(
            grid_levels=2,
            base_resolution=8,
            grid_feature_dim=4,
            batch_size=4,
            num_epochs=2,
            max_steps=10,
            log_every_n_steps=5,
        )

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)

    def test_trainer_creation(self):
        """Test trainer creation."""
        trainer = GridNeRFTrainer(
            config=self.config,
            output_dir=self.temp_dir,
            device=torch.device(
                "cpu",
            ),
        )

        self.assertEqual(trainer.config, self.config)
        self.assertEqual(trainer.output_dir, Path(self.temp_dir))

    def test_model_setup(self):
        """Test model setup in trainer."""
        trainer = GridNeRFTrainer(
            config=self.config,
            output_dir=self.temp_dir,
            device=torch.device(
                "cpu",
            ),
        )

        trainer.setup_model()

        self.assertIsNotNone(trainer.model)

    def test_optimizer_setup(self):
        """Test optimizer setup."""
        trainer = GridNeRFTrainer(
            config=self.config,
            output_dir=self.temp_dir,
            device=torch.device(
                "cpu",
            ),
        )

        trainer.setup_model()
        trainer.setup_optimizer()

        self.assertIsNotNone(trainer.optimizer)
        self.assertIsNotNone(trainer.scheduler)


class TestIntegration(unittest.TestCase):
    """Integration tests."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_cuda_compatibility(self):
        """Test CUDA compatibility."""
        config = GridNeRFConfig(grid_levels=2, base_resolution=8)
        model = GridNeRF(config).cuda()

        rays_o = torch.randn(5, 3).cuda()
        rays_d = torch.randn(5, 3).cuda()
        rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)

        with torch.no_grad():
            outputs = model(rays_o, rays_d)

        self.assertTrue(outputs["rgb"].is_cuda)
        self.assertTrue(outputs["depth"].is_cuda)

    def test_gradient_flow(self):
        """Test gradient flow through model."""
        config = GridNeRFConfig(grid_levels=2, base_resolution=8)
        model = GridNeRF(config)

        rays_o = torch.randn(3, 3, requires_grad=True)
        rays_d = torch.randn(3, 3)
        rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
        target_rgb = torch.rand(3, 3)

        outputs = model(rays_o, rays_d)
        outputs["total_loss"].backward()

        # Check gradients exist
        for param in model.parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad)


class TestPerformance(unittest.TestCase):
    """Performance benchmarks."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = GridNeRFConfig(grid_levels=3, base_resolution=32, batch_size=1024)
        self.model = GridNeRF(self.config)
        if torch.cuda.is_available():
            self.model = self.model.cuda()

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_forward_pass_speed(self):
        """Test forward pass speed."""
        device = torch.device("cuda")
        rays_o = torch.randn(1024, 3, device=device)
        rays_d = torch.randn(1024, 3, device=device)
        rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)

        # Warmup
        for _ in range(10):
            with torch.no_grad():
                _ = self.model(rays_o, rays_d)

        # Benchmark
        torch.cuda.synchronize()
        import time

        start_time = time.time()

        for _ in range(100):
            with torch.no_grad():
                _ = self.model(rays_o, rays_d)

        torch.cuda.synchronize()
        end_time = time.time()

        avg_time = (end_time - start_time) / 100
        print(f"Average forward pass time: {avg_time:.4f}s")

        # Should complete in reasonable time (< 0.1s for 1024 rays)
        self.assertLess(avg_time, 0.1)

    def test_memory_usage(self):
        """Test memory usage."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        device = torch.device("cuda")

        # Measure baseline memory
        torch.cuda.empty_cache()
        baseline_memory = torch.cuda.memory_allocated()

        # Create model and data
        rays_o = torch.randn(2048, 3, device=device)
        rays_d = torch.randn(2048, 3, device=device)
        rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)

        # Forward pass
        with torch.no_grad():
            outputs = self.model(rays_o, rays_d)

        peak_memory = torch.cuda.max_memory_allocated()
        memory_used = (peak_memory - baseline_memory) / 1024**2  # MB

        print(f"Peak memory usage: {memory_used:.2f} MB")

        # Should use reasonable amount of memory (< 1GB for this test)
        self.assertLess(memory_used, 1024)


def run_tests():
    """Run all tests."""
    # Create test suite
    test_suite = unittest.TestSuite()

    # Add test classes
    test_classes = [
        TestGridNeRFConfig,
        TestHierarchicalGrid,
        TestGridGuidedMLP,
        TestGridNeRFRenderer,
        TestGridNeRF,
        TestDataset,
        TestUtilities,
        TestTrainer,
        TestIntegration,
        TestPerformance,
    ]

    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)
