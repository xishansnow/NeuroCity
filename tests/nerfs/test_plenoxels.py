"""
Tests for Plenoxels implementation

This module contains unit tests for the core Plenoxels components.
"""

import unittest
import torch
import numpy as np
from typing import Dict, Any
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from nerfs.plenoxels.core import (
    PlenoxelConfig, SphericalHarmonics, VoxelGrid, PlenoxelModel, PlenoxelLoss
)
from nerfs.plenoxels.dataset import PlenoxelDatasetConfig
from nerfs.plenoxels.trainer import PlenoxelTrainerConfig


class TestSphericalHarmonics(unittest.TestCase):
    """Test spherical harmonics utilities."""
    
    def test_sh_coeffs_count(self):
        """Test SH coefficient counting."""
        self.assertEqual(SphericalHarmonics.get_num_coeffs(0), 1)
        self.assertEqual(SphericalHarmonics.get_num_coeffs(1), 4)
        self.assertEqual(SphericalHarmonics.get_num_coeffs(2), 9)
        self.assertEqual(SphericalHarmonics.get_num_coeffs(3), 16)
    
    def test_sh_evaluation(self):
        """Test spherical harmonics evaluation."""
        device = torch.device('cpu')
        
        # Test directions
        dirs = torch.tensor([
            [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]
        ], device=device)
        
        # Test degree 0
        sh_0 = SphericalHarmonics.eval_sh(0, dirs)
        self.assertEqual(sh_0.shape, (3, 1))
        
        # Test degree 1
        sh_1 = SphericalHarmonics.eval_sh(1, dirs)
        self.assertEqual(sh_1.shape, (3, 4))
        
        # Test degree 2
        sh_2 = SphericalHarmonics.eval_sh(2, dirs)
        self.assertEqual(sh_2.shape, (3, 9))
    
    def test_sh_color_evaluation(self):
        """Test SH color evaluation."""
        device = torch.device('cpu')
        
        dirs = torch.tensor([
            [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]
        ], device=device)
        
        # SH coefficients for RGB
        sh_coeffs = torch.randn(2, 3, 4, device=device)  # degree 1
        
        colors = SphericalHarmonics.eval_sh_color(sh_coeffs, dirs)
        self.assertEqual(colors.shape, (2, 3))
        
        # Colors should be in [0, 1] range due to sigmoid
        self.assertTrue(torch.all(colors >= 0))
        self.assertTrue(torch.all(colors <= 1))


class TestVoxelGrid(unittest.TestCase):
    """Test voxel grid functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device('cpu')
        self.resolution = (32, 32, 32)
        self.scene_bounds = (-1.0, -1.0, -1.0, 1.0, 1.0, 1.0)
        self.sh_degree = 1
        
        self.voxel_grid = VoxelGrid(
            self.resolution, self.scene_bounds, self.sh_degree
        ).to(self.device)
    
    def test_voxel_grid_creation(self):
        """Test voxel grid creation."""
        # Check parameter shapes
        self.assertEqual(self.voxel_grid.density.shape, self.resolution)
        
        expected_sh_shape = (*self.resolution, 3, 4)  # degree 1 has 4 coeffs
        self.assertEqual(self.voxel_grid.sh_coeffs.shape, expected_sh_shape)
    
    def test_coordinate_conversion(self):
        """Test world to voxel coordinate conversion."""
        # Test center point
        world_coords = torch.tensor([[0.0, 0.0, 0.0]], device=self.device)
        voxel_coords = self.voxel_grid.world_to_voxel_coords(world_coords)
        
        expected_center = torch.tensor([[15.5, 15.5, 15.5]], device=self.device)
        self.assertTrue(torch.allclose(voxel_coords, expected_center, atol=1e-5))
    
    def test_trilinear_interpolation(self):
        """Test trilinear interpolation."""
        # Test interpolation at center
        world_coords = torch.tensor([[0.0, 0.0, 0.0]], device=self.device)
        density, sh_coeffs = self.voxel_grid.trilinear_interpolation(world_coords)
        
        self.assertEqual(density.shape, (1, ))
        self.assertEqual(sh_coeffs.shape, (1, 3, 4))
    
    def test_occupancy_mask(self):
        """Test occupancy mask computation."""
        mask = self.voxel_grid.get_occupancy_mask(threshold=0.01)
        self.assertEqual(mask.shape, self.resolution)
        self.assertTrue(torch.all((mask == 0) | (mask == 1)))
    
    def test_total_variation_loss(self):
        """Test total variation loss computation."""
        tv_loss = self.voxel_grid.total_variation_loss()
        self.assertIsInstance(tv_loss, torch.Tensor)
        self.assertEqual(tv_loss.shape, ())
        self.assertTrue(tv_loss >= 0)
    
    def test_l1_loss(self):
        """Test L1 sparsity loss computation."""
        l1_loss = self.voxel_grid.l1_loss()
        self.assertIsInstance(l1_loss, torch.Tensor)
        self.assertEqual(l1_loss.shape, ())
        self.assertTrue(l1_loss >= 0)


class TestPlenoxelModel(unittest.TestCase):
    """Test Plenoxel model functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device('cpu')
        self.config = PlenoxelConfig(
            grid_resolution=(
                32,
                32,
                32,
            )
        )
        self.model = PlenoxelModel(self.config).to(self.device)
    
    def test_model_creation(self):
        """Test model creation."""
        self.assertIsInstance(self.model.voxel_grid, VoxelGrid)
        self.assertEqual(self.model.config, self.config)
    
    def test_forward_pass(self):
        """Test model forward pass."""
        # Create test rays
        num_rays = 10
        rays_o = torch.randn(num_rays, 3, device=self.device)
        rays_d = torch.randn(num_rays, 3, device=self.device)
        rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
        
        # Forward pass
        outputs = self.model(rays_o, rays_d, num_samples=32)
        
        # Check output shapes
        self.assertEqual(outputs['rgb'].shape, (num_rays, 3))
        self.assertEqual(outputs['depth'].shape, (num_rays, ))
        self.assertEqual(outputs['weights'].shape, (num_rays, 32))
        
        # Check value ranges
        self.assertTrue(torch.all(outputs['rgb'] >= 0))
        self.assertTrue(torch.all(outputs['rgb'] <= 1))
        self.assertTrue(torch.all(outputs['depth'] >= 0))
    
    def test_occupancy_stats(self):
        """Test occupancy statistics."""
        stats = self.model.get_occupancy_stats()
        
        required_keys = ['total_voxels', 'occupied_voxels', 'occupancy_ratio', 'sparsity_ratio']
        for key in required_keys:
            self.assertIn(key, stats)
        
        self.assertEqual(stats['total_voxels'], 32 * 32 * 32)
        self.assertTrue(0 <= stats['occupancy_ratio'] <= 1)
        self.assertTrue(0 <= stats['sparsity_ratio'] <= 1)
        self.assertAlmostEqual(stats['occupancy_ratio'] + stats['sparsity_ratio'], 1.0, places=5)


class TestPlenoxelLoss(unittest.TestCase):
    """Test Plenoxel loss function."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device('cpu')
        self.config = PlenoxelConfig()
        self.loss_fn = PlenoxelLoss(self.config)
    
    def test_color_loss(self):
        """Test color reconstruction loss."""
        # Create test outputs and targets
        num_rays = 100
        outputs = {
            'rgb': torch.rand(num_rays, 3, device=self.device)
        }
        targets = {
            'colors': torch.rand(num_rays, 3, device=self.device)
        }
        
        losses = self.loss_fn(outputs, targets)
        
        self.assertIn('color_loss', losses)
        self.assertIsInstance(losses['color_loss'], torch.Tensor)
        self.assertEqual(losses['color_loss'].shape, ())
        self.assertTrue(losses['color_loss'] >= 0)


class TestConfigurations(unittest.TestCase):
    """Test configuration classes."""
    
    def test_plenoxel_config(self):
        """Test PlenoxelConfig."""
        config = PlenoxelConfig()
        
        # Test default values
        self.assertEqual(config.grid_resolution, (256, 256, 256))
        self.assertEqual(config.sh_degree, 2)
        self.assertTrue(config.use_coarse_to_fine)
        
        # Test custom values
        custom_config = PlenoxelConfig(
            grid_resolution=(128, 128, 128), sh_degree=1, use_coarse_to_fine=False
        )
        
        self.assertEqual(custom_config.grid_resolution, (128, 128, 128))
        self.assertEqual(custom_config.sh_degree, 1)
        self.assertFalse(custom_config.use_coarse_to_fine)
    
    def test_dataset_config(self):
        """Test PlenoxelDatasetConfig."""
        config = PlenoxelDatasetConfig()
        
        # Test default values
        self.assertEqual(config.dataset_type, "blender")
        self.assertEqual(config.downsample_factor, 1)
        self.assertFalse(config.white_background)
    
    def test_trainer_config(self):
        """Test PlenoxelTrainerConfig."""
        config = PlenoxelTrainerConfig()
        
        # Test default values
        self.assertEqual(config.max_epochs, 10000)
        self.assertEqual(config.learning_rate, 0.1)
        self.assertTrue(config.use_tensorboard)


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete pipeline."""
    
    def test_end_to_end_forward(self):
        """Test end-to-end forward pass."""
        device = torch.device('cpu')
        
        # Create small model for testing
        config = PlenoxelConfig(
            grid_resolution=(16, 16, 16), sh_degree=1, near_plane=0.5, far_plane=2.0
        )
        model = PlenoxelModel(config).to(device)
        
        # Create batch of rays
        batch_size = 5
        rays_o = torch.randn(batch_size, 3, device=device)
        rays_d = torch.randn(batch_size, 3, device=device)
        rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
        
        # Forward pass
        with torch.no_grad():
            outputs = model(rays_o, rays_d, num_samples=16)
        
        # Verify outputs
        self.assertEqual(outputs['rgb'].shape, (batch_size, 3))
        self.assertEqual(outputs['depth'].shape, (batch_size, ))
        
        # Test loss computation
        loss_fn = PlenoxelLoss(config)
        target_colors = torch.rand(batch_size, 3, device=device)
        losses = loss_fn(outputs, {'colors': target_colors})
        
        self.assertIn('color_loss', losses)
        self.assertTrue(losses['color_loss'] >= 0)


def run_tests():
    """Run all tests."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestSphericalHarmonics, TestVoxelGrid, TestPlenoxelModel, TestPlenoxelLoss, TestConfigurations, TestIntegration
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    # Run tests
    success = run_tests()
    
    if success:
        print("\nAll tests passed! ✅")
    else:
        print("\nSome tests failed! ❌")
        exit(1) 