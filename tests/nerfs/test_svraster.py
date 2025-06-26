"""
Test script for SVRaster package.

This script contains unit tests and integration tests for the SVRaster
sparse voxel rasterization implementation.
"""

import unittest
import torch
import numpy as np
import tempfile
import os
from pathlib import Path
import sys

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from nerfs.svraster.core import SVRasterConfig, AdaptiveSparseVoxels, VoxelRasterizer, SVRasterModel, SVRasterLoss
from nerfs.svraster.dataset import SVRasterDatasetConfig, SVRasterDataset
from nerfs.svraster.trainer import SVRasterTrainerConfig, SVRasterTrainer
from nerfs.svraster.utils.morton_utils import morton_encode_3d, morton_decode_3d
from nerfs.svraster.utils.octree_utils import octree_subdivision, octree_pruning
from nerfs.svraster.utils.rendering_utils import ray_direction_dependent_ordering, depth_peeling
from nerfs.svraster.utils.voxel_utils import voxel_pruning, compute_voxel_bounds


class TestSVRasterCore(unittest.TestCase):
    """Test core SVRaster components."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = SVRasterConfig(
            max_octree_levels=8, base_resolution=32, scene_bounds=(-1.0, -1.0, -1.0, 1.0, 1.0, 1.0)
        )
        self.device = torch.device('cpu')
    
    def test_config_creation(self):
        """Test SVRaster configuration creation."""
        self.assertEqual(self.config.max_octree_levels, 8)
        self.assertEqual(self.config.base_resolution, 32)
        self.assertEqual(self.config.density_activation, "exp")
        self.assertEqual(self.config.color_activation, "sigmoid")
    
    def test_adaptive_sparse_voxels(self):
        """Test adaptive sparse voxels creation and operations."""
        sparse_voxels = AdaptiveSparseVoxels(self.config)
        
        # Check initialization
        self.assertTrue(len(sparse_voxels.voxel_positions) > 0)
        self.assertTrue(len(sparse_voxels.voxel_densities) > 0)
        
        # Check voxel count
        total_voxels = sparse_voxels.get_total_voxel_count()
        expected_voxels = self.config.base_resolution ** 3
        self.assertEqual(total_voxels, expected_voxels)
        
        # Test getting all voxels
        voxels = sparse_voxels.get_all_voxels()
        self.assertIn('positions', voxels)
        self.assertIn('densities', voxels)
        self.assertIn('colors', voxels)
        self.assertEqual(voxels['positions'].shape[0], total_voxels)
    
    def test_voxel_rasterizer(self):
        """Test voxel rasterizer."""
        rasterizer = VoxelRasterizer(self.config)
        
        # Create dummy voxel data
        num_voxels = 100
        voxels = {
            'positions': torch.randn(
                num_voxels,
                3,
            )
            'levels': torch.zeros(
                num_voxels,
                dtype=torch.int,
            )
        }
        
        # Create dummy rays
        num_rays = 50
        ray_origins = torch.randn(num_rays, 3)
        ray_directions = torch.randn(num_rays, 3)
        ray_directions = ray_directions / torch.norm(ray_directions, dim=1, keepdim=True)
        
        # Test rasterization
        outputs = rasterizer(voxels, ray_origins, ray_directions)
        
        self.assertIn('rgb', outputs)
        self.assertIn('depth', outputs)
        self.assertIn('alpha', outputs)
        self.assertEqual(outputs['rgb'].shape, (num_rays, 3))
        self.assertEqual(outputs['depth'].shape, (num_rays, ))
        self.assertEqual(outputs['alpha'].shape, (num_rays, ))
    
    def test_svraster_model(self):
        """Test complete SVRaster model."""
        model = SVRasterModel(self.config)
        
        # Create dummy rays
        num_rays = 20
        ray_origins = torch.randn(num_rays, 3)
        ray_directions = torch.randn(num_rays, 3)
        ray_directions = ray_directions / torch.norm(ray_directions, dim=1, keepdim=True)
        
        # Forward pass
        outputs = model(ray_origins, ray_directions)
        
        self.assertIn('rgb', outputs)
        self.assertEqual(outputs['rgb'].shape, (num_rays, 3))
        
        # Check output ranges
        self.assertTrue(torch.all(outputs['rgb'] >= 0))
        self.assertTrue(torch.all(outputs['rgb'] <= 1))
    
    def test_svraster_loss(self):
        """Test SVRaster loss function."""
        loss_fn = SVRasterLoss(self.config)
        
        # Create mock model
        model = SVRasterModel(self.config)
        
        # Create dummy outputs and targets
        num_rays = 50
        outputs = {
            'rgb': torch.rand(
                num_rays,
                3,
            )
        }
        targets = {
            'colors': torch.rand(num_rays, 3)
        }
        
        # Compute losses
        losses = loss_fn(outputs, targets, model)
        
        self.assertIn('rgb_loss', losses)
        self.assertIn('total_loss', losses)
        self.assertTrue(losses['total_loss'].item() >= 0)


class TestSVRasterUtils(unittest.TestCase):
    """Test SVRaster utility functions."""
    
    def test_morton_encoding(self):
        """Test Morton code encoding and decoding."""
        # Test basic encoding/decoding
        x, y, z = 5, 3, 7
        morton_code = morton_encode_3d(x, y, z)
        decoded_x, decoded_y, decoded_z = morton_decode_3d(morton_code)
        
        self.assertEqual(x, decoded_x)
        self.assertEqual(y, decoded_y)
        self.assertEqual(z, decoded_z)
        
        # Test multiple coordinates
        coords = [(0, 0, 0), (1, 1, 1), (10, 5, 15)]
        for x, y, z in coords:
            morton_code = morton_encode_3d(x, y, z)
            decoded = morton_decode_3d(morton_code)
            self.assertEqual((x, y, z), decoded)
    
    def test_octree_subdivision(self):
        """Test octree subdivision."""
        # Create test voxels
        num_voxels = 10
        positions = torch.randn(num_voxels, 3)
        sizes = torch.ones(num_voxels)
        subdivision_mask = torch.zeros(num_voxels, dtype=torch.bool)
        subdivision_mask[:3] = True  # Subdivide first 3 voxels
        
        child_positions, child_sizes = octree_subdivision(positions, sizes, subdivision_mask)
        
        # Should create 8 children per subdivided voxel
        expected_children = 3 * 8
        self.assertEqual(child_positions.shape[0], expected_children)
        self.assertEqual(child_sizes.shape[0], expected_children)
        
        # Child sizes should be half of parent sizes
        self.assertTrue(torch.allclose(child_sizes, torch.ones_like(child_sizes) * 0.5))
    
    def test_octree_pruning(self):
        """Test octree pruning."""
        # Create test densities
        densities = torch.tensor([-5.0, -1.0, 0.0, 1.0, 2.0])  # log densities
        threshold = 0.1
        
        keep_mask = octree_pruning(densities, threshold)
        
        # Only high-density voxels should be kept
        expected_keep = torch.exp(densities) > threshold
        self.assertTrue(torch.equal(keep_mask, expected_keep))
    
    def test_voxel_bounds(self):
        """Test voxel bounds computation."""
        positions = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        sizes = torch.tensor([1.0, 2.0])
        
        box_min, box_max = compute_voxel_bounds(positions, sizes)
        
        expected_min = torch.tensor([[-0.5, -0.5, -0.5], [0.0, 0.0, 0.0]])
        expected_max = torch.tensor([[0.5, 0.5, 0.5], [2.0, 2.0, 2.0]])
        
        self.assertTrue(torch.allclose(box_min, expected_min))
        self.assertTrue(torch.allclose(box_max, expected_max))


class TestSVRasterDataset(unittest.TestCase):
    """Test SVRaster dataset functionality."""
    
    def setUp(self):
        """Set up test dataset."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = SVRasterDatasetConfig(
            data_dir=self.temp_dir, dataset_type="blender", image_height=64, image_width=64, train_split=0.8, val_split=0.1, test_split=0.1
        )
    
    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def create_dummy_dataset(self):
        """Create dummy dataset files."""
        # Create dummy images
        from PIL import Image
        import json
        
        for i in range(10):
            img = Image.new('RGB', (64, 64), color=(i*25, i*25, i*25))
            img.save(os.path.join(self.temp_dir, f"image_{i:04d}.png"))
        
        # Create transforms_train.json for blender format
        transforms = {
            "camera_angle_x": 0.6911112070083618, "frames": []
        }
        
        for i in range(10):
            # Create identity transform matrix
            transform_matrix = [
                [1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 2.0], [0.0, 0.0, 0.0, 1.0]
            ]
            
            transforms["frames"].append({
                "file_path": f"image_{i:04d}", "transform_matrix": transform_matrix
            })
        
        with open(os.path.join(self.temp_dir, "transforms_train.json"), 'w') as f:
            json.dump(transforms, f)
    
    def test_dataset_creation(self):
        """Test dataset creation and loading."""
        self.create_dummy_dataset()
        
        # Create dataset
        dataset = SVRasterDataset(self.config, split="train")
        
        # Check dataset properties
        self.assertTrue(len(dataset) > 0)
        self.assertTrue(hasattr(dataset, 'images'))
        self.assertTrue(hasattr(dataset, 'poses'))
        self.assertTrue(hasattr(dataset, 'intrinsics'))
        
        # Test data loading
        sample = dataset[0]
        self.assertIn('rays_o', sample)
        self.assertIn('rays_d', sample)
        self.assertIn('colors', sample)


class TestSVRasterTrainer(unittest.TestCase):
    """Test SVRaster trainer functionality."""
    
    def setUp(self):
        """Set up test trainer."""
        self.model_config = SVRasterConfig(
            max_octree_levels=4, base_resolution=16
        )
        self.trainer_config = SVRasterTrainerConfig(
            num_epochs=2, batch_size=1, learning_rate=1e-3, device="cpu"
        )
        
        # Create dummy dataset
        self.temp_dir = tempfile.mkdtemp()
        self.dataset_config = SVRasterDatasetConfig(
            data_dir=self.temp_dir, dataset_type="blender", image_height=32, image_width=32
        )
    
    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def create_dummy_dataset(self):
        """Create dummy dataset for training."""
        from PIL import Image
        import json
        
        for i in range(5):
            img = Image.new('RGB', (32, 32), color=(50, 100, 150))
            img.save(os.path.join(self.temp_dir, f"image_{i:04d}.png"))
        
        # Create transforms_train.json
        transforms = {
            "camera_angle_x": 0.6911112070083618, "frames": []
        }
        
        for i in range(5):
            transform_matrix = [
                [1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 2.0], [0.0, 0.0, 0.0, 1.0]
            ]
            
            transforms["frames"].append({
                "file_path": f"image_{i:04d}", "transform_matrix": transform_matrix
            })
        
        with open(os.path.join(self.temp_dir, "transforms_train.json"), 'w') as f:
            json.dump(transforms, f)
    
    def test_trainer_creation(self):
        """Test trainer creation."""
        self.create_dummy_dataset()
        
        # Create dataset
        train_dataset = SVRasterDataset(self.dataset_config, split="train")
        
        # Create trainer
        trainer = SVRasterTrainer(
            self.model_config, self.trainer_config, train_dataset
        )
        
        # Check trainer properties
        self.assertIsNotNone(trainer.model)
        self.assertIsNotNone(trainer.optimizer)
        self.assertIsNotNone(trainer.loss_fn)


def run_tests():
    """Run all tests."""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTest(unittest.makeSuite(TestSVRasterCore))
    test_suite.addTest(unittest.makeSuite(TestSVRasterUtils))
    test_suite.addTest(unittest.makeSuite(TestSVRasterDataset))
    test_suite.addTest(unittest.makeSuite(TestSVRasterTrainer))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1) 