"""
Test script for Nerfacto package.

This script contains unit tests and integration tests for the Nerfacto
neural radiance fields implementation.
"""

import torch
import numpy as np
import unittest
import tempfile
import os
import json
from pathlib import Path
import sys

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from nerfs.nerfacto.core import NerfactoModel, NerfactoConfig, NerfactoLoss
from nerfs.nerfacto.dataset import NerfactoDataset, NerfactoDatasetConfig
from nerfs.nerfacto.trainer import NerfactoTrainer, NerfactoTrainerConfig
from nerfs.nerfacto.utils.camera_utils import generate_rays, sample_rays_uniform


class TestNerfactoCore(unittest.TestCase):
    """Test cases for Nerfacto core components."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.config = NerfactoConfig(
            num_levels=4, # Smaller for testing
            base_resolution=16, max_resolution=128, hidden_dim=32, num_layers=2
        )
    
    def test_model_creation(self):
        """Test model creation and basic properties."""
        model = NerfactoModel(self.config)
        
        # Check if model is created
        self.assertIsInstance(model, NerfactoModel)
        
        # Check if model has required components
        self.assertTrue(hasattr(model, 'field'))
        self.assertTrue(hasattr(model, 'renderer'))
        
        # Count parameters
        num_params = sum(p.numel() for p in model.parameters())
        self.assertGreater(num_params, 0)
        print(f"Model created with {num_params:, } parameters")
    
    def test_model_forward(self):
        """Test model forward pass."""
        model = NerfactoModel(self.config).to(self.device)
        
        # Create test input
        batch_size = 100
        ray_origins = torch.randn(batch_size, 3, device=self.device)
        ray_directions = torch.randn(batch_size, 3, device=self.device)
        ray_directions = ray_directions / torch.norm(ray_directions, dim=-1, keepdim=True)
        
        # Forward pass
        with torch.no_grad():
            outputs = model(ray_origins, ray_directions)
        
        # Check outputs
        self.assertIn('colors', outputs)
        self.assertIn('densities', outputs)
        self.assertIn('depths', outputs)
        
        # Check shapes
        self.assertEqual(outputs['colors'].shape, (batch_size, 3))
        self.assertEqual(outputs['densities'].shape, (batch_size, ))
        self.assertEqual(outputs['depths'].shape, (batch_size, ))
        
        # Check value ranges
        self.assertTrue(torch.all(outputs['colors'] >= 0))
        self.assertTrue(torch.all(outputs['colors'] <= 1))
        self.assertTrue(torch.all(outputs['densities'] >= 0))
        
        print("Forward pass test passed")
    
    def test_loss_computation(self):
        """Test loss function."""
        loss_fn = NerfactoLoss(self.config)
        
        batch_size = 100
        
        # Create test outputs and targets
        outputs = {
            'colors': torch.rand(
                batch_size,
                3,
            )
        }
        
        targets = {
            'colors': torch.rand(batch_size, 3)
        }
        
        # Compute loss
        losses = loss_fn(outputs, targets)
        
        # Check if losses are computed
        self.assertIn('color_loss', losses)
        self.assertIsInstance(losses['color_loss'], torch.Tensor)
        
        # Check if loss is scalar
        self.assertEqual(losses['color_loss'].dim(), 0)
        
        # Check if loss is positive
        self.assertGreaterEqual(losses['color_loss'].item(), 0)
        
        print("Loss computation test passed")


class TestNerfactoDataset(unittest.TestCase):
    """Test cases for Nerfacto dataset."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.create_sample_dataset()
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def create_sample_dataset(self):
        """Create a minimal sample dataset for testing."""
        # Create images directory
        images_dir = os.path.join(self.temp_dir, "images")
        os.makedirs(images_dir, exist_ok=True)
        
        # Create sample transforms.json
        transforms = {
            "camera_angle_x": 0.6911112070083618, "frames": [
                {
                    "file_path": "./images/frame_001", "transform_matrix": [
                        [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 4], [0, 0, 0, 1]
                    ]
                }, {
                    "file_path": "./images/frame_002", "transform_matrix": [
                        [0, 0, 1, 4], [0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 1]
                    ]
                }
            ]
        }
        
        # Save transforms file
        with open(os.path.join(self.temp_dir, "transforms.json"), 'w') as f:
            json.dump(transforms, f, indent=2)
        
        # Create dummy images
        from PIL import Image
        dummy_image = Image.new('RGB', (100, 100), color='red')
        dummy_image.save(os.path.join(images_dir, "frame_001.png"))
        dummy_image.save(os.path.join(images_dir, "frame_002.png"))
    
    def test_dataset_creation(self):
        """Test dataset creation."""
        config = NerfactoDatasetConfig(
            data_dir=self.temp_dir, data_format="instant_ngp"
        )
        
        dataset = NerfactoDataset(config, split="train")
        
        # Check if dataset is created
        self.assertIsInstance(dataset, NerfactoDataset)
        self.assertGreater(len(dataset), 0)
        
        print(f"Dataset created with {len(dataset)} samples")
    
    def test_dataset_getitem(self):
        """Test dataset item retrieval."""
        config = NerfactoDatasetConfig(
            data_dir=self.temp_dir, data_format="instant_ngp", image_width=50, # Small for testing
            image_height=50
        )
        
        dataset = NerfactoDataset(config, split="train")
        
        if len(dataset) > 0:
            # Get first item
            item = dataset[0]
            
            # Check required keys
            required_keys = ['image', 'pose', 'intrinsics', 'rays_o', 'rays_d', 'colors']
            for key in required_keys:
                self.assertIn(key, item)
            
            # Check shapes
            self.assertEqual(item['image'].shape, (50, 50, 3))
            self.assertEqual(item['pose'].shape, (4, 4))
            self.assertEqual(item['intrinsics'].shape, (4, ))
            
            print("Dataset getitem test passed")


class TestNerfactoUtils(unittest.TestCase):
    """Test cases for Nerfacto utilities."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def test_ray_generation(self):
        """Test ray generation utilities."""
        batch_size = 2
        image_height, image_width = 64, 64
        
        # Create test camera parameters
        camera_poses = torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1)
        camera_intrinsics = torch.tensor([
            [50, 50, 32, 32], # fx, fy, cx, cy
            [50, 50, 32, 32]
        ]).float()
        
        # Generate rays
        ray_origins, ray_directions = generate_rays(
            camera_poses, camera_intrinsics, image_height, image_width, self.device
        )
        
        # Check shapes
        expected_shape = (batch_size, image_height, image_width, 3)
        self.assertEqual(ray_origins.shape, expected_shape)
        self.assertEqual(ray_directions.shape, expected_shape)
        
        # Check ray direction normalization
        ray_norms = torch.norm(ray_directions, dim=-1)
        self.assertTrue(torch.allclose(ray_norms, torch.ones_like(ray_norms), atol=1e-6))
        
        print("Ray generation test passed")
    
    def test_ray_sampling(self):
        """Test ray sampling utilities."""
        batch_size = 2
        image_height, image_width = 32, 32
        num_rays = 100
        
        # Create test rays
        ray_origins = torch.randn(batch_size, image_height, image_width, 3)
        ray_directions = torch.randn(batch_size, image_height, image_width, 3)
        
        # Sample rays
        sampled_origins, sampled_directions = sample_rays_uniform(
            ray_origins, ray_directions, num_rays, self.device
        )
        
        # Check shapes
        self.assertEqual(sampled_origins.shape, (num_rays, 3))
        self.assertEqual(sampled_directions.shape, (num_rays, 3))
        
        print("Ray sampling test passed")


class TestNerfactoIntegration(unittest.TestCase):
    """Integration tests for Nerfacto."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.temp_dir = tempfile.mkdtemp()
        self.create_test_data()
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def create_test_data(self):
        """Create test data for integration test."""
        # Create a minimal dataset
        images_dir = os.path.join(self.temp_dir, "images")
        os.makedirs(images_dir, exist_ok=True)
        
        transforms = {
            "camera_angle_x": 0.6911112070083618, "frames": []
        }
        
        # Create a few frames
        for i in range(3):
            frame = {
                "file_path": f"./images/frame_{i:03d}", "transform_matrix": [
                    [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 4], [0, 0, 0, 1]
                ]
            }
            transforms["frames"].append(frame)
            
            # Create dummy image
            from PIL import Image
            dummy_image = Image.new('RGB', (64, 64), color=(i*80, 100, 200-i*50))
            dummy_image.save(os.path.join(images_dir, f"frame_{i:03d}.png"))
        
        with open(os.path.join(self.temp_dir, "transforms.json"), 'w') as f:
            json.dump(transforms, f, indent=2)
    
    def test_end_to_end_training(self):
        """Test end-to-end training pipeline."""
        # Create configurations
        model_config = NerfactoConfig(
            num_levels=4, base_resolution=16, max_resolution=64, hidden_dim=32, num_layers=2
        )
        
        dataset_config = NerfactoDatasetConfig(
            data_dir=self.temp_dir, data_format="instant_ngp", image_width=32, image_height=32
        )
        
        trainer_config = NerfactoTrainerConfig(
            max_epochs=2, # Very short for testing
            learning_rate=1e-3, batch_size=1, eval_every_n_epochs=1, save_every_n_epochs=1, output_dir=self.temp_dir, experiment_name="test_experiment", use_mixed_precision=False, # Disable for stability
            use_wandb=False
        )
        
        # Create trainer
        trainer = NerfactoTrainer(
            config=trainer_config, model_config=model_config, dataset_config=dataset_config, device=self.device
        )
        
        # Run short training
        try:
            trainer.train()
            print("Integration test passed - training completed")
        except Exception as e:
            self.fail(f"Integration test failed: {str(e)}")


def run_all_tests():
    """Run all Nerfacto tests."""
    print("=" * 60)
    print("Running Nerfacto Tests")
    print("=" * 60)
    
    # Create test suite
    test_classes = [
        TestNerfactoCore, TestNerfactoDataset, TestNerfactoUtils, TestNerfactoIntegration
    ]
    
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"  {test}: {traceback}")
    
    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"  {test}: {traceback}")
    
    success = len(result.failures) == 0 and len(result.errors) == 0
    print(f"\nOverall result: {'PASSED' if success else 'FAILED'}")
    
    return success


if __name__ == "__main__":
    # Run all tests
    run_all_tests() 