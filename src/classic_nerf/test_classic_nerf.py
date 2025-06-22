"""
Test suite for Classic NeRF implementation.

This module contains comprehensive tests for all components of the Classic NeRF
implementation including models, datasets, training, and utilities.
"""

import unittest
import torch
import numpy as np
import tempfile
import os
import json
from unittest.mock import patch, MagicMock

from classic_nerf import (
    NeRFConfig,
    NeRF,
    Embedder, 
    NeRFRenderer,
    NeRFLoss,
    BlenderDataset,
    NeRFTrainer,
    raw2outputs,
    sample_pdf,
    get_rays_np,
    pose_spherical,
    to8b,
    img2mse,
    mse2psnr
)


class TestNeRFConfig(unittest.TestCase):
    """Test NeRF configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = NeRFConfig()
        
        self.assertEqual(config.netdepth, 8)
        self.assertEqual(config.netwidth, 256)
        self.assertEqual(config.N_samples, 64)
        self.assertEqual(config.N_importance, 128)
        self.assertTrue(config.use_viewdirs)
        self.assertEqual(config.learning_rate, 5e-4)
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = NeRFConfig(
            netdepth=6,
            netwidth=128,
            N_samples=32,
            learning_rate=1e-3
        )
        
        self.assertEqual(config.netdepth, 6)
        self.assertEqual(config.netwidth, 128)
        self.assertEqual(config.N_samples, 32)
        self.assertEqual(config.learning_rate, 1e-3)


class TestEmbedder(unittest.TestCase):
    """Test positional encoding embedder."""
    
    def test_embedder_creation(self):
        """Test embedder creation."""
        embedder = Embedder(input_dims=3, max_freq_log2=9, num_freqs=10)
        
        self.assertEqual(embedder.input_dims, 3)
        self.assertEqual(embedder.max_freq_log2, 9)
        self.assertEqual(embedder.num_freqs, 10)
        self.assertEqual(embedder.out_dim, 3 + 3 * 10 * 2)  # input + freqs * sins/cos
    
    def test_embedder_forward(self):
        """Test embedder forward pass."""
        embedder = Embedder(input_dims=3, max_freq_log2=2, num_freqs=3)
        
        # Test input
        x = torch.randn(100, 3)
        embedded = embedder(x)
        
        self.assertEqual(embedded.shape, (100, embedder.out_dim))
        self.assertFalse(torch.isnan(embedded).any())
        self.assertFalse(torch.isinf(embedded).any())


class TestNeRF(unittest.TestCase):
    """Test NeRF model."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = NeRFConfig(netdepth=4, netwidth=64, multires=4, multires_views=2)
        self.model = NeRF(self.config)
    
    def test_model_creation(self):
        """Test model creation."""
        self.assertIsInstance(self.model, NeRF)
        self.assertEqual(len(self.model.pts_linears), self.config.netdepth)
        
        # Check parameter count
        total_params = sum(p.numel() for p in self.model.parameters())
        self.assertGreater(total_params, 1000)  # Should have reasonable number of parameters
    
    def test_forward_pass(self):
        """Test forward pass."""
        batch_size = 32
        
        # Create test input
        pts_embedded = torch.randn(batch_size, self.model.embed_fn.out_dim)
        
        if self.config.use_viewdirs:
            dirs_embedded = torch.randn(batch_size, self.model.embeddirs_fn.out_dim)
            test_input = torch.cat([pts_embedded, dirs_embedded], -1)
        else:
            test_input = pts_embedded
        
        # Forward pass
        output = self.model(test_input)
        
        self.assertEqual(output.shape, (batch_size, 4))  # RGB + density
        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())
    
    def test_model_without_viewdirs(self):
        """Test model without view directions."""
        config = NeRFConfig(use_viewdirs=False, netdepth=4, netwidth=64, multires=4)
        model = NeRF(config)
        
        batch_size = 32
        pts_embedded = torch.randn(batch_size, model.embed_fn.out_dim)
        
        output = model(pts_embedded)
        self.assertEqual(output.shape, (batch_size, 4))


class TestVolumeRendering(unittest.TestCase):
    """Test volume rendering functions."""
    
    def test_raw2outputs(self):
        """Test raw2outputs function."""
        batch_size = 10
        n_samples = 64
        
        # Create test data
        raw = torch.randn(batch_size, n_samples, 4)
        z_vals = torch.linspace(2.0, 6.0, n_samples).expand(batch_size, n_samples)
        rays_d = torch.randn(batch_size, 3)
        rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
        
        # Test rendering
        outputs = raw2outputs(raw, z_vals, rays_d)
        
        self.assertIn('rgb_map', outputs)
        self.assertIn('depth_map', outputs)
        self.assertIn('acc_map', outputs)
        self.assertIn('weights', outputs)
        
        self.assertEqual(outputs['rgb_map'].shape, (batch_size, 3))
        self.assertEqual(outputs['depth_map'].shape, (batch_size,))
        self.assertEqual(outputs['acc_map'].shape, (batch_size,))
        self.assertEqual(outputs['weights'].shape, (batch_size, n_samples))
    
    def test_sample_pdf(self):
        """Test PDF sampling function."""
        batch_size = 5
        n_bins = 32
        n_samples = 64
        
        # Create test data
        bins = torch.linspace(0, 1, n_bins).expand(batch_size, n_bins)
        weights = torch.rand(batch_size, n_bins - 1)
        
        # Test sampling
        samples = sample_pdf(bins, weights, n_samples, det=True)
        
        self.assertEqual(samples.shape, (batch_size, n_samples))
        self.assertTrue((samples >= 0).all())
        self.assertTrue((samples <= 1).all())


class TestDataset(unittest.TestCase):
    """Test dataset functionality."""
    
    def setUp(self):
        """Set up test dataset."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create minimal dataset structure
        for split in ['train', 'val', 'test']:
            transforms = {
                'camera_angle_x': 0.6911112070083618,
                'frames': [
                    {
                        'file_path': f'./{split}/r_{i:03d}',
                        'transform_matrix': np.eye(4).tolist()
                    } for i in range(5)
                ]
            }
            
            with open(os.path.join(self.temp_dir, f'transforms_{split}.json'), 'w') as f:
                json.dump(transforms, f)
    
    def test_blender_dataset_creation(self):
        """Test Blender dataset creation."""
        dataset = BlenderDataset(
            basedir=self.temp_dir,
            split='train',
            half_res=True,
            white_bkgd=True
        )
        
        self.assertGreater(len(dataset), 0)
        
        # Test data loading
        sample = dataset[0]
        self.assertIn('rays_o', sample)
        self.assertIn('rays_d', sample)
        self.assertIn('target', sample)
        
        self.assertEqual(sample['rays_o'].shape, (3,))
        self.assertEqual(sample['rays_d'].shape, (3,))
        self.assertEqual(sample['target'].shape, (3,))
    
    def tearDown(self):
        """Clean up test files."""
        import shutil
        shutil.rmtree(self.temp_dir)


class TestTrainer(unittest.TestCase):
    """Test NeRF trainer."""
    
    def setUp(self):
        """Set up trainer test."""
        self.config = NeRFConfig(
            netdepth=2, 
            netwidth=32, 
            N_samples=8, 
            N_importance=8,
            multires=2,
            multires_views=1
        )
        self.device = torch.device('cpu')  # Use CPU for testing
        self.trainer = NeRFTrainer(self.config, device=self.device)
    
    def test_trainer_creation(self):
        """Test trainer creation."""
        self.assertIsInstance(self.trainer, NeRFTrainer)
        self.assertIsInstance(self.trainer.model_coarse, NeRF)
        self.assertIsInstance(self.trainer.model_fine, NeRF)
        self.assertIsNotNone(self.trainer.optimizer)
    
    def test_train_step(self):
        """Test single training step."""
        batch_size = 16
        
        # Create synthetic batch
        batch = {
            'rays_o': torch.randn(batch_size, 3),
            'rays_d': torch.randn(batch_size, 3),
            'targets': torch.rand(batch_size, 3)
        }
        
        # Normalize ray directions
        batch['rays_d'] = batch['rays_d'] / torch.norm(batch['rays_d'], dim=-1, keepdim=True)
        
        # Test training step
        losses = self.trainer.train_step(batch)
        
        self.assertIsInstance(losses, dict)
        self.assertIn('total_loss', losses)
        self.assertIn('psnr', losses)
        self.assertGreater(losses['psnr'], 0)


class TestUtilities(unittest.TestCase):
    """Test utility functions."""
    
    def test_to8b(self):
        """Test to8b conversion."""
        x = torch.rand(100, 100, 3)
        result = to8b(x.numpy())
        
        self.assertEqual(result.dtype, np.uint8)
        self.assertEqual(result.shape, (100, 100, 3))
        self.assertTrue((result >= 0).all())
        self.assertTrue((result <= 255).all())
    
    def test_img2mse(self):
        """Test MSE calculation."""
        img1 = torch.rand(10, 10, 3)
        img2 = torch.rand(10, 10, 3)
        
        mse = img2mse(img1, img2)
        self.assertIsInstance(mse, torch.Tensor)
        self.assertEqual(mse.shape, ())
        self.assertGreaterEqual(mse.item(), 0)
    
    def test_mse2psnr(self):
        """Test PSNR calculation."""
        mse = torch.tensor(0.01)
        psnr = mse2psnr(mse)
        
        self.assertIsInstance(psnr, torch.Tensor)
        self.assertGreater(psnr.item(), 0)
    
    def test_pose_spherical(self):
        """Test spherical pose generation."""
        pose = pose_spherical(45.0, -30.0, 4.0)
        
        self.assertEqual(pose.shape, (4, 4))
        self.assertTrue(np.allclose(pose[3, :], [0, 0, 0, 1]))
    
    def test_get_rays_np(self):
        """Test ray generation."""
        H, W = 100, 100
        K = np.array([[100, 0, 50], [0, 100, 50], [0, 0, 1]])
        c2w = np.eye(4)
        
        rays_o, rays_d = get_rays_np(H, W, K, c2w)
        
        self.assertEqual(rays_o.shape, (H, W, 3))
        self.assertEqual(rays_d.shape, (H, W, 3))
        
        # Check ray directions are normalized
        norms = np.linalg.norm(rays_d, axis=-1)
        self.assertTrue(np.allclose(norms, 1.0))


class TestLoss(unittest.TestCase):
    """Test loss function."""
    
    def test_nerf_loss(self):
        """Test NeRF loss computation."""
        config = NeRFConfig()
        criterion = NeRFLoss(config)
        
        batch_size = 32
        
        # Create test predictions and targets
        pred = {
            'rgb_map': torch.rand(batch_size, 3),
            'rgb_map0': torch.rand(batch_size, 3)  # Coarse network output
        }
        target = torch.rand(batch_size, 3)
        
        # Compute loss
        losses = criterion(pred, target)
        
        self.assertIn('rgb_loss', losses)
        self.assertIn('rgb_loss0', losses)
        self.assertIn('total_loss', losses)
        self.assertIn('psnr', losses)
        
        self.assertGreater(losses['total_loss'].item(), 0)
        self.assertGreater(losses['psnr'].item(), 0)


def run_tests():
    """Run all tests."""
    # Create test suite
    test_classes = [
        TestNeRFConfig,
        TestEmbedder,
        TestNeRF,
        TestVolumeRendering,
        TestDataset,
        TestTrainer,
        TestUtilities,
        TestLoss
    ]
    
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result


if __name__ == '__main__':
    print("Running Classic NeRF Test Suite")
    print("=" * 50)
    
    result = run_tests()
    
    print("\n" + "=" * 50)
    if result.wasSuccessful():
        print("All tests passed! ✅")
    else:
        print(f"Tests failed: {len(result.failures)} failures, {len(result.errors)} errors ❌")
        
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
