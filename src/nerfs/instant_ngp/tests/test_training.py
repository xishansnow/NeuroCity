"""
Tests for Instant NGP training and inference
"""

import unittest
import torch
import numpy as np
import tempfile
import shutil
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Use relative imports for local testing
from core import (
    InstantNGPConfig,
    InstantNGPModel,
)
from trainer import (
    InstantNGPTrainer,
    InstantNGPTrainerConfig,
)
from renderer import (
    InstantNGPInferenceRenderer,
    InstantNGPRendererConfig
)


class TestInstantNGPTraining(unittest.TestCase):
    """Test Instant NGP training functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.temp_dir = tempfile.mkdtemp()
        
        # Small model for testing
        self.model_config = InstantNGPConfig(
            num_levels=4,
            base_resolution=16,
            finest_resolution=64,
            hidden_dim=32,
            num_layers=2
        )
        
        self.trainer_config = InstantNGPTrainerConfig(
            num_epochs=2,
            batch_size=512,
            learning_rate=1e-2,
            learning_rate_hash=1e-1,
            log_freq=50,
            checkpoint_dir=self.temp_dir
        )
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_trainer_creation(self):
        """Test trainer creation."""
        model = InstantNGPModel(self.model_config)
        trainer = InstantNGPTrainer(model, self.trainer_config, device=self.device)
        
        # Check trainer components
        self.assertIsNotNone(trainer.model)
        self.assertIsNotNone(trainer.optimizer)
        self.assertIsNotNone(trainer.scheduler)
        self.assertIsNotNone(trainer.criterion)
        
        # Check device
        self.assertEqual(trainer.device, self.device)
    
    def test_optimizer_setup(self):
        """Test optimizer parameter groups."""
        model = InstantNGPModel(self.model_config)
        trainer = InstantNGPTrainer(model, self.trainer_config, device=self.device)
        
        # Check parameter groups
        param_groups = trainer.optimizer.param_groups
        self.assertEqual(len(param_groups), 2)  # hash and mlp groups
        
        # Check learning rates
        hash_lr = param_groups[0]['lr']
        mlp_lr = param_groups[1]['lr']
        
        self.assertEqual(hash_lr, self.trainer_config.learning_rate_hash)
        self.assertEqual(mlp_lr, self.trainer_config.learning_rate)
    
    def test_training_step(self):
        """Test single training step."""
        model = InstantNGPModel(self.model_config)
        trainer = InstantNGPTrainer(model, self.trainer_config, device=self.device)
        
        # Create mock batch
        batch_size = 256
        batch = {
            "rays_o": torch.rand(batch_size, 3, device=self.device),
            "rays_d": torch.nn.functional.normalize(
                torch.randn(batch_size, 3, device=self.device), dim=-1
            ),
            "rgb": torch.rand(batch_size, 3, device=self.device)
        }
        
        # Training step
        losses = trainer.train_step(batch)
        
        # Check loss components
        self.assertIn("total_loss", losses)
        self.assertIn("rgb_loss", losses)
        self.assertIn("psnr", losses)
        
        # Check losses are reasonable
        self.assertGreater(losses["total_loss"], 0)
        self.assertGreater(losses["psnr"], 0)
    
    def test_volume_rendering(self):
        """Test volume rendering in trainer."""
        model = InstantNGPModel(self.model_config)
        trainer = InstantNGPTrainer(model, self.trainer_config, device=self.device)
        
        # Test input
        batch_size = 128
        rays_o = torch.rand(batch_size, 3, device=self.device)
        rays_d = torch.nn.functional.normalize(
            torch.randn(batch_size, 3, device=self.device), dim=-1
        )
        near = torch.full((batch_size, 1), 0.1, device=self.device)
        far = torch.full((batch_size, 1), 5.0, device=self.device)
        
        # Volume rendering
        with torch.no_grad():
            outputs = trainer._volume_render(rays_o, rays_d, near, far)
        
        # Check outputs
        self.assertIn("rgb", outputs)
        self.assertIn("depth", outputs)
        self.assertIn("acc", outputs)
        
        # Check shapes
        self.assertEqual(outputs["rgb"].shape, (batch_size, 3))
        self.assertEqual(outputs["depth"].shape, (batch_size,))
        self.assertEqual(outputs["acc"].shape, (batch_size,))
        
        # Check value ranges
        self.assertTrue((outputs["rgb"] >= 0).all())
        self.assertTrue((outputs["rgb"] <= 1).all())
        self.assertTrue((outputs["depth"] >= 0).all())
        self.assertTrue((outputs["acc"] >= 0).all())
        self.assertTrue((outputs["acc"] <= 1).all())
    
    def test_checkpoint_save_load(self):
        """Test checkpoint saving and loading."""
        model = InstantNGPModel(self.model_config)
        trainer = InstantNGPTrainer(model, self.trainer_config, device=self.device)
        
        # Save checkpoint
        checkpoint_path = "test_checkpoint.pth"
        trainer.save_checkpoint(checkpoint_path)
        
        full_path = Path(self.temp_dir) / checkpoint_path
        self.assertTrue(full_path.exists())
        
        # Load checkpoint
        new_model = InstantNGPModel(self.model_config)
        new_trainer = InstantNGPTrainer(new_model, self.trainer_config, device=self.device)
        new_trainer.load_checkpoint(str(full_path))
        
        # Check model states are identical
        original_state = trainer.model.state_dict()
        loaded_state = new_trainer.model.state_dict()
        
        for key in original_state:
            torch.testing.assert_close(
                original_state[key], loaded_state[key], rtol=1e-5, atol=1e-6
            )


class TestInstantNGPInference(unittest.TestCase):
    """Test Instant NGP inference functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.temp_dir = tempfile.mkdtemp()
        
        # Small model for testing
        self.model_config = InstantNGPConfig(
            num_levels=4,
            base_resolution=16,
            finest_resolution=64,
            hidden_dim=32,
            num_layers=2
        )
        
        self.renderer_config = InstantNGPRendererConfig(
            num_samples=32,
            num_samples_fine=16,
            chunk_size=1024,
            output_dir=self.temp_dir
        )
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_renderer_creation(self):
        """Test renderer creation."""
        model = InstantNGPModel(self.model_config)
        renderer = InstantNGPInferenceRenderer(
            model, self.renderer_config, device=self.device
        )
        
        # Check renderer components
        self.assertIsNotNone(renderer.model)
        self.assertEqual(renderer.device, self.device)
        self.assertTrue(Path(self.renderer_config.output_dir).exists())
    
    def test_ray_rendering(self):
        """Test ray rendering."""
        model = InstantNGPModel(self.model_config)
        renderer = InstantNGPInferenceRenderer(
            model, self.renderer_config, device=self.device
        )
        
        # Test input
        batch_size = 256
        rays_o = torch.rand(batch_size, 3, device=self.device)
        rays_d = torch.nn.functional.normalize(
            torch.randn(batch_size, 3, device=self.device), dim=-1
        )
        
        # Render rays
        with torch.no_grad():
            outputs = renderer.render_rays(rays_o, rays_d)
        
        # Check outputs
        self.assertIn("rgb", outputs)
        self.assertIn("depth", outputs)
        self.assertIn("acc", outputs)
        
        # Check shapes
        self.assertEqual(outputs["rgb"].shape, (batch_size, 3))
        self.assertEqual(outputs["depth"].shape, (batch_size,))
        self.assertEqual(outputs["acc"].shape, (batch_size,))
        
        # Check value ranges
        self.assertTrue((outputs["rgb"] >= 0).all())
        self.assertTrue((outputs["rgb"] <= 1).all())
    
    def test_image_rendering(self):
        """Test image rendering."""
        model = InstantNGPModel(self.model_config)
        renderer = InstantNGPInferenceRenderer(
            model, self.renderer_config, device=self.device
        )
        
        # Camera parameters
        width, height = 64, 64  # Small for testing
        camera_pose = torch.eye(4, device=self.device)
        camera_pose[2, 3] = 3.0  # Move camera back
        
        focal = width * 0.8
        intrinsics = torch.tensor([
            [focal, 0, width/2],
            [0, focal, height/2],
            [0, 0, 1]
        ], device=self.device, dtype=torch.float32)
        
        # Render image
        with torch.no_grad():
            result = renderer.render_image(camera_pose, intrinsics, width, height)
        
        # Check outputs
        self.assertIn("rgb", result)
        self.assertIn("depth", result)
        
        # Check shapes
        self.assertEqual(result["rgb"].shape, (height, width, 3))
        self.assertEqual(result["depth"].shape, (height, width))
        
        # Check value ranges
        self.assertTrue((result["rgb"] >= 0).all())
        self.assertTrue((result["rgb"] <= 1).all())
    
    def test_chunked_rendering(self):
        """Test chunked rendering for large batches."""
        model = InstantNGPModel(self.model_config)
        
        # Small chunk size for testing
        config = InstantNGPRendererConfig(
            num_samples=16,
            chunk_size=64  # Small chunk
        )
        renderer = InstantNGPInferenceRenderer(model, config, device=self.device)
        
        # Large batch that will be chunked
        batch_size = 200  # Larger than chunk_size
        rays_o = torch.rand(batch_size, 3, device=self.device)
        rays_d = torch.nn.functional.normalize(
            torch.randn(batch_size, 3, device=self.device), dim=-1
        )
        
        # Render rays (should be chunked internally)
        with torch.no_grad():
            outputs = renderer.render_rays(rays_o, rays_d)
        
        # Check outputs are complete
        self.assertEqual(outputs["rgb"].shape, (batch_size, 3))
        self.assertEqual(outputs["depth"].shape, (batch_size,))
    
    def test_hierarchical_sampling(self):
        """Test hierarchical sampling."""
        model = InstantNGPModel(self.model_config)
        
        # Enable hierarchical sampling
        config = InstantNGPRendererConfig(
            num_samples=16,
            num_samples_fine=8,
            use_hierarchical_sampling=True
        )
        renderer = InstantNGPInferenceRenderer(model, config, device=self.device)
        
        # Test input
        batch_size = 64
        rays_o = torch.rand(batch_size, 3, device=self.device)
        rays_d = torch.nn.functional.normalize(
            torch.randn(batch_size, 3, device=self.device), dim=-1
        )
        
        # Render with hierarchical sampling
        with torch.no_grad():
            outputs = renderer.render_rays(rays_o, rays_d)
        
        # Check that fine sampling outputs exist
        self.assertIn("rgb_fine", outputs)
        self.assertIn("depth_fine", outputs)
        
        # Check shapes
        self.assertEqual(outputs["rgb_fine"].shape, (batch_size, 3))
        self.assertEqual(outputs["depth_fine"].shape, (batch_size,))
    
    def test_early_termination(self):
        """Test early termination optimization."""
        model = InstantNGPModel(self.model_config)
        
        # Enable early termination
        config = InstantNGPRendererConfig(
            num_samples=32,
            use_early_termination=True,
            early_termination_threshold=0.95
        )
        renderer = InstantNGPInferenceRenderer(model, config, device=self.device)
        
        # Test with high density to trigger early termination
        batch_size = 32
        rays_o = torch.zeros(batch_size, 3, device=self.device)  # Center rays
        rays_d = torch.tensor([[0, 0, -1]], device=self.device).expand(batch_size, -1)
        
        # Render
        with torch.no_grad():
            outputs = renderer.render_rays(rays_o, rays_d)
        
        # Should still produce valid outputs
        self.assertEqual(outputs["rgb"].shape, (batch_size, 3))
        self.assertTrue((outputs["rgb"] >= 0).all())
        self.assertTrue((outputs["rgb"] <= 1).all())


if __name__ == "__main__":
    unittest.main(verbosity=2)
