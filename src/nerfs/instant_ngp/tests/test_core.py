"""
Tests for Instant NGP core functionality
"""

import unittest
import torch
import numpy as np
from pathlib import Path
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Use relative imports for local testing
from core import InstantNGPConfig, InstantNGPModel, HashEncoder, SHEncoder, InstantNGPLoss


class TestInstantNGPCore(unittest.TestCase):
    """Test core Instant NGP functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = InstantNGPConfig(
            num_levels=8,  # Smaller for testing
            base_resolution=16,
            finest_resolution=256,
            hidden_dim=32,
            num_layers=2,
        )

    def test_config_creation(self):
        """Test configuration creation."""
        config = InstantNGPConfig()

        # Check default values
        self.assertEqual(config.num_levels, 16)
        self.assertEqual(config.base_resolution, 16)
        self.assertEqual(config.finest_resolution, 512)
        self.assertEqual(config.hidden_dim, 64)
        self.assertEqual(config.num_layers, 3)

    def test_hash_encoder_creation(self):
        """Test hash encoder creation."""
        encoder = HashEncoder(self.config)

        self.assertEqual(encoder.num_levels, self.config.num_levels)
        self.assertEqual(encoder.feature_dim, self.config.feature_dim)

        # Test on device
        encoder = encoder.to(self.device)
        self.assertEqual(next(encoder.parameters()).device, self.device)

    def test_hash_encoder_forward(self):
        """Test hash encoder forward pass."""
        encoder = HashEncoder(self.config).to(self.device)

        # Test input
        batch_size = 1024
        positions = torch.rand(batch_size, 3, device=self.device) * 2 - 1  # [-1, 1]

        # Forward pass
        features = encoder(positions)

        # Check output shape
        expected_dim = self.config.num_levels * self.config.feature_dim
        self.assertEqual(features.shape, (batch_size, expected_dim))

        # Check features are finite
        self.assertTrue(torch.isfinite(features).all())

    def test_sh_encoder_creation(self):
        """Test spherical harmonics encoder creation."""
        encoder = SHEncoder(degree=4)

        self.assertEqual(encoder.degree, 4)
        self.assertEqual(encoder.output_dim, 16)  # (degree+1)^2

    def test_sh_encoder_forward(self):
        """Test SH encoder forward pass."""
        encoder = SHEncoder(degree=2).to(self.device)

        # Test input
        batch_size = 512
        directions = torch.randn(batch_size, 3, device=self.device)
        directions = torch.nn.functional.normalize(directions, dim=-1)

        # Forward pass
        sh_features = encoder(directions)

        # Check output shape
        expected_dim = (2 + 1) ** 2  # 9 for degree 2
        self.assertEqual(sh_features.shape, (batch_size, expected_dim))

        # Check features are finite
        self.assertTrue(torch.isfinite(sh_features).all())

    def test_model_creation(self):
        """Test model creation."""
        model = InstantNGPModel(self.config).to(self.device)

        # Check components exist
        self.assertIsNotNone(model.encoding)
        self.assertIsNotNone(model.direction_encoder)
        self.assertIsNotNone(model.sigma_net)
        self.assertIsNotNone(model.color_net)

        # Check parameter count
        total_params = sum(p.numel() for p in model.parameters())
        self.assertGreater(total_params, 0)
        print(f"Model has {total_params:,} parameters")

    def test_model_forward(self):
        """Test model forward pass."""
        model = InstantNGPModel(self.config).to(self.device)

        # Test input
        batch_size = 256
        positions = torch.rand(batch_size, 3, device=self.device) * 2 - 1
        directions = torch.randn(batch_size, 3, device=self.device)
        directions = torch.nn.functional.normalize(directions, dim=-1)

        # Forward pass
        outputs = model(positions, directions)

        # Check output shape
        self.assertEqual(outputs.shape, (batch_size, 4))  # density + rgb

        # Check outputs are finite
        self.assertTrue(torch.isfinite(outputs).all())

        # Check density is non-negative after activation
        density = outputs[..., 0]
        self.assertTrue((density >= 0).all())

        # Check rgb is in valid range after activation
        rgb = outputs[..., 1:4]
        self.assertTrue((rgb >= 0).all())
        self.assertTrue((rgb <= 1).all())

    def test_loss_function(self):
        """Test loss function."""
        loss_fn = InstantNGPLoss()

        batch_size = 128
        # Mock prediction and target
        prediction = {
            "rgb": torch.rand(batch_size, 3, device=self.device),
            "depth": torch.rand(batch_size, device=self.device),
            "acc": torch.rand(batch_size, device=self.device),
        }
        target = torch.rand(batch_size, 3, device=self.device)

        # Compute loss
        losses = loss_fn(prediction, target)

        # Check loss components
        self.assertIn("rgb_loss", losses)
        self.assertIn("total_loss", losses)
        self.assertIn("psnr", losses)

        # Check losses are finite and positive
        for key, value in losses.items():
            if key != "psnr":  # PSNR can be negative
                self.assertTrue(torch.isfinite(value))
                if "loss" in key:
                    self.assertGreaterEqual(value.item(), 0)

    def test_model_gradients(self):
        """Test gradient computation."""
        model = InstantNGPModel(self.config).to(self.device)
        loss_fn = InstantNGPLoss()

        # Enable gradients
        model.train()

        # Test input
        batch_size = 64
        positions = torch.rand(batch_size, 3, device=self.device, requires_grad=True)
        directions = torch.randn(batch_size, 3, device=self.device)
        directions = torch.nn.functional.normalize(directions, dim=-1)
        target = torch.rand(batch_size, 3, device=self.device)

        # Forward pass
        outputs = model(positions, directions)

        # Mock render outputs for loss
        prediction = {
            "rgb": outputs[..., 1:4],
            "depth": torch.rand(batch_size, device=self.device),
            "acc": torch.rand(batch_size, device=self.device),
        }

        # Compute loss
        losses = loss_fn(prediction, target)
        loss = losses["total_loss"]

        # Backward pass
        loss.backward()

        # Check gradients exist and are finite
        for name, param in model.named_parameters():
            if param.grad is not None:
                self.assertTrue(torch.isfinite(param.grad).all(), f"Invalid gradients in {name}")
                self.assertGreater(param.grad.abs().sum().item(), 0, f"Zero gradients in {name}")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_cuda_functionality(self):
        """Test CUDA-specific functionality."""
        device = torch.device("cuda")
        model = InstantNGPModel(self.config).to(device)

        # Test memory efficiency
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated()

        # Large batch test
        batch_size = 4096
        positions = torch.rand(batch_size, 3, device=device)
        directions = torch.randn(batch_size, 3, device=device)
        directions = torch.nn.functional.normalize(directions, dim=-1)

        outputs = model(positions, directions)

        peak_memory = torch.cuda.max_memory_allocated()
        memory_used = (peak_memory - initial_memory) / 1024**2  # MB

        print(f"CUDA memory used: {memory_used:.2f} MB for batch size {batch_size}")

        # Cleanup
        del outputs, positions, directions
        torch.cuda.empty_cache()

    def test_model_save_load(self):
        """Test model saving and loading."""
        model = InstantNGPModel(self.config).to(self.device)

        # Save model state
        state_dict = model.state_dict()

        # Create new model and load state
        new_model = InstantNGPModel(self.config).to(self.device)
        new_model.load_state_dict(state_dict)

        # Test that outputs are identical
        batch_size = 32
        positions = torch.rand(batch_size, 3, device=self.device)
        directions = torch.randn(batch_size, 3, device=self.device)
        directions = torch.nn.functional.normalize(directions, dim=-1)

        with torch.no_grad():
            outputs1 = model(positions, directions)
            outputs2 = new_model(positions, directions)

        # Check outputs are identical
        torch.testing.assert_close(outputs1, outputs2, rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
