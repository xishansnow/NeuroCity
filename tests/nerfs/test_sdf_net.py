"""Test suite for SDF-Net implementation."""

import unittest
import torch
import numpy as np
from src.nerfs.sdf_net.core import SDFNet, SDFNetConfig


class TestSDFNet(unittest.TestCase):
    """Test cases for SDF-Net."""

    def setUp(self):
        """Set up test cases."""
        self.config = SDFNetConfig()
        self.model = SDFNet(self.config)
        self.batch_size = 4
        self.num_points = 100

        # Generate test data
        self.coords = torch.randn(self.batch_size, self.num_points, 3)
        self.normals = torch.randn(self.batch_size, self.num_points, 3)
        self.normals = self.normals / torch.norm(self.normals, dim=-1, keepdim=True)

    def test_model_initialization(self):
        """Test model initialization."""
        self.assertIsInstance(self.model, SDFNet)
        self.assertEqual(self.model.config, self.config)

    def test_forward_pass(self):
        """Test forward pass."""
        outputs = self.model(self.coords)

        # Check outputs
        self.assertIn("sdf", outputs)
        self.assertIn("gradients", outputs)

        # Check shapes
        self.assertEqual(outputs["sdf"].shape, (self.batch_size, self.num_points, 1))
        self.assertEqual(outputs["gradients"].shape, (self.batch_size, self.num_points, 3))

    def test_training_step(self):
        """Test training step."""
        # Create optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)

        # Create batch
        batch = {
            "coords": self.coords,
            "normals": self.normals,
            "sdf": torch.rand(self.batch_size, self.num_points, 1),
        }

        # Run training step
        loss, metrics = self.model.training_step(batch, optimizer)

        # Check outputs
        self.assertIsInstance(loss, torch.Tensor)
        self.assertIsInstance(metrics, dict)
        self.assertIn("total_loss", metrics)

    def test_validation_step(self):
        """Test validation step."""
        # Create batch
        batch = {
            "coords": self.coords,
            "normals": self.normals,
            "sdf": torch.rand(self.batch_size, self.num_points, 1),
        }

        # Run validation step
        metrics = self.model.validation_step(batch)

        # Check outputs
        self.assertIsInstance(metrics, dict)
        self.assertIn("total_loss", metrics)

    def test_learning_rate_update(self):
        """Test learning rate update."""
        # Create optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        initial_lr = optimizer.param_groups[0]["lr"]

        # Update learning rate
        self.model.update_learning_rate(optimizer, step=self.config.learning_rate_decay_steps)

        # Check learning rate decay
        updated_lr = optimizer.param_groups[0]["lr"]
        self.assertLess(updated_lr, initial_lr)

    def test_amp_support(self):
        """Test automatic mixed precision support."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        self.model.cuda()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)

        # Create batch
        batch = {
            "coords": self.coords.cuda(),
            "normals": self.normals.cuda(),
            "sdf": torch.rand(self.batch_size, self.num_points, 1).cuda(),
        }

        # Run training step with AMP
        with torch.cuda.amp.autocast():
            outputs = self.model(batch["coords"])

        # Check outputs
        self.assertIn("sdf", outputs)
        self.assertIn("gradients", outputs)

    def test_non_blocking_transfer(self):
        """Test non-blocking memory transfer."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        self.model.cuda()

        # Test non-blocking transfer
        outputs = self.model(self.coords.cuda(non_blocking=True))

        # Check outputs
        self.assertIn("sdf", outputs)
        self.assertIn("gradients", outputs)

    def test_gradient_checkpointing(self):
        """Test gradient checkpointing."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        self.model.cuda()
        self.model.enable_gradient_checkpointing()

        outputs = self.model(self.coords.cuda())

        # Check outputs
        self.assertIn("sdf", outputs)
        self.assertIn("gradients", outputs)


if __name__ == "__main__":
    unittest.main()
