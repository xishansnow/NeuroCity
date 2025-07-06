"""Test suite for CNC-NeRF implementation."""

import unittest
import torch
import numpy as np
from src.nerfs.cnc_nerf.core import CNCNeRF, CNCNeRFConfig


class TestCNCNeRF(unittest.TestCase):
    """Test cases for CNC-NeRF."""

    def setUp(self):
        """Set up test cases."""
        self.config = CNCNeRFConfig()
        self.model = CNCNeRF(self.config)
        self.batch_size = 4
        self.num_points = 100

        # Generate test data
        self.coords = torch.randn(self.batch_size, self.num_points, 3)
        self.view_dirs = torch.randn(self.batch_size, self.num_points, 3)
        self.view_dirs = self.view_dirs / torch.norm(self.view_dirs, dim=-1, keepdim=True)

    def test_model_initialization(self):
        """Test model initialization."""
        self.assertIsInstance(self.model, CNCNeRF)
        self.assertEqual(self.model.config, self.config)

    def test_forward_pass(self):
        """Test forward pass."""
        outputs = self.model(self.coords, self.view_dirs)

        # Check outputs
        self.assertIn("rgb", outputs)
        self.assertIn("density", outputs)

        # Check shapes
        self.assertEqual(outputs["rgb"].shape, (self.batch_size, self.num_points, 3))
        self.assertEqual(outputs["density"].shape, (self.batch_size, self.num_points, 1))

    def test_training_step(self):
        """Test training step."""
        # Create optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)

        # Create batch
        batch = {
            "coords": self.coords,
            "view_dirs": self.view_dirs,
            "rgb": torch.rand_like(self.coords),
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
            "view_dirs": self.view_dirs,
            "rgb": torch.rand_like(self.coords),
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
            "view_dirs": self.view_dirs.cuda(),
            "rgb": torch.rand_like(self.coords).cuda(),
        }

        # Run training step with AMP
        with torch.cuda.amp.autocast():
            outputs = self.model(batch["coords"], batch["view_dirs"])

        # Check outputs
        self.assertIn("rgb", outputs)
        self.assertIn("density", outputs)

    def test_non_blocking_transfer(self):
        """Test non-blocking memory transfer."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        self.model.cuda()

        # Test non-blocking transfer
        outputs = self.model(
            self.coords.cuda(non_blocking=True), self.view_dirs.cuda(non_blocking=True)
        )

        # Check outputs
        self.assertIn("rgb", outputs)
        self.assertIn("density", outputs)

    def test_gradient_checkpointing(self):
        """Test gradient checkpointing."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        self.model.cuda()
        self.model.enable_gradient_checkpointing()

        outputs = self.model(self.coords.cuda(), self.view_dirs.cuda())

        # Check outputs
        self.assertIn("rgb", outputs)
        self.assertIn("density", outputs)


if __name__ == "__main__":
    unittest.main()
