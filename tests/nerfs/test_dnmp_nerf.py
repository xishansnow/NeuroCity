"""Test suite for DNMP-NeRF implementation."""

import unittest
import torch
import numpy as np
from src.nerfs.dnmp_nerf.core import DNMP, DNMPConfig


class TestDNMPNeRF(unittest.TestCase):
    """Test cases for DNMP-NeRF."""

    def setUp(self):
        """Set up test cases."""
        self.config = DNMPConfig()
        # DNMP requires a mesh_autoencoder parameter
        from src.nerfs.dnmp_nerf.mesh_autoencoder import MeshAutoEncoder

        self.mesh_autoencoder = MeshAutoEncoder(self.config)
        self.model = DNMP(self.config, self.mesh_autoencoder)
        self.batch_size = 4
        self.num_points = 100

        # Generate test data
        self.coords = torch.randn(self.batch_size, self.num_points, 3)
        self.view_dirs = torch.randn(self.batch_size, self.num_points, 3)
        self.view_dirs = self.view_dirs / torch.norm(self.view_dirs, dim=-1, keepdim=True)

    def test_model_initialization(self):
        """Test model initialization."""
        self.assertIsInstance(self.model, DNMP)
        self.assertEqual(self.model.config, self.config)

    def test_forward_pass(self):
        """Test forward pass."""
        # Initialize scene with some test points
        point_cloud = torch.randn(100, 3)
        self.model.initialize_scene(point_cloud)

        # Create a mock rasterizer
        from src.nerfs.dnmp_nerf.rasterizer import DNMPRasterizer

        rasterizer = DNMPRasterizer(self.config)

        outputs = self.model(self.coords, self.view_dirs, rasterizer)

        # Check outputs
        self.assertIn("rgb", outputs)
        self.assertIn("depth", outputs)

        # Check shapes
        self.assertEqual(outputs["rgb"].shape, (self.batch_size * self.num_points, 3))
        self.assertEqual(outputs["depth"].shape, (self.batch_size * self.num_points,))

    def test_training_step(self):
        """Test training step."""
        # Create optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.geometry_lr)

        # Initialize scene
        point_cloud = torch.randn(100, 3)
        self.model.initialize_scene(point_cloud)

        # Create mock rasterizer
        from src.nerfs.dnmp_nerf.rasterizer import DNMPRasterizer

        rasterizer = DNMPRasterizer(self.config)

        # Create batch
        batch = {
            "rays_o": self.coords,
            "rays_d": self.view_dirs,
            "rgb": torch.rand_like(self.coords),
            "rasterizer": rasterizer,
        }

        # Run training step
        loss, metrics = self.model.training_step(batch, optimizer)

        # Check outputs
        self.assertIsInstance(loss, torch.Tensor)
        self.assertIsInstance(metrics, dict)
        self.assertIn("total_loss", metrics)

    def test_validation_step(self):
        """Test validation step."""
        # Initialize scene
        point_cloud = torch.randn(100, 3)
        self.model.initialize_scene(point_cloud)

        # Create mock rasterizer
        from src.nerfs.dnmp_nerf.rasterizer import DNMPRasterizer

        rasterizer = DNMPRasterizer(self.config)

        # Create batch
        batch = {
            "rays_o": self.coords,
            "rays_d": self.view_dirs,
            "rgb": torch.rand_like(self.coords),
            "rasterizer": rasterizer,
        }

        # Run validation step
        metrics = self.model.validation_step(batch)

        # Check outputs
        self.assertIsInstance(metrics, dict)
        self.assertIn("total_loss", metrics)

    def test_learning_rate_update(self):
        """Test learning rate update."""
        # Create optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.geometry_lr)
        initial_lr = optimizer.param_groups[0]["lr"]

        # Update learning rate
        from torch.optim.lr_scheduler import ExponentialLR

        scheduler = ExponentialLR(optimizer, gamma=0.1)
        scheduler.step()

        # Check learning rate decay
        updated_lr = optimizer.param_groups[0]["lr"]
        self.assertLess(updated_lr, initial_lr)

    def test_amp_support(self):
        """Test automatic mixed precision support."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        self.model.cuda()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.geometry_lr)

        # Initialize scene
        point_cloud = torch.randn(100, 3).cuda()
        self.model.initialize_scene(point_cloud)

        # Create mock rasterizer
        from src.nerfs.dnmp_nerf.rasterizer import DNMPRasterizer

        rasterizer = DNMPRasterizer(self.config).cuda()

        # Create batch
        batch = {
            "rays_o": self.coords.cuda(),
            "rays_d": self.view_dirs.cuda(),
            "rgb": torch.rand_like(self.coords).cuda(),
            "rasterizer": rasterizer,
        }

        # Run training step with AMP
        with torch.cuda.amp.autocast():
            outputs = self.model(batch["rays_o"], batch["rays_d"], batch["rasterizer"])

        # Check outputs
        self.assertIn("rgb", outputs)
        self.assertIn("depth", outputs)

    def test_non_blocking_transfer(self):
        """Test non-blocking memory transfer."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        self.model.cuda()

        # Initialize scene
        point_cloud = torch.randn(100, 3).cuda(non_blocking=True)
        self.model.initialize_scene(point_cloud)

        # Create mock rasterizer
        from src.nerfs.dnmp_nerf.rasterizer import DNMPRasterizer

        rasterizer = DNMPRasterizer(self.config).cuda()

        # Test non-blocking transfer
        outputs = self.model(
            self.coords.cuda(non_blocking=True), self.view_dirs.cuda(non_blocking=True), rasterizer
        )

        # Check outputs
        self.assertIn("rgb", outputs)
        self.assertIn("depth", outputs)

    def test_gradient_checkpointing(self):
        """Test gradient checkpointing."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        self.model.cuda()

        # Initialize scene
        point_cloud = torch.randn(100, 3).cuda()
        self.model.initialize_scene(point_cloud)

        # Create mock rasterizer
        from src.nerfs.dnmp_nerf.rasterizer import DNMPRasterizer

        rasterizer = DNMPRasterizer(self.config).cuda()

        # Enable gradient checkpointing
        self.model.renderer.train()  # Must be in training mode
        torch.utils.checkpoint.checkpoint_sequential(
            [self.model.renderer], 2, self.coords.cuda(), self.view_dirs.cuda(), rasterizer
        )

        outputs = self.model(self.coords.cuda(), self.view_dirs.cuda(), rasterizer)

        # Check outputs
        self.assertIn("rgb", outputs)
        self.assertIn("depth", outputs)


if __name__ == "__main__":
    unittest.main()
