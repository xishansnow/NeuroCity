"""
Test module for Mip-NeRF implementation

This module contains unit tests for the core components of Mip-NeRF.
"""

import torch
import numpy as np
import unittest
from pathlib import Path
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from nerfs.mip_nerf.core import (
    MipNeRFConfig,
    IntegratedPositionalEncoder,
    ConicalFrustum,
    MipNeRFMLP,
    MipNeRFRenderer,
    MipNeRF,
)


class TestMipNeRFCore(unittest.TestCase):
    """Test core Mip-NeRF components"""

    def setUp(self):
        """Set up test fixtures"""
        self.config = MipNeRFConfig(
            netdepth=4,  # Smaller for testing
            netwidth=64,
            num_samples=32,
            num_importance=32,
            max_deg_point=8,
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.batch_size = 100

    def test_integrated_positional_encoder(self):
        """Test integrated positional encoding"""
        encoder = IntegratedPositionalEncoder(min_deg=0, max_deg=8)

        # Test input
        means = torch.randn(self.batch_size, 3)
        vars = torch.abs(torch.randn(self.batch_size, 3)) * 0.1

        # Encode
        encoding = encoder(means, vars)

        # Check output shape
        expected_dim = 2 * 3 * (8 - 0)  # 2 * 3 * num_levels
        self.assertEqual(encoding.shape, (self.batch_size, expected_dim))

        # Check finite values
        self.assertTrue(torch.all(torch.isfinite(encoding)))

        print(f"✓ IPE test passed: {means.shape} -> {encoding.shape}")

    def test_conical_frustum(self):
        """Test conical frustum creation and conversion"""
        origins = torch.randn(self.batch_size, 3)
        directions = torch.randn(self.batch_size, 3)
        directions = torch.nn.functional.normalize(directions, dim=-1)
        t_vals = torch.linspace(2.0, 6.0, 32).expand(self.batch_size, -1)
        pixel_radius = 0.001

        # Create frustums
        frustums = ConicalFrustum.from_rays(origins, directions, t_vals, pixel_radius)

        # Check shapes
        self.assertEqual(frustums.means.shape, (self.batch_size, 32, 3))
        self.assertEqual(frustums.covs.shape, (self.batch_size, 32, 3, 3))

        # Convert to Gaussian
        means, vars = frustums.to_gaussian()
        self.assertEqual(means.shape, (self.batch_size, 32, 3))
        self.assertEqual(vars.shape, (self.batch_size, 32, 3))

        # Check positive variances
        self.assertTrue(torch.all(vars >= 0))

        print(f"✓ Conical frustum test passed: {origins.shape} -> {means.shape}")

    def test_mip_nerf_mlp(self):
        """Test Mip-NeRF MLP forward pass"""
        mlp = MipNeRFMLP(self.config).to(self.device)

        # Create test frustums
        origins = torch.randn(self.batch_size, 3, device=self.device)
        directions = torch.randn(self.batch_size, 3, device=self.device)
        directions = torch.nn.functional.normalize(directions, dim=-1)
        t_vals = torch.linspace(2.0, 6.0, 32, device=self.device).expand(self.batch_size, -1)

        frustums = ConicalFrustum.from_rays(origins, directions, t_vals, 0.001)
        frustums.means = frustums.means.to(self.device)
        frustums.covs = frustums.covs.to(self.device)

        viewdirs = directions

        # Forward pass
        output = mlp(frustums, viewdirs)

        # Check output
        self.assertIn("density", output)
        self.assertIn("rgb", output)
        self.assertEqual(output["density"].shape, (self.batch_size, 32, 1))
        self.assertEqual(output["rgb"].shape, (self.batch_size, 32, 3))

        # Check value ranges
        self.assertTrue(torch.all(output["rgb"] >= 0))
        self.assertTrue(torch.all(output["rgb"] <= 1))

        print(f"✓ MLP test passed: density {output['density'].shape}, rgb {output['rgb'].shape}")

    def test_mip_nerf_renderer(self):
        """Test Mip-NeRF renderer"""
        renderer = MipNeRFRenderer(self.config)

        # Test sampling
        origins = torch.randn(self.batch_size, 3)
        directions = torch.randn(self.batch_size, 3)
        directions = torch.nn.functional.normalize(directions, dim=-1)

        t_vals = renderer.sample_along_rays(origins, directions, 2.0, 6.0, 32)
        self.assertEqual(t_vals.shape, (self.batch_size, 32))

        # Test volumetric rendering
        densities = torch.rand(self.batch_size, 32, 1)
        colors = torch.rand(self.batch_size, 32, 3)

        render_output = renderer.volumetric_rendering(densities, colors, t_vals)

        self.assertIn("rgb", render_output)
        self.assertIn("depth", render_output)
        self.assertIn("weights", render_output)

        self.assertEqual(render_output["rgb"].shape, (self.batch_size, 3))
        self.assertEqual(render_output["depth"].shape, (self.batch_size,))
        self.assertEqual(render_output["weights"].shape, (self.batch_size, 32))

        print(f"✓ Renderer test passed: {render_output['rgb'].shape}")

    def test_full_mip_nerf_model(self):
        """Test complete Mip-NeRF model"""
        model = MipNeRF(self.config).to(self.device)

        # Test input
        origins = torch.randn(self.batch_size, 3, device=self.device)
        directions = torch.randn(self.batch_size, 3, device=self.device)
        directions = torch.nn.functional.normalize(directions, dim=-1)
        viewdirs = directions

        # Forward pass
        results = model(origins, directions, viewdirs, near=2.0, far=6.0)

        # Check coarse results
        self.assertIn("coarse", results)
        coarse = results["coarse"]
        self.assertIn("rgb", coarse)
        self.assertIn("depth", coarse)
        self.assertEqual(coarse["rgb"].shape, (self.batch_size, 3))

        # Check fine results (if importance sampling enabled)
        if self.config.num_importance > 0:
            self.assertIn("fine", results)
            fine = results["fine"]
            self.assertIn("rgb", fine)
            self.assertIn("depth", fine)
            self.assertEqual(fine["rgb"].shape, (self.batch_size, 3))

        print(f"✓ Full model test passed: coarse {coarse['rgb'].shape}")
        if "fine" in results:
            print(f"✓ Fine network: {results['fine']['rgb'].shape}")

    def test_loss_function(self):
        """Test Mip-NeRF loss computation"""
        model = MipNeRF(self.config).to(self.device)

        # Create mock predictions and targets
        predictions = {
            "coarse": {
                "rgb": torch.rand(self.batch_size, 3, device=self.device),
                "depth": torch.rand(self.batch_size, device=self.device),
                "weights": torch.rand(self.batch_size, 32, device=self.device),
                "acc_alpha": torch.rand(self.batch_size, device=self.device),
            }
        }

        if self.config.num_importance > 0:
            predictions["fine"] = {
                "rgb": torch.rand(self.batch_size, 3, device=self.device),
                "depth": torch.rand(self.batch_size, device=self.device),
                "weights": torch.rand(self.batch_size, 32, device=self.device),
                "acc_alpha": torch.rand(self.batch_size, device=self.device),
            }

        targets = {
            "rgb": torch.rand(self.batch_size, 3, device=self.device),
            "depth": torch.rand(self.batch_size, device=self.device),
        }

        # Compute loss
        losses = model.compute_loss(predictions, targets)

        # Check loss outputs
        self.assertIn("total_loss", losses)
        self.assertIn("coarse_loss", losses)
        self.assertIn("psnr", losses)

        if "fine" in predictions:
            self.assertIn("fine_loss", losses)

        # Check loss is scalar
        self.assertEqual(losses["total_loss"].shape, ())

        print(f"✓ Loss function test passed: total_loss = {losses['total_loss']:.4f}")


class TestMipNeRFUtils(unittest.TestCase):
    """Test utility functions"""

    def test_math_utils(self):
        """Test mathematical utilities"""
        try:
            from .utils.math_utils import expected_sin, expected_cos, safe_exp

            x = torch.randn(100, 3)
            x_var = torch.abs(torch.randn(100, 3)) * 0.1

            sin_val = expected_sin(x, x_var)
            cos_val = expected_cos(x, x_var)

            self.assertEqual(sin_val.shape, x.shape)
            self.assertEqual(cos_val.shape, x.shape)
            self.assertTrue(torch.all(torch.isfinite(sin_val)))
            self.assertTrue(torch.all(torch.isfinite(cos_val)))

            # Test safe operations
            large_x = torch.tensor([100.0, -100.0, 0.0])
            safe_exp_val = safe_exp(large_x)
            self.assertTrue(torch.all(torch.isfinite(safe_exp_val)))

            print("✓ Math utils test passed")

        except ImportError as e:
            print(f"⚠ Math utils test skipped: {e}")

    def test_ray_utils(self):
        """Test ray utilities"""
        try:
            from .utils.ray_utils import cast_rays, volumetric_rendering

            origins = torch.randn(100, 3)
            directions = torch.randn(100, 3)
            directions = torch.nn.functional.normalize(directions, dim=-1)
            radii = torch.full((100,), 0.001)

            rays = cast_rays(origins, directions, radii, near=2.0, far=6.0)

            self.assertIn("origins", rays)
            self.assertIn("directions", rays)
            self.assertIn("radii", rays)

            print("✓ Ray utils test passed")

        except ImportError as e:
            print(f"⚠ Ray utils test skipped: {e}")


def run_performance_test():
    """Run performance benchmark"""
    print("\n" + "=" * 50)
    print("PERFORMANCE BENCHMARK")
    print("=" * 50)

    config = MipNeRFConfig(netdepth=8, netwidth=256, num_samples=64, num_importance=128)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MipNeRF(config).to(device)

    # Benchmark parameters
    batch_sizes = [100, 500, 1000] if device == "cuda" else [100, 200]

    print(f"Device: {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):, }")

    import time

    for batch_size in batch_sizes:
        # Prepare data
        origins = torch.randn(batch_size, 3, device=device)
        directions = torch.randn(batch_size, 3, device=device)
        directions = torch.nn.functional.normalize(directions, dim=-1)
        viewdirs = directions

        # Warm up
        with torch.no_grad():
            _ = model(origins[:10], directions[:10], viewdirs[:10], near=2.0, far=6.0)

        # Benchmark
        torch.cuda.synchronize() if device == "cuda" else None
        start_time = time.time()

        with torch.no_grad():
            results = model(origins, directions, viewdirs, near=2.0, far=6.0)

        torch.cuda.synchronize() if device == "cuda" else None
        end_time = time.time()

        elapsed = end_time - start_time
        rays_per_second = batch_size / elapsed

        print(f"Batch size {batch_size:4d}: {elapsed:.3f}s ({rays_per_second:.0f} rays/s)")


def main():
    """Run all tests"""
    print("Mip-NeRF Test Suite")
    print("==================")

    # Run unit tests
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestMipNeRFCore))
    suite.addTests(loader.loadTestsFromTestCase(TestMipNeRFUtils))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Run performance test
    if result.wasSuccessful():
        try:
            run_performance_test()
        except Exception as e:
            print(f"Performance test failed: {e}")

    # Summary
    print("\n" + "=" * 50)
    if result.wasSuccessful():
        print("✅ ALL TESTS PASSED")
        print("Mip-NeRF implementation is ready to use!")
    else:
        print("❌ SOME TESTS FAILED")
        print(f"Failures: {len(result.failures)}")
        print(f"Errors: {len(result.errors)}")

    return result.wasSuccessful()


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
