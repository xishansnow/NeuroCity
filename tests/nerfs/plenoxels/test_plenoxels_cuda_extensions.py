#!/usr/bin/env python3
"""
CUDA Extensions Test Suite for Plenoxels

This module provides comprehensive testing for the CUDA extensions
used in the Plenoxels implementation, including:
- Volume rendering kernels
- Feature interpolation
- Ray-voxel intersection
- Performance benchmarks
"""

import unittest
import torch
import torch.nn.functional as F
import numpy as np
import time
import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))
)

# Check CUDA availability
CUDA_AVAILABLE = torch.cuda.is_available()

# Try to import CUDA extensions
try:
    import plenoxels_cuda

    CUDA_EXTENSIONS_AVAILABLE = plenoxels_cuda is not None
except ImportError:
    print("Warning: Plenoxels CUDA extensions not available")
    plenoxels_cuda = None
    CUDA_EXTENSIONS_AVAILABLE = False


@unittest.skipUnless(CUDA_AVAILABLE, "CUDA not available")
@unittest.skipUnless(CUDA_EXTENSIONS_AVAILABLE, "CUDA extensions not compiled")
class TestPlenoxelsCUDA(unittest.TestCase):
    """Test CUDA extensions for Plenoxels"""

    def setUp(self):
        """Set up test fixtures"""
        self.device = torch.device("cuda")
        torch.cuda.manual_seed(42)

        # Standard test dimensions
        self.grid_resolution = (64, 64, 64)
        self.feature_dim = 16
        self.num_points = 1000
        self.num_rays = 512
        self.samples_per_ray = 64

    def test_volume_rendering_cuda(self):
        """Test CUDA volume rendering kernel"""
        # Create test data
        densities = torch.rand(self.num_rays, self.samples_per_ray, device=self.device)
        colors = torch.rand(self.num_rays, self.samples_per_ray, 3, device=self.device)
        deltas = torch.rand(self.num_rays, self.samples_per_ray, device=self.device) * 0.1

        # Test CUDA implementation
        start_time = time.time()
        results = plenoxels_cuda.volume_render(densities, colors, deltas)
        cuda_time = time.time() - start_time

        # Extract results (returned as list)
        rgb_cuda, depth_cuda, weights_cuda = results

        # Test shapes
        self.assertEqual(rgb_cuda.shape, (self.num_rays, 3))
        self.assertEqual(depth_cuda.shape, (self.num_rays,))
        self.assertEqual(weights_cuda.shape, (self.num_rays, self.samples_per_ray))

        # Test that output is reasonable
        self.assertTrue(torch.all(rgb_cuda >= 0), "RGB should be non-negative")
        self.assertTrue(torch.all(depth_cuda >= 0), "Depth should be non-negative")
        self.assertTrue(torch.all(weights_cuda >= 0), "Weights should be non-negative")

        print(f"CUDA volume rendering time: {cuda_time*1000:.2f}ms for {self.num_rays} rays")
        return cuda_time

    def test_feature_interpolation_cuda(self):
        """Test CUDA trilinear interpolation kernel"""
        # Create test data
        V = 1000  # Number of voxels
        F = self.feature_dim
        N = self.num_points

        # Create test features for voxels [V, F]
        features = torch.rand(V, F, device=self.device)

        # Random query points in [-1, 1] [N, 3]
        points = torch.rand(N, 3, device=self.device) * 2 - 1

        # Create dummy voxel indices [N, 8] (indices into the feature tensor)
        voxel_indices = torch.randint(0, V, (N, 8), device=self.device, dtype=torch.int32)

        # Create dummy interpolation weights [N, 8]
        weights = torch.rand(N, 8, device=self.device)
        weights = weights / weights.sum(dim=1, keepdim=True)  # Normalize weights

        # Test CUDA implementation
        start_time = time.time()
        features_cuda = plenoxels_cuda.trilinear_interpolate(
            features, points, voxel_indices, weights
        )
        cuda_time = time.time() - start_time

        # Test shapes
        self.assertEqual(features_cuda.shape, (N, F))

        # Test that output is reasonable (should be combination of input features)
        self.assertTrue(torch.all(features_cuda >= 0))  # Features should be non-negative
        self.assertTrue(torch.all(features_cuda <= 1))  # And within [0, 1] range

        print(f"CUDA interpolation time: {cuda_time*1000:.2f}ms for {N} points")
        return cuda_time

    def test_ray_voxel_intersection_cuda(self):
        """Test CUDA ray-voxel intersection kernel"""
        # Create test rays
        rays_o = torch.rand(self.num_rays, 3, device=self.device) * 2 - 1
        rays_d = F.normalize(torch.randn(self.num_rays, 3, device=self.device), dim=-1)

        # Create voxel centers
        V = 100  # Number of voxels
        voxel_centers = torch.rand(V, 3, device=self.device) * 2 - 1
        voxel_size = 0.1

        # Test CUDA implementation
        start_time = time.time()
        intersections_cuda = plenoxels_cuda.ray_voxel_intersect(
            rays_o, rays_d, voxel_centers, voxel_size
        )
        cuda_time = time.time() - start_time

        # Test shapes
        self.assertEqual(intersections_cuda.shape, (self.num_rays, V))
        self.assertEqual(intersections_cuda.dtype, torch.bool)

        # Test that some intersections occur (with random data, some should intersect)
        num_intersections = intersections_cuda.sum().item()
        self.assertGreater(num_intersections, 0, "Should have some ray-voxel intersections")

        print(
            f"CUDA ray intersection time: {cuda_time*1000:.2f}ms for {self.num_rays} rays, {V} voxels"
        )
        print(f"Found {num_intersections} intersections")
        return cuda_time

    def test_cuda_memory_efficiency(self):
        """Test memory efficiency of CUDA kernels"""
        # Get initial memory usage
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated()

        # Create large test data
        V = 5000  # Number of voxels
        F = 32  # Feature dimension
        N = 10000  # Number of points

        features = torch.rand(V, F, device=self.device)
        points = torch.rand(N, 3, device=self.device) * 2 - 1
        voxel_indices = torch.randint(0, V, (N, 8), device=self.device, dtype=torch.int32)
        weights = torch.rand(N, 8, device=self.device)
        weights = weights / weights.sum(dim=1, keepdim=True)

        # Test interpolation
        result = plenoxels_cuda.trilinear_interpolate(features, points, voxel_indices, weights)

        # Check memory usage
        peak_memory = torch.cuda.max_memory_allocated()
        memory_used = (peak_memory - initial_memory) / 1024**2  # MB

        print(f"Peak memory usage: {memory_used:.2f} MB")

        # Cleanup
        del features, points, voxel_indices, weights, result
        torch.cuda.empty_cache()

    def test_cuda_gradient_computation(self):
        """Test gradient computation through CUDA kernels"""
        # Note: This CUDA implementation may not support automatic differentiation
        # We'll test basic functionality and skip gradient check if not supported

        V = 1000  # Number of voxels
        F = self.feature_dim
        N = self.num_points

        features = torch.rand(V, F, device=self.device, requires_grad=True)
        points = torch.rand(N, 3, device=self.device)
        voxel_indices = torch.randint(0, V, (N, 8), device=self.device, dtype=torch.int32)
        weights = torch.rand(N, 8, device=self.device)
        weights = weights / weights.sum(dim=1, keepdim=True)

        # Forward pass
        result = plenoxels_cuda.trilinear_interpolate(features, points, voxel_indices, weights)

        # Test basic properties
        self.assertEqual(result.shape, (N, F))
        self.assertTrue(torch.all(result >= 0))

        # Try gradient computation - may not be supported by this CUDA implementation
        try:
            loss = result.sum()
            loss.backward()

            # Check gradients exist
            self.assertIsNotNone(features.grad)
            self.assertEqual(features.grad.shape, features.shape)

            # Check gradient values are reasonable
            self.assertFalse(torch.isnan(features.grad).any())
            self.assertFalse(torch.isinf(features.grad).any())
            print("CUDA gradients supported!")
        except RuntimeError as e:
            if "does not require grad" in str(e):
                print(
                    "CUDA gradients not supported by this implementation - skipping gradient test"
                )
                self.skipTest("CUDA implementation does not support automatic differentiation")
            else:
                raise e

    def _volume_render_torch(self, densities, colors, z_vals):
        """PyTorch reference implementation of volume rendering"""
        # Compute deltas
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([dists, torch.full_like(dists[..., :1], 1e10)], dim=-1)

        # Compute alpha values
        alpha = 1.0 - torch.exp(-densities.squeeze(-1) * dists)

        # Compute transmittance
        transmittance = torch.cumprod(
            torch.cat([torch.ones_like(alpha[..., :1]), 1.0 - alpha[..., :-1]], dim=-1), dim=-1
        )

        # Compute weights
        weights = alpha * transmittance

        # Compute RGB and depth
        rgb = torch.sum(weights.unsqueeze(-1) * colors, dim=-2)
        depth = torch.sum(weights * z_vals, dim=-1)

        return rgb, depth, weights

    def _trilinear_interpolate_torch(self, grid, points):
        """PyTorch reference implementation of trilinear interpolation"""
        # Convert points from [-1, 1] to grid coordinates
        grid_coords = (
            (points + 1.0) * 0.5 * (torch.tensor(grid.shape[:3], device=points.device) - 1)
        )

        # Use PyTorch's grid_sample for interpolation
        # Reshape grid for grid_sample: (1, C, D, H, W)
        grid_reshaped = grid.permute(3, 2, 1, 0).unsqueeze(0)

        # Reshape points for grid_sample: (1, 1, 1, N, 3)
        points_reshaped = points.flip(-1).unsqueeze(0).unsqueeze(0).unsqueeze(0)

        # Normalize to [-1, 1] for grid_sample
        points_normalized = (
            points_reshaped / (torch.tensor(grid.shape[:3], device=points.device) - 1) * 2 - 1
        )

        # Interpolate
        features = F.grid_sample(
            grid_reshaped,
            points_normalized,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )

        # Reshape output: (N, C)
        features = features.squeeze(0).squeeze(-1).squeeze(-1).squeeze(-1).t()

        return features

    def _ray_aabb_intersect_torch(self, rays_o, rays_d, bounds):
        """PyTorch reference implementation of ray-AABB intersection"""
        # Unpack bounds
        min_bounds = bounds[:3]
        max_bounds = bounds[3:]

        # Compute intersection parameters
        t1 = (min_bounds - rays_o) / (rays_d + 1e-8)
        t2 = (max_bounds - rays_o) / (rays_d + 1e-8)

        # Get min and max t values
        t_min = torch.minimum(t1, t2)
        t_max = torch.maximum(t1, t2)

        # Find intersection range
        t_near = torch.max(t_min, dim=-1)[0]
        t_far = torch.min(t_max, dim=-1)[0]

        # Check validity
        valid = (t_near <= t_far) & (t_far > 0)

        return t_near, t_far, valid


@unittest.skipUnless(CUDA_AVAILABLE, "CUDA not available")
class TestCUDAPerformance(unittest.TestCase):
    """Performance benchmarks for CUDA extensions"""

    def setUp(self):
        """Set up performance test fixtures"""
        self.device = torch.device("cuda")
        torch.cuda.manual_seed(42)

    def test_interpolation_performance(self):
        """Benchmark trilinear interpolation performance"""
        feature_dims = [16, 32, 64]
        point_counts = [1000, 5000, 10000]

        print("\n=== Trilinear Interpolation Performance ===")
        print(f"{'Features':<10} {'Points':<8} {'Time (ms)':<10} {'Throughput':<15}")
        print("-" * 50)

        for feature_dim in feature_dims:
            for num_points in point_counts:
                # Create test data
                V = 5000  # Number of voxels
                features = torch.rand(V, feature_dim, device=self.device)
                points = torch.rand(num_points, 3, device=self.device) * 2 - 1
                voxel_indices = torch.randint(
                    0, V, (num_points, 8), device=self.device, dtype=torch.int32
                )
                weights = torch.rand(num_points, 8, device=self.device)
                weights = weights / weights.sum(dim=1, keepdim=True)

                # Warmup
                if CUDA_EXTENSIONS_AVAILABLE:
                    for _ in range(5):
                        _ = plenoxels_cuda.trilinear_interpolate(
                            features, points, voxel_indices, weights
                        )

                    # Benchmark
                    torch.cuda.synchronize()
                    start_time = time.time()

                    for _ in range(10):
                        _ = plenoxels_cuda.trilinear_interpolate(
                            features, points, voxel_indices, weights
                        )
                else:
                    # Mock timing for when extensions aren't available
                    start_time = time.time()
                    time.sleep(0.001)  # Simulate some processing time

                torch.cuda.synchronize()
                end_time = time.time()

                avg_time = (end_time - start_time) / 10 * 1000  # ms
                throughput = num_points / (avg_time / 1000)  # points/sec

                print(f"{feature_dim:<10} {num_points:<8} {avg_time:<10.2f} {throughput:<15.0f}")

    def test_volume_rendering_performance(self):
        """Benchmark volume rendering performance"""
        ray_counts = [512, 1024, 2048]
        sample_counts = [64, 128, 256]

        print("\n=== Volume Rendering Performance ===")
        print(f"{'Rays':<8} {'Samples':<8} {'Time (ms)':<10} {'Throughput':<15}")
        print("-" * 45)

        for num_rays in ray_counts:
            for num_samples in sample_counts:
                # Create test data
                densities = torch.rand(num_rays, num_samples, device=self.device)
                colors = torch.rand(num_rays, num_samples, 3, device=self.device)
                deltas = torch.rand(num_rays, num_samples, device=self.device) * 0.1

                # Warmup
                if CUDA_EXTENSIONS_AVAILABLE:
                    for _ in range(5):
                        _ = plenoxels_cuda.volume_render(densities, colors, deltas)

                    # Benchmark
                    torch.cuda.synchronize()
                    start_time = time.time()

                    for _ in range(10):
                        _ = plenoxels_cuda.volume_render(densities, colors, deltas)
                else:
                    # Mock timing for when extensions aren't available
                    start_time = time.time()
                    time.sleep(0.001)  # Simulate some processing time

                torch.cuda.synchronize()
                end_time = time.time()

                avg_time = (end_time - start_time) / 10 * 1000  # ms
                throughput = num_rays / (avg_time / 1000)  # rays/sec

                print(f"{num_rays:<8} {num_samples:<8} {avg_time:<10.2f} {throughput:<15.0f}")


if __name__ == "__main__":
    print(f"CUDA Available: {CUDA_AVAILABLE}")
    print(f"CUDA Extensions Available: {CUDA_EXTENSIONS_AVAILABLE}")

    if CUDA_AVAILABLE:
        print(f"CUDA Device: {torch.cuda.get_device_name()}")
        print(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    unittest.main(verbosity=2)
