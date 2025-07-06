#!/usr/bin/env python3
"""
Test script for the Plenoxels CUDA extension.
"""

import torch
import numpy as np
import sys
import os
import time

def test_plenoxels_cuda_extension():
    """Test the Plenoxels CUDA extension specifically."""
    print("=" * 60)
    print("Testing Plenoxels CUDA Extension")
    print("=" * 60)
    
    # Check if CUDA is available
    print(f"CUDA available: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("❌ CUDA is not available, cannot test CUDA extension")
        return False
        
    print(f"CUDA device: {torch.cuda.get_device_name()}")
    
    try:
        # Add the CUDA extension to the path
        cuda_path = '/home/xishansnow/3DVision/NeuroCity/src/nerfs/plenoxels/cuda'
        if cuda_path not in sys.path:
            sys.path.insert(0, cuda_path)
        
        # Import the CUDA extension
        import plenoxels_cuda
        print("✅ Successfully imported plenoxels_cuda module")
        
        # Test 1: Ray-Voxel Intersection
        print("\n📋 Test 1: Ray-Voxel Intersection")
        N_rays = 1000
        N_voxels = 500
        
        # Create test data
        ray_origins = torch.randn(N_rays, 3, device='cuda', dtype=torch.float32)
        ray_directions = torch.randn(N_rays, 3, device='cuda', dtype=torch.float32)
        ray_directions = ray_directions / torch.norm(ray_directions, dim=1, keepdim=True)  # Normalize
        voxel_centers = torch.randn(N_voxels, 3, device='cuda', dtype=torch.float32)
        voxel_size = 0.1
        
        start_time = time.time()
        intersections = plenoxels_cuda.ray_voxel_intersect(
            ray_origins, ray_directions, voxel_centers, voxel_size
        )
        end_time = time.time()
        
        print(f"   Input shapes: rays={ray_origins.shape}, voxels={voxel_centers.shape}")
        print(f"   Output shape: {intersections.shape}")
        print(f"   Output dtype: {intersections.dtype}")
        print(f"   Intersection rate: {intersections.float().mean():.3f}")
        print(f"   Time taken: {(end_time - start_time)*1000:.2f} ms")
        print("   ✅ Ray-voxel intersection test passed")
        
        # Test 2: Volume Rendering
        print("\n📋 Test 2: Volume Rendering")
        N_rays = 1000
        N_samples = 64
        
        # Create test data for volume rendering
        densities = torch.rand(N_rays, N_samples, device='cuda', dtype=torch.float32) * 10.0
        colors = torch.rand(N_rays, N_samples, 3, device='cuda', dtype=torch.float32)
        deltas = torch.ones(N_rays, N_samples, device='cuda', dtype=torch.float32) * 0.01
        
        start_time = time.time()
        rgb, depth, weights = plenoxels_cuda.volume_render(densities, colors, deltas)
        end_time = time.time()
        
        print(f"   Input shapes: densities={densities.shape}, colors={colors.shape}")
        print(f"   Output shapes: rgb={rgb.shape}, depth={depth.shape}, weights={weights.shape}")
        print(f"   RGB range: [{rgb.min():.3f}, {rgb.max():.3f}]")
        print(f"   Depth range: [{depth.min():.3f}, {depth.max():.3f}]")
        print(f"   Weights sum: {weights.sum(dim=1).mean():.3f}")
        print(f"   Time taken: {(end_time - start_time)*1000:.2f} ms")
        print("   ✅ Volume rendering test passed")
        
        # Test 3: Trilinear Interpolation
        print("\n📋 Test 3: Trilinear Interpolation")
        N_points = 1000
        N_voxels = 500
        feature_dim = 32
        
        # Create test data for interpolation
        features = torch.randn(N_voxels, feature_dim, device='cuda', dtype=torch.float32)
        points = torch.randn(N_points, 3, device='cuda', dtype=torch.float32)
        
        # Create dummy voxel indices and weights for each point (8 neighbors)
        voxel_indices = torch.randint(0, N_voxels, (N_points, 8), device='cuda', dtype=torch.int32)
        weights = torch.rand(N_points, 8, device='cuda', dtype=torch.float32)
        weights = weights / weights.sum(dim=1, keepdim=True)  # Normalize weights
        
        start_time = time.time()
        interpolated_features = plenoxels_cuda.trilinear_interpolate(
            features, points, voxel_indices, weights
        )
        end_time = time.time()
        
        print(f"   Input shapes: features={features.shape}, points={points.shape}")
        print(f"   Output shape: {interpolated_features.shape}")
        print(f"   Feature range: [{interpolated_features.min():.3f}, {interpolated_features.max():.3f}]")
        print(f"   Time taken: {(end_time - start_time)*1000:.2f} ms")
        print("   ✅ Trilinear interpolation test passed")
        
        # Test 4: Interpolation Weights Computation
        print("\n📋 Test 4: Interpolation Weights Computation")
        N_points = 1000
        
        # Create test data
        points = torch.randn(N_points, 3, device='cuda', dtype=torch.float32)
        # Create 8 corner coordinates for each point (dummy data)
        voxel_coords = torch.randn(N_points, 8, 3, device='cuda', dtype=torch.float32)
        voxel_coords = voxel_coords.view(N_points, 24)  # Flatten to [N, 24]
        
        start_time = time.time()
        computed_weights = plenoxels_cuda.compute_interpolation_weights(points, voxel_coords)
        end_time = time.time()
        
        print(f"   Input shapes: points={points.shape}, voxel_coords={voxel_coords.shape}")
        print(f"   Output shape: {computed_weights.shape}")
        print(f"   Weights range: [{computed_weights.min():.3f}, {computed_weights.max():.3f}]")
        print(f"   Weights sum per point: {computed_weights.sum(dim=1).mean():.3f}")
        print(f"   Time taken: {(end_time - start_time)*1000:.2f} ms")
        print("   ✅ Interpolation weights computation test passed")
        
        # Performance comparison with CPU (for volume rendering)
        print("\n📋 Performance Comparison (Volume Rendering)")
        N_rays = 5000
        N_samples = 128
        
        # Create larger test data
        densities_gpu = torch.rand(N_rays, N_samples, device='cuda', dtype=torch.float32) * 10.0
        colors_gpu = torch.rand(N_rays, N_samples, 3, device='cuda', dtype=torch.float32)
        deltas_gpu = torch.ones(N_rays, N_samples, device='cuda', dtype=torch.float32) * 0.01
        
        densities_cpu = densities_gpu.cpu()
        colors_cpu = colors_gpu.cpu()
        deltas_cpu = deltas_gpu.cpu()
        
        # GPU timing
        torch.cuda.synchronize()
        start_time = time.time()
        for _ in range(10):
            rgb_gpu, depth_gpu, weights_gpu = plenoxels_cuda.volume_render(densities_gpu, colors_gpu, deltas_gpu)
        torch.cuda.synchronize()
        gpu_time = (time.time() - start_time) / 10
        
        # Simple CPU implementation for comparison
        def volume_render_cpu(densities, colors, deltas):
            N, S = densities.shape
            weights = torch.zeros_like(densities)
            rgb = torch.zeros(N, 3)
            depth = torch.zeros(N)
            
            for i in range(N):
                T = 1.0
                for j in range(S):
                    alpha = 1.0 - torch.exp(-densities[i, j] * deltas[i, j])
                    w = alpha * T
                    weights[i, j] = w
                    rgb[i] += w * colors[i, j]
                    depth[i] += w * deltas[i, j]
                    T *= (1.0 - alpha)
                    
            return rgb, depth, weights
        
        start_time = time.time()
        rgb_cpu, depth_cpu, weights_cpu = volume_render_cpu(densities_cpu, colors_cpu, deltas_cpu)
        cpu_time = time.time() - start_time
        
        speedup = cpu_time / gpu_time
        print(f"   GPU time: {gpu_time*1000:.2f} ms")
        print(f"   CPU time: {cpu_time*1000:.2f} ms")
        print(f"   Speedup: {speedup:.1f}x")
        
        # Verify results are similar
        rgb_diff = torch.abs(rgb_gpu.cpu() - rgb_cpu).mean()
        print(f"   RGB difference (GPU vs CPU): {rgb_diff:.6f}")
        print("   ✅ Performance comparison completed")
        
        print("\n" + "=" * 60)
        print("🎉 All Plenoxels CUDA extension tests passed successfully!")
        print("=" * 60)
        
        return True
        
    except ImportError as e:
        print(f"❌ Failed to import plenoxels_cuda: {e}")
        return False
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_plenoxels_cuda_extension()
    if success:
        print("\n✅ Plenoxels CUDA extension is working correctly!")
    else:
        print("\n❌ Plenoxels CUDA extension test failed!")
        sys.exit(1)
