#!/usr/bin/env python3
"""
Test script to verify SVRaster rasterization fixes.
"""

import torch
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from nerfs.svraster.core import SVRasterConfig, VoxelRasterizer

def test_rasterization_fixes():
    """Test the fixed rasterization logic."""
    print("Testing SVRaster rasterization fixes...")
    
    # Create configuration
    config = SVRasterConfig(
        max_octree_levels=8,
        base_resolution=16,
        scene_bounds=(-1.0, -1.0, -1.0, 1.0, 1.0, 1.0),
        morton_ordering=True
    )
    
    # Create rasterizer
    rasterizer = VoxelRasterizer(config)
    
    # Create dummy voxel data with correct color shape
    num_voxels = 50  # Reduced number for testing
    color_dim = 3 * (config.sh_degree + 1) ** 2  # Calculate correct color dimension
    
    voxels = {
        'positions': torch.randn(num_voxels, 3) * 0.5,  # Random positions in [-0.5, 0.5]
        'sizes': torch.ones(num_voxels) * 0.1,
        'densities': torch.randn(num_voxels) * 0.1,
        'colors': torch.rand(num_voxels, color_dim) * 0.5 + 0.25,  # Correct color shape
        'levels': torch.zeros(num_voxels, dtype=torch.int),
        'morton_codes': torch.arange(num_voxels, dtype=torch.long)
    }
    
    # Create dummy rays
    num_rays = 5  # Reduced number for testing
    ray_origins = torch.randn(num_rays, 3) * 2  # Random origins
    ray_directions = torch.randn(num_rays, 3)
    ray_directions = ray_directions / torch.norm(ray_directions, dim=1, keepdim=True)
    
    print(f"Testing with {num_voxels} voxels and {num_rays} rays...")
    print(f"Color tensor shape: {voxels['colors'].shape}")
    
    try:
        # Test rasterization
        outputs = rasterizer(voxels, ray_origins, ray_directions)
        
        # Check outputs
        assert 'rgb' in outputs, "Missing RGB output"
        assert 'depth' in outputs, "Missing depth output"
        assert 'weights' in outputs, "Missing weights output"
        
        assert outputs['rgb'].shape == (num_rays, 3), f"RGB shape mismatch: {outputs['rgb'].shape}"
        assert outputs['depth'].shape == (num_rays, 1), f"Depth shape mismatch: {outputs['depth'].shape}"
        assert outputs['weights'].shape == (num_rays, 1), f"Weights shape mismatch: {outputs['weights'].shape}"
        
        # Check value ranges
        assert torch.all(outputs['rgb'] >= 0), "RGB values should be non-negative"
        assert torch.all(outputs['rgb'] <= 1), "RGB values should be <= 1"
        assert torch.all(outputs['weights'] >= 0), "Weights should be non-negative"
        assert torch.all(outputs['weights'] <= 1), "Weights should be <= 1"
        
        print("✅ Rasterization test passed!")
        print(f"   RGB range: [{outputs['rgb'].min():.3f}, {outputs['rgb'].max():.3f}]")
        print(f"   Depth range: [{outputs['depth'].min():.3f}, {outputs['depth'].max():.3f}]")
        print(f"   Weights range: [{outputs['weights'].min():.3f}, {outputs['weights'].max():.3f}]")
        
        return True
        
    except Exception as e:
        print(f"❌ Rasterization test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_vectorized_intersection():
    """Test the vectorized ray-voxel intersection."""
    print("\nTesting vectorized ray-voxel intersection...")
    
    config = SVRasterConfig()
    rasterizer = VoxelRasterizer(config)
    
    # Create test data
    voxels = {
        'positions': torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        'sizes': torch.tensor([0.5, 0.5, 0.5])
    }
    
    ray_o = torch.tensor([-1.0, 0.0, 0.0])
    ray_d = torch.tensor([1.0, 0.0, 0.0])
    
    try:
        intersections = rasterizer._ray_voxel_intersections(ray_o, ray_d, voxels)
        
        print(f"Found {len(intersections)} intersections")
        for i, (voxel_idx, t_near, t_far) in enumerate(intersections):
            print(f"  Intersection {i}: voxel {voxel_idx}, t_near={t_near:.3f}, t_far={t_far:.3f}")
        
        # Should find intersection with voxel 0 (at origin)
        assert len(intersections) > 0, "Should find at least one intersection"
        assert intersections[0][0] == 0, "First intersection should be with voxel 0"
        
        print("✅ Vectorized intersection test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Vectorized intersection test failed: {e}")
        return False

if __name__ == "__main__":
    print("SVRaster Rasterization Fixes Test")
    print("=" * 40)
    
    success1 = test_rasterization_fixes()
    success2 = test_vectorized_intersection()
    
    if success1 and success2:
        print("\n🎉 All tests passed! Rasterization fixes are working correctly.")
    else:
        print("\n❌ Some tests failed. Please check the implementation.")
        sys.exit(1) 