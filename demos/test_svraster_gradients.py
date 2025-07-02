#!/usr/bin/env python3
"""
Test SVRaster gradients

Simple test to diagnose gradient issues in SVRaster.
"""

import sys
import os
import torch
import numpy as np

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from nerfs.svraster.core import SVRasterConfig, SVRasterModel, SVRasterLoss


def test_gradients():
    """Test if gradients are properly computed."""
    print("Testing SVRaster gradients...")
    
    # Create simple configuration
    config = SVRasterConfig(
        max_octree_levels=4,
        base_resolution=8,
        scene_bounds=(-1.0, -1.0, -1.0, 1.0, 1.0, 1.0),
        ray_samples_per_voxel=4,
        morton_ordering=True,
        sh_degree=2,
        use_ssim_loss=False,
        use_distortion_loss=False,
        use_pointwise_rgb_loss=False,
        use_opacity_regularization=False
    )
    
    # Create model and loss function
    model = SVRasterModel(config)
    loss_fn = SVRasterLoss(config)
    
    # Check if model parameters require grad
    print("Checking model parameters...")
    for name, param in model.named_parameters():
        print(f"  {name}: requires_grad={param.requires_grad}")
    
    # Create simple test data
    num_rays = 16
    ray_origins = torch.randn(num_rays, 3)
    ray_directions = torch.randn(num_rays, 3)
    ray_directions = ray_directions / torch.norm(ray_directions, dim=1, keepdim=True)
    target_colors = torch.rand(num_rays, 3)
    
    print(f"\nTest data shapes:")
    print(f"  ray_origins: {ray_origins.shape}")
    print(f"  ray_directions: {ray_directions.shape}")
    print(f"  target_colors: {target_colors.shape}")
    
    # Forward pass
    print("\nRunning forward pass...")
    try:
        outputs = model(ray_origins, ray_directions)
        print(f"  Output keys: {list(outputs.keys())}")
        for key, value in outputs.items():
            print(f"  {key}: {value.shape}, requires_grad={value.requires_grad}")
    except Exception as e:
        print(f"  Forward pass failed: {e}")
        return
    
    # Loss computation
    print("\nComputing loss...")
    try:
        losses = loss_fn(outputs, {'rgb': target_colors}, model)
        print(f"  Loss keys: {list(losses.keys())}")
        for key, value in losses.items():
            print(f"  {key}: {value.item():.6f}, requires_grad={value.requires_grad}")
    except Exception as e:
        print(f"  Loss computation failed: {e}")
        return
    
    # Backward pass
    print("\nRunning backward pass...")
    try:
        total_loss = losses['total_loss']
        total_loss.backward()
        print("  Backward pass successful!")
        
        # Check gradients
        print("\nChecking gradients...")
        for name, param in model.named_parameters():
            if param.grad is not None:
                print(f"  {name}: grad_norm={param.grad.norm().item():.6f}")
            else:
                print(f"  {name}: no gradient")
                
    except Exception as e:
        print(f"  Backward pass failed: {e}")
        return
    
    print("\nGradient test completed successfully!")


if __name__ == "__main__":
    test_gradients() 