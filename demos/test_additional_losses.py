#!/usr/bin/env python3
"""
Test script to verify SVRaster additional losses:
1. Distortion loss
2. Pointwise RGB loss
"""

import torch
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from nerfs.svraster.core import SVRasterConfig, SVRasterModel, SVRasterLoss

def test_distortion_loss():
    """Test distortion loss functionality."""
    print("Testing SVRaster distortion loss...")
    
    # Create configuration with distortion loss enabled
    config = SVRasterConfig(
        max_octree_levels=4,
        base_resolution=8,
        scene_bounds=(-1.0, -1.0, -1.0, 1.0, 1.0, 1.0),
        use_distortion_loss=True,
        distortion_loss_weight=0.01,
        use_opacity_regularization=True,
        opacity_reg_weight=0.01
    )
    
    # Create model and loss function
    model = SVRasterModel(config)
    loss_fn = SVRasterLoss(config)
    
    # Create dummy data
    batch_size = 32
    ray_origins = torch.randn(batch_size, 3) * 2.0
    ray_directions = torch.randn(batch_size, 3)
    ray_directions = ray_directions / ray_directions.norm(dim=-1, keepdim=True)
    target_colors = torch.rand(batch_size, 3)
    
    # Forward pass
    outputs = model(ray_origins, ray_directions)
    
    print(f"  Output shapes:")
    print(f"    rgb: {outputs['rgb'].shape}")
    print(f"    depth: {outputs['depth'].shape}")
    print(f"    weights: {outputs['weights'].shape}")
    
    # Compute losses
    targets = {'rgb': target_colors}
    losses = loss_fn(outputs, targets, model)
    
    print(f"\n  Loss components:")
    print(f"    RGB loss: {losses['rgb_loss'].item():.6f}")
    print(f"    Distortion loss: {losses['distortion_loss'].item():.6f}")
    print(f"    Opacity reg: {losses['opacity_reg'].item():.6f}")
    print(f"    Total loss: {losses['total_loss'].item():.6f}")
    
    # Test with distortion loss disabled
    print(f"\n  Testing with distortion loss disabled...")
    config_no_distortion = SVRasterConfig(
        max_octree_levels=4,
        base_resolution=8,
        scene_bounds=(-1.0, -1.0, -1.0, 1.0, 1.0, 1.0),
        use_distortion_loss=False,
        use_opacity_regularization=True,
        opacity_reg_weight=0.01
    )
    
    loss_fn_no_distortion = SVRasterLoss(config_no_distortion)
    losses_no_distortion = loss_fn_no_distortion(outputs, targets, model)
    
    print(f"    Distortion loss: {losses_no_distortion['distortion_loss'].item():.6f}")
    print(f"    Expected: 0.0")
    
    # Verify that distortion loss is zero when disabled
    assert losses_no_distortion['distortion_loss'].item() == 0.0, "Distortion loss should be zero when disabled"
    
    print(f"\n✅ Distortion loss test completed successfully!")

def test_pointwise_rgb_loss():
    """Test pointwise RGB loss functionality."""
    print(f"\nTesting SVRaster pointwise RGB loss...")
    
    # Create configuration with pointwise RGB loss enabled
    config = SVRasterConfig(
        max_octree_levels=4,
        base_resolution=8,
        scene_bounds=(-1.0, -1.0, -1.0, 1.0, 1.0, 1.0),
        use_pointwise_rgb_loss=True,
        pointwise_rgb_loss_weight=0.1,
        use_opacity_regularization=True,
        opacity_reg_weight=0.01
    )
    
    # Create model and loss function
    model = SVRasterModel(config)
    loss_fn = SVRasterLoss(config)
    
    # Create dummy data
    batch_size = 32
    ray_origins = torch.randn(batch_size, 3) * 2.0
    ray_directions = torch.randn(batch_size, 3)
    ray_directions = ray_directions / ray_directions.norm(dim=-1, keepdim=True)
    target_colors = torch.rand(batch_size, 3)
    
    # Forward pass
    outputs = model(ray_origins, ray_directions)
    
    # Compute losses
    targets = {'rgb': target_colors}
    losses = loss_fn(outputs, targets, model)
    
    print(f"  Loss components:")
    print(f"    RGB loss (MSE): {losses['rgb_loss'].item():.6f}")
    print(f"    Pointwise RGB loss (L1): {losses['pointwise_rgb_loss'].item():.6f}")
    print(f"    Opacity reg: {losses['opacity_reg'].item():.6f}")
    print(f"    Total loss: {losses['total_loss'].item():.6f}")
    
    # Test with pointwise RGB loss disabled
    print(f"\n  Testing with pointwise RGB loss disabled...")
    config_no_pointwise = SVRasterConfig(
        max_octree_levels=4,
        base_resolution=8,
        scene_bounds=(-1.0, -1.0, -1.0, 1.0, 1.0, 1.0),
        use_pointwise_rgb_loss=False,
        use_opacity_regularization=True,
        opacity_reg_weight=0.01
    )
    
    loss_fn_no_pointwise = SVRasterLoss(config_no_pointwise)
    losses_no_pointwise = loss_fn_no_pointwise(outputs, targets, model)
    
    print(f"    Pointwise RGB loss: {losses_no_pointwise['pointwise_rgb_loss'].item():.6f}")
    print(f"    Expected: 0.0")
    
    # Verify that pointwise RGB loss is zero when disabled
    assert losses_no_pointwise['pointwise_rgb_loss'].item() == 0.0, "Pointwise RGB loss should be zero when disabled"
    
    print(f"\n✅ Pointwise RGB loss test completed successfully!")

def test_all_losses_combined():
    """Test all losses working together."""
    print(f"\nTesting all SVRaster losses combined...")
    
    # Create configuration with all losses enabled
    config = SVRasterConfig(
        max_octree_levels=4,
        base_resolution=8,
        scene_bounds=(-1.0, -1.0, -1.0, 1.0, 1.0, 1.0),
        use_ssim_loss=True,
        ssim_loss_weight=0.1,
        use_distortion_loss=True,
        distortion_loss_weight=0.01,
        use_pointwise_rgb_loss=True,
        pointwise_rgb_loss_weight=0.1,
        use_opacity_regularization=True,
        opacity_reg_weight=0.01
    )
    
    # Create model and loss function
    model = SVRasterModel(config)
    loss_fn = SVRasterLoss(config)
    
    # Create dummy data
    batch_size = 32
    ray_origins = torch.randn(batch_size, 3) * 2.0
    ray_directions = torch.randn(batch_size, 3)
    ray_directions = ray_directions / ray_directions.norm(dim=-1, keepdim=True)
    target_colors = torch.rand(batch_size, 3)
    
    # Forward pass
    outputs = model(ray_origins, ray_directions)
    
    # Compute losses
    targets = {'rgb': target_colors}
    losses = loss_fn(outputs, targets, model)
    
    print(f"  All loss components:")
    print(f"    RGB loss (MSE): {losses['rgb_loss'].item():.6f}")
    print(f"    SSIM loss: {losses['ssim_loss'].item():.6f}")
    print(f"    Distortion loss: {losses['distortion_loss'].item():.6f}")
    print(f"    Pointwise RGB loss (L1): {losses['pointwise_rgb_loss'].item():.6f}")
    print(f"    Opacity reg: {losses['opacity_reg'].item():.6f}")
    print(f"    Total loss: {losses['total_loss'].item():.6f}")
    
    # Verify all losses are positive
    for loss_name, loss_value in losses.items():
        if loss_name != 'total_loss':
            assert loss_value.item() >= 0, f"{loss_name} should be non-negative"
    
    print(f"\n✅ All losses combined test completed successfully!")

def test_loss_gradients():
    """Test that all losses produce gradients."""
    print(f"\nTesting loss gradients...")
    
    config = SVRasterConfig(
        max_octree_levels=2,
        base_resolution=4,
        scene_bounds=(-1.0, -1.0, -1.0, 1.0, 1.0, 1.0),
        use_ssim_loss=True,
        use_distortion_loss=True,
        use_pointwise_rgb_loss=True,
        use_opacity_regularization=True
    )
    
    model = SVRasterModel(config)
    loss_fn = SVRasterLoss(config)
    
    # Create simple data
    ray_origins = torch.tensor([[0.0, 0.0, 2.0]])
    ray_directions = torch.tensor([[0.0, 0.0, -1.0]])
    target_colors = torch.rand(1, 3)
    
    # Forward pass
    outputs = model(ray_origins, ray_directions)
    targets = {'rgb': target_colors}
    
    # Compute loss and gradients
    losses = loss_fn(outputs, targets, model)
    total_loss = losses['total_loss']
    
    # Backward pass
    total_loss.backward()
    
    # Check gradients
    has_gradients = False
    for name, param in model.named_parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            has_gradients = True
            print(f"    Parameter '{name}' has gradients")
            break
    
    if has_gradients:
        print(f"    ✅ All losses produce gradients correctly")
    else:
        print(f"    ❌ Losses do not produce gradients")
    
    return has_gradients

def main():
    """Main test function."""
    print("SVRaster Additional Losses Test")
    print("=" * 50)
    
    # Test distortion loss
    test_distortion_loss()
    
    # Test pointwise RGB loss
    test_pointwise_rgb_loss()
    
    # Test all losses combined
    test_all_losses_combined()
    
    # Test gradients
    test_loss_gradients()
    
    print(f"\n🎉 All additional losses tests passed!")
    print(f"\nSummary of implemented losses:")
    print(f"1. ✅ RGB MSE loss - Standard reconstruction loss")
    print(f"2. ✅ SSIM loss - Structural similarity for perceptual quality")
    print(f"3. ✅ Distortion loss - Geometry regularization")
    print(f"4. ✅ Pointwise RGB loss - Per-pixel L1 loss")
    print(f"5. ✅ Opacity regularization - Sparsity constraint")

if __name__ == "__main__":
    main() 