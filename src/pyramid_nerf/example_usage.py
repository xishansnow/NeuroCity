#!/usr/bin/env python3
"""
Example usage of PyNeRF package
"""

import torch
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from pyramid_nerf import (
    PyNeRF, PyNeRFConfig, PyramidEncoder, PyramidRenderer,
    create_pyramid_hierarchy, compute_psnr
)


def basic_usage_example():
    """Basic usage example"""
    print("=== PyNeRF Basic Usage Example ===")
    
    # 1. Create configuration
    config = PyNeRFConfig(
        num_levels=6,
        base_resolution=16,
        max_resolution=512,
        batch_size=1024,
        learning_rate=1e-3
    )
    print(f"Created config with {config.num_levels} pyramid levels")
    
    # 2. Create model
    model = PyNeRF(config)
    print(f"Created model with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # 3. Create sample data
    batch_size = 1024
    rays_o = torch.randn(batch_size, 3)  # Ray origins
    rays_d = torch.randn(batch_size, 3)  # Ray directions
    rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)  # Normalize directions
    bounds = torch.tensor([[2.0, 6.0]]).expand(batch_size, 2)  # Near/far bounds
    
    # 4. Forward pass
    model.eval()
    with torch.no_grad():
        outputs = model(rays_o, rays_d, bounds)
    
    print(f"Model output shapes:")
    print(f"  RGB: {outputs['rgb'].shape}")
    if 'depth' in outputs:
        print(f"  Depth: {outputs['depth'].shape}")
    if 'acc' in outputs:
        print(f"  Accumulation: {outputs['acc'].shape}")
    
    print("✓ Basic usage example completed successfully!\n")


def pyramid_encoder_example():
    """Pyramid encoder example"""
    print("=== Pyramid Encoder Example ===")
    
    # Create pyramid encoder
    encoder = PyramidEncoder(
        num_levels=4,
        base_resolution=16,
        max_resolution=256,
        features_per_level=4
    )
    
    # Encode positions
    positions = torch.randn(1000, 3)
    features = encoder(positions)
    
    print(f"Encoder input shape: {positions.shape}")
    print(f"Encoder output shape: {features.shape}")
    print(f"Features per level: {encoder.features_per_level}")
    print(f"Total features: {features.shape[-1]}")
    
    print("✓ Pyramid encoder example completed successfully!\n")


def pyramid_hierarchy_example():
    """Pyramid hierarchy example"""
    print("=== Pyramid Hierarchy Example ===")
    
    # Create different pyramid hierarchies
    hierarchies = [
        create_pyramid_hierarchy(6, 16, 2.0, 512),
        create_pyramid_hierarchy(8, 8, 1.5, 256),
        create_pyramid_hierarchy(4, 32, 3.0, 1024)
    ]
    
    for i, hierarchy in enumerate(hierarchies):
        print(f"Hierarchy {i+1}: {hierarchy}")
    
    print("✓ Pyramid hierarchy example completed successfully!\n")


def volume_rendering_example():
    """Volume rendering example"""
    print("=== Volume Rendering Example ===")
    
    # Create renderer
    config = PyNeRFConfig()
    renderer = PyramidRenderer(config)
    
    # Create sample volume data
    batch_size = 512
    num_samples = 64
    
    rgb = torch.rand(batch_size, num_samples, 3)  # RGB values
    sigma = torch.rand(batch_size, num_samples)   # Density values
    z_vals = torch.linspace(2.0, 6.0, num_samples).expand(batch_size, num_samples)
    rays_d = torch.randn(batch_size, 3)
    rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
    
    # Render
    outputs = renderer.render(rgb, sigma, z_vals, rays_d)
    
    print(f"Volume rendering outputs:")
    print(f"  RGB map: {outputs['rgb'].shape}")
    print(f"  Depth map: {outputs['depth'].shape}")
    print(f"  Accumulation map: {outputs['acc'].shape}")
    print(f"  Weights: {outputs['weights'].shape}")
    
    print("✓ Volume rendering example completed successfully!\n")


def metrics_example():
    """Metrics computation example"""
    print("=== Metrics Example ===")
    
    # Create sample images
    height, width = 128, 128
    pred_image = torch.rand(height, width, 3)
    gt_image = torch.rand(height, width, 3)
    
    # Compute metrics
    psnr = compute_psnr(pred_image, gt_image)
    
    print(f"Image shape: {pred_image.shape}")
    print(f"PSNR: {psnr:.4f} dB")
    
    # Create similar images for better PSNR
    similar_image = gt_image + 0.1 * torch.randn_like(gt_image)
    similar_psnr = compute_psnr(similar_image, gt_image)
    print(f"Similar image PSNR: {similar_psnr:.4f} dB")
    
    print("✓ Metrics example completed successfully!\n")


def main():
    """Run all examples"""
    print("PyNeRF Package Usage Examples")
    print("=" * 50)
    
    examples = [
        basic_usage_example,
        pyramid_encoder_example,
        pyramid_hierarchy_example,
        volume_rendering_example,
        metrics_example
    ]
    
    for example in examples:
        try:
            example()
        except Exception as e:
            print(f"❌ Example failed: {e}\n")
    
    print("=" * 50)
    print("All examples completed!")
    print("\nNext steps:")
    print("1. Prepare your dataset (NeRF synthetic or LLFF format)")
    print("2. Run training: python train_pyramid_nerf.py --data_dir /path/to/data")
    print("3. Render results: python render_pyramid_nerf.py --checkpoint /path/to/checkpoint")


if __name__ == "__main__":
    main()
