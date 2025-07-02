#!/usr/bin/env python3
"""
Simple SVRaster Ablation Demo

Quick demonstration of ablation studies for SVRaster components.
"""

import sys
import os
import torch
import numpy as np
from pathlib import Path

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from nerfs.svraster.core import SVRasterConfig, SVRasterModel, SVRasterLoss


def create_simple_test_data():
    """Create simple synthetic test data."""
    num_rays = 256
    image_size = int(np.sqrt(num_rays))
    
    # Create rays pointing towards origin
    u = torch.linspace(-0.5, 0.5, image_size)
    v = torch.linspace(-0.5, 0.5, image_size)
    u, v = torch.meshgrid(u, v, indexing='ij')
    
    # Camera position
    camera_pos = torch.tensor([0.0, 0.0, 2.0])
    
    # Create rays
    ray_origins = camera_pos.unsqueeze(0).repeat(num_rays, 1)
    ray_directions = torch.stack([
        u.flatten(),
        v.flatten(),
        -torch.ones(num_rays)
    ], dim=-1)
    ray_directions = ray_directions / torch.norm(ray_directions, dim=-1, keepdim=True)
    
    # Create target colors
    target_colors = torch.stack([
        0.5 + 0.5 * torch.sin(2 * np.pi * u.flatten()),
        0.5 + 0.5 * torch.cos(2 * np.pi * v.flatten()),
        0.5 + 0.25 * torch.sin(4 * np.pi * (u + v).flatten())
    ], dim=-1)
    
    return ray_origins, ray_directions, target_colors


def compute_metrics(pred_rgb, target_rgb):
    """Compute evaluation metrics."""
    with torch.no_grad():
        # PSNR
        mse = torch.mean((pred_rgb - target_rgb) ** 2)
        psnr = -10.0 * torch.log10(mse + 1e-8)
        
        # SSIM (simplified)
        pred_gray = 0.299 * pred_rgb[:, 0] + 0.587 * pred_rgb[:, 1] + 0.114 * pred_rgb[:, 2]
        target_gray = 0.299 * target_rgb[:, 0] + 0.587 * target_rgb[:, 1] + 0.114 * target_rgb[:, 2]
        
        mu1, mu2 = torch.mean(pred_gray), torch.mean(target_gray)
        sigma1_sq, sigma2_sq = torch.var(pred_gray), torch.var(target_gray)
        sigma12 = torch.mean((pred_gray - mu1) * (target_gray - mu2))
        
        c1, c2 = 0.01 ** 2, 0.03 ** 2
        ssim = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / \
               ((mu1**2 + mu2**2 + c1) * (sigma1_sq + sigma2_sq + c2))
        
        return {
            'psnr': psnr.item(),
            'ssim': ssim.item(),
            'mse': mse.item()
        }


def train_and_evaluate(config, experiment_name, rays_o, rays_d, target_colors, num_epochs=2):
    """Train model and evaluate performance."""
    print(f"Running: {experiment_name}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model and loss function
    model = SVRasterModel(config).to(device)
    loss_fn = SVRasterLoss(config)
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        
        # Forward pass
        outputs = model(rays_o.to(device), rays_d.to(device))
        losses = loss_fn(outputs, {'rgb': target_colors.to(device)}, model)
        loss = losses['total_loss']
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if epoch % 1 == 0:
            print(f"  Epoch {epoch}: Loss={loss.item():.4f}")
    
    # Final evaluation
    model.eval()
    with torch.no_grad():
        outputs = model(rays_o.to(device), rays_d.to(device))
        metrics = compute_metrics(outputs['rgb'], target_colors.to(device))
    
    return metrics


def run_loss_ablation():
    """Run loss function ablation study."""
    print("\n" + "="*50)
    print("LOSS FUNCTION ABLATION STUDY")
    print("="*50)
    
    # Create test data
    rays_o, rays_d, target_colors = create_simple_test_data()
    
    # Base configuration
    base_config = SVRasterConfig(
        max_octree_levels=4,
        base_resolution=8,
        scene_bounds=(-1.0, -1.0, -1.0, 1.0, 1.0, 1.0),
        ray_samples_per_voxel=4,
        morton_ordering=True,
        sh_degree=2
    )
    
    # Experiments
    experiments = [
        ("RGB_MSE_only", {
            'use_ssim_loss': False,
            'use_distortion_loss': False,
            'use_pointwise_rgb_loss': False,
            'use_opacity_regularization': False
        }),
        ("RGB_MSE_SSIM", {
            'use_ssim_loss': True,
            'use_distortion_loss': False,
            'use_pointwise_rgb_loss': False,
            'use_opacity_regularization': False
        }),
        ("All_Losses", {
            'use_ssim_loss': True,
            'use_distortion_loss': True,
            'use_pointwise_rgb_loss': True,
            'use_opacity_regularization': True
        })
    ]
    
    results = {}
    for exp_name, config_updates in experiments:
        config = SVRasterConfig(**{**base_config.__dict__, **config_updates})
        results[exp_name] = train_and_evaluate(config, exp_name, rays_o, rays_d, target_colors)
    
    return results


def run_component_ablation():
    """Run component ablation study."""
    print("\n" + "="*50)
    print("COMPONENT ABLATION STUDY")
    print("="*50)
    
    # Create test data
    rays_o, rays_d, target_colors = create_simple_test_data()
    
    # Base configuration
    base_config = SVRasterConfig(
        max_octree_levels=4,
        base_resolution=8,
        scene_bounds=(-1.0, -1.0, -1.0, 1.0, 1.0, 1.0),
        ray_samples_per_voxel=4,
        morton_ordering=True,
        sh_degree=2
    )
    
    # Experiments
    experiments = [
        ("No_Morton_Ordering", {'morton_ordering': False}),
        ("With_Morton_Ordering", {'morton_ordering': True}),
        ("SH_Degree_0", {'sh_degree': 0}),
        ("SH_Degree_2", {'sh_degree': 2}),
        ("Low_Sampling", {'ray_samples_per_voxel': 2}),
        ("High_Sampling", {'ray_samples_per_voxel': 8})
    ]
    
    results = {}
    for exp_name, config_updates in experiments:
        config = SVRasterConfig(**{**base_config.__dict__, **config_updates})
        results[exp_name] = train_and_evaluate(config, exp_name, rays_o, rays_d, target_colors)
    
    return results


def print_summary(loss_results, component_results):
    """Print summary of ablation results."""
    print("\n" + "="*60)
    print("ABLATION STUDY SUMMARY")
    print("="*60)
    
    # Loss ablation summary
    print("\nLoss Function Ablation:")
    print("-" * 30)
    for exp_name, metrics in loss_results.items():
        print(f"{exp_name:20s}: PSNR={metrics['psnr']:6.2f}, SSIM={metrics['ssim']:6.4f}")
    
    # Component ablation summary
    print("\nComponent Ablation:")
    print("-" * 20)
    for exp_name, metrics in component_results.items():
        print(f"{exp_name:20s}: PSNR={metrics['psnr']:6.2f}, SSIM={metrics['ssim']:6.4f}")
    
    print("\n" + "="*60)


def main():
    """Main function."""
    print("SVRaster Simple Ablation Demo")
    print("=" * 60)
    
    # Run ablation studies
    loss_results = run_loss_ablation()
    component_results = run_component_ablation()
    
    # Print summary
    print_summary(loss_results, component_results)
    
    print("\nAblation demo completed!")


if __name__ == "__main__":
    main() 