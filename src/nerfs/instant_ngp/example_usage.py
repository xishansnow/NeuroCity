"""
Example usage of Instant NGP implementation.

This script demonstrates how to use the Instant NGP model for training and inference
on NeRF-style datasets. It includes examples for both synthetic and real datasets.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import json
import time
from tqdm import tqdm

from . import (
    InstantNGPConfig, InstantNGP, InstantNGPTrainer, InstantNGPDataset, create_instant_ngp_dataloader
)


def train_instant_ngp_example():
    """Example of training Instant NGP on a dataset."""
    print("🚀 Training Instant NGP Example")
    print("=" * 50)
    
    # Configuration
    config = InstantNGPConfig(
        num_levels=16, level_dim=2, base_resolution=16, desired_resolution=2048, hidden_dim=64, learning_rate=5e-4, lambda_entropy=1e-4, lambda_tv=1e-4
    )
    
    print("Configuration:")
    print(f"  - Hash levels: {config.num_levels}")
    print(f"  - Base resolution: {config.base_resolution}")
    print(f"  - Target resolution: {config.desired_resolution}")
    print(f"  - Learning rate: {config.learning_rate}")
    
    # Create model and trainer
    trainer = InstantNGPTrainer(config)
    print(f"✓ Model created with {
        sum(p.numel() for p in trainer.model.parameters()):,
    }
    
    # Create dummy dataset (replace with real dataset path)
    data_root = "data/nerf_synthetic/lego"  # Example path
    
    try:
        # Create dataloaders
        train_loader = create_instant_ngp_dataloader(
            data_root=data_root, split='train', batch_size=8192, img_wh=(400, 400), num_workers=4
        )
        
        val_loader = create_instant_ngp_dataloader(
            data_root=data_root, split='val', batch_size=1, img_wh=(
                400,
                400,
            )
        )
        
        print(f"✓ Datasets loaded: {
            len(train_loader.dataset),
        }
        
        # Train model
        print("\n🎯 Starting training...")
        trainer.train(
            train_loader=train_loader, val_loader=val_loader, num_epochs=20, save_dir="outputs/instant_ngp_example"
        )
        
    except FileNotFoundError:
        print("⚠️  Dataset not found. Creating synthetic example instead...")
        synthetic_training_example()


def synthetic_training_example():
    """Train on synthetic data for demonstration."""
    print("\n🎭 Synthetic Training Example")
    print("-" * 30)
    
    config = InstantNGPConfig(
        num_levels=8, level_dim=2, base_resolution=16, desired_resolution=512, hidden_dim=32, learning_rate=1e-3
    )
    
    # Create model
    model = InstantNGP(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    
    print(f"✓ Synthetic model created")
    
    # Generate synthetic training data
    num_rays = 10000
    positions = torch.randn(num_rays, 3) * 0.5  # Random positions
    directions = torch.randn(num_rays, 3)
    directions = directions / torch.norm(directions, dim=-1, keepdim=True)
    
    # Create target RGB (simple function)
    target_rgb = torch.sigmoid(positions.sum(dim=-1, keepdim=True) * 2).expand(-1, 3)
    target_rgb = target_rgb + torch.randn_like(target_rgb) * 0.1  # Add noise
    target_rgb = torch.clamp(target_rgb, 0, 1)
    
    print(f"✓ Generated {num_rays} synthetic training rays")
    
    # Training loop
    model.train()
    losses = []
    
    print("\n🎯 Training on synthetic data...")
    for epoch in tqdm(range(100), desc="Training"):
        # Shuffle data
        idx = torch.randperm(num_rays)
        pos_batch = positions[idx]
        dir_batch = directions[idx] 
        target_batch = target_rgb[idx]
        
        # Forward pass
        pred_rgb, pred_density = model(pos_batch, dir_batch)
        
        # Loss
        loss = torch.nn.functional.mse_loss(pred_rgb, target_batch)
        losses.append(loss.item())
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if epoch % 20 == 0:
            tqdm.write(f"Epoch {epoch}: Loss = {loss.item():.6f}")
    
    print(f"✓ Training completed! Final loss: {losses[-1]:.6f}")
    
    # Test inference
    print("\n🔍 Testing inference...")
    model.eval()
    with torch.no_grad():
        test_positions = torch.randn(100, 3) * 0.5
        test_directions = torch.randn(100, 3)
        test_directions = test_directions / torch.norm(test_directions, dim=-1, keepdim=True)
        
        pred_rgb, pred_density = model(test_positions, test_directions)
        
        print(f"✓ Inference successful!")
        print(f"  - RGB range: [{pred_rgb.min():.3f}, {pred_rgb.max():.3f}]")
        print(f"  - Density range: [{pred_density.min():.3f}, {pred_density.max():.3f}]")


def rendering_example():
    """Example of rendering with Instant NGP."""
    print("\n🎨 Rendering Example")
    print("-" * 20)
    
    from .core import InstantNGPRenderer
    
    config = InstantNGPConfig()
    model = InstantNGP(config)
    renderer = InstantNGPRenderer(config)
    
    print("✓ Renderer created")
    
    # Create test rays (camera looking down -z axis)
    H, W = 64, 64
    focal = 50.0
    
    # Create pixel coordinates
    i, j = torch.meshgrid(
        torch.arange(W, dtype=torch.float32), torch.arange(H, dtype=torch.float32), indexing='xy'
    )
    
    # Camera rays
    directions = torch.stack([
        (i - W * 0.5) / focal, -(j - H * 0.5) / focal, -torch.ones_like(i)
    ], dim=-1)
    
    # Normalize
    directions = directions / torch.norm(directions, dim=-1, keepdim=True)
    
    # Ray origins (camera position)
    rays_o = torch.zeros(H, W, 3)
    rays_o[:, :, 2] = 5.0  # Move camera back
    
    # Flatten for rendering
    rays_o_flat = rays_o.reshape(-1, 3)
    rays_d_flat = directions.reshape(-1, 3)
    
    near = torch.ones(len(rays_o_flat)) * 2.0
    far = torch.ones(len(rays_o_flat)) * 6.0
    
    print(f"✓ Created {len(rays_o_flat)} test rays")
    
    # Render
    print("🎨 Rendering image...")
    model.eval()
    with torch.no_grad():
        results = renderer.render_rays(
            model, rays_o_flat, rays_d_flat, near, far, num_samples=64
        )
    
    # Reshape to image
    rgb_image = results['rgb'].reshape(H, W, 3).cpu().numpy()
    depth_image = results['depth'].reshape(H, W).cpu().numpy()
    
    print("✓ Rendering completed!")
    print(f"  - RGB image shape: {rgb_image.shape}")
    print(f"  - Depth range: [{depth_image.min():.3f}, {depth_image.max():.3f}]")
    
    return rgb_image, depth_image


def hash_encoding_example():
    """Example of hash encoding functionality."""
    print("\n🔢 Hash Encoding Example")
    print("-" * 25)
    
    from .core import HashEncoder
    
    config = InstantNGPConfig(
        num_levels=4, level_dim=2, base_resolution=16, per_level_scale=2.0
    )
    
    encoder = HashEncoder(config)
    print(f"✓ Hash encoder created:")
    print(f"  - Levels: {encoder.num_levels}")
    print(f"  - Features per level: {encoder.level_dim}")
    print(f"  - Output dimension: {encoder.output_dim}")
    print(f"  - Resolutions: {encoder.resolutions}")
    
    # Test encoding
    positions = torch.tensor([
        [0.0, 0.0, 0.0], # Origin
        [0.5, 0.5, 0.5], # Corner
        [-0.8, 0.3, -0.1], # Random point
    ])
    
    encoded = encoder(positions)
    print(f"\n✓ Encoded {len(positions)} positions:")
    print(f"  - Input shape: {positions.shape}")
    print(f"  - Output shape: {encoded.shape}")
    print(f"  - Feature range: [{encoded.min():.4f}, {encoded.max():.4f}]")
    
    # Test consistency
    encoded2 = encoder(positions)
    is_consistent = torch.allclose(encoded, encoded2)
    print(f"  - Consistency check: {'✓ PASS' if is_consistent else '✗ FAIL'}")


def spherical_harmonics_example():
    """Example of spherical harmonics encoding."""
    print("\n🌐 Spherical Harmonics Example")
    print("-" * 32)
    
    from .core import SHEncoder
    
    encoder = SHEncoder(degree=4)
    print(f"✓ SH encoder created (degree={encoder.degree}, output_dim={encoder.output_dim})")
    
    # Test directions
    directions = torch.tensor([
        [1.0, 0.0, 0.0], # +X direction
        [0.0, 1.0, 0.0], # +Y direction  
        [0.0, 0.0, 1.0], # +Z direction
        [0.707, 0.707, 0.0], # Diagonal
    ])
    
    encoded = encoder(directions)
    print(f"\n✓ Encoded {len(directions)} directions:")
    print(f"  - Input shape: {directions.shape}")
    print(f"  - Output shape: {encoded.shape}")
    
    # Show some SH coefficients
    print(f"  - SH coefficients for +X direction: {encoded[0, :4].tolist()}")
    print(f"  - SH coefficients for +Y direction: {encoded[1, :4].tolist()}")


def benchmark_example():
    """Benchmark Instant NGP performance."""
    print("\n⚡ Performance Benchmark")
    print("-" * 23)
    
    config = InstantNGPConfig()
    model = InstantNGP(config)
    
    # Test different batch sizes
    batch_sizes = [100, 1000, 10000]
    
    for batch_size in batch_sizes:
        positions = torch.randn(batch_size, 3) * 0.5
        directions = torch.randn(batch_size, 3)
        directions = directions / torch.norm(directions, dim=-1, keepdim=True)
        
        # Warmup
        with torch.no_grad():
            model(positions[:10], directions[:10])
        
        # Benchmark
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.time()
        
        with torch.no_grad():
            rgb, density = model(positions, directions)
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end_time = time.time()
        
        elapsed = end_time - start_time
        rays_per_second = batch_size / elapsed
        
        print(f"  - Batch size {batch_size:5d}: {rays_per_second:8.0f} rays/sec ({elapsed:.4f}s)")


def main():
    """Main example runner."""
    parser = argparse.ArgumentParser(description="Instant NGP Examples")
    parser.add_argument(
        '--example',
        type=str,
        default='all',
        choices=['all',
        'train',
        'render',
        'hash',
        'sh',
        'benchmark'],
        help='Which example to run',
    )
    parser.add_argument(
        '--data_root',
        type=str,
        default=None,
        help='Path to dataset for training example',
    )
    
    args = parser.parse_args()
    
    print("🎯 Instant NGP Examples")
    print("=" * 50)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name()}")
    print("=" * 50)
    
    if args.example in ['all', 'hash']:
        hash_encoding_example()
        
    if args.example in ['all', 'sh']:
        spherical_harmonics_example()
        
    if args.example in ['all', 'render']:
        rgb_img, depth_img = rendering_example()
        
    if args.example in ['all', 'benchmark']:
        benchmark_example()
        
    if args.example in ['all', 'train']:
        if args.data_root:
            # Use provided dataset
            pass  # Would implement real training here
        else:
            train_instant_ngp_example()
    
    print("\n🎉 All examples completed!")


if __name__ == "__main__":
    main()

