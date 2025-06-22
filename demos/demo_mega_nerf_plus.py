#!/usr/bin/env python3
"""
Mega-NeRF++ Demo Script

This script demonstrates the key capabilities of Mega-NeRF++ for large-scale
photogrammetric reconstruction. It shows how to:
1. Create a basic configuration
2. Generate synthetic data for testing
3. Train a small model
4. Render novel views
5. Evaluate performance

Run this script to verify that your Mega-NeRF++ installation is working correctly.
"""

import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time
import argparse

# Import Mega-NeRF++ components
from src.nerfs.mega_nerf_plus.core import MegaNeRFPlus, MegaNeRFPlusConfig
from src.nerfs.mega_nerf_plus.trainer import MegaNeRFPlusTrainer
from src.nerfs.mega_nerf_plus.spatial_partitioner import AdaptiveOctree, PartitionConfig
from src.nerfs.mega_nerf_plus.memory_manager import MemoryManager
from src.nerfs.mega_nerf_plus.multires_renderer import PhotogrammetricVolumetricRenderer


def create_demo_config() -> MegaNeRFPlusConfig:
    """Create a demonstration configuration suitable for quick testing"""
    
    return MegaNeRFPlusConfig(
        # Small network for demo
        num_levels=4,
        base_resolution=16,
        max_resolution=128,
        netdepth=4,
        netwidth=64,
        
        # Demo training parameters
        batch_size=512,
        lr_init=1e-3,
        lr_final=1e-4,
        lr_decay_steps=5000,
        
        # Rendering parameters
        num_samples=32,
        num_importance=64,
        
        # Memory settings
        max_memory_gb=4.0,
        use_mixed_precision=True,
        
        # Multi-resolution
        num_lods=3,
        progressive_upsampling=False,  # Disable for demo
        
        # Photogrammetric
        max_image_resolution=512,
        downsample_factor=2
    )


def generate_synthetic_scene():
    """Generate a simple synthetic scene for demonstration"""
    
    print("Generating synthetic scene...")
    
    # Scene parameters
    num_cameras = 20
    scene_radius = 3.0
    
    # Generate camera positions on a sphere
    theta = np.linspace(0, 2*np.pi, num_cameras, endpoint=False)
    phi = np.linspace(np.pi/4, 3*np.pi/4, num_cameras)
    
    camera_positions = []
    camera_orientations = []
    
    for i in range(num_cameras):
        # Camera position
        x = scene_radius * np.sin(phi[i]) * np.cos(theta[i])
        y = scene_radius * np.sin(phi[i]) * np.sin(theta[i])
        z = scene_radius * np.cos(phi[i])
        
        cam_pos = np.array([x, y, z])
        camera_positions.append(cam_pos)
        
        # Camera orientation (looking at origin)
        forward = -cam_pos / np.linalg.norm(cam_pos)
        up = np.array([0, 0, 1])
        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)
        up = np.cross(right, forward)
        
        orientation = np.stack([right, up, forward], axis=1)
        camera_orientations.append(orientation)
    
    # Convert to tensors
    camera_positions = torch.tensor(camera_positions, dtype=torch.float32)
    camera_orientations = torch.tensor(camera_orientations, dtype=torch.float32)
    
    # Scene bounds
    scene_bounds = torch.tensor([[-2, -2, -2], [2, 2, 2]], dtype=torch.float32)
    
    return {
        'camera_positions': camera_positions,
        'camera_orientations': camera_orientations,
        'scene_bounds': scene_bounds,
        'num_cameras': num_cameras
    }


def create_synthetic_dataset(scene_data, config):
    """Create a synthetic dataset for training"""
    
    print("Creating synthetic dataset...")
    
    # Simple synthetic RGB function (colored sphere)
    def synthetic_rgb(points):
        # Distance from origin
        distances = torch.norm(points, dim=-1, keepdim=True)
        
        # Create a colored sphere
        sphere_radius = 1.0
        sphere_mask = distances < sphere_radius
        
        # Color based on position
        colors = torch.sigmoid(points)  # Map to [0, 1]
        
        # Apply sphere mask
        colors = colors * sphere_mask.float()
        
        return colors
    
    # Generate rays and RGB values
    all_rays = []
    all_rgbs = []
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    for i in range(scene_data['num_cameras']):
        # Camera parameters
        cam_pos = scene_data['camera_positions'][i]
        cam_rot = scene_data['camera_orientations'][i]
        
        # Generate rays for this camera (simplified)
        image_size = (32, 32)  # Small for demo
        rays_o, rays_d, rgb_values = generate_camera_rays(
            cam_pos, cam_rot, image_size, synthetic_rgb
        )
        
        all_rays.append(torch.cat([rays_o, rays_d], dim=-1))
        all_rgbs.append(rgb_values)
    
    # Combine all data
    rays = torch.cat(all_rays, dim=0)
    rgbs = torch.cat(all_rgbs, dim=0)
    
    print(f"Generated {len(rays)} ray samples")
    
    return {
        'rays': rays,
        'rgbs': rgbs,
        'scene_bounds': scene_data['scene_bounds']
    }


def generate_camera_rays(cam_pos, cam_rot, image_size, rgb_function):
    """Generate rays for a single camera"""
    
    h, w = image_size
    
    # Generate pixel coordinates
    i_coords, j_coords = torch.meshgrid(
        torch.linspace(-1, 1, w),
        torch.linspace(-1, 1, h),
        indexing='xy'
    )
    
    # Convert to ray directions (simplified pinhole camera)
    focal_length = 1.0
    dirs = torch.stack([
        i_coords / focal_length,
        -j_coords / focal_length,
        -torch.ones_like(i_coords)
    ], dim=-1)
    
    # Transform to world coordinates
    dirs = torch.sum(dirs[..., None, :] * cam_rot, dim=-1)
    dirs = dirs / torch.norm(dirs, dim=-1, keepdim=True)
    
    # Ray origins
    origins = cam_pos.expand(dirs.shape)
    
    # Sample points along rays and get colors
    t_vals = torch.linspace(0.5, 3.0, 16)  # Sample from 0.5 to 3.0
    points = origins[..., None, :] + dirs[..., None, :] * t_vals[..., None]
    
    # Get RGB values using the synthetic function
    rgb_values = rgb_function(points.reshape(-1, 3))
    rgb_values = rgb_values.reshape(*points.shape[:-1], 3)
    
    # Volume rendering (simplified)
    weights = torch.softmax(-torch.norm(points, dim=-1), dim=-1)
    final_rgb = torch.sum(weights[..., None] * rgb_values, dim=-2)
    
    # Flatten spatial dimensions
    rays_o = origins.reshape(-1, 3)
    rays_d = dirs.reshape(-1, 3)
    rgb_flat = final_rgb.reshape(-1, 3)
    
    return rays_o, rays_d, rgb_flat


class SyntheticDataset(torch.utils.data.Dataset):
    """Simple dataset for synthetic data"""
    
    def __init__(self, rays, rgbs):
        self.rays = rays
        self.rgbs = rgbs
    
    def __len__(self):
        return len(self.rays)
    
    def __getitem__(self, idx):
        return {
            'rays': self.rays[idx],
            'rgbs': self.rgbs[idx]
        }


def demo_basic_functionality():
    """Demonstrate basic Mega-NeRF++ functionality"""
    
    print("\n" + "="*60)
    print("MEGA-NERF++ BASIC FUNCTIONALITY DEMO")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Create configuration
    print("\n1. Creating configuration...")
    config = create_demo_config()
    print(f"   Network levels: {config.num_levels}")
    print(f"   Batch size: {config.batch_size}")
    print(f"   LOD levels: {config.num_lods}")
    
    # 2. Create model
    print("\n2. Creating Mega-NeRF++ model...")
    model = MegaNeRFPlus(config).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Total parameters: {total_params:,}")
    
    # 3. Test forward pass
    print("\n3. Testing forward pass...")
    batch_size = 64
    rays_o = torch.randn(batch_size, 3, device=device)
    rays_d = torch.nn.functional.normalize(torch.randn(batch_size, 3, device=device), dim=-1)
    
    start_time = time.time()
    with torch.no_grad():
        results = model.render_rays(rays_o, rays_d, near=0.1, far=5.0)
    end_time = time.time()
    
    print(f"   Rendered {batch_size} rays in {(end_time - start_time)*1000:.1f}ms")
    print(f"   Output shape: {results['coarse']['rgb'].shape}")
    print(f"   RGB range: [{results['coarse']['rgb'].min():.3f}, {results['coarse']['rgb'].max():.3f}]")
    
    return model, config


def demo_spatial_partitioning():
    """Demonstrate spatial partitioning capabilities"""
    
    print("\n" + "="*60)
    print("SPATIAL PARTITIONING DEMO")
    print("="*60)
    
    # Generate scene
    scene_data = generate_synthetic_scene()
    
    # Create partitioner
    partition_config = PartitionConfig(
        max_partition_size=128,
        min_partition_size=32,
        max_depth=3
    )
    
    partitioner = AdaptiveOctree(partition_config)
    
    # Partition scene
    print(f"\nPartitioning scene with {scene_data['num_cameras']} cameras...")
    partitions = partitioner.partition_scene(
        scene_data['scene_bounds'],
        scene_data['camera_positions']
    )
    
    print(f"Created {len(partitions)} partitions")
    
    # Analyze partitions
    for i, partition in enumerate(partitions[:3]):  # Show first 3
        bounds = partition['bounds']
        size = torch.norm(bounds[1] - bounds[0])
        num_cameras = len(partition['cameras'])
        print(f"  Partition {i}: size={size:.2f}, cameras={num_cameras}")
    
    return partitions


def demo_memory_management():
    """Demonstrate memory management features"""
    
    print("\n" + "="*60)
    print("MEMORY MANAGEMENT DEMO")
    print("="*60)
    
    # Create memory manager
    memory_manager = MemoryManager(max_memory_gb=8.0)
    
    # Show initial memory stats
    stats = memory_manager.get_memory_stats()
    print(f"\nMemory Statistics:")
    for key, value in stats.items():
        if 'gpu' in key and torch.cuda.is_available():
            print(f"  {key}: {value:.2f} GB")
        elif 'cpu' in key:
            print(f"  {key}: {value:.1f}%")
    
    # Test tensor caching
    print(f"\nTesting tensor caching...")
    test_tensor = torch.randn(1000, 1000)
    memory_manager.cache_tensor('demo_tensor', test_tensor)
    
    cached_tensor = memory_manager.get_cached_tensor('demo_tensor')
    cache_hit = cached_tensor is not None
    print(f"  Cache hit: {cache_hit}")
    
    # Test memory cleanup
    print(f"  Running memory cleanup...")
    memory_manager.cleanup_cache()
    print(f"  Cleanup completed")
    
    return memory_manager


def demo_training():
    """Demonstrate training on synthetic data"""
    
    print("\n" + "="*60)
    print("TRAINING DEMO")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create synthetic data
    scene_data = generate_synthetic_scene()
    config = create_demo_config()
    dataset_dict = create_synthetic_dataset(scene_data, config)
    
    # Create dataset and dataloader
    dataset = SyntheticDataset(dataset_dict['rays'], dataset_dict['rgbs'])
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=config.batch_size, shuffle=True
    )
    
    # Create model
    model = MegaNeRFPlus(config).to(device)
    
    # Simple training loop
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr_init)
    model.train()
    
    print(f"Training on {len(dataset)} samples for 10 steps...")
    
    losses = []
    for step, batch in enumerate(dataloader):
        if step >= 10:  # Only train for 10 steps in demo
            break
            
        # Move to device
        rays = batch['rays'].to(device)
        target_rgb = batch['rgbs'].to(device)
        
        # Forward pass
        rays_o = rays[..., :3]
        rays_d = rays[..., 3:6]
        
        results = model.render_rays(rays_o, rays_d, near=0.1, far=5.0)
        
        # Compute loss
        pred_rgb = results['coarse']['rgb']
        loss = torch.nn.functional.mse_loss(pred_rgb, target_rgb)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if step % 5 == 0:
            print(f"  Step {step}: loss = {loss.item():.6f}")
    
    print(f"Training complete. Final loss: {losses[-1]:.6f}")
    
    return model, losses


def demo_inference():
    """Demonstrate inference and novel view synthesis"""
    
    print("\n" + "="*60)
    print("INFERENCE DEMO")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Use a pre-created model (simplified)
    config = create_demo_config()
    model = MegaNeRFPlus(config).to(device)
    model.eval()
    
    # Generate novel view
    print("Generating novel view...")
    
    # Camera parameters for novel view
    cam_pos = torch.tensor([2.0, 2.0, 2.0], device=device)
    look_at = torch.tensor([0.0, 0.0, 0.0], device=device)
    up = torch.tensor([0.0, 0.0, 1.0], device=device)
    
    # Create camera matrix
    forward = torch.nn.functional.normalize(look_at - cam_pos, dim=0)
    right = torch.nn.functional.normalize(torch.cross(forward, up), dim=0)
    up = torch.cross(right, forward)
    
    # Generate rays for small image
    image_size = (64, 64)
    h, w = image_size
    
    # Pixel coordinates
    i_coords, j_coords = torch.meshgrid(
        torch.linspace(-1, 1, w, device=device),
        torch.linspace(-1, 1, h, device=device),
        indexing='xy'
    )
    
    # Ray directions
    focal_length = 1.0
    dirs = torch.stack([
        i_coords / focal_length,
        -j_coords / focal_length,
        -torch.ones_like(i_coords, device=device)
    ], dim=-1)
    
    # Transform to world coordinates
    camera_matrix = torch.stack([right, up, forward], dim=1)
    dirs = torch.sum(dirs[..., None, :] * camera_matrix, dim=-1)
    dirs = torch.nn.functional.normalize(dirs, dim=-1)
    
    origins = cam_pos.expand(dirs.shape)
    
    # Render in chunks
    chunk_size = 256
    rendered_pixels = []
    
    rays_o_flat = origins.reshape(-1, 3)
    rays_d_flat = dirs.reshape(-1, 3)
    
    print(f"Rendering {len(rays_o_flat)} rays...")
    
    with torch.no_grad():
        for i in range(0, len(rays_o_flat), chunk_size):
            chunk_rays_o = rays_o_flat[i:i+chunk_size]
            chunk_rays_d = rays_d_flat[i:i+chunk_size]
            
            chunk_results = model.render_rays(
                chunk_rays_o, chunk_rays_d, near=0.1, far=5.0
            )
            
            chunk_rgb = chunk_results['coarse']['rgb']
            rendered_pixels.append(chunk_rgb)
    
    # Combine results
    rendered_flat = torch.cat(rendered_pixels, dim=0)
    rendered_image = rendered_flat.reshape(h, w, 3)
    
    print(f"Rendered image shape: {rendered_image.shape}")
    print(f"Pixel value range: [{rendered_image.min():.3f}, {rendered_image.max():.3f}]")
    
    return rendered_image


def demo_performance_benchmark():
    """Run performance benchmarks"""
    
    print("\n" + "="*60)
    print("PERFORMANCE BENCHMARK")
    print("="*60)
    
    if not torch.cuda.is_available():
        print("CUDA not available - skipping GPU benchmarks")
        return
    
    device = torch.device('cuda')
    config = create_demo_config()
    model = MegaNeRFPlus(config).to(device)
    model.eval()
    
    # Benchmark different batch sizes
    batch_sizes = [64, 128, 256, 512]
    
    print("Batch Size | Rays/Second | GPU Memory (MB)")
    print("-" * 45)
    
    for batch_size in batch_sizes:
        # Generate random rays
        rays_o = torch.randn(batch_size, 3, device=device)
        rays_d = torch.nn.functional.normalize(
            torch.randn(batch_size, 3, device=device), dim=-1
        )
        
        # Warm up
        with torch.no_grad():
            for _ in range(5):
                _ = model.render_rays(rays_o, rays_d, near=0.1, far=5.0)
        
        # Benchmark
        torch.cuda.synchronize()
        start_time = time.time()
        
        num_runs = 10
        with torch.no_grad():
            for _ in range(num_runs):
                _ = model.render_rays(rays_o, rays_d, near=0.1, far=5.0)
        
        torch.cuda.synchronize()
        end_time = time.time()
        
        # Calculate metrics
        total_rays = batch_size * num_runs
        total_time = end_time - start_time
        rays_per_second = total_rays / total_time
        gpu_memory_mb = torch.cuda.memory_allocated() / (1024**2)
        
        print(f"{batch_size:10d} | {rays_per_second:11.0f} | {gpu_memory_mb:13.1f}")


def main():
    """Main demo function"""
    
    parser = argparse.ArgumentParser(description='Mega-NeRF++ Demo')
    parser.add_argument('--quick', action='store_true', 
                       help='Run quick demo (skip training and benchmarks)')
    parser.add_argument('--no-cuda', action='store_true',
                       help='Force CPU usage')
    
    args = parser.parse_args()
    
    if args.no_cuda:
        # Force CPU usage
        import os
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    
    print("🚀 Welcome to the Mega-NeRF++ Demo!")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    try:
        # Basic functionality demo
        model, config = demo_basic_functionality()
        
        # Spatial partitioning demo
        partitions = demo_spatial_partitioning()
        
        # Memory management demo
        memory_manager = demo_memory_management()
        
        if not args.quick:
            # Training demo
            trained_model, losses = demo_training()
            
            # Inference demo
            rendered_image = demo_inference()
            
            # Performance benchmark
            demo_performance_benchmark()
        
        print("\n" + "="*60)
        print("🎉 DEMO COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\nMega-NeRF++ is working correctly on your system.")
        print("You can now proceed to train on your own photogrammetric datasets.")
        
        # Provide next steps
        print("\n📖 Next Steps:")
        print("1. Prepare your photogrammetric dataset")
        print("2. Use the example_usage.py script for training")
        print("3. Check the README.md for detailed documentation")
        print("4. Visit the repository for more examples")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {str(e)}")
        print("\nPlease check your installation and try again.")
        print("If the problem persists, please report an issue.")
        raise


if __name__ == '__main__':
    main() 