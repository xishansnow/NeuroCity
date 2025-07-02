#!/usr/bin/env python3
"""
Test script for GPU-optimized SVRaster implementation
"""

import torch
import numpy as np
import time
import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from src.nerfs.svraster.cuda.svraster_gpu import SVRasterGPU, SVRasterGPUTrainer
from src.nerfs.svraster.core import SVRasterConfig

def test_cuda_availability():
    """Test if CUDA is available"""
    print("Testing CUDA availability...")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print()

def test_model_creation():
    """Test SVRaster GPU model creation"""
    print("Testing SVRaster GPU model creation...")
    
    config = SVRasterConfig(
        max_octree_levels=8,
        base_resolution=32,
        scene_bounds=(-1.0, -1.0, -1.0, 1.0, 1.0, 1.0),
        subdivision_threshold=0.01,
        pruning_threshold=0.001
    )
    
    model = SVRasterGPU(config)
    print(f"Model created successfully")
    print(f"Device: {model.device}")
    
    stats = model.get_voxel_statistics()
    print(f"Total voxels: {stats['total_voxels']}")
    print(f"Number of levels: {stats['num_levels']}")
    print()

def test_forward_pass():
    """Test forward pass"""
    print("Testing forward pass...")
    
    config = SVRasterConfig(
        max_octree_levels=8,
        base_resolution=32,
        scene_bounds=(-1.0, -1.0, -1.0, 1.0, 1.0, 1.0)
    )
    
    model = SVRasterGPU(config)
    
    # Create test rays
    num_rays = 1000
    ray_origins = torch.randn(num_rays, 3, device=model.device)
    ray_directions = torch.randn(num_rays, 3, device=model.device)
    ray_directions = ray_directions / ray_directions.norm(dim=1, keepdim=True)
    
    # Forward pass
    start_time = time.time()
    outputs = model(ray_origins, ray_directions)
    forward_time = time.time() - start_time
    
    print(f"Forward pass completed in {forward_time:.4f} seconds")
    print(f"Output shapes:")
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")
        else:
            print(f"  {key}: {type(value)}")
    print()

def test_training_step():
    """Test training step"""
    print("Testing training step...")
    
    config = SVRasterConfig(
        max_octree_levels=8,
        base_resolution=32,
        scene_bounds=(-1.0, -1.0, -1.0, 1.0, 1.0, 1.0),
        learning_rate=0.001,
        subdivision_interval=10,
        pruning_interval=20
    )
    
    model = SVRasterGPU(config)
    trainer = SVRasterGPUTrainer(model, config)
    
    # Create test data
    num_rays = 1000
    ray_origins = torch.randn(num_rays, 3, device=model.device)
    ray_directions = torch.randn(num_rays, 3, device=model.device)
    ray_directions = ray_directions / ray_directions.norm(dim=1, keepdim=True)
    target_colors = torch.rand(num_rays, 3, device=model.device)
    
    # Training step
    start_time = time.time()
    metrics = trainer.train_step(ray_origins, ray_directions, target_colors)
    step_time = time.time() - start_time
    
    print(f"Training step completed in {step_time:.4f} seconds")
    print(f"Loss: {metrics['loss']:.6f}")
    print(f"PSNR: {metrics['psnr']:.2f}")
    print()

def test_adaptive_operations():
    """Test adaptive subdivision and pruning"""
    print("Testing adaptive operations...")
    
    config = SVRasterConfig(
        max_octree_levels=8,
        base_resolution=32,
        scene_bounds=(-1.0, -1.0, -1.0, 1.0, 1.0, 1.0),
        subdivision_threshold=0.01,
        pruning_threshold=0.001
    )
    
    model = SVRasterGPU(config)
    
    # Get initial stats
    initial_stats = model.get_voxel_statistics()
    print(f"Initial voxels: {initial_stats['total_voxels']}")
    
    # Test subdivision
    print("Testing adaptive subdivision...")
    subdivision_criteria = torch.randn(initial_stats['total_voxels'], device=model.device)
    
    start_time = time.time()
    model.adaptive_subdivision(subdivision_criteria)
    subdivision_time = time.time() - start_time
    
    print(f"Subdivision completed in {subdivision_time:.4f} seconds")
    
    # Test pruning
    print("Testing voxel pruning...")
    start_time = time.time()
    model.voxel_pruning()
    pruning_time = time.time() - start_time
    
    print(f"Pruning completed in {pruning_time:.4f} seconds")
    
    # Get final stats
    final_stats = model.get_voxel_statistics()
    print(f"Final voxels: {final_stats['total_voxels']}")
    print()

def test_performance():
    """Test performance with different batch sizes"""
    print("Testing performance...")
    
    config = SVRasterConfig(
        max_octree_levels=8,
        base_resolution=32,
        scene_bounds=(-1.0, -1.0, -1.0, 1.0, 1.0, 1.0)
    )
    
    model = SVRasterGPU(config)
    
    batch_sizes = [100, 1000, 5000, 10000]
    
    for batch_size in batch_sizes:
        print(f"Testing batch size: {batch_size}")
        
        # Create test data
        ray_origins = torch.randn(batch_size, 3, device=model.device)
        ray_directions = torch.randn(batch_size, 3, device=model.device)
        ray_directions = ray_directions / ray_directions.norm(dim=1, keepdim=True)
        
        # Warmup
        for _ in range(3):
            _ = model(ray_origins, ray_directions)
        
        # Benchmark
        torch.cuda.synchronize()
        start_time = time.time()
        
        for _ in range(10):
            outputs = model(ray_origins, ray_directions)
        
        torch.cuda.synchronize()
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 10
        rays_per_second = batch_size / avg_time
        
        print(f"  Average time: {avg_time:.4f} seconds")
        print(f"  Rays per second: {rays_per_second:.0f}")
        print()

def test_memory_usage():
    """Test memory usage"""
    print("Testing memory usage...")
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping memory test")
        return
    
    torch.cuda.empty_cache()
    initial_memory = torch.cuda.memory_allocated() / 1e6
    
    config = SVRasterConfig(
        max_octree_levels=8,
        base_resolution=64,  # Larger resolution
        scene_bounds=(-1.0, -1.0, -1.0, 1.0, 1.0, 1.0)
    )
    
    model = SVRasterGPU(config)
    
    model_memory = torch.cuda.memory_allocated() / 1e6
    print(f"Model memory usage: {model_memory:.1f} MB")
    
    # Test with different batch sizes
    batch_sizes = [1000, 5000, 10000]
    
    for batch_size in batch_sizes:
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated() / 1e6
        
        ray_origins = torch.randn(batch_size, 3, device=model.device)
        ray_directions = torch.randn(batch_size, 3, device=model.device)
        ray_directions = ray_directions / ray_directions.norm(dim=1, keepdim=True)
        
        outputs = model(ray_origins, ray_directions)
        
        peak_memory = torch.cuda.max_memory_allocated() / 1e6
        print(f"Batch size {batch_size}: Peak memory {peak_memory:.1f} MB")
    
    print()

def main():
    """Run all tests"""
    print("SVRaster GPU Implementation Tests")
    print("=" * 50)
    print()
    
    try:
        test_cuda_availability()
        test_model_creation()
        test_forward_pass()
        test_training_step()
        test_adaptive_operations()
        test_performance()
        test_memory_usage()
        
        print("All tests completed successfully!")
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 