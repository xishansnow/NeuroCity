"""
CNC-NeRF Example Usage

This module provides examples of how to use the CNC-NeRF implementation.
"""

import torch
import numpy as np
from pathlib import Path

from .core import CNCNeRF, CNCNeRFConfig
from .dataset import CNCNeRFDataset, CNCNeRFDatasetConfig, create_synthetic_dataset
from .trainer import CNCNeRFTrainer, CNCNeRFTrainerConfig, create_cnc_nerf_trainer


def basic_usage_example():
    """Basic usage example of CNC-NeRF."""
    print("=== CNC-NeRF Basic Usage Example ===")
    
    # Create model configuration
    model_config = CNCNeRFConfig(
        feature_dim=8, num_levels=8, base_resolution=16, max_resolution=256, use_binarization=True, compression_lambda=0.001
    )
    
    # Create model
    model = CNCNeRF(model_config)
    print(f"Created CNC-NeRF model with {sum(p.numel() for p in model.parameters()):, } parameters")
    
    # Test forward pass
    coords = torch.rand(1000, 3)
    view_dirs = torch.rand(1000, 3)
    
    with torch.no_grad():
        output = model(coords, view_dirs)
        print(f"Forward pass output shapes:")
        print(f"  Density: {output['density'].shape}")
        print(f"  Color: {output['color'].shape}")
        print(f"  Features: {output['features'].shape}")
    
    # Test compression
    print("\nTesting compression...")
    compression_info = model.compress_model()
    stats = model.get_compression_stats()
    
    print(f"Compression results:")
    print(f"  Original size: {stats['original_size_mb']:.2f} MB")
    print(f"  Compressed size: {stats['compressed_size_mb']:.2f} MB")
    print(f"  Compression ratio: {stats['compression_ratio']:.1f}x")
    print(f"  Size reduction: {stats['size_reduction_percent']:.1f}%")
    
    return model, compression_info


def training_example():
    """Example of training CNC-NeRF on synthetic data."""
    print("\n=== CNC-NeRF Training Example ===")
    
    # Create dataset configuration
    dataset_config = CNCNeRFDatasetConfig(
        data_root="cnc_synthetic_data", image_width=200, image_height=150, pyramid_levels=3, use_pyramid_loss=True, num_rays_per_batch=512
    )
    
    # Create synthetic dataset
    print("Creating synthetic dataset...")
    dataset = create_synthetic_dataset(dataset_config, scene_type='lego')
    
    # Create model configuration
    model_config = CNCNeRFConfig(
        feature_dim=4, num_levels=6, base_resolution=16, max_resolution=128, use_binarization=True, compression_lambda=0.01
    )
    
    # Create trainer configuration
    trainer_config = CNCNeRFTrainerConfig(
        num_epochs=50, learning_rate=1e-3, compression_loss_weight=0.01, val_every=5, save_every=10, log_every=10
    )
    
    # Create trainer
    trainer = create_cnc_nerf_trainer(model_config, dataset_config, trainer_config)
    
    # Train for a few epochs (for demonstration)
    print("Starting training...")
    trainer.config.num_epochs = 5  # Just a few epochs for demo
    
    trainer.train()
    
    # Test compression after training
    print("\nTesting compression after training...")
    compression_results = trainer.compress_and_evaluate()
    
    print("Training example completed!")
    
    return trainer, compression_results


def compression_analysis_example():
    """Example of analyzing compression performance."""
    print("\n=== CNC-NeRF Compression Analysis Example ===")
    
    # Test different compression settings
    compression_configs = [
        {
            'use_binarization': False,
            'compression_lambda': 0.0,
            'name': 'No compression',
        },
        {
            'use_binarization': True,
            'compression_lambda': 0.001,
            'name': 'Binarization',
        },
        {
            'use_binarization': True,
            'compression_lambda': 0.01,
            'name': 'Binarization + Compression',
        },
    ]
    
    results = []
    
    for config in compression_configs:
        print(f"\nTesting {config['name']}...")
        
        # Create model with specific compression settings
        model_config = CNCNeRFConfig(
            feature_dim=8, num_levels=8, base_resolution=16, max_resolution=256, use_binarization=config['use_binarization'], compression_lambda=config['compression_lambda']
        )
        
        model = CNCNeRF(model_config)
        
        # Test compression
        compression_info = model.compress_model()
        stats = model.get_compression_stats()
        
        result = {
            'name': config['name'], 'original_size_mb': stats['original_size_mb'], 'compressed_size_mb': stats['compressed_size_mb'], 'compression_ratio': stats['compression_ratio'], 'size_reduction_percent': stats['size_reduction_percent']
        }
        
        results.append(result)
        
        print(f"  Original size: {result['original_size_mb']:.2f} MB")
        print(f"  Compressed size: {result['compressed_size_mb']:.2f} MB")
        print(f"  Compression ratio: {result['compression_ratio']:.1f}x")
    
    return results


def rendering_speed_benchmark():
    """Benchmark rendering speed with different configurations."""
    print("\n=== CNC-NeRF Rendering Speed Benchmark ===")
    
    import time
    
    configs = [
        {
            'num_levels': 4,
            'max_resolution': 128,
            'name': 'Low quality',
        },
        {
            'num_levels': 8,
            'max_resolution': 256,
            'name': 'High quality',
        },
    ]
    
    num_rays = 1000
    num_samples = 64
    num_runs = 5
    
    for config in configs:
        print(f"\nTesting {config['name']}...")
        
        model_config = CNCNeRFConfig(
            feature_dim=8, num_levels=config['num_levels'], base_resolution=16, max_resolution=config['max_resolution'], use_binarization=True
        )
        
        model = CNCNeRF(model_config)
        model.eval()
        
        # Prepare test data
        coords = torch.rand(num_rays * num_samples, 3)
        view_dirs = torch.rand(num_rays * num_samples, 3)
        
        # Benchmark
        times = []
        with torch.no_grad():
            for _ in range(num_runs):
                start_time = time.time()
                output = model(coords, view_dirs)
                end_time = time.time()
                times.append(end_time - start_time)
        
        avg_time = np.mean(times)
        rays_per_second = num_rays / avg_time
        
        print(f"  Average time: {avg_time:.4f} seconds")
        print(f"  Rays per second: {rays_per_second:.0f}")
        print(f"  Parameters: {sum(p.numel() for p in model.parameters()):, }")


def main():
    """Run all examples."""
    print("CNC-NeRF Examples")
    print("================")
    
    # Run examples
    model, compression_info = basic_usage_example()
    
    try:
        training_results = training_example()
        compression_results = compression_analysis_example()
        rendering_speed_benchmark()
        
        print("\n=== All Examples Completed Successfully! ===")
        
    except Exception as e:
        print(f"\nError during examples: {e}")
        print("Some examples may require additional setup or resources.")


if __name__ == "__main__":
    main() 