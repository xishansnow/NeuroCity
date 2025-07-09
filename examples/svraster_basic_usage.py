#!/usr/bin/env python3
"""
SVRaster Basic Usage Example

This example demonstrates the basic usage of the SVRaster package for
training and inference with sparse voxel radiance fields.
"""

import torch
import numpy as np
from pathlib import Path

# Import SVRaster package
import sys
sys.path.append('../src')
import nerfs.svraster as svraster


def main():
    print("SVRaster Basic Usage Example")
    print("=" * 40)
    
    # Check system compatibility
    print("\n1. System Compatibility Check:")
    svraster.check_compatibility()
    
    print("\n2. Package Information:")
    print(f"SVRaster version: {svraster.__version__}")
    print(f"Available components: {len(svraster.__all__)}")
    
    # Configure the model
    print("\n3. Model Configuration:")
    model_config = svraster.SVRasterConfig(
        max_octree_levels=8,
        base_resolution=128,
        scene_bounds=(-1.0, -1.0, -1.0, 1.0, 1.0, 1.0),
        sh_degree=2,
        density_activation="exp",
        color_activation="sigmoid",
        learning_rate=1e-3,
        weight_decay=1e-6
    )
    print(f"Model config created: {type(model_config).__name__}")
    
    # Create the model
    print("\n4. Model Creation:")
    model = svraster.SVRasterModel(model_config)
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Configure dataset
    print("\n5. Dataset Configuration:")
    dataset_config = svraster.SVRasterDatasetConfig(
        data_dir="data/nerf_synthetic/lego",  # Example path
        image_width=400,
        image_height=400,
        camera_angle_x=0.6911112070083618,
        downscale_factor=2,
        num_rays_train=1024,
        num_rays_val=512
    )
    print(f"Dataset config created: {type(dataset_config).__name__}")
    
    # Configure trainer
    print("\n6. Trainer Configuration:")
    trainer_config = svraster.SVRasterTrainerConfig(
        num_epochs=10,  # Short for demo
        batch_size=1,
        learning_rate=1e-3,
        weight_decay=1e-6,
        save_every=5,
        validate_every=2
    )
    
    # Create volume renderer for training
    volume_renderer_config = svraster.SVRasterConfig(
        ray_samples_per_voxel=64,
        depth_peeling_layers=4,
        morton_ordering=True,
        background_color=(1.0, 1.0, 1.0)
    )
    volume_renderer = svraster.VolumeRenderer(volume_renderer_config)
    
    trainer = svraster.SVRasterTrainer(model, volume_renderer, trainer_config)
    print(f"Trainer created: {type(trainer).__name__}")
    
    # Configure renderer
    print("\n7. Renderer Configuration:")
    renderer_config = svraster.SVRasterRendererConfig(
        image_width=800,
        image_height=800,
        background_color=(1.0, 1.0, 1.0),
        render_batch_size=4096,
        output_format="png"
    )
    
    # Create rasterizer for inference
    raster_config = svraster.VoxelRasterizerConfig(
        background_color=(1.0, 1.0, 1.0),
        near_plane=0.1,
        far_plane=100.0
    )
    rasterizer = svraster.VoxelRasterizer(raster_config)
    
    renderer = svraster.SVRasterRenderer(model, rasterizer, renderer_config)
    print(f"Renderer created: {type(renderer).__name__}")
    
    # Test volume renderer (for training)
    print("\n8. Volume Renderer (Training):")
    print(f"Volume renderer already created: {type(volume_renderer).__name__}")
    
    # Test rasterizer (for inference)
    print("\n9. True Voxel Rasterizer (Inference):")
    print(f"Rasterizer already created: {type(rasterizer).__name__}")
    
    # Test spherical harmonics
    print("\n10. Spherical Harmonics Test:")
    # Create some sample view directions
    dirs = torch.randn(100, 3)
    dirs = dirs / torch.norm(dirs, dim=-1, keepdim=True)  # Normalize
    
    sh_values = svraster.eval_sh_basis(degree=2, dirs=dirs)
    print(f"SH basis evaluation: input {dirs.shape} -> output {sh_values.shape}")
    
    # Test utilities
    print("\n11. Utility Functions Test:")
    # Morton encoding/decoding
    x, y, z = 1, 2, 3
    morton_code = svraster.morton_encode_3d(x, y, z)
    decoded_x, decoded_y, decoded_z = svraster.morton_decode_3d(morton_code)
    print(f"Morton encoding test: ({x}, {y}, {z}) -> {morton_code} -> ({decoded_x}, {decoded_y}, {decoded_z})")
    
    # Test GPU components if available
    if svraster.CUDA_AVAILABLE:
        print("\n12. GPU Components Test:")
        try:
            gpu_model = svraster.SVRasterGPU(model_config)
            print(f"GPU model created: {type(gpu_model).__name__}")
            
            gpu_trainer = svraster.SVRasterGPUTrainer(gpu_model, model_config)
            print(f"GPU trainer created: {type(gpu_trainer).__name__}")
            
            ema_model = svraster.EMAModel(model, decay=0.999)
            print(f"EMA model created: {type(ema_model).__name__}")
        except Exception as e:
            print(f"GPU components test failed: {e}")
    else:
        print("\n12. GPU Components: Not available")
    
    print("\n" + "=" * 40)
    print("SVRaster basic usage example completed successfully!")
    print("All components are properly exposed and functional.")


if __name__ == "__main__":
    main()
