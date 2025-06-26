"""
Example usage of Mega-NeRF++ for large-scale photogrammetric reconstruction

This script demonstrates how to use Mega-NeRF++ for training and inference
on high-resolution photogrammetric datasets.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import argparse
import json
import time

from .core import MegaNeRFPlus, MegaNeRFPlusConfig
from .dataset import create_meganerf_plus_dataset, create_photogrammetric_dataloader
from .trainer import MegaNeRFPlusTrainer, MultiScaleTrainer, DistributedTrainer
from .spatial_partitioner import PhotogrammetricPartitioner, PartitionConfig
from .memory_manager import MemoryManager
from .multires_renderer import PhotogrammetricVolumetricRenderer


def create_sample_config() -> MegaNeRFPlusConfig:
    """Create a sample configuration for Mega-NeRF++"""
    
    return MegaNeRFPlusConfig(
        # Network architecture
        num_levels=8, base_resolution=32, max_resolution=2048, # MLP parameters
        netdepth=8, netwidth=256, netdepth_fine=8, netwidth_fine=256, # Spatial partitioning
        max_partition_size=1024, min_partition_size=64, overlap_ratio=0.1, adaptive_partitioning=True, # Multi-resolution parameters
        num_lods=4, lod_threshold=0.01, # Photogrammetric parameters
        max_image_resolution=4096, downsample_factor=2, progressive_upsampling=True, # Training parameters
        batch_size=4096, chunk_size=1024, lr_init=5e-4, lr_final=5e-6, lr_decay_steps=200000, # Memory management
        max_memory_gb=12.0, use_mixed_precision=True, gradient_checkpointing=True, # Rendering parameters
        num_samples=64, num_importance=128, use_viewdirs=True, # Loss parameters
        lambda_rgb=1.0, lambda_depth=0.1, lambda_semantic=0.0, lambda_distortion=0.01
    )


def basic_training_example(data_dir: str, output_dir: str):
    """
    Basic training example with Mega-NeRF++
    
    Args:
        data_dir: Path to dataset directory
        output_dir: Path to output directory
    """
    
    print("=== Basic Mega-NeRF++ Training Example ===")
    
    # Create configuration
    config = create_sample_config()
    
    # Create dataset
    print("Loading dataset...")
    train_dataset = create_meganerf_plus_dataset(
        data_dir, dataset_type='photogrammetric', split='train', max_image_resolution=config.max_image_resolution, downsample_factor=config.downsample_factor
    )
    
    val_dataset = create_meganerf_plus_dataset(
        data_dir, dataset_type='photogrammetric', split='val', max_image_resolution=config.max_image_resolution, downsample_factor=config.downsample_factor
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Create model
    print("Creating model...")
    model = MegaNeRFPlus(config)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:, }")
    print(f"Trainable parameters: {trainable_params:, }")
    
    # Create trainer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    trainer = MegaNeRFPlusTrainer(
        config=config, model=model, train_dataset=train_dataset, val_dataset=val_dataset, device=device, log_dir=Path(
            output_dir,
        )
    )
    
    # Start training
    print("Starting training...")
    start_time = time.time()
    
    trainer.train(num_epochs=50)
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")


def large_scene_training_example(data_dir: str, output_dir: str):
    """
    Large scene training example with spatial partitioning
    
    Args:
        data_dir: Path to dataset directory  
        output_dir: Path to output directory
    """
    
    print("=== Large Scene Mega-NeRF++ Training Example ===")
    
    # Create configuration for large scenes
    config = create_sample_config()
    config.max_image_resolution = 8192  # Higher resolution
    config.batch_size = 2048  # Smaller batch size for memory
    config.max_memory_gb = 16.0  # More memory
    
    # Create large scene dataset with partitioning
    print("Loading large scene dataset...")
    partition_config = {
        'max_partition_size': 2048, 'min_partition_size': 128, 'overlap_ratio': 0.15, 'adaptive_threshold': 0.02
    }
    
    train_dataset = create_meganerf_plus_dataset(
        data_dir, dataset_type='large_scene', split='train', partition_config=partition_config, streaming_mode=True, max_image_resolution=config.max_image_resolution, downsample_factor=1  # Full resolution
    )
    
    print(f"Dataset partitions: {
        len(train_dataset.partitions) if hasattr(train_dataset,
        'partitions') else 'N/A',
    }
    
    # Create model with memory optimization
    print("Creating optimized model...")
    model = MegaNeRFPlus(config)
    
    # Apply memory optimizations
    from .memory_manager import MemoryOptimizer
    model = MemoryOptimizer.optimize_model_memory(
        model, use_checkpointing=True, use_mixed_precision=True
    )
    
    # Create multi-scale trainer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trainer = MultiScaleTrainer(
        config=config, model=model, train_dataset=train_dataset, val_dataset=None, # Skip validation for large scenes
        device=device, log_dir=Path(
            output_dir,
        )
    )
    
    # Start training
    print("Starting large scene training...")
    trainer.train(num_epochs=100)


def inference_example(model_path: str, data_dir: str, output_dir: str):
    """
    Inference example for rendering novel views
    
    Args:
        model_path: Path to trained model checkpoint
        data_dir: Path to dataset directory
        output_dir: Path to output directory
    """
    
    print("=== Mega-NeRF++ Inference Example ===")
    
    # Load configuration
    checkpoint = torch.load(model_path, map_location='cpu')
    config = checkpoint.get('config', create_sample_config())
    
    # Create model and load weights
    print("Loading model...")
    model = MegaNeRFPlus(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    
    # Load test dataset
    test_dataset = create_meganerf_plus_dataset(
        data_dir, dataset_type='photogrammetric', split='test', max_image_resolution=config.max_image_resolution, use_cached_rays=False  # Load full images for inference
    )
    
    # Create renderer
    renderer = PhotogrammetricVolumetricRenderer(config)
    
    # Render test images
    output_path = Path(output_dir) / 'rendered_images'
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Rendering {len(test_dataset)} test images...")
    
    with torch.no_grad():
        for i in range(len(test_dataset)):
            print(f"Rendering image {i+1}/{len(test_dataset)}")
            
            data = test_dataset[i]
            if 'image' not in data:
                continue
            
            target_image = data['image'].to(device)
            pose = data['pose'].to(device)
            intrinsic = data['intrinsics'].to(device)
            
            # Render image
            rendered_image = render_full_image(
                model, renderer, pose, intrinsic, target_image.shape[:2], config
            )
            
            # Save images
            target_np = target_image.cpu().numpy()
            rendered_np = rendered_image.cpu().numpy()
            
            # Convert to uint8
            target_uint8 = (np.clip(target_np, 0, 1) * 255).astype(np.uint8)
            rendered_uint8 = (np.clip(rendered_np, 0, 1) * 255).astype(np.uint8)
            
            # Save
            from PIL import Image
            Image.fromarray(target_uint8).save(output_path / f'target_{i:04d}.png')
            Image.fromarray(rendered_uint8).save(output_path / f'rendered_{i:04d}.png')
            
            # Compute metrics
            psnr = compute_psnr(rendered_image, target_image)
            print(f"  PSNR: {psnr:.2f} dB")


def render_full_image(
    model: nn.Module,
    renderer,
    pose: torch.Tensor,
    intrinsic: torch.Tensor,
    image_size: tuple,
    config: MegaNeRFPlusConfig,
)
    """Render a full image using the trained model"""
    
    h, w = image_size
    device = pose.device
    
    # Generate rays
    i_coords, j_coords = torch.meshgrid(
        torch.arange(
            w,
            dtype=torch.float32,
            device=device,
        )
    )
    
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]
    
    dirs = torch.stack([
        (i_coords - cx) / fx, -(j_coords - cy) / fy, -torch.ones_like(i_coords)
    ], dim=-1)
    
    dirs = torch.sum(dirs[..., None, :] * pose[:3, :3], dim=-1)
    origins = pose[:3, 3].expand(dirs.shape)
    
    # Render in chunks
    rays_o_flat = origins.reshape(-1, 3)
    rays_d_flat = dirs.reshape(-1, 3)
    
    rendered_pixels = []
    chunk_size = config.chunk_size
    
    for i in range(0, len(rays_o_flat), chunk_size):
        chunk_rays_o = rays_o_flat[i:i+chunk_size]
        chunk_rays_d = rays_d_flat[i:i+chunk_size]
        
        # Render chunk
        chunk_results = model.render_rays(
            chunk_rays_o, chunk_rays_d, near=0.1, far=100.0, lod=0, white_bkgd=True
        )
        
        if 'fine' in chunk_results:
            chunk_rgb = chunk_results['fine']['rgb']
        else:
            chunk_rgb = chunk_results['coarse']['rgb']
        
        rendered_pixels.append(chunk_rgb)
    
    # Combine and reshape
    rendered_flat = torch.cat(rendered_pixels, dim=0)
    rendered_image = rendered_flat.reshape(h, w, 3)
    
    return rendered_image


def compute_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Compute PSNR between prediction and target"""
    mse = torch.mean((pred - target) ** 2)
    psnr = -10.0 * torch.log10(mse)
    return psnr.item()


def distributed_training_example(data_dir: str, output_dir: str):
    """
    Distributed training example for multi-GPU setups
    
    Args:
        data_dir: Path to dataset directory
        output_dir: Path to output directory
    """
    
    print("=== Distributed Mega-NeRF++ Training Example ===")
    
    # This would typically be launched with torchrun
    # torchrun --nproc_per_node=4 example_usage.py --distributed
    
    config = create_sample_config()
    config.batch_size = 8192  # Larger batch size for distributed training
    
    # Create dataset
    train_dataset = create_meganerf_plus_dataset(
        data_dir, dataset_type='photogrammetric', split='train', max_image_resolution=config.max_image_resolution
    )
    
    # Create model
    model = MegaNeRFPlus(config)
    
    # Create distributed trainer
    trainer = DistributedTrainer(
        config=config, model=model, train_dataset=train_dataset, device='cuda', log_dir=Path(
            output_dir,
        )
    )
    
    # Start training
    trainer.train(num_epochs=200)


def main():
    """Main function with command line interface"""
    
    parser = argparse.ArgumentParser(description='Mega-NeRF++ Examples')
    parser.add_argument(
        '--mode',
        type=str,
        required=True,
        choices=['basic',
        'large_scene',
        'inference',
        'distributed'],
        help='Example mode to run',
    )
    parser.add_argument('--data_dir', type=str, required=True, help='Path to dataset directory')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to output directory')
    parser.add_argument('--model_path', type=str, help='Path to model checkpoint (for inference)')
    
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Run selected example
    if args.mode == 'basic':
        basic_training_example(args.data_dir, args.output_dir)
    elif args.mode == 'large_scene':
        large_scene_training_example(args.data_dir, args.output_dir)
    elif args.mode == 'inference':
        if not args.model_path:
            print("Error: --model_path required for inference mode")
            return
        inference_example(args.model_path, args.data_dir, args.output_dir)
    elif args.mode == 'distributed':
        distributed_training_example(args.data_dir, args.output_dir)


if __name__ == '__main__':
    main() 