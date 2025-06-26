"""
Grid-NeRF Example Usage

This script demonstrates how to use Grid-NeRF for training and evaluation
on large urban scenes. It includes examples for different datasets and
configurations.
"""

import os
import sys
import warnings
import logging
import torch
import argparse
from pathlib import Path
import multiprocessing as mp
from typing import List, Optional, Tuple
import numpy as np

# Add the project root to Python path to enable imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Suppress specific warnings for optional dependencies
warnings.filterwarnings('ignore', message='.*OpenVDB.*')
warnings.filterwarnings('ignore', message='.*trimesh.*')

# Import from the nerfs package using absolute imports
try:
    from src.nerfs.grid_nerf import (
        GridNeRF, GridNeRFConfig, GridNeRFTrainer, create_dataset, create_grid_nerf_model, get_default_config, quick_setup
    )
except ImportError as e:
    print(f"Import error: {e}")
    print("Please make sure you're running from the project root directory")
    sys.exit(1)


def basic_example():
    """Basic example with default configuration."""
    print("=== Basic Grid-NeRF Example ===")
    
    # Create default configuration
    config = get_default_config()
    
    # Update with specific settings (using correct parameter names)
    config.update({
        "scene_bounds": (-50, -50, -5, 50, 50, 20), # Tuple format
        "num_grid_levels": 3, # Correct parameter name
        "base_grid_resolution": 32, # Correct parameter name
        "learning_rate": 1e-3, # Use existing parameter
        "num_samples_coarse": 32, # Use existing parameter
    })
    
    # Create model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = create_grid_nerf_model(config, device)
    
    print(f"Created Grid-NeRF model on {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):, }")
    
    return model, config


def kitti360_example(data_path: str, output_dir: str):
    """Example training on KITTI-360 dataset."""
    print("=== KITTI-360 Grid-NeRF Training ===")
    
    # Configuration for KITTI-360 (using correct parameter names)
    config = {
        # Scene bounds for KITTI-360 (tuple format)
        "scene_bounds": (-200, -200, -10, 200, 200, 50), # Grid configuration
        "num_grid_levels": 4, # Correct parameter name
        "base_grid_resolution": 64, # Correct parameter name
        "max_grid_resolution": 512, # Correct parameter name
        "grid_feature_dim": 32, # Network architecture
        "mlp_num_layers": 4, # Correct parameter name
        "mlp_hidden_dim": 256, # Correct parameter name
        "view_dependent": True, # Training settings
        "learning_rate": 1e-2, "weight_decay": 1e-6, # Rendering
        "num_samples_coarse": 64, # Correct parameter name
        "num_samples_fine": 128, # Correct parameter name
        "near_plane": 0.1, "far_plane": 1000.0, # Loss weights
        "color_loss_weight": 1.0, "depth_loss_weight": 0.1, "grid_regularization_weight": 1e-4, }
    
    # Quick setup
    model, trainer, dataset = quick_setup(
        data_path=data_path, output_dir=output_dir, config=config, device="cuda" if torch.cuda.is_available(
        )
    )
    
    print(f"Dataset: {len(dataset)} samples")
    print(f"Output directory: {output_dir}")
    
    # Start training
    trainer.train(
        train_dataset=dataset, val_dataset=None, # Could create validation split
        test_dataset=None   # Could create test split
    )
    
    return model, trainer


def custom_dataset_example(data_path: str, output_dir: str):
    """Example with custom dataset configuration."""
    print("=== Custom Dataset Grid-NeRF Training ===")
    
    # Configuration for custom urban dataset (using correct parameters)
    config = GridNeRFConfig(
        # Scene bounds - adjust based on your data
        scene_bounds=(-100, -100, -10, 100, 100, 30), # Grid configuration
        num_grid_levels=4, # Correct parameter name
        base_grid_resolution=64, # Correct parameter name
        max_grid_resolution=512, # Correct parameter name
        grid_feature_dim=32, # Network architecture
        mlp_num_layers=3, # Correct parameter name
        mlp_hidden_dim=256, # Correct parameter name
        view_dependent=True, # Rendering
        num_samples_coarse=64, # Correct parameter name
        num_samples_fine=128, # Correct parameter name
        near_plane=0.1, far_plane=1000.0, # Training
        learning_rate=5e-4, weight_decay=1e-6, # Loss weights
        color_loss_weight=1.0, depth_loss_weight=0.1, grid_regularization_weight=1e-4, )
    
    # Create model and trainer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = GridNeRF(config).to(device)
    trainer = GridNeRFTrainer(
        config=config, output_dir=output_dir, device=device
    )
    
    # Create dataset
    train_dataset = create_dataset(
        data_path=data_path, split='train', config=config
    )
    
    # Optional: Create validation and test datasets
    val_dataset = None
    test_dataset = None
    
    if os.path.exists(os.path.join(data_path, 'val')):
        val_dataset = create_dataset(
            data_path=data_path, split='val', config=config
        )
    
    if os.path.exists(os.path.join(data_path, 'test')):
        test_dataset = create_dataset(
            data_path=data_path, split='test', config=config
        )
    
    print(f"Training samples: {len(train_dataset)}")
    if val_dataset:
        print(f"Validation samples: {len(val_dataset)}")
    if test_dataset:
        print(f"Test samples: {len(test_dataset)}")
    
    # Start training
    trainer.train(
        train_dataset=train_dataset, val_dataset=val_dataset, test_dataset=test_dataset
    )
    
    return model, trainer


def evaluation_example(model_path: str, data_path: str, output_dir: str):
    """Example of model evaluation and rendering."""
    print("=== Grid-NeRF Evaluation Example ===")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load trained model
    checkpoint = torch.load(model_path, map_location=device)
    config = GridNeRFConfig(**checkpoint['config'])
    
    model = GridNeRF(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Loaded model from {model_path}")
    print(f"Training epochs: {checkpoint.get('epoch', 'unknown')}")
    print(f"Best PSNR: {checkpoint.get('best_psnr', 'unknown')}")
    
    # Create test dataset
    test_dataset = create_dataset(
        data_path=data_path, split='test', config=config
    )
    
    # Render test images
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    with torch.no_grad():
        for i in range(min(10, len(test_dataset))):
            sample = test_dataset[i]
            
            # Generate rays for the image
            H, W = sample['image_height'], sample['image_width']
            rays_o, rays_d = test_dataset.get_rays(i)
            
            # Flatten rays
            rays_o_flat = rays_o.reshape(-1, 3).to(device)
            rays_d_flat = rays_d.reshape(-1, 3).to(device)
            
            # Render in chunks
            chunk_size = 1024  # Use default chunk size
            rgb_chunks = []
            depth_chunks = []
            
            for j in range(0, rays_o_flat.shape[0], chunk_size):
                rays_o_chunk = rays_o_flat[j:j+chunk_size]
                rays_d_chunk = rays_d_flat[j:j+chunk_size]
                
                outputs = model(rays_o_chunk, rays_d_chunk)
                rgb_chunks.append(outputs['rgb'].cpu())
                depth_chunks.append(outputs['depth'].cpu())
            
            # Combine chunks and reshape
            rgb_pred = torch.cat(rgb_chunks, dim=0).reshape(H, W, 3)
            depth_pred = torch.cat(depth_chunks, dim=0).reshape(H, W)
            
            # Save images
            from src.nerfs.grid_nerf.utils import save_image
            save_image(rgb_pred, output_path / f"render_{i:03d}.png")
            
            # Save depth as colormap
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 8))
            plt.imshow(depth_pred.numpy(), cmap='plasma')
            plt.colorbar()
            plt.title(f"Depth Map {i}")
            plt.savefig(output_path / f"depth_{i:03d}.png", dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"Rendered image {i+1}/{min(10, len(test_dataset))}")
    
    print(f"Evaluation results saved to {output_path}")


def multi_gpu_training_example(data_path: str, output_dir: str, num_gpus: int = 2):
    """Example of multi-GPU distributed training."""
    print(f"=== Multi-GPU Grid-NeRF Training ({num_gpus} GPUs) ===")
    
    # Configuration for distributed training
    config = get_default_config()
    config.update({
        "learning_rate": 1e-3, })
    
    grid_config = GridNeRFConfig(**config)
    
    # Data configuration
    data_config = {
        'train_data_path': data_path, 'train_kwargs': {}, }
    
    # Launch distributed training
    from src.nerfs.grid_nerf.trainer import main_worker
    
    mp.spawn(
        main_worker, args=(
            num_gpus,
            grid_config,
            output_dir,
            data_config,
        )
    )
    
    print("Multi-GPU training completed!")


def create_spiral_video_example(model_path: str, output_dir: str):
    """Example of creating a spiral video from trained model."""
    print("=== Spiral Video Generation ===")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    checkpoint = torch.load(model_path, map_location=device)
    config = GridNeRFConfig(**checkpoint['config'])
    
    model = GridNeRF(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Create spiral camera path
    import numpy as np
    
    center = np.array([0.0, 0.0, 10.0])  # Adjust based on scene
    radius = 20.0
    height_range = (5.0, 15.0)
    num_frames = 120
    
    # Generate camera poses
    poses = []
    for i in range(num_frames):
        angle = 2 * np.pi * i / num_frames
        height_ratio = i / (num_frames - 1)
        height = height_range[0] + height_ratio * (height_range[1] - height_range[0])
        
        # Camera position
        x = center[0] + radius * np.cos(angle)
        y = center[1] + radius * np.sin(angle)
        z = center[2] + height
        cam_pos = np.array([x, y, z])
        
        # Create camera pose matrix (simplified)
        pose = np.eye(4)
        pose[:3, 3] = cam_pos
        poses.append(pose)
    
    # Render frames
    output_path = Path(output_dir)
    frames_dir = output_path / "spiral_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    
    H, W = 800, 800  # Output resolution
    focal = 800.0    # Focal length
    
    # Use absolute import for utils
    from src.nerfs.grid_nerf.utils import get_ray_directions
    
    with torch.no_grad():
        for i, pose in enumerate(poses):
            # Generate rays for this pose
            directions = get_ray_directions(H, W, focal)
            
            # Transform rays to world coordinates
            rays_d = directions.reshape(-1, 3)  # Simplified
            rays_o = torch.tensor(pose[:3, 3]).float().expand_as(rays_d)
            
            # Move to device
            rays_o = rays_o.to(device)
            rays_d = rays_d.to(device)
            
            # Render in chunks
            chunk_size = 1024
            rgb_chunks = []
            
            for j in range(0, rays_o.shape[0], chunk_size):
                rays_o_chunk = rays_o[j:j+chunk_size]
                rays_d_chunk = rays_d[j:j+chunk_size]
                
                outputs = model(rays_o_chunk, rays_d_chunk)
                rgb_chunks.append(outputs['rgb'].cpu())
            
            # Combine and reshape
            rgb_frame = torch.cat(rgb_chunks, dim=0).reshape(H, W, 3)
            
            # Save frame
            from src.nerfs.grid_nerf.utils import save_image
            save_image(rgb_frame, frames_dir / f"frame_{i:04d}.png")
            
            print(f"Rendered frame {i+1}/{num_frames}")
    
    # Create video from frames
    from src.nerfs.grid_nerf.utils import create_video_from_images
    create_video_from_images(
        image_dir=frames_dir, output_path=output_path / "spiral_video.mp4", fps=30
    )
    
    print(f"Spiral video saved to {output_path / 'spiral_video.mp4'}")


def train_grid_nerf(
    data_path: str, output_dir: str, num_epochs: int = 100, batch_size: int = 4096, learning_rate: float = 1e-3, **kwargs
) -> dict[str, float]:
    """Train Grid NeRF model."""
    # Create model and trainer
    model, trainer = setup_grid_nerf(data_path, output_dir, batch_size, learning_rate)
    
    # Train model
    metrics = trainer.train(num_epochs)
    
    return {
        "loss": float(
            metrics["loss"],
        )
    }


def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(description="Grid-NeRF Example Usage")
    parser.add_argument(
        "--example",
        type=str,
        default="basic",
        choices=["basic",
        "kitti360",
        "custom",
        "eval",
        "multi_gpu",
        "spiral"],
        help="Example to run",
    )
    parser.add_argument("--data_path", type=str, help="Path to dataset")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Output directory")
    parser.add_argument("--model_path", type=str, help="Path to trained model")
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=2,
        help="Number of GPUs for multi-GPU training",
    )
    
    args = parser.parse_args()
    
    if args.example == "basic":
        model, config = basic_example()
        print("Basic example completed successfully!")
        
    elif args.example == "kitti360":
        if not args.data_path:
            print("Error: --data_path required for KITTI-360 example")
            return
        model, trainer = kitti360_example(args.data_path, args.output_dir)
        print("KITTI-360 training completed!")
        
    elif args.example == "custom":
        if not args.data_path:
            print("Error: --data_path required for custom dataset example")
            return
        model, trainer = custom_dataset_example(args.data_path, args.output_dir)
        print("Custom dataset training completed!")
        
    elif args.example == "eval":
        if not args.model_path or not args.data_path:
            print("Error: --model_path and --data_path required for evaluation")
            return
        evaluation_example(args.model_path, args.data_path, args.output_dir)
        print("Evaluation completed!")
        
    elif args.example == "multi_gpu":
        if not args.data_path:
            print("Error: --data_path required for multi-GPU training")
            return
        if not torch.cuda.is_available() or torch.cuda.device_count() < args.num_gpus:
            print(f"Error: {args.num_gpus} GPUs not available")
            return
        multi_gpu_training_example(args.data_path, args.output_dir, args.num_gpus)
        print("Multi-GPU training completed!")
        
    elif args.example == "spiral":
        if not args.model_path:
            print("Error: --model_path required for spiral video generation")
            return
        create_spiral_video_example(args.model_path, args.output_dir)
        print("Spiral video generation completed!")


if __name__ == "__main__":
    main() 