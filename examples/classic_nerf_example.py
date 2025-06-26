"""
Classic NeRF Example Usage

This script demonstrates how to use Classic NeRF for training and evaluation.
"""

import os
import sys
import torch
import argparse
from pathlib import Path

# Add the project root to Python path to enable imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from src.nerfs.classic_nerf import (
        NeRF, NeRFConfig, NeRFTrainer, create_nerf_dataloader
    )
except ImportError as e:
    print(f"Import error: {e}")
    print("Please make sure you're running from the project root directory")
    sys.exit(1)


def basic_classic_nerf_example():
    """Basic Classic NeRF example."""
    print("=== Basic Classic NeRF Example ===")
    
    # Create configuration with only valid parameters
    config = NeRFConfig(
        # Network architecture
        netdepth=8, netwidth=256, netdepth_fine=8, netwidth_fine=256, # Positional encoding
        multires=10, multires_views=4, # Sampling
        N_samples=64, N_importance=128, perturb=True, use_viewdirs=True, # Rendering
        raw_noise_std=0.0, white_bkgd=True, # Training
        learning_rate=5e-4, lrate_decay=250, # Scene bounds
        near=2.0, far=6.0, # Loss weights
        rgb_loss_weight=1.0, # Optimizer
        beta1=0.9, beta2=0.999, epsilon=1e-7
    )
    
    # Create model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NeRF(config).to(device)
    
    print(f"Created Classic NeRF model on {device}")  
    # print(f"Model parameters: {sum(p.numel() for p in model.parameters()):, d}")  # noqa: F541
    
    return model, config


def synthetic_scene_example(data_path: str, output_dir: str):
    """Example training on synthetic NeRF dataset."""
    print("=== Synthetic Scene Classic NeRF Training ===")
    
    config = NeRFConfig(
        # Network architecture
        netdepth=8, netwidth=256, netdepth_fine=8, netwidth_fine=256, # Positional encoding
        multires=10, multires_views=4, # Sampling
        N_samples=64, N_importance=128, perturb=True, use_viewdirs=True, # Rendering
        raw_noise_std=1e0, # Add noise for regularization
        white_bkgd=True, # Synthetic scenes typically have white background
        
        # Training
        learning_rate=5e-4, lrate_decay=250, # Scene bounds for synthetic scenes
        near=2.0, far=6.0, # Loss weights
        rgb_loss_weight=1.0
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create model and trainer
    model = NeRF(config).to(device)
    trainer = NeRFTrainer(config)
    
    # Create dataset
    train_loader = create_nerf_dataloader(
        'blender', data_path, split='train', batch_size=1, shuffle=True
    )
    
    val_loader = create_nerf_dataloader(
        'blender', data_path, split='val', batch_size=1, shuffle=False
    )
    
    print(f"Training dataset: {len(train_loader)} batches")
    print(f"Validation dataset: {len(val_loader)} batches")
    print(f"Output directory: {output_dir}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Start training
    trainer.train(
        train_loader=train_loader, val_loader=val_loader, num_epochs=100, output_dir=output_dir, device=device
    )
    
    return model, trainer


def real_world_example(data_path: str, output_dir: str):
    """Example training on real-world LLFF dataset."""
    print("=== Real-World Classic NeRF Training ===")
    
    config = NeRFConfig(
        # Network architecture
        netdepth=8, netwidth=256, netdepth_fine=8, netwidth_fine=256, # Positional encoding
        multires=10, multires_views=4, # Sampling
        N_samples=64, N_importance=128, perturb=True, use_viewdirs=True, # Rendering
        raw_noise_std=0.0, white_bkgd=False, # Real scenes typically don't have white background
        
        # Training
        learning_rate=5e-4, lrate_decay=250, # Scene bounds for real scenes (adjust based on dataset)
        near=0.1, far=100.0, # Loss weights
        rgb_loss_weight=1.0
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create model and trainer
    model = NeRF(config).to(device)
    trainer = NeRFTrainer(config)
    
    # Create dataset (assuming LLFF format)
    train_loader = create_nerf_dataloader(
        'llff', data_path, split='train', batch_size=1, shuffle=True
    )
    
    val_loader = create_nerf_dataloader(
        'llff', data_path, split='val', batch_size=1, shuffle=False
    )
    
    print(f"Training dataset: {len(train_loader)} batches")
    print(f"Validation dataset: {len(val_loader)} batches")
    print(f"Output directory: {output_dir}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Start training
    trainer.train(
        train_loader=train_loader, val_loader=val_loader, num_epochs=200, # More epochs for real scenes
        output_dir=output_dir, device=device
    )
    
    return model, trainer


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Classic NeRF Example")
    parser.add_argument(
        "--example",
        type=str,
        default="basic",
        choices=["basic",
        "synthetic",
        "real_world"],
        help="Example to run",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="./data/nerf_synthetic/lego",
        help="Path to dataset",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs/classic_nerf",
        help="Output directory",
    )
    
    args = parser.parse_args()
    
    if args.example == "basic":
        model, config = basic_classic_nerf_example()
        print("Basic Classic NeRF example completed!")
        
    elif args.example == "synthetic":
        if not os.path.exists(args.data_path):
            print(f"Error: Data path {args.data_path} does not exist")
            print("Please download the NeRF synthetic dataset:")
            print("wget http://cseweb.ucsd.edu/~viscomp/projects/LF/papers/ECCV20/nerf/nerf_synthetic.zip")
            return
        
        model, trainer = synthetic_scene_example(args.data_path, args.output_dir)
        print("Synthetic scene training completed!")
        
    elif args.example == "real_world":
        if not os.path.exists(args.data_path):
            print(f"Error: Data path {args.data_path} does not exist")
            print("Please download the NeRF LLFF dataset:")
            print("wget http://cseweb.ucsd.edu/~viscomp/projects/LF/papers/ECCV20/nerf/nerf_llff_data.zip")
            return
        
        model, trainer = real_world_example(args.data_path, args.output_dir)
        print("Real-world training completed!")


if __name__ == "__main__":
    main() 