"""
Mip-NeRF Example Usage

This script demonstrates how to use Mip-NeRF for multiscale neural
radiance fields with anti-aliasing.
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
    from src.nerfs.mip_nerf import (
        MipNeRFConfig, MipNeRFModel, MipNeRFTrainer, create_mip_nerf_dataloader, create_mip_nerf_dataset
    )
except ImportError as e:
    print(f"Import error: {e}")
    print("Please make sure you're running from the project root directory")
    sys.exit(1)


def basic_example():
    """Basic Mip-NeRF example with default configuration."""
    print("=== Basic Mip-NeRF Example ===")
    
    config = MipNeRFConfig(
        # Multiscale representation
        num_levels=5, max_deg_point=16, # Network architecture
        netdepth=8, netwidth=256, # Integrated positional encoding
        use_integrated_encoding=True, # Training settings
        learning_rate=5e-4, weight_decay=1e-6, # Scene bounds
        near_plane=2.0, far_plane=6.0
    )
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MipNeRFModel(config).to(device)
    
    print(f"Created Mip-NeRF model on {device}")
    print(f"Multiscale levels: {config.num_levels}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):, }")
    
    return model, config


def anti_aliasing_example(data_path: str, output_dir: str):
    """Example demonstrating anti-aliasing capabilities."""
    print("=== Mip-NeRF Anti-Aliasing Training ===")
    
    config = MipNeRFConfig(
        # Enhanced multiscale representation
        num_levels=8, max_deg_point=20, # Network architecture
        netdepth=8, netwidth=256, netdepth_fine=8, netwidth_fine=256, # Integrated positional encoding with anti-aliasing
        use_integrated_encoding=True, use_multiscale_training=True, # Training settings
        learning_rate=1e-3, weight_decay=1e-6, scheduler_type="exponential", # Scene bounds
        near_plane=0.1, far_plane=100.0, # Loss weights
        rgb_loss_weight=1.0, coarse_loss_weight=0.1, # Anti-aliasing settings
        blur_loss_weight=0.01, distortion_loss_weight=0.01
    )
    
    # Create dataset
    dataset = create_mip_nerf_dataset(
        data_path=data_path, dataset_type="blender", config=config, multiscale_training=True
    )
    
    train_loader = create_mip_nerf_dataloader(
        dataset=dataset, split='train', batch_size=1, shuffle=True
    )
    
    val_loader = create_mip_nerf_dataloader(
        dataset=dataset, split='val', batch_size=1, shuffle=False
    )
    
    # Create model and trainer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MipNeRFModel(config).to(device)
    trainer = MipNeRFTrainer(config)
    
    print(f"Training Mip-NeRF with anti-aliasing on {len(train_loader)} images")
    print(f"Using {config.num_levels} multiscale levels")
    print(f"Output directory: {output_dir}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Train model
    trainer.train(
        train_loader=train_loader, val_loader=val_loader, output_dir=output_dir, device=device
    )
    
    return model, trainer


def main():
    """Main function to run examples."""
    parser = argparse.ArgumentParser(description="Mip-NeRF Example Usage")
    parser.add_argument(
        "--example",
        type=str,
        default="basic",
        choices=["basic",
        "anti_aliasing"],
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
        default="./outputs/mip_nerf",
        help="Output directory",
    )
    
    args = parser.parse_args()
    
    if args.example == "basic":
        model, config = basic_example()
        print("Basic example completed successfully!")
        
    elif args.example == "anti_aliasing":
        if not os.path.exists(args.data_path):
            print(f"Error: Data path {args.data_path} does not exist")
            return
        
        model, trainer = anti_aliasing_example(args.data_path, args.output_dir)
        print("Anti-aliasing training completed!")


if __name__ == "__main__":
    main() 