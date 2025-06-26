"""
Pyramid-NeRF Example Usage

This script demonstrates how to use Pyramid-NeRF for hierarchical
neural radiance fields with pyramid sampling.
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
    from src.nerfs.pyramid_nerf import (
        PyramidNeRFConfig, PyramidNeRFModel, PyramidNeRFTrainer, create_pyramid_nerf_dataloader, create_pyramid_nerf_dataset
    )
except ImportError as e:
    print(f"Import error: {e}")
    print("Please make sure you're running from the project root directory")
    sys.exit(1)


def basic_example():
    """Basic Pyramid-NeRF example with default configuration."""
    print("=== Basic Pyramid-NeRF Example ===")
    
    config = PyramidNeRFConfig(
        # Pyramid structure
        num_pyramid_levels=4, pyramid_scale_factor=2, # Network architecture
        netdepth=8, netwidth=256, # Hierarchical sampling
        hierarchical_sampling=True, # Training settings
        learning_rate=5e-4, weight_decay=1e-6, # Scene bounds
        near_plane=2.0, far_plane=6.0
    )
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PyramidNeRFModel(config).to(device)
    
    print(f"Created Pyramid-NeRF model on {device}")
    print(f"Pyramid levels: {config.num_pyramid_levels}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):, }")
    
    return model, config


def hierarchical_training_example(data_path: str, output_dir: str):
    """Example with hierarchical pyramid training."""
    print("=== Hierarchical Pyramid-NeRF Training ===")
    
    config = PyramidNeRFConfig(
        # Enhanced pyramid structure
        num_pyramid_levels=6, pyramid_scale_factor=2, # Network architecture
        netdepth=8, netwidth=256, netdepth_fine=8, netwidth_fine=256, # Advanced hierarchical sampling
        hierarchical_sampling=True, adaptive_pyramid_weights=True, # Training settings
        learning_rate=1e-3, weight_decay=1e-6, scheduler_type="step", # Scene bounds
        near_plane=0.1, far_plane=100.0, # Loss weights
        rgb_loss_weight=1.0, pyramid_consistency_weight=0.1, level_loss_weights=[1.0, 0.8, 0.6, 0.4, 0.2, 0.1]
    )
    
    # Create dataset
    dataset = create_pyramid_nerf_dataset(
        data_path=data_path, dataset_type="blender", config=config
    )
    
    train_loader = create_pyramid_nerf_dataloader(
        dataset=dataset, split='train', batch_size=1, shuffle=True
    )
    
    val_loader = create_pyramid_nerf_dataloader(
        dataset=dataset, split='val', batch_size=1, shuffle=False
    )
    
    # Create model and trainer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PyramidNeRFModel(config).to(device)
    trainer = PyramidNeRFTrainer(config)
    
    print(f"Training Pyramid-NeRF with {config.num_pyramid_levels} levels")
    print(f"Output directory: {output_dir}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Train model
    trainer.train(
        train_loader=train_loader, val_loader=val_loader, output_dir=output_dir, device=device
    )
    
    return model, trainer


def main():
    """Main function to run examples."""
    parser = argparse.ArgumentParser(description="Pyramid-NeRF Example Usage")
    parser.add_argument(
        "--example",
        type=str,
        default="basic",
        choices=["basic",
        "hierarchical"],
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
        default="./outputs/pyramid_nerf",
        help="Output directory",
    )
    
    args = parser.parse_args()
    
    if args.example == "basic":
        model, config = basic_example()
        print("Basic example completed successfully!")
        
    elif args.example == "hierarchical":
        if not os.path.exists(args.data_path):
            print(f"Error: Data path {args.data_path} does not exist")
            return
        
        model, trainer = hierarchical_training_example(args.data_path, args.output_dir)
        print("Hierarchical training completed!")


if __name__ == "__main__":
    main() 