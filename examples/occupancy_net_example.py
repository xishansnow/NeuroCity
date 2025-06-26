"""
Occupancy Networks Example Usage

This script demonstrates how to use Occupancy Networks for 3D shape
representation and reconstruction.
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
    from src.nerfs.occupancy_net import (
        OccupancyNetConfig, OccupancyNetModel, OccupancyNetTrainer, create_occupancy_dataloader, create_occupancy_dataset
    )
except ImportError as e:
    print(f"Import error: {e}")
    print("Please make sure you're running from the project root directory")
    sys.exit(1)


def basic_example():
    """Basic Occupancy Networks example with default configuration."""
    print("=== Basic Occupancy Networks Example ===")
    
    config = OccupancyNetConfig(
        # Network architecture
        netdepth=5, netwidth=128, # Occupancy-specific settings
        threshold=0.5, use_batch_norm=True, # Training settings
        learning_rate=1e-4, weight_decay=1e-6, # Mesh extraction
        resolution=128, use_octree=True
    )
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = OccupancyNetModel(config).to(device)
    
    print(f"Created Occupancy Networks model on {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):, }")
    
    return model, config


def shape_reconstruction_example(data_path: str, output_dir: str):
    """Example for 3D shape reconstruction."""
    print("=== Occupancy Networks Shape Reconstruction ===")
    
    config = OccupancyNetConfig(
        # Enhanced network architecture
        netdepth=8, netwidth=256, # Occupancy-specific settings
        threshold=0.5, use_batch_norm=True, use_residual_blocks=True, # Training settings
        learning_rate=5e-4, weight_decay=1e-6, scheduler_type="step", # Loss settings
        occupancy_loss_weight=1.0, boundary_loss_weight=0.1, # Mesh extraction
        resolution=256, use_octree=True, octree_depth=6
    )
    
    # Create dataset
    dataset = create_occupancy_dataset(
        data_path=data_path, dataset_type="shapenet", config=config
    )
    
    train_loader = create_occupancy_dataloader(
        dataset=dataset, split='train', batch_size=1, shuffle=True
    )
    
    val_loader = create_occupancy_dataloader(
        dataset=dataset, split='val', batch_size=1, shuffle=False
    )
    
    # Create model and trainer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = OccupancyNetModel(config).to(device)
    trainer = OccupancyNetTrainer(config)
    
    print(f"Training Occupancy Networks for shape reconstruction")
    print(f"Mesh extraction resolution: {config.resolution}")
    print(f"Output directory: {output_dir}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Train model
    trainer.train(
        train_loader=train_loader, val_loader=val_loader, output_dir=output_dir, device=device
    )
    
    return model, trainer


def main():
    """Main function to run examples."""
    parser = argparse.ArgumentParser(description="Occupancy Networks Example Usage")
    parser.add_argument(
        "--example",
        type=str,
        default="basic",
        choices=["basic",
        "shape_reconstruction"],
        help="Example to run",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="./data/occupancy_net/shapes",
        help="Path to dataset",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs/occupancy_net",
        help="Output directory",
    )
    
    args = parser.parse_args()
    
    if args.example == "basic":
        model, config = basic_example()
        print("Basic example completed successfully!")
        
    elif args.example == "shape_reconstruction":
        if not os.path.exists(args.data_path):
            print(f"Error: Data path {args.data_path} does not exist")
            return
        
        model, trainer = shape_reconstruction_example(args.data_path, args.output_dir)
        print("Shape reconstruction completed!")


if __name__ == "__main__":
    main() 