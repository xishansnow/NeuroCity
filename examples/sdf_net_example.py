"""
SDF-Net Example Usage

This script demonstrates how to use SDF-Net for signed distance field
learning and surface reconstruction.
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
    from src.nerfs.sdf_net import (
        SDFNetConfig, SDFNetModel, SDFNetTrainer, create_sdf_dataloader, create_sdf_dataset
    )
except ImportError as e:
    print(f"Import error: {e}")
    print("Please make sure you're running from the project root directory")
    sys.exit(1)


def basic_example():
    """Basic SDF-Net example with default configuration."""
    print("=== Basic SDF-Net Example ===")
    
    config = SDFNetConfig(
        # Network architecture
        netdepth=8, netwidth=256, skip_layers=[4], # SDF-specific settings
        use_geometric_initialization=True, beta_init=0.1, # Training settings
        learning_rate=1e-4, weight_decay=1e-6, # Surface extraction
        marching_cubes_resolution=256
    )
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SDFNetModel(config).to(device)
    
    print(f"Created SDF-Net model on {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):, }")
    
    return model, config


def surface_reconstruction_example(data_path: str, output_dir: str):
    """Example for surface reconstruction from point clouds."""
    print("=== SDF-Net Surface Reconstruction ===")
    
    config = SDFNetConfig(
        # Enhanced network architecture
        netdepth=8, netwidth=512, skip_layers=[4], # SDF-specific settings
        use_geometric_initialization=True, beta_init=0.1, use_lipschitz_regularization=True, # Training settings
        learning_rate=5e-4, weight_decay=1e-6, scheduler_type="step", # Loss weights
        sdf_loss_weight=1.0, eikonal_loss_weight=0.1, inter_loss_weight=0.01, # Surface extraction
        marching_cubes_resolution=512, surface_threshold=0.0
    )
    
    # Create dataset
    dataset = create_sdf_dataset(
        data_path=data_path, dataset_type="point_cloud", config=config
    )
    
    train_loader = create_sdf_dataloader(
        dataset=dataset, split='train', batch_size=1, shuffle=True
    )
    
    val_loader = create_sdf_dataloader(
        dataset=dataset, split='val', batch_size=1, shuffle=False
    )
    
    # Create model and trainer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SDFNetModel(config).to(device)
    trainer = SDFNetTrainer(config)
    
    print(f"Training SDF-Net for surface reconstruction")
    print(f"Marching cubes resolution: {config.marching_cubes_resolution}")
    print(f"Output directory: {output_dir}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Train model
    trainer.train(
        train_loader=train_loader, val_loader=val_loader, output_dir=output_dir, device=device
    )
    
    return model, trainer


def main():
    """Main function to run examples."""
    parser = argparse.ArgumentParser(description="SDF-Net Example Usage")
    parser.add_argument(
        "--example",
        type=str,
        default="basic",
        choices=["basic",
        "surface_reconstruction"],
        help="Example to run",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="./data/sdf_net/bunny",
        help="Path to dataset",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs/sdf_net",
        help="Output directory",
    )
    
    args = parser.parse_args()
    
    if args.example == "basic":
        model, config = basic_example()
        print("Basic example completed successfully!")
        
    elif args.example == "surface_reconstruction":
        if not os.path.exists(args.data_path):
            print(f"Error: Data path {args.data_path} does not exist")
            return
        
        model, trainer = surface_reconstruction_example(args.data_path, args.output_dir)
        print("Surface reconstruction completed!")


if __name__ == "__main__":
    main() 