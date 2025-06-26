"""
DNMP-NeRF Example Usage

This script demonstrates how to use DNMP-NeRF for dynamic neural
radiance fields with motion and deformation modeling.
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
    from src.nerfs.dnmp_nerf import (
        DNMPNeRFConfig, DNMPNeRFModel, DNMPNeRFTrainer, create_dnmp_nerf_dataloader, create_dnmp_nerf_dataset
    )
except ImportError as e:
    print(f"Import error: {e}")
    print("Please make sure you're running from the project root directory")
    sys.exit(1)


def basic_example():
    """Basic DNMP-NeRF example with default configuration."""
    print("=== Basic DNMP-NeRF Example ===")
    
    config = DNMPNeRFConfig(
        # Network architecture
        netdepth=8, netwidth=256, # Dynamic features
        num_frames=100, use_temporal_encoding=True, temporal_encoding_freq=6, # Motion modeling
        use_motion_field=True, motion_field_dim=64, # Training settings
        learning_rate=5e-4, weight_decay=1e-6, # Scene bounds
        near_plane=2.0, far_plane=6.0
    )
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DNMPNeRFModel(config).to(device)
    
    print(f"Created DNMP-NeRF model on {device}")
    print(f"Number of frames: {config.num_frames}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):, }")
    
    return model, config


def dynamic_scene_example(data_path: str, output_dir: str):
    """Example for dynamic scene modeling."""
    print("=== DNMP-NeRF Dynamic Scene Training ===")
    
    config = DNMPNeRFConfig(
        # Enhanced network architecture
        netdepth=8, netwidth=256, netdepth_fine=8, netwidth_fine=256, # Advanced dynamic features
        num_frames=200, use_temporal_encoding=True, temporal_encoding_freq=10, # Advanced motion modeling
        use_motion_field=True, motion_field_dim=128, use_deformation_field=True, deformation_field_dim=64, # Training settings
        learning_rate=1e-3, weight_decay=1e-6, scheduler_type="step", # Scene bounds
        near_plane=0.1, far_plane=100.0, # Loss weights
        rgb_loss_weight=1.0, motion_consistency_weight=0.1, temporal_smoothness_weight=0.05, deformation_regularization_weight=0.01
    )
    
    # Create dataset
    dataset = create_dnmp_nerf_dataset(
        data_path=data_path, dataset_type="dynamic_scene", config=config
    )
    
    train_loader = create_dnmp_nerf_dataloader(
        dataset=dataset, split='train', batch_size=1, shuffle=True
    )
    
    val_loader = create_dnmp_nerf_dataloader(
        dataset=dataset, split='val', batch_size=1, shuffle=False
    )
    
    # Create model and trainer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DNMPNeRFModel(config).to(device)
    trainer = DNMPNeRFTrainer(config)
    
    print(f"Training DNMP-NeRF on dynamic scene")
    print(f"Temporal frames: {config.num_frames}")
    print(f"Output directory: {output_dir}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Train model
    trainer.train(
        train_loader=train_loader, val_loader=val_loader, output_dir=output_dir, device=device
    )
    
    return model, trainer


def main():
    """Main function to run examples."""
    parser = argparse.ArgumentParser(description="DNMP-NeRF Example Usage")
    parser.add_argument(
        "--example",
        type=str,
        default="basic",
        choices=["basic",
        "dynamic_scene"],
        help="Example to run",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="./data/dnmp_nerf/dynamic",
        help="Path to dataset",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs/dnmp_nerf",
        help="Output directory",
    )
    
    args = parser.parse_args()
    
    if args.example == "basic":
        model, config = basic_example()
        print("Basic example completed successfully!")
        
    elif args.example == "dynamic_scene":
        if not os.path.exists(args.data_path):
            print(f"Error: Data path {args.data_path} does not exist")
            return
        
        model, trainer = dynamic_scene_example(args.data_path, args.output_dir)
        print("Dynamic scene training completed!")


if __name__ == "__main__":
    main() 