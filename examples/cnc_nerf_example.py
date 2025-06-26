"""
CNC-NeRF Example Usage

This script demonstrates how to use CNC-NeRF for controllable neural
radiance fields with conditional rendering.
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
    from src.nerfs.cnc_nerf import (
        CNCNeRFConfig, CNCNeRFModel, CNCNeRFTrainer, create_cnc_nerf_dataloader, create_cnc_nerf_dataset
    )
except ImportError as e:
    print(f"Import error: {e}")
    print("Please make sure you're running from the project root directory")
    sys.exit(1)


def basic_example():
    """Basic CNC-NeRF example with default configuration."""
    print("=== Basic CNC-NeRF Example ===")
    
    config = CNCNeRFConfig(
        # Network architecture
        netdepth=8, netwidth=256, # Controllable features
        num_control_codes=16, control_code_dim=32, use_conditional_rendering=True, # Training settings
        learning_rate=5e-4, weight_decay=1e-6, # Scene bounds
        near_plane=2.0, far_plane=6.0
    )
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CNCNeRFModel(config).to(device)
    
    print(f"Created CNC-NeRF model on {device}")
    print(f"Control codes: {config.num_control_codes}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):, }")
    
    return model, config


def controllable_rendering_example(data_path: str, output_dir: str):
    """Example for controllable rendering with conditions."""
    print("=== CNC-NeRF Controllable Rendering ===")
    
    config = CNCNeRFConfig(
        # Enhanced network architecture
        netdepth=8, netwidth=256, netdepth_fine=8, netwidth_fine=256, # Advanced controllable features
        num_control_codes=32, control_code_dim=64, use_conditional_rendering=True, use_style_transfer=True, # Training settings
        learning_rate=1e-3, weight_decay=1e-6, scheduler_type="cosine", # Scene bounds
        near_plane=0.1, far_plane=100.0, # Loss weights
        rgb_loss_weight=1.0, control_consistency_weight=0.1, style_loss_weight=0.05
    )
    
    # Create dataset
    dataset = create_cnc_nerf_dataset(
        data_path=data_path, dataset_type="controllable", config=config
    )
    
    train_loader = create_cnc_nerf_dataloader(
        dataset=dataset, split='train', batch_size=1, shuffle=True
    )
    
    val_loader = create_cnc_nerf_dataloader(
        dataset=dataset, split='val', batch_size=1, shuffle=False
    )
    
    # Create model and trainer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CNCNeRFModel(config).to(device)
    trainer = CNCNeRFTrainer(config)
    
    print(f"Training CNC-NeRF with controllable rendering")
    print(f"Control codes: {config.num_control_codes}")
    print(f"Output directory: {output_dir}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Train model
    trainer.train(
        train_loader=train_loader, val_loader=val_loader, output_dir=output_dir, device=device
    )
    
    return model, trainer


def main():
    """Main function to run examples."""
    parser = argparse.ArgumentParser(description="CNC-NeRF Example Usage")
    parser.add_argument(
        "--example",
        type=str,
        default="basic",
        choices=["basic",
        "controllable"],
        help="Example to run",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="./data/cnc_nerf/scene",
        help="Path to dataset",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs/cnc_nerf",
        help="Output directory",
    )
    
    args = parser.parse_args()
    
    if args.example == "basic":
        model, config = basic_example()
        print("Basic example completed successfully!")
        
    elif args.example == "controllable":
        if not os.path.exists(args.data_path):
            print(f"Error: Data path {args.data_path} does not exist")
            return
        
        model, trainer = controllable_rendering_example(args.data_path, args.output_dir)
        print("Controllable rendering completed!")


if __name__ == "__main__":
    main() 