"""
Inf-NeRF Example Usage

This script demonstrates how to use Inf-NeRF for infinite neural
radiance fields with unbounded scene representation.
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
    from src.nerfs.inf_nerf import (
        InfNeRFConfig, InfNeRFModel, InfNeRFTrainer, create_inf_nerf_dataloader, create_inf_nerf_dataset
    )
except ImportError as e:
    print(f"Import error: {e}")
    print("Please make sure you're running from the project root directory")
    sys.exit(1)


def basic_example():
    """Basic Inf-NeRF example with default configuration."""
    print("=== Basic Inf-NeRF Example ===")
    
    config = InfNeRFConfig(
        # Network architecture
        netdepth=8, netwidth=256, # Infinite scene features
        use_unbounded_scene=True, scene_contraction="sphere", background_model="infinite", # Training settings
        learning_rate=5e-4, weight_decay=1e-6, # Unbounded scene bounds
        inner_radius=1.0, outer_radius=1000.0
    )
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = InfNeRFModel(config).to(device)
    
    print(f"Created Inf-NeRF model on {device}")
    print(f"Scene contraction: {config.scene_contraction}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):, }")
    
    return model, config


def unbounded_scene_example(data_path: str, output_dir: str):
    """Example for unbounded scene reconstruction."""
    print("=== Inf-NeRF Unbounded Scene Training ===")
    
    config = InfNeRFConfig(
        # Enhanced network architecture
        netdepth=8, netwidth=256, netdepth_fine=8, netwidth_fine=256, # Advanced infinite scene features
        use_unbounded_scene=True, scene_contraction="sphere", background_model="learned_infinite", use_hierarchical_sampling=True, # Training settings
        learning_rate=1e-3, weight_decay=1e-6, scheduler_type="cosine", # Unbounded scene bounds
        inner_radius=2.0, outer_radius=10000.0, # Loss weights
        rgb_loss_weight=1.0, background_loss_weight=0.1, contraction_loss_weight=0.01
    )
    
    # Create dataset
    dataset = create_inf_nerf_dataset(
        data_path=data_path, dataset_type="unbounded", config=config
    )
    
    train_loader = create_inf_nerf_dataloader(
        dataset=dataset, split='train', batch_size=1, shuffle=True
    )
    
    val_loader = create_inf_nerf_dataloader(
        dataset=dataset, split='val', batch_size=1, shuffle=False
    )
    
    # Create model and trainer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = InfNeRFModel(config).to(device)
    trainer = InfNeRFTrainer(config)
    
    print(f"Training Inf-NeRF on unbounded scene")
    print(f"Inner radius: {config.inner_radius}, Outer radius: {config.outer_radius}")
    print(f"Output directory: {output_dir}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Train model
    trainer.train(
        train_loader=train_loader, val_loader=val_loader, output_dir=output_dir, device=device
    )
    
    return model, trainer


def main():
    """Main function to run examples."""
    parser = argparse.ArgumentParser(description="Inf-NeRF Example Usage")
    parser.add_argument(
        "--example",
        type=str,
        default="basic",
        choices=["basic",
        "unbounded_scene"],
        help="Example to run",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="./data/inf_nerf/outdoor",
        help="Path to dataset",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs/inf_nerf",
        help="Output directory",
    )
    
    args = parser.parse_args()
    
    if args.example == "basic":
        model, config = basic_example()
        print("Basic example completed successfully!")
        
    elif args.example == "unbounded_scene":
        if not os.path.exists(args.data_path):
            print(f"Error: Data path {args.data_path} does not exist")
            return
        
        model, trainer = unbounded_scene_example(args.data_path, args.output_dir)
        print("Unbounded scene training completed!")


if __name__ == "__main__":
    main() 