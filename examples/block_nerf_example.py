"""
Block-NeRF Example Usage

This script demonstrates how to use Block-NeRF for large-scale
city scene reconstruction with block decomposition.
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
    from src.nerfs.block_nerf import (
        BlockNeRFConfig, BlockNeRFModel, BlockNeRFTrainer, create_block_nerf_dataloader, create_block_nerf_dataset
    )
except ImportError as e:
    print(f"Import error: {e}")
    print("Please make sure you're running from the project root directory")
    sys.exit(1)


def basic_example():
    """Basic Block-NeRF example with default configuration."""
    print("=== Basic Block-NeRF Example ===")
    
    config = BlockNeRFConfig(
        # Block decomposition
        num_blocks=(2, 2, 1), # 2x2x1 grid
        block_size=(50.0, 50.0, 20.0), overlap_ratio=0.1, # Network architecture
        netdepth=8, netwidth=256, # Training settings
        learning_rate=5e-4, weight_decay=1e-6, # Scene bounds
        scene_bounds=(-50.0, -50.0, -5.0, 50.0, 50.0, 15.0), near_plane=0.1, far_plane=1000.0
    )
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = BlockNeRFModel(config).to(device)
    
    print(f"Created Block-NeRF model on {device}")
    print(f"Block grid: {config.num_blocks}")
    print(f"Total blocks: {config.num_blocks[0] * config.num_blocks[1] * config.num_blocks[2]}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):, d}")
    
    return model, config


def city_scale_example(data_path: str, output_dir: str):
    """Example for city-scale reconstruction."""
    print("=== City-Scale Block-NeRF Training ===")
    
    config = BlockNeRFConfig(
        # Large block decomposition for city scale
        num_blocks=(4, 4, 2), # 4x4x2 grid = 32 blocks
        block_size=(100.0, 100.0, 50.0), overlap_ratio=0.15, # Network architecture
        netdepth=8, netwidth=256, netdepth_fine=8, netwidth_fine=256, # Advanced features
        use_appearance_embedding=True, appearance_embed_dim=32, use_transient_embedding=True, # Training settings
        learning_rate=1e-3, weight_decay=1e-6, scheduler_type="cosine", # Large scene bounds
        scene_bounds=(
            -200.0,
            -200.0,
            -10.0,
            200.0,
            200.0,
            100.0,
        ),
        rgb_loss_weight=1.0, depth_loss_weight=0.1, block_consistency_weight=0.05, # Memory optimization
        use_gradient_checkpointing=True, max_blocks_per_batch=4
    )
    
    # Create dataset
    dataset = create_block_nerf_dataset(
        data_path=data_path, dataset_type="city_scale", config=config
    )
    
    train_loader = create_block_nerf_dataloader(
        dataset=dataset, split='train', batch_size=1, shuffle=True
    )
    
    val_loader = create_block_nerf_dataloader(
        dataset=dataset, split='val', batch_size=1, shuffle=False
    )
    
    # Create model and trainer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = BlockNeRFModel(config).to(device)
    trainer = BlockNeRFTrainer(config)
    
    print(f"Training Block-NeRF on city-scale data")
    print(f"Block decomposition: {config.num_blocks}")
    print(f"Scene bounds: {config.scene_bounds}")
    print(f"Output directory: {output_dir}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Train model
    trainer.train(
        train_loader=train_loader, val_loader=val_loader, output_dir=output_dir, device=device
    )
    
    return model, trainer


def main():
    """Main function to run examples."""
    parser = argparse.ArgumentParser(description="Block-NeRF Example Usage")
    parser.add_argument(
        "--example",
        type=str,
        default="basic",
        choices=["basic",
        "city_scale"],
        help="Example to run",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="./data/block_nerf/city",
        help="Path to dataset",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs/block_nerf",
        help="Output directory",
    )
    
    args = parser.parse_args()
    
    if args.example == "basic":
        model, config = basic_example()
        print("Basic example completed successfully!")
        
    elif args.example == "city_scale":
        if not os.path.exists(args.data_path):
            print(f"Error: Data path {args.data_path} does not exist")
            return
        
        model, trainer = city_scale_example(args.data_path, args.output_dir)
        print("City-scale training completed!")


if __name__ == "__main__":
    main() 