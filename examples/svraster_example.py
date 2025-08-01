from __future__ import annotations

"""
SVRaster Example Usage

This script demonstrates how to use SVRaster for sparse voxel rasterization
and efficient neural rendering.
"""

from typing import Any, Optional

import os
import sys
import torch
import argparse
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from src.nerfs.svraster import (
        SVRasterConfig, SVRasterModel, SVRasterTrainer, SVRasterTrainerConfig,
        SVRasterDataset, SVRasterDatasetConfig, create_svraster_dataloader, create_svraster_dataset
    )
except ImportError as e:
    print(f"Import error: {e}")
    print("Please make sure you're running from the project root directory")
    sys.exit(1)


def basic_svraster_example():
    """Basic SVRaster example."""
    print("=== Basic SVRaster Example ===")

    # Create configuration
    config = SVRasterConfig(
        max_octree_levels=12, base_resolution=64, scene_bounds=(
            -2.0,
            -2.0,
            -2.0,
            2.0,
            2.0,
            2.0,
        )
    )

    # Create model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SVRasterModel(config).to(device)

    print(f"Created SVRaster model on {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,d}")

    return model, config


def train_svraster_example(data_path: str, output_dir: str):
    """Training example for SVRaster."""
    print("=== SVRaster Training ===")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create model configuration
    model_config = SVRasterConfig(
        max_octree_levels=12,
        base_resolution=64,
        scene_bounds=(
            -2.0,
            -2.0,
            -2.0,
            2.0,
            2.0,
            2.0,
        )
    )
    
    # Create dataset configuration
    dataset_config = SVRasterDatasetConfig(
        data_dir=data_path,
        dataset_type='blender',
        image_height=800,
        image_width=800,
        num_rays_train=1024,
        num_rays_val=512,
    )
    
    # Create trainer configuration
    trainer_config = SVRasterTrainerConfig(
        num_epochs=100,
        batch_size=1,
        learning_rate=1e-3,
        weight_decay=1e-4,
        optimizer_type="adam",
        scheduler_type="cosine",
        rgb_loss_weight=1.0,
        subdivision_start_epoch=10,
        subdivision_interval=5,
        enable_pruning=True,
        checkpoint_dir=os.path.join(output_dir, "checkpoints"),
        log_dir=os.path.join(output_dir, "logs"),
        device=str(device),
    )

    # Create datasets
    train_dataset = create_svraster_dataset(dataset_config, split='train')
    val_dataset = create_svraster_dataset(dataset_config, split='val')

    # Create data loaders
    train_loader = create_svraster_dataloader(
        config=dataset_config,
        split='train',
        batch_size=1,
        shuffle=True,
        num_workers=4,
    )

    val_loader = create_svraster_dataloader(
        config=dataset_config,
        split='val',
        batch_size=1,
        shuffle=False,
        num_workers=4,
    )

    # Create model and trainer
    model = SVRasterModel(model_config).to(device)
    trainer = SVRasterTrainer(
        model_config=model_config,
        trainer_config=trainer_config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )

    print(f"Training dataset: {len(train_loader)} batches")
    print(f"Validation dataset: {len(val_loader)} batches")
    print(f"Output directory: {output_dir}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Start training
    trainer.train()

    return model, trainer


def large_scene_example(data_path: str, output_dir: str):
    """Example for large scene reconstruction."""
    print("=== Large Scene SVRaster Training ===")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create model configuration for large scenes
    model_config = SVRasterConfig(
        max_octree_levels=16,  # Deeper octree for large scenes
        base_resolution=128,   # Larger base resolution
        scene_bounds=(-50.0, -50.0, -10.0, 50.0, 50.0, 30.0),  # Larger bounds
        density_activation="exp",
        color_activation="sigmoid",
        sh_degree=3,  # Higher SH degree for better appearance
        subdivision_threshold=0.005,  # More aggressive subdivision
        pruning_threshold=0.0005,  # More aggressive pruning
        ray_samples_per_voxel=12,
        background_color=(0.5, 0.7, 1.0),  # Sky color
        use_view_dependent_color=True,
        use_opacity_regularization=True,
        opacity_reg_weight=0.01,
    )
    
    # Create dataset configuration
    dataset_config = SVRasterDatasetConfig(
        data_dir=data_path,
        dataset_type='large_scale',
        image_height=256,
        image_width=256,
        num_rays_train=2048,
        num_rays_val=1024,
    )
    
    # Create trainer configuration for large scenes
    trainer_config = SVRasterTrainerConfig(
        num_epochs=200,
        batch_size=2,
        learning_rate=2e-3,
        weight_decay=1e-5,
        optimizer_type="adamw",
        scheduler_type="cosine",
        rgb_loss_weight=1.0,
        subdivision_start_epoch=5,
        subdivision_interval=3,
        enable_pruning=True,
        pruning_start_epoch=10,
        pruning_interval=5,
        val_interval=10,
        log_interval=50,
        save_interval=500,
        checkpoint_dir=os.path.join(output_dir, "checkpoints"),
        log_dir=os.path.join(output_dir, "logs"),
        device=str(device),
    )

    # Create datasets
    train_dataset = create_svraster_dataset(dataset_config, split='train')
    val_dataset = create_svraster_dataset(dataset_config, split='val')

    # Create data loaders
    train_loader = create_svraster_dataloader(
        config=dataset_config,
        split='train',
        batch_size=2,
        shuffle=True,
        num_workers=4,
    )

    val_loader = create_svraster_dataloader(
        config=dataset_config,
        split='val',
        batch_size=1,
        shuffle=False,
        num_workers=4,
    )

    # Create model and trainer
    model = SVRasterModel(model_config).to(device)
    trainer = SVRasterTrainer(
        model_config=model_config,
        trainer_config=trainer_config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )

    print(f"Training large scene SVRaster on {len(train_loader)} samples")
    print(f"Scene bounds: {model_config.scene_bounds}")
    print(f"Max octree levels: {model_config.max_octree_levels}")
    print(f"Output directory: {output_dir}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Start training
    trainer.train()

    return model, trainer


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="SVRaster Example")
    parser.add_argument(
        "--example",
        type=str,
        default="basic",
        choices=["basic",
                 "train",
                 "large_scene",
                 "visualize"],
        help="Example to run",
    )
    parser.add_argument("--data_path", type=str, help="Path to dataset")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs/svraster",
        help="Output directory",
    )

    args = parser.parse_args()

    if args.example == "basic":
        model, config = basic_svraster_example()
        print("Basic SVRaster example completed!")

    elif args.example == "train":
        if not args.data_path:
            print("Error: --data_path required for training")
            return
        model, trainer = train_svraster_example(
            args.data_path, args.output_dir)
        print("SVRaster training completed!")

    elif args.example == "large_scene":
        if not args.data_path:
            print("Error: --data_path required for large scene training")
            return
        model, trainer = large_scene_example(args.data_path, args.output_dir)
        print("Large scene SVRaster training completed!")
    elif args.example == "visualize":
        # Only visualize model structure
        model, config = basic_svraster_example()
        model.visualize_structure(args.output_dir)
        print("Model visualization completed!")


if __name__ == "__main__":
    main()