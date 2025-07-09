"""
Basic Plenoxels Usage Examples

This file demonstrates the essential usage patterns of the Plenoxels package
for training and inference.
"""

import torch
from pathlib import Path

# Import the Plenoxels components
from nerfs.plenoxels import (
    PlenoxelTrainer,
    PlenoxelRenderer,
    PlenoxelTrainingConfig,
    PlenoxelInferenceConfig,
    ExampleConfigs,
    quick_train,
    quick_render,
)

from nerfs.plenoxels.dataset import PlenoxelDataset, PlenoxelDatasetConfig


def basic_training_example():
    """Basic training example."""
    print("=== Basic Training Example ===")
    
    # 1. Create configuration
    config = PlenoxelTrainingConfig(
        grid_resolution=(256, 256, 256),
        num_epochs=10000,
        learning_rate=0.1,
        use_coarse_to_fine=True,
        experiment_name="basic_plenoxel",
    )
    
    # 2. Setup dataset
    dataset_config = PlenoxelDatasetConfig(
        data_root="data/nerf_synthetic/lego",
        white_background=True,
    )
    
    train_dataset = PlenoxelDataset(dataset_config, split="train")
    val_dataset = PlenoxelDataset(dataset_config, split="val")
    
    # 3. Train
    trainer = PlenoxelTrainer(config, train_dataset, val_dataset)
    renderer = trainer.train()
    
    print(f"Training completed! Best PSNR: {trainer.best_psnr:.2f}")
    return renderer


def basic_inference_example():
    """Basic inference example."""
    print("=== Basic Inference Example ===")
    
    # Load trained model
    renderer = PlenoxelRenderer.from_checkpoint("checkpoints/basic_plenoxel/best.pth")
    
    # Setup camera parameters
    camera_matrix = torch.tensor([
        [512.0, 0.0, 256.0],
        [0.0, 512.0, 256.0],
        [0.0, 0.0, 1.0]
    ])
    
    camera_pose = torch.eye(4)
    camera_pose[2, 3] = -3.0  # Move camera back
    
    # Render image
    outputs = renderer.render_image(
        height=512,
        width=512,
        camera_matrix=camera_matrix,
        camera_pose=camera_pose,
    )
    
    print(f"Rendered image shape: {outputs['rgb'].shape}")
    return outputs


def quick_usage_example():
    """Example using the quick interface."""
    print("=== Quick Usage Example ===")
    
    # Setup dataset
    dataset_config = PlenoxelDatasetConfig(data_root="data/nerf_synthetic/lego")
    train_dataset = PlenoxelDataset(dataset_config, split="train")
    val_dataset = PlenoxelDataset(dataset_config, split="val")
    
    # Quick training with defaults
    renderer = quick_train(
        train_dataset,
        val_dataset,
        num_epochs=1000,
        grid_resolution=(128, 128, 128),
    )
    
    # Quick rendering
    camera_matrix = torch.tensor([[256.0, 0.0, 128.0], [0.0, 256.0, 128.0], [0.0, 0.0, 1.0]])
    camera_pose = torch.eye(4)
    camera_pose[2, 3] = -2.0
    
    outputs = quick_render(
        "checkpoints/quick_train/final.pth",
        height=256,
        width=256,
        camera_matrix=camera_matrix,
        camera_pose=camera_pose,
    )
    
    print(f"Quick render completed: {outputs['rgb'].shape}")


if __name__ == "__main__":
    print("Plenoxels Basic Usage Examples")
    print("=" * 40)
    
    # Show how to use preset configs
    print("\n=== Available Preset Configurations ===")
    fast_config = ExampleConfigs.fast_training()
    print(f"Fast training: {fast_config.grid_resolution}, {fast_config.num_epochs} epochs")
    
    hq_config = ExampleConfigs.high_quality_training()
    print(f"High quality: {hq_config.grid_resolution}, {hq_config.num_epochs} epochs")
    
    print("\nNote: Run individual functions with real datasets for full examples.")
    print("See README.md for complete setup instructions.")
