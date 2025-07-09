"""
Plenoxels Example Usage

This script demonstrates how to use Plenoxels for training and evaluation
with sparse 3D voxel grids.
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
    from src.nerfs.plenoxels import (
        PlenoxelConfig, PlenoxelModel, PlenoxelTrainer, PlenoxelTrainingConfig,
        PlenoxelDatasetConfig, create_plenoxel_dataloader, create_plenoxel_dataset
    )
except ImportError as e:
    print(f"Import error: {e}")
    print("Please make sure you're running from the project root directory")
    sys.exit(1)


def basic_example():
    """Basic Plenoxels example with default settings."""
    print("=== Basic Plenoxels Example ===")
    
    # Create configuration with correct parameters
    config = PlenoxelConfig(
        grid_resolution=(256, 256, 256), # Voxel grid resolution
        sh_degree=2,  # Spherical harmonics degree
        use_coarse_to_fine=True, # Coarse-to-fine training
        coarse_resolutions=[(64, 64, 64)],  # Coarse resolution for coarse-to-fine training
        sparsity_threshold=0.01,  # Sparsity threshold for voxel grid
        tv_lambda=1e-6,  # Total variation regularization
        l1_lambda=1e-8,  # L1 regularization
        step_size=0.01, # Step size for rendering
        sigma_thresh=1e-8, # Sigma threshold for rendering
        stop_thresh=1e-4,  # Optimization
        learning_rate=0.1, # Learning rate
        weight_decay=0.0,  # Weight decay
        near_plane=0.1, # Near plane
        far_plane=10.0, # Far plane
    )
    
    print(f"Configuration: {config}")
    
    # Create model
    model = PlenoxelModel(config)
    print(f"Model created with voxel grid resolution: {config.grid_resolution}")
    print(f"SH degree: {config.sh_degree}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Test forward pass with dummy data
    batch_size = 1024
    ray_origins = torch.randn(batch_size, 3)
    ray_directions = torch.randn(batch_size, 3)
    ray_directions = ray_directions / torch.norm(ray_directions, dim=-1, keepdim=True)
    
    with torch.no_grad():
        outputs = model(ray_origins, ray_directions, num_samples=64)
        print(f"Forward pass successful!")
        print(f"Output keys: {list(outputs.keys())}")
        if 'rgb' in outputs:
            print(f"RGB shape: {outputs['rgb'].shape}")
            print(f"RGB range: [{outputs['rgb'].min():.4f}, {outputs['rgb'].max():.4f}]")
        if 'depth' in outputs:
            print(f"Depth shape: {outputs['depth'].shape}")
            print(f"Depth range: [{outputs['depth'].min():.4f}, {outputs['depth'].max():.4f}]")
    
    return model, config


def advanced_example():
    """Advanced Plenoxels example with custom settings."""
    print("\n=== Advanced Plenoxels Example ===")
    
    # Create configuration with custom parameters
    config = PlenoxelConfig(
        # Higher resolution voxel grid
        grid_resolution=(512, 512, 512),
        sh_degree=3,  # More aggressive coarse-to-fine
        use_coarse_to_fine=True,
        coarse_resolutions=[(32, 32, 32)],
        sparsity_threshold=0.005,
        tv_lambda=1e-5,
        l1_lambda=1e-7,  # Finer rendering
        step_size=0.005,
        sigma_thresh=1e-9,
        stop_thresh=1e-5,  # Different optimization
        learning_rate=0.05,
        weight_decay=1e-6,  # Extended scene bounds
        near_plane=0.05,
        far_plane=20.0,
    )
    
    print(f"Advanced configuration: {config}")
    
    # Create model
    model = PlenoxelModel(config)
    print(f"Advanced model created with voxel grid resolution: {config.grid_resolution}")
    print(f"SH degree: {config.sh_degree}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    
    return model, config


def training_example(data_path=None):
    """Example of training Plenoxels."""
    print("\n=== Training Example ===")
    
    if data_path is None:
        print("No data path provided, using synthetic data")
        return
    
    # Create configurations
    model_config = PlenoxelConfig(
        learning_rate=0.1,
        use_coarse_to_fine=True,
    )
    
    trainer_config = PlenoxelTrainingConfig(
        num_epochs=100,
        learning_rate=0.1,
        experiment_name="plenoxel_training_example",
        checkpoint_dir="outputs/plenoxels",
    )
    
    dataset_config = PlenoxelDatasetConfig(
        data_dir=data_path,
        dataset_type='blender',
        num_rays_train=1024,
    )
    
    try:
        # Create trainer
        trainer = PlenoxelTrainer(
            model_config=model_config,
            trainer_config=trainer_config,
            dataset_config=dataset_config,
        )
        
        # Train for a few steps
        print("Starting training...")
        trainer.train()
        print("Training completed!")
        
    except Exception as e:
        print(f"Training failed: {e}")


def evaluation_example(model_path=None):
    """Example of evaluating Plenoxels."""
    print("\n=== Evaluation Example ===")
    
    if model_path is None:
        print("No model path provided, using basic model")
        model, config = basic_example()
    else:
        # Load trained model
        config = PlenoxelConfig()
        model = PlenoxelModel(config)
        model.load_state_dict(torch.load(model_path))
        print(f"Model loaded from {model_path}")
    
    # Evaluation metrics
    model.eval()
    with torch.no_grad():
        # Test on random data
        ray_origins = torch.randn(100, 3)
        ray_directions = torch.randn(100, 3)
        ray_directions = ray_directions / torch.norm(ray_directions, dim=-1, keepdim=True)
        
        outputs = model(ray_origins, ray_directions, num_samples=64)
        
        print(f"Evaluation completed!")
        if 'rgb' in outputs:
            print(f"Average RGB: {outputs['rgb'].mean(dim=0)}")
        if 'depth' in outputs:
            print(f"Average depth: {outputs['depth'].mean():.4f}")
        
        # Occupancy statistics
        if hasattr(model, 'get_occupancy_stats'):
            stats = model.get_occupancy_stats()
            print(f"Occupancy stats: {stats}")


def main():
    """Main function to run examples."""
    parser = argparse.ArgumentParser(description="Plenoxels Examples")
    parser.add_argument(
        "--example",
        type=str,
        default="basic",
        choices=["basic",
        "advanced",
        "training",
        "evaluation"],
        help="Type of example to run",
    )
    parser.add_argument("--data_path", type=str, help="Path to training data")
    parser.add_argument("--model_path", type=str, help="Path to trained model")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/plenoxels",
        help="Output directory",
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("🚀 Plenoxels Example Runner")
    print(f"Example type: {args.example}")
    print(f"Output directory: {args.output_dir}")
    
    if args.example == "basic":
        model, config = basic_example()
    elif args.example == "advanced":
        model, config = advanced_example()
    elif args.example == "training":
        training_example(args.data_path)
    elif args.example == "evaluation":
        evaluation_example(args.model_path)
    
    print("\n✅ Example completed successfully!")


if __name__ == "__main__":
    main() 