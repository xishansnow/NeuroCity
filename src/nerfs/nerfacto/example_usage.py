"""
Nerfacto Example Usage

This script demonstrates how to use the Nerfacto package for training
and evaluating neural radiance fields on real-world data.
"""

import torch
import argparse
import os
from pathlib import Path

from .core import NerfactoModel, NerfactoConfig
from .dataset import NerfactoDataset, NerfactoDatasetConfig
from .trainer import NerfactoTrainer, NerfactoTrainerConfig, create_nerfacto_trainer


def main():
    """Main function for Nerfacto example usage."""
    parser = argparse.ArgumentParser(description="Nerfacto Example Usage")
    
    # Data arguments
    parser.add_argument("--data_dir", type=str, required=True,
                       help="Path to the dataset directory")
    parser.add_argument("--data_format", type=str, default="colmap",
                       choices=["colmap", "blender", "instant_ngp"],
                       help="Data format")
    
    # Training arguments
    parser.add_argument("--output_dir", type=str, default="outputs",
                       help="Output directory for results")
    parser.add_argument("--experiment_name", type=str, default="nerfacto_experiment",
                       help="Experiment name")
    parser.add_argument("--max_epochs", type=int, default=30000,
                       help="Maximum number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=5e-4,
                       help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=1,
                       help="Batch size")
    
    # Model arguments
    parser.add_argument("--num_levels", type=int, default=16,
                       help="Number of hash levels")
    parser.add_argument("--base_resolution", type=int, default=16,
                       help="Base hash grid resolution")
    parser.add_argument("--max_resolution", type=int, default=2048,
                       help="Maximum hash grid resolution")
    
    # Hardware arguments
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use (cuda/cpu)")
    parser.add_argument("--num_workers", type=int, default=4,
                       help="Number of data loading workers")
    
    # Evaluation arguments
    parser.add_argument("--eval_only", action="store_true",
                       help="Only run evaluation")
    parser.add_argument("--checkpoint_path", type=str, default=None,
                       help="Path to checkpoint for evaluation")
    
    args = parser.parse_args()
    
    # Check if data directory exists
    if not os.path.exists(args.data_dir):
        raise FileNotFoundError(f"Data directory not found: {args.data_dir}")
    
    print("=" * 60)
    print("Nerfacto - Neural Radiance Fields Training")
    print("=" * 60)
    print(f"Data directory: {args.data_dir}")
    print(f"Data format: {args.data_format}")
    print(f"Output directory: {args.output_dir}")
    print(f"Device: {args.device}")
    print("=" * 60)
    
    # Create configurations
    model_config = NerfactoConfig(
        num_levels=args.num_levels,
        base_resolution=args.base_resolution,
        max_resolution=args.max_resolution
    )
    
    dataset_config = NerfactoDatasetConfig(
        data_dir=args.data_dir,
        data_format=args.data_format,
        auto_scale=True
    )
    
    trainer_config = NerfactoTrainerConfig(
        max_epochs=args.max_epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        experiment_name=args.experiment_name,
        use_mixed_precision=True,
        use_wandb=False  # Set to True if you want to use Weights & Biases
    )
    
    # Create trainer
    trainer = NerfactoTrainer(
        config=trainer_config,
        model_config=model_config,
        dataset_config=dataset_config,
        device=args.device
    )
    
    if args.eval_only:
        # Evaluation mode
        if args.checkpoint_path is None:
            # Find latest checkpoint
            checkpoint_dir = os.path.join(args.output_dir, args.experiment_name, "checkpoints")
            if os.path.exists(checkpoint_dir):
                checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
                if checkpoints:
                    latest_checkpoint = max(checkpoints, key=lambda x: os.path.getctime(
                        os.path.join(checkpoint_dir, x)))
                    args.checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
        
        if args.checkpoint_path and os.path.exists(args.checkpoint_path):
            print(f"Loading checkpoint: {args.checkpoint_path}")
            trainer.load_checkpoint(args.checkpoint_path)
            
            # Run evaluation
            print("Running evaluation...")
            val_metrics = trainer.validate()
            
            print("Evaluation Results:")
            for key, value in val_metrics.items():
                print(f"  {key}: {value:.6f}")
        else:
            print("No checkpoint found for evaluation!")
    else:
        # Training mode
        print("Starting training...")
        trainer.train()
        print("Training completed!")


def demo_basic_usage():
    """Demonstrate basic usage of Nerfacto components."""
    print("=" * 50)
    print("Nerfacto Basic Usage Demo")
    print("=" * 50)
    
    # 1. Create model configuration
    print("1. Creating model configuration...")
    config = NerfactoConfig(
        num_levels=16,
        base_resolution=16,
        max_resolution=2048,
        features_per_level=2,
        hidden_dim=64,
        num_layers=2
    )
    print(f"   Model config: {config}")
    
    # 2. Create model
    print("2. Creating Nerfacto model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = NerfactoModel(config).to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"   Model created with {num_params:,} parameters")
    
    # 3. Test forward pass
    print("3. Testing forward pass...")
    batch_size = 1024
    ray_origins = torch.randn(batch_size, 3, device=device)
    ray_directions = torch.randn(batch_size, 3, device=device)
    ray_directions = ray_directions / torch.norm(ray_directions, dim=-1, keepdim=True)
    
    with torch.no_grad():
        outputs = model(ray_origins, ray_directions)
    
    print(f"   Input rays: {batch_size}")
    print("   Outputs:")
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor):
            print(f"     {key}: {value.shape}")
    
    print("Demo completed successfully!")


def create_sample_dataset():
    """Create a sample synthetic dataset for testing."""
    print("Creating sample synthetic dataset...")
    
    # This would create a simple synthetic dataset
    # for testing purposes - simplified implementation
    
    output_dir = "sample_data"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create transforms.json file
    transforms = {
        "camera_angle_x": 0.6911112070083618,
        "frames": []
    }
    
    # Generate some sample camera poses
    import math
    num_frames = 20
    radius = 4.0
    
    for i in range(num_frames):
        angle = 2 * math.pi * i / num_frames
        
        # Camera position
        x = radius * math.cos(angle)
        y = 0.0
        z = radius * math.sin(angle)
        
        # Look at origin
        transform_matrix = [
            [math.cos(angle + math.pi/2), 0, math.sin(angle + math.pi/2), x],
            [0, 1, 0, y],
            [-math.sin(angle + math.pi/2), 0, math.cos(angle + math.pi/2), z],
            [0, 0, 0, 1]
        ]
        
        frame = {
            "file_path": f"./images/frame_{i:04d}",
            "transform_matrix": transform_matrix
        }
        
        transforms["frames"].append(frame)
    
    # Save transforms file
    import json
    with open(os.path.join(output_dir, "transforms.json"), 'w') as f:
        json.dump(transforms, f, indent=2)
    
    print(f"Sample dataset created in: {output_dir}")
    print("Note: You'll need to add actual images to the images/ subdirectory")


if __name__ == "__main__":
    # You can run different demos by commenting/uncommenting
    
    # Run main training/evaluation
    # main()
    
    # Run basic usage demo
    demo_basic_usage()
    
    # Create sample dataset
    # create_sample_dataset() 