"""
Mega-NeRF Example Usage

This script demonstrates how to use Mega-NeRF for large-scale scene reconstruction
using spatial decomposition and clustering techniques.
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
    from src.nerfs.mega_nerf import (
        MegaNeRF, MegaNeRFConfig, MegaNeRFTrainer, MegaNeRFDataset, CameraTrajectory
    )
except ImportError as e:
    print(f"Import error: {e}")
    print("Please make sure you're running from the project root directory")
    sys.exit(1)


def basic_example():
    """Basic Mega-NeRF example with default settings."""
    print("=== Basic Mega-NeRF Example ===")
    
    # Create configuration with correct parameters
    config = MegaNeRFConfig(
        # Scene decomposition parameters
        num_submodules=8, grid_size=(4, 2), # 2D grid decomposition
        overlap_factor=0.15, # Network parameters
        hidden_dim=256, num_layers=8, skip_connections=[4], use_viewdirs=True, # Training parameters
        batch_size=1024, learning_rate=5e-4, lr_decay=0.1, max_iterations=500000, # Sampling parameters
        num_coarse=256, num_fine=512, near=0.1, far=1000.0, # Appearance embedding
        use_appearance_embedding=True, appearance_dim=48, # Boundary parameters
        scene_bounds=(-100, -100, -10, 100, 100, 50), foreground_ratio=0.8
    )
    
    print(f"Configuration: {config}")
    
    # Create model
    model = MegaNeRF(config)
    print(f"Model created with {config.num_submodules} submodules")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Test forward pass with dummy data
    batch_size = 512
    points = torch.randn(batch_size, 3) * 50.0  # Random points in scene bounds
    viewdirs = torch.randn(batch_size, 3)
    viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)  # Normalize
    appearance_idx = torch.randint(0, 10, (batch_size, ))  # Random appearance indices
    
    with torch.no_grad():
        density, color = model(points, viewdirs, appearance_idx)
        print(f"Forward pass successful!")
        print(f"Density shape: {density.shape}, Color shape: {color.shape}")
        print(f"Density range: [{density.min():.4f}, {density.max():.4f}]")
        print(f"Color range: [{color.min():.4f}, {color.max():.4f}]")
    
    return model, config


def advanced_example():
    """Advanced Mega-NeRF example with custom settings."""
    print("\n=== Advanced Mega-NeRF Example ===")
    
    # Create configuration with custom parameters
    config = MegaNeRFConfig(
        # Larger scene decomposition
        num_submodules=16, grid_size=(4, 4), overlap_factor=0.2, # Larger network
        hidden_dim=512, num_layers=12, skip_connections=[4, 8], use_viewdirs=True, # Training parameters
        batch_size=2048, learning_rate=1e-3, lr_decay=0.05, max_iterations=1000000, # More samples
        num_coarse=512, num_fine=1024, near=0.05, far=2000.0, # Larger appearance embedding
        use_appearance_embedding=True, appearance_dim=64, # Larger scene bounds
        scene_bounds=(-500, -500, -50, 500, 500, 100), foreground_ratio=0.9
    )
    
    print(f"Advanced configuration: {config}")
    
    # Create model
    model = MegaNeRF(config)
    print(f"Advanced model created with {config.num_submodules} submodules")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Show submodule information
    centroids = model.get_submodule_centroids()
    print(f"Submodule centroids: {centroids.shape}")
    
    return model, config


def training_example(data_path=None):
    """Example of training Mega-NeRF."""
    print("\n=== Training Example ===")
    
    if data_path is None:
        print("No data path provided, using synthetic data")
        return
    
    # Create configuration
    config = MegaNeRFConfig(
        learning_rate=5e-4, batch_size=1024
    )
    
    try:
        # Create dataset
        dataset = MegaNeRFDataset(data_path, split='train')
        
        # Create trainer
        trainer = MegaNeRFTrainer(config)
        
        # Train for a few steps
        print("Starting training...")
        trainer.train(dataset, num_epochs=1, max_steps=10)
        print("Training completed!")
        
    except Exception as e:
        print(f"Training failed: {e}")


def evaluation_example(model_path=None):
    """Example of evaluating Mega-NeRF."""
    print("\n=== Evaluation Example ===")
    
    if model_path is None:
        print("No model path provided, using basic model")
        model, config = basic_example()
    else:
        # Load trained model
        config = MegaNeRFConfig()
        model = MegaNeRF(config)
        model.load_state_dict(torch.load(model_path))
        print(f"Model loaded from {model_path}")
    
    # Evaluation metrics
    model.eval()
    with torch.no_grad():
        # Test on random data
        points = torch.randn(100, 3) * 50.0
        viewdirs = torch.randn(100, 3)
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        appearance_idx = torch.randint(0, 10, (100, ))
        
        density, color = model(points, viewdirs, appearance_idx)
        
        print(f"Evaluation completed!")
        print(f"Average density: {density.mean():.4f}")
        print(f"Average color: {color.mean(dim=0)}")


def main():
    """Main function to run examples."""
    parser = argparse.ArgumentParser(description="Mega-NeRF Examples")
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
        default="outputs/mega_nerf",
        help="Output directory",
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("🚀 Mega-NeRF Example Runner")
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