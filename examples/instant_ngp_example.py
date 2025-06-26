"""
Instant-NGP Example Usage

This script demonstrates how to use Instant-NGP for ultra-fast NeRF training
using multiresolution hash encoding.
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
    from src.nerfs.instant_ngp import (
        InstantNGP, InstantNGPConfig, InstantNGPTrainer, create_instant_ngp_dataloader
    )
except ImportError as e:
    print(f"Import error: {e}")
    print("Please make sure you're running from the project root directory")
    sys.exit(1)


def basic_example():
    """Basic Instant-NGP example with default settings."""
    print("=== Basic Instant-NGP Example ===")
    
    # Create configuration with correct parameters
    config = InstantNGPConfig(
        # Hash encoding parameters
        num_levels=16, level_dim=2, per_level_scale=2.0, base_resolution=16, log2_hashmap_size=19, desired_resolution=2048, # Network architecture
        geo_feat_dim=15, hidden_dim=64, hidden_dim_color=64, num_layers=2, num_layers_color=3, # Positional encoding for directions
        dir_pe=4, # Training parameters
        learning_rate=1e-2, learning_rate_decay=0.33, decay_step=1000, weight_decay=1e-6, # Rendering parameters
        density_activation='exp', density_bias=-1.0, rgb_activation='sigmoid', # Scene bounds
        bound=2.0, # Loss parameters
        lambda_entropy=1e-4, lambda_tv=1e-4
    )
    
    print(f"Configuration: {config}")
    
    # Create model
    model = InstantNGP(config)
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Test forward pass with dummy data
    batch_size = 1024
    positions = torch.randn(batch_size, 3) * 2.0  # Random positions in scene bounds
    directions = torch.randn(batch_size, 3)
    directions = directions / torch.norm(directions, dim=-1, keepdim=True)  # Normalize
    
    with torch.no_grad():
        density, color = model(positions, directions)
        print(f"Forward pass successful!")
        print(f"Density shape: {density.shape}, Color shape: {color.shape}")
        print(f"Density range: [{density.min():.4f}, {density.max():.4f}]")
        print(f"Color range: [{color.min():.4f}, {color.max():.4f}]")
    
    return model, config


def advanced_example():
    """Advanced Instant-NGP example with custom settings."""
    print("\n=== Advanced Instant-NGP Example ===")
    
    # Create configuration with custom parameters
    config = InstantNGPConfig(
        # Larger hash encoding
        num_levels=20, level_dim=4, per_level_scale=1.5, base_resolution=32, log2_hashmap_size=20, desired_resolution=4096, # Larger network
        geo_feat_dim=31, hidden_dim=128, hidden_dim_color=128, num_layers=3, num_layers_color=4, # Higher direction encoding
        dir_pe=6, # Different training parameters
        learning_rate=5e-3, learning_rate_decay=0.5, decay_step=2000, weight_decay=1e-5, # Different activations
        density_activation='exp', density_bias=-2.0, rgb_activation='sigmoid', # Larger scene
        bound=5.0, # Different regularization
        lambda_entropy=1e-3, lambda_tv=1e-3
    )
    
    print(f"Advanced configuration: {config}")
    
    # Create model
    model = InstantNGP(config)
    print(f"Advanced model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    return model, config


def training_example(data_path=None):
    """Example of training Instant-NGP."""
    print("\n=== Training Example ===")
    
    if data_path is None:
        print("No data path provided, using synthetic data")
        return
    
    # Create configuration
    config = InstantNGPConfig(
        learning_rate=1e-2, weight_decay=1e-6
    )
    
    try:
        # Create data loader
        train_loader = create_instant_ngp_dataloader(
            data_path, split='train', batch_size=4096, num_workers=4
        )
        
        # Create trainer
        trainer = InstantNGPTrainer(config)
        
        # Train for a few steps
        print("Starting training...")
        trainer.train(train_loader, num_epochs=1, max_steps=10)
        print("Training completed!")
        
    except Exception as e:
        print(f"Training failed: {e}")


def evaluation_example(model_path=None):
    """Example of evaluating Instant-NGP."""
    print("\n=== Evaluation Example ===")
    
    if model_path is None:
        print("No model path provided, using basic model")
        model, config = basic_example()
    else:
        # Load trained model
        config = InstantNGPConfig()
        model = InstantNGP(config)
        model.load_state_dict(torch.load(model_path))
        print(f"Model loaded from {model_path}")
    
    # Evaluation metrics
    model.eval()
    with torch.no_grad():
        # Test on random data
        positions = torch.randn(100, 3) * 2.0
        directions = torch.randn(100, 3)
        directions = directions / torch.norm(directions, dim=-1, keepdim=True)
        
        density, color = model(positions, directions)
        
        print(f"Evaluation completed!")
        print(f"Average density: {density.mean():.4f}")
        print(f"Average color: {color.mean(dim=0)}")


def main():
    """Main function to run examples."""
    parser = argparse.ArgumentParser(description="Instant-NGP Examples")
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
        default="outputs/instant_ngp",
        help="Output directory",
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("🚀 Instant-NGP Example Runner")
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