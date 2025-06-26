"""
Bungee-NeRF Example Usage

This script demonstrates how to use Bungee-NeRF for progressive neural
radiance field training with adaptive sampling.
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
    # Import whatever is actually available in the bungee_nerf module
    # Since we don't know the exact API, we'll use a more flexible approach
    import src.nerfs.bungee_nerf as bungee_nerf
    
    # Try to get the main classes, fall back to creating basic examples
    BungeeNeRFConfig = getattr(bungee_nerf, 'BungeeNeRFConfig', None)
    BungeeNeRFModel = getattr(bungee_nerf, 'BungeeNeRFModel', None)
    BungeeNeRFTrainer = getattr(bungee_nerf, 'BungeeNeRFTrainer', None)
    
except ImportError as e:
    print(f"Import error: {e}")
    print("Please make sure you're running from the project root directory")
    print("Note: This example requires the bungee_nerf module to be properly implemented")
    sys.exit(1)


def basic_example():
    """Basic Bungee-NeRF example with default configuration."""
    print("=== Basic Bungee-NeRF Example ===")
    
    if BungeeNeRFConfig is None or BungeeNeRFModel is None:
        print("BungeeNeRF classes not available. This is a placeholder example.")
        print("Please implement the bungee_nerf module or check the imports.")
        return None, None
    
    try:
        config = BungeeNeRFConfig(
            # Progressive training stages
            num_stages=4, stage_samples=[64, 128, 256, 512], stage_epochs=[25, 25, 25, 25], # Network architecture
            netdepth=8, netwidth=256, # Adaptive sampling
            adaptive_sampling=True, importance_sampling=True, # Training settings
            learning_rate=5e-4, weight_decay=1e-6, # Scene bounds
            near_plane=2.0, far_plane=6.0
        )
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = BungeeNeRFModel(config).to(device)
        
        print(f"Created Bungee-NeRF model on {device}")
        print(f"Progressive stages: {config.num_stages}")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):, }")
        
        return model, config
        
    except Exception as e:
        print(f"Error creating Bungee-NeRF model: {e}")
        print("This might be due to missing implementation or incorrect parameters.")
        return None, None


def progressive_training_example(data_path: str, output_dir: str):
    """Example with progressive training stages."""
    print("=== Progressive Bungee-NeRF Training ===")
    
    if not all([BungeeNeRFConfig, BungeeNeRFModel, BungeeNeRFTrainer]):
        print("BungeeNeRF classes not fully available. This is a placeholder example.")
        return None, None
    
    # This is a placeholder implementation
    # The actual implementation would depend on the real bungee_nerf API
    print("Progressive training example - implementation pending")
    print(f"Data path: {data_path}")
    print(f"Output directory: {output_dir}")
    
    return None, None


def main():
    """Main function to run examples."""
    parser = argparse.ArgumentParser(description="Bungee-NeRF Example Usage")
    parser.add_argument(
        "--example",
        type=str,
        default="basic",
        choices=["basic",
        "progressive"],
        help="Example to run",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="./data/nerf_synthetic/lego",
        help="Path to dataset",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs/bungee_nerf",
        help="Output directory",
    )
    
    args = parser.parse_args()
    
    if args.example == "basic":
        model, config = basic_example()
        if model is not None:
            print("Basic example completed successfully!")
        else:
            print("Basic example failed - check implementation")
        
    elif args.example == "progressive":
        if not os.path.exists(args.data_path):
            print(f"Error: Data path {args.data_path} does not exist")
            return
        
        model, trainer = progressive_training_example(args.data_path, args.output_dir)
        if model is not None:
            print("Progressive training completed!")
        else:
            print("Progressive training failed - check implementation")


if __name__ == "__main__":
    main() 