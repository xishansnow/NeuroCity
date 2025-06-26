"""
Instant NGP Example Usage

This script demonstrates how to use Instant NGP for fast NeRF training.
"""

import os
import sys
import torch
import argparse
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np

# Add the project root to Python path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from src.nerfs.instant_ngp import (
        InstantNGP, InstantNGPConfig, InstantNGPTrainer, create_dataset
    )
except ImportError as e:
    print(f"Import error: {e}")
    print("Please make sure you're running from the project root directory")
    sys.exit(1)


def basic_instant_ngp_example():
    """Basic Instant NGP example."""
    print("=== Basic Instant NGP Example ===")
    
    # Create configuration
    config = InstantNGPConfig(
        scene_bounds=(
            -2,
            -2,
            -2,
            2,
            2,
            2,
        )
    )
    
    # Create model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = InstantNGP(config).to(device)
    
    print(f"Created Instant NGP model on {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):, }")
    
    return model, config


def train_instant_ngp_example(data_path: str, output_dir: str):
    """Training example for Instant NGP."""
    print("=== Instant NGP Training ===")
    
    config = InstantNGPConfig(
        scene_bounds=(
            -2,
            -2,
            -2,
            2,
            2,
            2,
        )
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create model and trainer
    model = InstantNGP(config).to(device)
    trainer = InstantNGPTrainer(config, output_dir, device)
    
    # Create dataset
    train_dataset = create_dataset(data_path, 'train', config)
    
    print(f"Training dataset: {len(train_dataset)} samples")
    
    # Start training
    trainer.train(train_dataset)
    
    return model, trainer


def train_instant_ngp(
    data_path: str, output_dir: str, num_epochs: int = 100, batch_size: int = 4096, learning_rate: float = 1e-3, **kwargs
) -> dict[str, float]:
    """Train Instant-NGP model."""
    # Create model and trainer
    model, trainer = setup_instant_ngp(data_path, output_dir, batch_size, learning_rate)
    
    # Train model
    metrics = trainer.train(num_epochs)
    
    return {
        "loss": float(
            metrics["loss"],
        )
    }


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Instant NGP Example")
    parser.add_argument(
        "--example",
        type=str,
        default="basic",
        choices=["basic",
        "train"],
        help="Example to run",
    )
    parser.add_argument("--data_path", type=str, help="Path to dataset")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs/instant_ngp",
        help="Output directory",
    )
    
    args = parser.parse_args()
    
    if args.example == "basic":
        model, config = basic_instant_ngp_example()
        print("Basic Instant NGP example completed!")
        
    elif args.example == "train":
        if not args.data_path:
            print("Error: --data_path required for training")
            return
        model, trainer = train_instant_ngp_example(args.data_path, args.output_dir)
        print("Instant NGP training completed!")


if __name__ == "__main__":
    main() 