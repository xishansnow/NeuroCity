"""
Classic NeRF Example Usage

This script demonstrates how to use Classic NeRF for training and evaluation.
"""

import os
import sys
import torch
import argparse
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np
import torch.nn as nn

# Add the project root to Python path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from src.nerfs.classic_nerf import (
        ClassicNeRF, ClassicNeRFConfig, ClassicNeRFTrainer, create_dataset
    )
except ImportError as e:
    print(f"Import error: {e}")
    print("Please make sure you're running from the project root directory")
    sys.exit(1)


def basic_classic_nerf_example():
    """Basic Classic NeRF example."""
    print("=== Basic Classic NeRF Example ===")
    
    # Create configuration
    config = ClassicNeRFConfig(
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
    model = ClassicNeRF(config).to(device)
    
    print(f"Created Classic NeRF model on {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):, }")
    
    return model, config


def train_classic_nerf_example(data_path: str, output_dir: str):
    """Training example for Classic NeRF."""
    print("=== Classic NeRF Training ===")
    
    config = ClassicNeRFConfig(
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
    model = ClassicNeRF(config).to(device)
    trainer = ClassicNeRFTrainer(config, output_dir, device)
    
    # Create dataset
    train_dataset = create_dataset(data_path, 'train', config)
    
    print(f"Training dataset: {len(train_dataset)} samples")
    
    # Start training
    trainer.train(train_dataset)
    
    return model, trainer


def train_classic_nerf(
    data_path: str, output_dir: str, num_epochs: int = 100, batch_size: int = 4096, learning_rate: float = 1e-3, **kwargs
) -> dict[str, float]:
    """Train classic NeRF model."""
    # Create model and trainer
    model, trainer = setup_classic_nerf(data_path, output_dir, batch_size, learning_rate)
    
    # Train model
    metrics = trainer.train(num_epochs)
    
    return {
        "loss": float(
            metrics["loss"],
        )
    }


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Classic NeRF Example")
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
        default="./outputs/classic_nerf",
        help="Output directory",
    )
    
    args = parser.parse_args()
    
    if args.example == "basic":
        model, config = basic_classic_nerf_example()
        print("Basic Classic NeRF example completed!")
        
    elif args.example == "train":
        if not args.data_path:
            print("Error: --data_path required for training")
            return
        model, trainer = train_classic_nerf_example(args.data_path, args.output_dir)
        print("Classic NeRF training completed!")


if __name__ == "__main__":
    main() 