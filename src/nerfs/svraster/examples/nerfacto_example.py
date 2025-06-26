"""
Nerfacto Example Usage

This script demonstrates how to use Nerfacto for NeRF training and evaluation.
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
    from src.nerfs.nerfacto import (
        Nerfacto, NerfactoConfig, NerfactoTrainer, create_dataset
    )
except ImportError as e:
    print(f"Import error: {e}")
    print("Please make sure you're running from the project root directory")
    sys.exit(1)


def basic_nerfacto_example():
    """Basic Nerfacto example."""
    print("=== Basic Nerfacto Example ===")
    
    # Create configuration
    config = NerfactoConfig(
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
    model = Nerfacto(config).to(device)
    
    print(f"Created Nerfacto model on {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):, }")
    
    return model, config


def train_nerfacto_example(data_path: str, output_dir: str):
    """Training example for Nerfacto."""
    print("=== Nerfacto Training ===")
    
    config = NerfactoConfig(
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
    model = Nerfacto(config).to(device)
    trainer = NerfactoTrainer(config, output_dir, device)
    
    # Create dataset
    train_dataset = create_dataset(data_path, 'train', config)
    
    print(f"Training dataset: {len(train_dataset)} samples")
    
    # Start training
    trainer.train(train_dataset)
    
    return model, trainer


def train_nerfacto(
    data_path: str, output_dir: str, num_epochs: int = 100, batch_size: int = 4096, learning_rate: float = 1e-3, **kwargs
) -> dict[str, float]:
    """Train Nerfacto model."""
    # Create model and trainer
    model, trainer = setup_nerfacto(data_path, output_dir, batch_size, learning_rate)
    
    # Train model
    metrics = trainer.train(num_epochs)
    
    return {
        "loss": float(
            metrics["loss"],
        )
    }


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Nerfacto Example")
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
        default="./outputs/nerfacto",
        help="Output directory",
    )
    
    args = parser.parse_args()
    
    if args.example == "basic":
        model, config = basic_nerfacto_example()
        print("Basic Nerfacto example completed!")
        
    elif args.example == "train":
        if not args.data_path:
            print("Error: --data_path required for training")
            return
        model, trainer = train_nerfacto_example(args.data_path, args.output_dir)
        print("Nerfacto training completed!")


if __name__ == "__main__":
    main() 