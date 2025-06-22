#!/usr/bin/env python3
"""
Training script for PyNeRF
"""

import argparse
import os
import sys
import logging
import torch
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from . import (
    PyNeRF, PyNeRFConfig, PyNeRFTrainer, MultiScaleTrainer,
    PyNeRFDataset, MultiScaleDataset
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train PyNeRF model")
    
    # Data arguments
    parser.add_argument("--data_dir", type=str, required=True,
                       help="Path to dataset directory")
    parser.add_argument("--dataset_type", type=str, default="nerf_synthetic",
                       choices=["nerf_synthetic", "llff"],
                       help="Dataset type")
    parser.add_argument("--img_downscale", type=int, default=1,
                       help="Image downscale factor")
    parser.add_argument("--white_background", action="store_true",
                       help="Use white background for transparent images")
    
    # Model arguments
    parser.add_argument("--num_levels", type=int, default=8,
                       help="Number of pyramid levels")
    parser.add_argument("--base_resolution", type=int, default=16,
                       help="Base resolution for pyramid")
    parser.add_argument("--max_resolution", type=int, default=2048,
                       help="Maximum resolution for pyramid")
    parser.add_argument("--hash_table_size", type=int, default=2**20,
                       help="Hash table size")
    parser.add_argument("--features_per_level", type=int, default=4,
                       help="Features per level")
    
    # Training arguments
    parser.add_argument("--max_steps", type=int, default=20000,
                       help="Maximum training steps")
    parser.add_argument("--batch_size", type=int, default=4096,
                       help="Batch size (number of rays)")
    parser.add_argument("--learning_rate", type=float, default=1e-3,
                       help="Learning rate")
    parser.add_argument("--num_samples", type=int, default=64,
                       help="Number of samples per ray")
    parser.add_argument("--num_importance", type=int, default=128,
                       help="Number of importance samples")
    
    # Multi-scale training
    parser.add_argument("--multiscale", action="store_true",
                       help="Use multi-scale training")
    parser.add_argument("--scales", type=int, nargs="+", default=[1, 2, 4, 8],
                       help="Scales for multi-scale training")
    
    # Output arguments
    parser.add_argument("--log_dir", type=str, default="./logs",
                       help="Log directory")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints",
                       help="Checkpoint directory")
    parser.add_argument("--experiment_name", type=str, default="pyramid_nerf",
                       help="Experiment name")
    
    # Device arguments
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use")
    parser.add_argument("--num_workers", type=int, default=4,
                       help="Number of data loading workers")
    
    # Resume training
    parser.add_argument("--resume", type=str, default=None,
                       help="Path to checkpoint to resume from")
    
    return parser.parse_args()


def create_config(args):
    """Create PyNeRF configuration from arguments"""
    return PyNeRFConfig(
        # Pyramid structure
        num_levels=args.num_levels,
        base_resolution=args.base_resolution,
        max_resolution=args.max_resolution,
        
        # Hash encoding
        hash_table_size=args.hash_table_size,
        features_per_level=args.features_per_level,
        
        # Training
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_samples=args.num_samples,
        num_importance=args.num_importance
    )


def create_datasets(args):
    """Create training and validation datasets"""
    
    if args.multiscale:
        # Multi-scale datasets
        train_dataset = MultiScaleDataset(
            data_dir=args.data_dir,
            split="train",
            scales=args.scales,
            img_downscale=args.img_downscale,
            white_background=args.white_background
        )
        
        val_dataset = MultiScaleDataset(
            data_dir=args.data_dir,
            split="val",
            scales=args.scales,
            img_downscale=args.img_downscale,
            white_background=args.white_background
        )
    else:
        # Regular datasets
        train_dataset = PyNeRFDataset(
            data_dir=args.data_dir,
            split="train",
            img_downscale=args.img_downscale,
            white_background=args.white_background
        )
        
        val_dataset = PyNeRFDataset(
            data_dir=args.data_dir,
            split="val",
            img_downscale=args.img_downscale,
            white_background=args.white_background
        )
    
    return train_dataset, val_dataset


def main():
    """Main training function"""
    args = parse_args()
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create configuration
    config = create_config(args)
    logger.info(f"PyNeRF Configuration: {config}")
    
    # Create datasets
    logger.info("Loading datasets...")
    train_dataset, val_dataset = create_datasets(args)
    logger.info(f"Train dataset: {len(train_dataset)} images")
    logger.info(f"Val dataset: {len(val_dataset)} images")
    
    # Create model
    logger.info("Creating PyNeRF model...")
    model = PyNeRF(config)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create output directories
    experiment_dir = os.path.join(args.log_dir, args.experiment_name)
    checkpoint_dir = os.path.join(args.checkpoint_dir, args.experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Create trainer
    if args.multiscale:
        logger.info("Using multi-scale trainer...")
        trainer = MultiScaleTrainer(
            model=model,
            config=config,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            device=device,
            log_dir=experiment_dir,
            checkpoint_dir=checkpoint_dir
        )
    else:
        logger.info("Using standard trainer...")
        trainer = PyNeRFTrainer(
            model=model,
            config=config,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            device=device,
            log_dir=experiment_dir,
            checkpoint_dir=checkpoint_dir
        )
    
    # Resume from checkpoint if specified
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'epoch' in checkpoint:
            trainer.epoch = checkpoint['epoch']
        if 'global_step' in checkpoint:
            trainer.global_step = checkpoint['global_step']
    
    # Start training
    logger.info("Starting training...")
    try:
        trainer.train()
        logger.info("Training completed successfully!")
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        trainer.save_checkpoint()
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise


if __name__ == "__main__":
    main()
