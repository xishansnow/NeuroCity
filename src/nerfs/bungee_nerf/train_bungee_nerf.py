"""
Training script for BungeeNeRF
"""

import argparse
import os
import sys
import logging
from pathlib import Path

import torch
import numpy as np

from .core import BungeeNeRF, BungeeNeRFConfig
from .dataset import BungeeNeRFDataset, MultiScaleDataset, GoogleEarthDataset
from .trainer import BungeeNeRFTrainer, ProgressiveTrainer, MultiScaleTrainer
from .utils import create_progressive_schedule

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train BungeeNeRF model")
    
    # Data arguments
    parser.add_argument("--data_dir", type=str, required=True, help="Path to dataset directory")
    parser.add_argument(
        "--dataset_type",
        type=str,
        default="auto",
        choices=["auto",
        "nerf_synthetic",
        "llff",
        "google_earth"],
        help="Dataset type",
    )
    parser.add_argument("--img_downscale", type=int, default=1, help="Image downscale factor")
    parser.add_argument("--white_background", action="store_true", help="Use white background")
    
    # Model arguments
    parser.add_argument("--num_stages", type=int, default=4, help="Number of progressive stages")
    parser.add_argument(
        "--base_resolution",
        type=int,
        default=16,
        help="Base resolution for progressive encoding",
    )
    parser.add_argument(
        "--max_resolution",
        type=int,
        default=2048,
        help="Maximum resolution for progressive encoding",
    )
    parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden dimension for MLP")
    parser.add_argument("--num_layers", type=int, default=8, help="Number of MLP layers")
    parser.add_argument(
        "--num_freqs_base",
        type=int,
        default=4,
        help="Base number of frequency bands",
    )
    parser.add_argument(
        "--num_freqs_max",
        type=int,
        default=10,
        help="Maximum number of frequency bands",
    )
    
    # Training arguments
    parser.add_argument(
        "--trainer_type",
        type=str,
        default="progressive",
        choices=["base",
        "progressive",
        "multiscale"],
        help="Type of trainer to use",
    )
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--max_steps", type=int, default=200000, help="Maximum training steps")
    
    # Progressive training arguments
    parser.add_argument(
        "--steps_per_stage",
        type=int,
        default=50000,
        help="Training steps per progressive stage",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=1000,
        help="Warmup steps for each stage",
    )
    
    # Sampling arguments
    parser.add_argument("--num_samples", type=int, default=64, help="Number of samples per ray")
    parser.add_argument(
        "--num_importance",
        type=int,
        default=128,
        help="Number of importance samples",
    )
    parser.add_argument(
        "--perturb",
        action="store_true",
        default=True,
        help="Add perturbation to sampling",
    )
    
    # Loss arguments
    parser.add_argument(
        "--color_loss_weight",
        type=float,
        default=1.0,
        help="Weight for color loss",
    )
    parser.add_argument(
        "--depth_loss_weight",
        type=float,
        default=0.1,
        help="Weight for depth loss",
    )
    parser.add_argument(
        "--progressive_loss_weight",
        type=float,
        default=0.05,
        help="Weight for progressive loss",
    )
    
    # Logging and saving arguments
    parser.add_argument("--log_dir", type=str, default="./logs", help="Directory for logging")
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./checkpoints",
        help="Directory for saving checkpoints",
    )
    parser.add_argument("--log_interval", type=int, default=100, help="Logging interval")
    parser.add_argument("--val_interval", type=int, default=1000, help="Validation interval")
    parser.add_argument("--save_interval", type=int, default=5000, help="Model saving interval")
    
    # Device arguments
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for training")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loading workers")
    
    # Resume training
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--start_stage",
        type=int,
        default=0,
        help="Starting stage for progressive training",
    )
    
    return parser.parse_args()


def create_config(args) -> BungeeNeRFConfig:
    """Create BungeeNeRF configuration from arguments"""
    
    config = BungeeNeRFConfig(
        # Progressive structure
        num_stages=args.num_stages, base_resolution=args.base_resolution, max_resolution=args.max_resolution, # Positional encoding
        num_freqs_base=args.num_freqs_base, num_freqs_max=args.num_freqs_max, include_input=True, # MLP architecture
        hidden_dim=args.hidden_dim, num_layers=args.num_layers, skip_layers=[4], # Progressive blocks
        block_hidden_dim=128, block_num_layers=4, # Training parameters
        batch_size=args.batch_size, learning_rate=args.learning_rate, max_steps=args.max_steps, # Sampling
        num_samples=args.num_samples, num_importance=args.num_importance, perturb=args.perturb, # Loss weights
        color_loss_weight=args.color_loss_weight, depth_loss_weight=args.depth_loss_weight, progressive_loss_weight=args.progressive_loss_weight
    )
    
    return config


def create_dataset(args, split: str):
    """Create dataset based on arguments"""
    
    if args.dataset_type == "google_earth":
        dataset_class = GoogleEarthDataset
    elif args.trainer_type in ["progressive", "multiscale"]:
        dataset_class = MultiScaleDataset
    else:
        dataset_class = BungeeNeRFDataset
    
    dataset = dataset_class(
        data_dir=args.data_dir, split=split, img_downscale=args.img_downscale, white_background=args.white_background
    )
    
    return dataset


def create_trainer(model, config, train_dataset, val_dataset, args):
    """Create trainer based on arguments"""
    
    if args.trainer_type == "progressive":
        # Create progressive schedule
        progressive_schedule = create_progressive_schedule(
            num_stages=args.num_stages, steps_per_stage=args.steps_per_stage, warmup_steps=args.warmup_steps
        )
        
        trainer = ProgressiveTrainer(
            model=model, config=config, train_dataset=train_dataset, val_dataset=val_dataset, device=args.device, progressive_schedule=progressive_schedule
        )
        
    elif args.trainer_type == "multiscale":
        trainer = MultiScaleTrainer(
            model=model, config=config, train_dataset=train_dataset, val_dataset=val_dataset, device=args.device
        )
        
    else:
        trainer = BungeeNeRFTrainer(
            model=model, config=config, train_dataset=train_dataset, val_dataset=val_dataset, device=args.device
        )
    
    return trainer


def main():
    """Main training function"""
    args = parse_args()
    
    # Check device
    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU")
        args.device = "cpu"
    
    logger.info(f"Using device: {args.device}")
    
    # Create datasets
    logger.info("Loading datasets...")
    train_dataset = create_dataset(args, "train")
    val_dataset = create_dataset(args, "val") if os.path.exists(
        os.path.join(args.data_dir, "transforms_val.json")
    ) else None
    
    logger.info(f"Train dataset: {len(train_dataset)} images")
    if val_dataset:
        logger.info(f"Val dataset: {len(val_dataset)} images")
    
    # Create configuration
    config = create_config(args)
    logger.info(f"Model configuration: {config}")
    
    # Create model
    logger.info("Creating model...")
    model = BungeeNeRF(config)
    
    # Set starting stage
    if args.start_stage > 0:
        model.set_current_stage(args.start_stage)
        logger.info(f"Set starting stage to {args.start_stage}")
    
    # Create trainer
    trainer = create_trainer(model, config, train_dataset, val_dataset, args)
    
    # Setup logging
    trainer.setup_logging(args.log_dir)
    
    # Resume from checkpoint if specified
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Start training
    logger.info("Starting training...")
    trainer.train(
        num_epochs=args.num_epochs, batch_size=args.batch_size, log_interval=args.log_interval, val_interval=args.val_interval, save_interval=args.save_interval, save_dir=args.save_dir
    )
    
    logger.info("Training completed!")


if __name__ == "__main__":
    main()
