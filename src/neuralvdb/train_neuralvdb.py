#!/usr/bin/env python3
"""
NeuralVDB Training Script

This script provides a command-line interface for training NeuralVDB models
with various configurations and datasets.
"""

import argparse
import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import numpy as np
from torch.utils.data import DataLoader

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from neuralvdb import (
    NeuralVDB, NeuralVDBConfig, AdvancedNeuralVDB, AdvancedNeuralVDBConfig, create_sample_data, load_training_data, visualize_training_data, NeuralVDBDataset, create_data_loaders
)

# Setup logging
logging.basicConfig(
    level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train NeuralVDB models')
    
    # Model configuration
    parser.add_argument(
        '--model-type',
        type=str,
        default='basic',
        choices=['basic',
        'advanced'],
        help='Model type to train',
    )
    
    parser.add_argument('--feature-dim', type=int, default=32, help='Feature dimension')
    
    parser.add_argument('--max-depth', type=int, default=8, help='Maximum octree depth')
    
    parser.add_argument(
        '--hidden-dims',
        type=int,
        nargs='+',
        default=[256,
        512,
        512,
        256,
        128],
        help='Hidden layer dimensions',
    )
    
    # Training configuration
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    
    parser.add_argument('--batch-size', type=int, default=1024, help='Batch size')
    
    parser.add_argument('--learning-rate', type=float, default=1e-3, help='Learning rate')
    
    parser.add_argument('--weight-decay', type=float, default=1e-5, help='Weight decay')
    
    parser.add_argument('--train-ratio', type=float, default=0.8, help='Training data ratio')
    
    # Data configuration
    parser.add_argument(
        '--data-type',
        type=str,
        default='synthetic',
        choices=['synthetic',
        'file',
        'tiles'],
        help='Data source type',
    )
    
    parser.add_argument('--data-path', type=str, help='Path to training data')
    
    parser.add_argument(
        '--num-points',
        type=int,
        default=50000,
        help='Number of points for synthetic data',
    )
    
    parser.add_argument(
        '--scene-type',
        type=str,
        default='mixed',
        choices=['sphere',
        'cube',
        'mixed',
        'urban'],
        help='Scene type for synthetic data',
    )
    
    # Output configuration
    parser.add_argument('--output-dir', type=str, default='./outputs', help='Output directory')
    
    parser.add_argument(
        '--model-name',
        type=str,
        default='neuralvdb_model',
        help='Model name for saving',
    )
    
    parser.add_argument(
        '--save-visualizations',
        action='store_true',
        help='Save training visualizations',
    )
    
    # Device configuration
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto',
        'cpu',
        'cuda'],
        help='Device to use for training',
    )
    
    # Advanced options
    parser.add_argument('--config-file', type=str, help='JSON config file to load parameters')
    
    parser.add_argument('--resume-from', type=str, help='Resume training from checkpoint')
    
    return parser.parse_args()


def load_config_from_file(config_path: str) -> dict[str, Any]:
    """Load configuration from JSON file"""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def create_model_config(args) -> NeuralVDBConfig | AdvancedNeuralVDBConfig:
    """Create model configuration from arguments"""
    common_params = {
        'feature_dim': args.feature_dim, 'max_depth': args.max_depth, 'hidden_dims': args.hidden_dims, 'learning_rate': args.learning_rate, 'weight_decay': args.weight_decay, 'batch_size': args.batch_size
    }
    
    if args.model_type == 'basic':
        return NeuralVDBConfig(**common_params)
    else:
        # Advanced model specific parameters
        advanced_params = {
            'adaptive_resolution': True, 'multi_scale_features': True, 'progressive_training': True, 'feature_compression': True
        }
        return AdvancedNeuralVDBConfig(**common_params, **advanced_params)


def load_data(args) -> tuple[DataLoader, DataLoader, Optional[np.ndarray], Optional[np.ndarray]]:
def load_data(args):
    """Load training data based on configuration"""
    if args.data_type == 'synthetic':
        logger.info(f"Generating synthetic data: {args.scene_type}, {args.num_points} points")
        points, occupancies = create_sample_data(
            n_points=args.num_points, scene_type=args.scene_type
        )
        
        # Create dataset and data loaders
        dataset = NeuralVDBDataset(points, occupancies)
        train_loader, val_loader = create_data_loaders(
            dataset, train_ratio=args.train_ratio, batch_size=args.batch_size
        )
        
        return train_loader, val_loader, points, occupancies
        
    elif args.data_type == 'file':
        if not args.data_path:
            raise ValueError("--data-path required for file data type")
        
        logger.info(f"Loading data from: {args.data_path}")
        train_loader, val_loader = load_training_data(
            args.data_path, task_type='occupancy', train_ratio=args.train_ratio
        )
        
        return train_loader, val_loader, None, None
        
    elif args.data_type == 'tiles':
        if not args.data_path:
            raise ValueError("--data-path required for tiles data type")
        
        logger.info(f"Loading tile data from: {args.data_path}")
        from neuralvdb.dataset import TileDataset
        
        tile_dataset = TileDataset(args.data_path, load_in_memory=False)
        train_loader, val_loader = create_data_loaders(
            tile_dataset, train_ratio=args.train_ratio, batch_size=args.batch_size
        )
        
        return train_loader, val_loader, None, None
    
    else:
        raise ValueError(f"Unknown data type: {args.data_type}")


def train_model(model, train_loader, val_loader, args, config):
    """Train the NeuralVDB model"""
    logger.info("Starting model training...")
    
    # Setup output directory
    os.makedirs(args.output_dir, exist_ok=True)
    model_path = os.path.join(args.output_dir, f"{args.model_name}.pth")
    
    # Train the model
    training_stats = model.fit(
        points=None, # Will use data loaders directly
        occupancies=None, train_ratio=args.train_ratio, num_epochs=args.epochs, save_path=model_path, device=args.device
    )
    
    # Save training statistics
    stats_path = os.path.join(args.output_dir, f"{args.model_name}_stats.json")
    with open(stats_path, 'w') as f:
        json.dump(training_stats, f, indent=2)
    
    logger.info(f"Training completed. Model saved to: {model_path}")
    return training_stats


def save_config(config, args):
    """Save configuration to file"""
    config_dict = {
        'model_type': args.model_type, 'model_config': config.__dict__, 'training_args': vars(args)
    }
    
    config_path = os.path.join(args.output_dir, f"{args.model_name}_config.json")
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=2, default=str)
    
    logger.info(f"Configuration saved to: {config_path}")


def main():
    """Main training function"""
    args = parse_args()
    
    # Load config from file if specified
    if args.config_file:
        file_config = load_config_from_file(args.config_file)
        # Update args with file config
        for key, value in file_config.items():
            if hasattr(args, key):
                setattr(args, key, value)
    
    # Create model configuration
    config = create_model_config(args)
    
    # Create model
    if args.model_type == 'basic':
        model = NeuralVDB(config)
    else:
        model = AdvancedNeuralVDB(config)
    
    logger.info(f"Created {args.model_type} NeuralVDB model")
    logger.info(f"Configuration: {config}")
    
    # Load data
    train_loader, val_loader, points, occupancies = load_data(args)
    
    # Save visualizations if requested
    if args.save_visualizations and points is not None:
        vis_path = os.path.join(args.output_dir, f"{args.model_name}_training_data.png")
        visualize_training_data(points, occupancies, save_path=vis_path, show_plot=False)
        logger.info(f"Training data visualization saved to: {vis_path}")
    
    # Resume from checkpoint if specified
    if args.resume_from:
        logger.info(f"Resuming training from: {args.resume_from}")
        model.load(args.resume_from)
    
    # Train model
    if args.data_type == 'synthetic':
        # For synthetic data, use the fit method with points and occupancies
        training_stats = model.fit(
            points=points, occupancies=occupancies, train_ratio=args.train_ratio, num_epochs=args.epochs, save_path=os.path.join(
                args.output_dir,
                f"{args.model_name}.pth",
            )
        )
    else:
        # For other data types, would need to modify the training loop
        # This is a simplified version
        logger.warning("Direct data loader training not fully implemented yet")
        # You would need to implement a training loop that uses the data loaders
        return
    
    # Save configuration
    save_config(config, args)
    
    # Print final statistics
    logger.info("Training Statistics:")
    logger.info(f"Final train loss: {training_stats['final_train_loss']:.6f}")
    if 'final_val_loss' in training_stats:
        logger.info(f"Final validation loss: {training_stats['final_val_loss']:.6f}")
    if 'best_val_loss' in training_stats:
        logger.info(f"Best validation loss: {training_stats['best_val_loss']:.6f}")
    
    logger.info("Training completed successfully!")


if __name__ == '__main__':
    main() 