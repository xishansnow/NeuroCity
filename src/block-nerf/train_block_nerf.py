#!/usr/bin/env python3
"""
Block-NeRF Training Script

This script trains Block-NeRF models on large-scale datasets.
"""

import argparse
import os
import sys
import torch
import numpy as np
import json
import wandb
from pathlib import Path
from typing import Dict, Any

# Add the parent directory to the path to import our modules
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.block_nerf import (
    BlockManager, BlockNeRFTrainer, BlockNeRFDataset,
    BlockCompositor, VisibilityNetwork
)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train Block-NeRF')
    
    # Dataset arguments
    parser.add_argument('--data_root', type=str, required=True,
                       help='Root directory of the dataset')
    parser.add_argument('--dataset_type', type=str, default='colmap',
                       choices=['colmap', 'llff', 'custom'],
                       help='Type of dataset format')
    parser.add_argument('--img_scale', type=float, default=0.5,
                       help='Image scale factor')
    
    # Scene configuration
    parser.add_argument('--scene_bounds', type=str, default=None,
                       help='Scene bounds as JSON string: [[[x_min,x_max],[y_min,y_max],[z_min,z_max]]]')
    parser.add_argument('--block_size', type=float, default=75.0,
                       help='Block size (radius in meters)')
    parser.add_argument('--overlap_ratio', type=float, default=0.5,
                       help='Overlap ratio between adjacent blocks')
    
    # Network configuration
    parser.add_argument('--hidden_dim', type=int, default=256,
                       help='Hidden dimension of NeRF networks')
    parser.add_argument('--num_layers', type=int, default=8,
                       help='Number of layers in NeRF networks')
    parser.add_argument('--pos_encoding_levels', type=int, default=16,
                       help='Number of positional encoding levels')
    parser.add_argument('--dir_encoding_levels', type=int, default=4,
                       help='Number of directional encoding levels')
    parser.add_argument('--appearance_dim', type=int, default=32,
                       help='Appearance embedding dimension')
    
    # Training configuration
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Batch size for training')
    parser.add_argument('--ray_batch_size', type=int, default=1024,
                       help='Number of rays per batch')
    parser.add_argument('--num_epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=5e-4,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                       help='Weight decay')
    
    # Rendering configuration
    parser.add_argument('--num_samples', type=int, default=64,
                       help='Number of samples per ray (coarse)')
    parser.add_argument('--num_fine_samples', type=int, default=128,
                       help='Number of fine samples per ray')
    parser.add_argument('--near', type=float, default=0.1,
                       help='Near plane distance')
    parser.add_argument('--far', type=float, default=100.0,
                       help='Far plane distance')
    
    # Loss weights
    parser.add_argument('--visibility_weight', type=float, default=0.1,
                       help='Weight for visibility loss')
    parser.add_argument('--pose_reg_weight', type=float, default=0.01,
                       help='Weight for pose regularization')
    parser.add_argument('--appearance_reg_weight', type=float, default=0.001,
                       help='Weight for appearance regularization')
    
    # Training options
    parser.add_argument('--use_pose_refinement', action='store_true',
                       help='Use learned pose refinement')
    parser.add_argument('--use_appearance_matching', action='store_true',
                       help='Use appearance matching between blocks')
    parser.add_argument('--train_visibility', action='store_true',
                       help='Train visibility network')
    
    # Optimization
    parser.add_argument('--scheduler', type=str, default='exponential',
                       choices=['exponential', 'cosine', 'step'],
                       help='Learning rate scheduler')
    parser.add_argument('--scheduler_gamma', type=float, default=0.99,
                       help='Scheduler gamma for exponential decay')
    parser.add_argument('--grad_clip', type=float, default=0.0,
                       help='Gradient clipping threshold (0 to disable)')
    
    # Logging and saving
    parser.add_argument('--exp_name', type=str, default='block_nerf_exp',
                       help='Experiment name')
    parser.add_argument('--save_dir', type=str, default='./experiments',
                       help='Directory to save experiments')
    parser.add_argument('--log_interval', type=int, default=100,
                       help='Logging interval (steps)')
    parser.add_argument('--val_interval', type=int, default=1000,
                       help='Validation interval (steps)')
    parser.add_argument('--save_interval', type=int, default=5000,
                       help='Checkpoint saving interval (steps)')
    
    # Wandb logging
    parser.add_argument('--use_wandb', action='store_true',
                       help='Use Weights & Biases for logging')
    parser.add_argument('--wandb_project', type=str, default='block-nerf',
                       help='Wandb project name')
    
    # System
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use for training')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    # Resume training
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    
    return parser.parse_args()


def setup_config(args) -> Dict[str, Any]:
    """Setup configuration dictionaries"""
    
    # Network configuration
    network_config = {
        'hidden_dim': args.hidden_dim,
        'num_layers': args.num_layers,
        'pos_encoding_levels': args.pos_encoding_levels,
        'dir_encoding_levels': args.dir_encoding_levels,
        'appearance_dim': args.appearance_dim,
        'use_integrated_encoding': True,
        'skip_connections': [4]
    }
    
    # Training configuration
    training_config = {
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'scheduler': args.scheduler,
        'scheduler_gamma': args.scheduler_gamma,
        'grad_clip': args.grad_clip,
        'visibility_weight': args.visibility_weight,
        'pose_reg_weight': args.pose_reg_weight,
        'appearance_reg_weight': args.appearance_reg_weight,
        'use_pose_refinement': args.use_pose_refinement,
        'use_appearance_matching': args.use_appearance_matching,
        'train_visibility': args.train_visibility,
        'interpolation_method': 'inverse_distance',
        'interpolation_power': 2.0,
        'render_kwargs': {
            'near': args.near,
            'far': args.far,
            'num_samples': args.num_samples,
            'num_fine_samples': args.num_fine_samples
        }
    }
    
    return network_config, training_config


def setup_scene_bounds(args, dataset) -> tuple:
    """Setup scene bounds from arguments or dataset"""
    if args.scene_bounds:
        scene_bounds = json.loads(args.scene_bounds)
        scene_bounds = tuple(tuple(bound) for bound in scene_bounds)
    else:
        # Compute from dataset
        min_bounds, max_bounds = dataset.get_scene_bounds()
        scene_bounds = tuple(zip(min_bounds, max_bounds))
    
    return scene_bounds


def main():
    """Main training function"""
    args = parse_args()
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create experiment directory
    exp_dir = Path(args.save_dir) / args.exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Save arguments
    with open(exp_dir / 'args.json', 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Setup wandb
    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.exp_name,
            config=vars(args),
            dir=str(exp_dir)
        )
    
    # Load datasets
    print("Loading datasets...")
    train_dataset = BlockNeRFDataset(
        data_root=args.data_root,
        split='train',
        img_scale=args.img_scale,
        ray_batch_size=args.ray_batch_size,
        use_cache=True,
        load_appearance_ids=True,
        load_exposure=True
    )
    
    val_dataset = BlockNeRFDataset(
        data_root=args.data_root,
        split='val',
        img_scale=args.img_scale,
        ray_batch_size=args.ray_batch_size,
        use_cache=True,
        load_appearance_ids=True,
        load_exposure=True
    )
    
    # Create data loaders
    train_loader = train_dataset.create_dataloader(
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    
    val_loader = val_dataset.create_dataloader(
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers
    )
    
    # Setup scene bounds
    scene_bounds = setup_scene_bounds(args, train_dataset)
    print(f"Scene bounds: {scene_bounds}")
    
    # Create block manager
    print("Setting up block manager...")
    block_manager = BlockManager(
        scene_bounds=scene_bounds,
        block_size=args.block_size,
        overlap_ratio=args.overlap_ratio,
        device=device
    )
    
    # Optimize block layout based on camera positions
    camera_positions = torch.from_numpy(train_dataset.get_camera_positions()).to(device)
    optimization_stats = block_manager.optimize_block_layout(camera_positions)
    print(f"Block layout optimization: {optimization_stats}")
    
    # Save block layout
    layout_path = exp_dir / 'block_layout.json'
    block_manager.save_block_layout(str(layout_path))
    
    # Setup configurations
    network_config, training_config = setup_config(args)
    
    # Create trainer
    print("Setting up trainer...")
    trainer = BlockNeRFTrainer(
        block_manager=block_manager,
        network_config=network_config,
        training_config=training_config,
        device=device
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Start training
    print("Starting training...")
    print(f"Training for {args.num_epochs} epochs")
    print(f"Total training samples: {len(train_dataset)}")
    print(f"Total validation samples: {len(val_dataset)}")
    
    try:
        trainer.train(
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            num_epochs=args.num_epochs,
            save_dir=str(exp_dir),
            log_interval=args.log_interval,
            val_interval=args.val_interval,
            save_interval=args.save_interval
        )
    except KeyboardInterrupt:
        print("Training interrupted by user")
    except Exception as e:
        print(f"Training failed with error: {e}")
        raise
    
    # Save final results
    print("Saving final results...")
    
    # Save block manager state
    final_blocks_dir = exp_dir / 'final_blocks'
    block_manager.save_blocks(str(final_blocks_dir))
    
    # Save scene statistics
    scene_stats = block_manager.get_scene_statistics()
    with open(exp_dir / 'scene_statistics.json', 'w') as f:
        json.dump(scene_stats, f, indent=2)
    
    # Close wandb
    if args.use_wandb:
        wandb.finish()
    
    print(f"Training completed! Results saved to {exp_dir}")


if __name__ == '__main__':
    main() 