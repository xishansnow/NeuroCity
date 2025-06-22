#!/usr/bin/env python3
"""
Mega-NeRF Training Script

This script trains Mega-NeRF models on large-scale datasets with spatial partitioning.
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

from src.mega_nerf import (
    MegaNeRF, MegaNeRFConfig, MegaNeRFTrainer, ParallelTrainer,
    MegaNeRFDataset, CameraDataset, GridPartitioner, GeometryAwarePartitioner
)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train Mega-NeRF')
    
    # Dataset arguments
    parser.add_argument('--data_root', type=str, required=True,
                       help='Root directory of the dataset')
    parser.add_argument('--dataset_type', type=str, default='colmap',
                       choices=['colmap', 'llff', 'nerf', 'synthetic'],
                       help='Type of dataset format')
    parser.add_argument('--image_scale', type=float, default=0.5,
                       help='Image scale factor')
    
    # Scene configuration
    parser.add_argument('--scene_bounds', type=str, default=None,
                       help='Scene bounds as JSON string: [x_min, y_min, z_min, x_max, y_max, z_max]')
    parser.add_argument('--num_submodules', type=int, default=8,
                       help='Number of submodules')
    parser.add_argument('--grid_size', type=str, default='4,2',
                       help='Grid size as "x,y"')
    parser.add_argument('--overlap_factor', type=float, default=0.15,
                       help='Overlap factor between submodules')
    
    # Partitioning strategy
    parser.add_argument('--partitioning_strategy', type=str, default='grid',
                       choices=['grid', 'geometry_aware'],
                       help='Spatial partitioning strategy')
    parser.add_argument('--use_kmeans', action='store_true',
                       help='Use k-means clustering for geometry-aware partitioning')
    
    # Network configuration
    parser.add_argument('--hidden_dim', type=int, default=256,
                       help='Hidden dimension of NeRF networks')
    parser.add_argument('--num_layers', type=int, default=8,
                       help='Number of layers in NeRF networks')
    parser.add_argument('--use_viewdirs', action='store_true', default=True,
                       help='Use view directions')
    parser.add_argument('--use_appearance_embedding', action='store_true', default=True,
                       help='Use appearance embeddings')
    parser.add_argument('--appearance_dim', type=int, default=48,
                       help='Appearance embedding dimension')
    
    # Training configuration
    parser.add_argument('--training_mode', type=str, default='sequential',
                       choices=['sequential', 'parallel'],
                       help='Training mode')
    parser.add_argument('--num_parallel_workers', type=int, default=4,
                       help='Number of parallel workers')
    parser.add_argument('--iterations_per_submodule', type=int, default=10000,
                       help='Number of iterations per submodule')
    parser.add_argument('--batch_size', type=int, default=1024,
                       help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=5e-4,
                       help='Learning rate')
    parser.add_argument('--lr_decay', type=float, default=0.1,
                       help='Learning rate decay factor')
    
    # Rendering configuration
    parser.add_argument('--num_coarse_samples', type=int, default=256,
                       help='Number of coarse samples per ray')
    parser.add_argument('--num_fine_samples', type=int, default=512,
                       help='Number of fine samples per ray')
    parser.add_argument('--near', type=float, default=0.1,
                       help='Near plane distance')
    parser.add_argument('--far', type=float, default=1000.0,
                       help='Far plane distance')
    
    # Logging and saving
    parser.add_argument('--exp_name', type=str, default='mega_nerf_exp',
                       help='Experiment name')
    parser.add_argument('--output_dir', type=str, default='./experiments',
                       help='Output directory for experiments')
    parser.add_argument('--log_interval', type=int, default=100,
                       help='Logging interval (iterations)')
    parser.add_argument('--val_interval', type=int, default=1000,
                       help='Validation interval (iterations)')
    parser.add_argument('--save_interval', type=int, default=5000,
                       help='Checkpoint saving interval (iterations)')
    
    # Wandb logging
    parser.add_argument('--use_wandb', action='store_true',
                       help='Use Weights & Biases for logging')
    parser.add_argument('--wandb_project', type=str, default='mega-nerf',
                       help='Wandb project name')
    
    # System
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use for training')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    # Resume training
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    
    return parser.parse_args()


def setup_config(args) -> MegaNeRFConfig:
    """Setup Mega-NeRF configuration"""
    
    # Parse grid size
    grid_x, grid_y = map(int, args.grid_size.split(','))
    
    # Parse scene bounds
    if args.scene_bounds:
        scene_bounds = tuple(json.loads(args.scene_bounds))
    else:
        scene_bounds = (-100, -100, -10, 100, 100, 50)  # Default bounds
    
    # Skip connections
    skip_connections = [4] if args.num_layers >= 5 else []
    
    config = MegaNeRFConfig(
        # Scene decomposition
        num_submodules=args.num_submodules,
        grid_size=(grid_x, grid_y),
        overlap_factor=args.overlap_factor,
        
        # Network parameters
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        skip_connections=skip_connections,
        use_viewdirs=args.use_viewdirs,
        
        # Training parameters
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        lr_decay=args.lr_decay,
        max_iterations=args.iterations_per_submodule,
        
        # Sampling parameters
        num_coarse=args.num_coarse_samples,
        num_fine=args.num_fine_samples,
        near=args.near,
        far=args.far,
        
        # Appearance embedding
        use_appearance_embedding=args.use_appearance_embedding,
        appearance_dim=args.appearance_dim,
        
        # Scene bounds
        scene_bounds=scene_bounds
    )
    
    return config


def create_partitioner(args, camera_positions: np.ndarray):
    """Create spatial partitioner based on configuration"""
    
    # Parse scene bounds
    if args.scene_bounds:
        scene_bounds = tuple(json.loads(args.scene_bounds))
    else:
        # Estimate from camera positions
        min_bounds = camera_positions.min(axis=0) - 20
        max_bounds = camera_positions.max(axis=0) + 20
        scene_bounds = (min_bounds[0], min_bounds[1], min_bounds[2],
                       max_bounds[0], max_bounds[1], max_bounds[2])
    
    if args.partitioning_strategy == 'grid':
        grid_x, grid_y = map(int, args.grid_size.split(','))
        partitioner = GridPartitioner(
            scene_bounds=scene_bounds,
            grid_size=(grid_x, grid_y),
            overlap_factor=args.overlap_factor
        )
    elif args.partitioning_strategy == 'geometry_aware':
        partitioner = GeometryAwarePartitioner(
            scene_bounds=scene_bounds,
            camera_positions=camera_positions,
            num_partitions=args.num_submodules,
            overlap_factor=args.overlap_factor,
            use_kmeans=args.use_kmeans
        )
    else:
        raise ValueError(f"Unknown partitioning strategy: {args.partitioning_strategy}")
    
    return partitioner


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
    exp_dir = Path(args.output_dir) / args.exp_name
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
    
    # Load camera dataset
    print("Loading camera dataset...")
    camera_dataset = CameraDataset(
        data_root=args.data_root,
        split='train',
        image_scale=args.image_scale,
        load_images=True
    )
    
    # Get camera positions for partitioning
    camera_positions = camera_dataset.get_camera_positions()
    print(f"Loaded {len(camera_positions)} cameras")
    
    # Create spatial partitioner
    print("Creating spatial partitioner...")
    partitioner = create_partitioner(args, camera_positions)
    
    # Visualize partitioning
    partition_viz_path = exp_dir / 'partitioning_visualization.png'
    if hasattr(partitioner, 'visualize_partitions'):
        partitioner.visualize_partitions(str(partition_viz_path))
    elif hasattr(partitioner, 'visualize_partitions_with_cameras'):
        partitioner.visualize_partitions_with_cameras(str(partition_viz_path))
    
    # Create Mega-NeRF dataset
    print("Creating Mega-NeRF dataset...")
    dataset = MegaNeRFDataset(
        data_root=args.data_root,
        partitioner=partitioner,
        split='train',
        ray_batch_size=args.batch_size,
        image_scale=args.image_scale,
        use_cache=True,
        cache_dir=str(exp_dir / 'cache')
    )
    
    # Setup configuration
    config = setup_config(args)
    
    # Save configuration
    with open(exp_dir / 'config.json', 'w') as f:
        json.dump({
            'num_submodules': config.num_submodules,
            'grid_size': config.grid_size,
            'overlap_factor': config.overlap_factor,
            'hidden_dim': config.hidden_dim,
            'num_layers': config.num_layers,
            'scene_bounds': config.scene_bounds,
            'batch_size': config.batch_size,
            'learning_rate': config.learning_rate
        }, f, indent=2)
    
    # Create Mega-NeRF model
    print("Creating Mega-NeRF model...")
    model = MegaNeRF(config)
    
    # Print model info
    model_info = model.get_model_info()
    print(f"Model info: {model_info}")
    
    # Create trainer
    print("Setting up trainer...")
    if args.training_mode == 'sequential':
        trainer = MegaNeRFTrainer(
            config=config,
            model=model,
            dataset=dataset,
            output_dir=str(exp_dir),
            device=device
        )
    elif args.training_mode == 'parallel':
        trainer = ParallelTrainer(
            config=config,
            model=model,
            dataset=dataset,
            output_dir=str(exp_dir),
            device=device,
            num_parallel_workers=args.num_parallel_workers
        )
    else:
        raise ValueError(f"Unknown training mode: {args.training_mode}")
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        if hasattr(trainer, 'load_checkpoint'):
            trainer.load_checkpoint(args.resume)
    
    # Start training
    print("Starting training...")
    print(f"Training mode: {args.training_mode}")
    print(f"Iterations per submodule: {args.iterations_per_submodule}")
    
    try:
        if args.training_mode == 'sequential':
            training_stats = trainer.train_sequential(
                num_iterations_per_submodule=args.iterations_per_submodule,
                log_interval=args.log_interval,
                val_interval=args.val_interval
            )
        else:  # parallel
            training_stats = trainer.train_parallel(
                num_iterations_per_submodule=args.iterations_per_submodule,
                save_interval=args.save_interval
            )
        
        print("Training completed successfully!")
        
        # Save training statistics
        with open(exp_dir / 'training_stats.json', 'w') as f:
            json.dump(training_stats, f, indent=2)
        
    except KeyboardInterrupt:
        print("Training interrupted by user")
    except Exception as e:
        print(f"Training failed with error: {e}")
        raise
    
    # Save final model
    print("Saving final model...")
    if hasattr(trainer, 'save_model'):
        trainer.save_model(str(exp_dir / 'final_model.pth'))
    
    # Save partitioner information
    if hasattr(partitioner, 'get_camera_coverage_stats'):
        coverage_stats = partitioner.get_camera_coverage_stats()
        with open(exp_dir / 'coverage_stats.json', 'w') as f:
            json.dump(coverage_stats, f, indent=2)
    
    # Close wandb
    if args.use_wandb:
        wandb.finish()
    
    print(f"Training completed! Results saved to {exp_dir}")


if __name__ == '__main__':
    main() 