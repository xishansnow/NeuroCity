#!/usr/bin/env python3
"""
Grid-NeRF Training Script

A standalone script for training Grid-NeRF models with command-line interface.
Supports single-GPU and multi-GPU distributed training.

Usage:
    python train_grid_nerf.py --data_path /path/to/data --output_dir ./outputs
    python train_grid_nerf.py --config configs/urban_scene.yaml
    python -m torch.distributed.launch --nproc_per_node=4 train_grid_nerf.py --distributed
"""

import os
import sys
import argparse
import yaml
import torch
import torch.multiprocessing as mp
from pathlib import Path

# Add the parent directory to the path so we can import grid_nerf
sys.path.append(str(Path(__file__).parent.parent))

from grid_nerf import (
    GridNeRF, GridNeRFConfig, GridNeRFTrainer,
    create_dataset, get_default_config, setup_logging
)
from grid_nerf.trainer import main_worker, setup_distributed_training


def load_config_file(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def save_config_file(config: dict, output_path: str) -> None:
    """Save configuration to YAML file."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def create_default_config_file(output_path: str) -> None:
    """Create a template configuration file."""
    config = {
        # Data configuration
        'data': {
            'data_path': '/path/to/your/dataset',
            'dataset_type': 'GridNeRFDataset',  # or 'KITTI360GridDataset'
            'image_extension': '.png',
            'load_depth': False,
            'train_split': 0.8,
            'val_split': 0.1,
            'test_split': 0.1
        },
        
        # Scene configuration
        'scene': {
            'scene_bounds': {
                'min_bound': [-100, -100, -10],
                'max_bound': [100, 100, 50]
            }
        },
        
        # Grid configuration
        'grid': {
            'grid_levels': 4,
            'base_resolution': 64,
            'resolution_multiplier': 2,
            'grid_feature_dim': 32
        },
        
        # Network architecture
        'network': {
            'density_layers': 3,
            'density_hidden_dim': 256,
            'color_layers': 2,
            'color_hidden_dim': 128,
            'position_encoding_levels': 10,
            'direction_encoding_levels': 4
        },
        
        # Rendering configuration
        'rendering': {
            'num_samples': 64,
            'num_importance_samples': 128,
            'perturb': True,
            'white_background': False,
            'chunk_size': 1024
        },
        
        # Training configuration
        'training': {
            'batch_size': 1024,
            'num_epochs': 200,
            'max_steps': 200000,
            'grid_lr': 0.01,
            'mlp_lr': 0.0005,
            'weight_decay': 1e-6,
            'grad_clip_norm': 1.0,
            'scheduler_type': 'cosine',
            'warmup_steps': 2000
        },
        
        # Loss configuration
        'loss': {
            'color_weight': 1.0,
            'depth_weight': 0.1,
            'grid_regularization_weight': 0.0001
        },
        
        # Evaluation configuration
        'evaluation': {
            'eval_batch_size': 256,
            'eval_every_n_epochs': 5,
            'save_every_n_epochs': 10,
            'render_every_n_epochs': 20
        },
        
        # Logging configuration
        'logging': {
            'log_every_n_steps': 100,
            'use_tensorboard': True,
            'save_debug_images': False
        }
    }
    
    save_config_file(config, output_path)
    print(f"Created template configuration file: {output_path}")
    print("Please edit the configuration file and run again.")


def merge_configs(base_config: dict, override_config: dict) -> dict:
    """Merge configuration dictionaries recursively."""
    result = base_config.copy()
    
    for key, value in override_config.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    
    return result


def config_dict_to_gridnerf_config(config_dict: dict) -> GridNeRFConfig:
    """Convert configuration dictionary to GridNeRFConfig object."""
    # Flatten nested configuration
    flat_config = {}
    
    # Process each section
    for section_name, section_config in config_dict.items():
        if isinstance(section_config, dict):
            flat_config.update(section_config)
        else:
            flat_config[section_name] = section_config
    
    return GridNeRFConfig(**flat_config)


def single_gpu_training(args, config_dict: dict) -> None:
    """Run single GPU training."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Setup logging
    setup_logging(Path(args.output_dir) / "logs" / "training.log")
    
    # Convert config
    config = config_dict_to_gridnerf_config(config_dict)
    
    # Create trainer
    trainer = GridNeRFTrainer(
        config=config,
        output_dir=args.output_dir,
        device=device,
        use_tensorboard=config_dict.get('logging', {}).get('use_tensorboard', True)
    )
    
    # Create datasets
    data_config = config_dict.get('data', {})
    train_dataset = create_dataset(
        data_path=data_config.get('data_path', args.data_path),
        split='train',
        config=config
    )
    
    val_dataset = None
    test_dataset = None
    
    # Create validation dataset if requested
    if config_dict.get('data', {}).get('val_split', 0) > 0:
        try:
            val_dataset = create_dataset(
                data_path=data_config.get('data_path', args.data_path),
                split='val',
                config=config
            )
        except Exception as e:
            print(f"Warning: Could not create validation dataset: {e}")
    
    # Create test dataset if requested
    if config_dict.get('data', {}).get('test_split', 0) > 0:
        try:
            test_dataset = create_dataset(
                data_path=data_config.get('data_path', args.data_path),
                split='test',
                config=config
            )
        except Exception as e:
            print(f"Warning: Could not create test dataset: {e}")
    
    print(f"Training dataset: {len(train_dataset)} samples")
    if val_dataset:
        print(f"Validation dataset: {len(val_dataset)} samples")
    if test_dataset:
        print(f"Test dataset: {len(test_dataset)} samples")
    
    # Start training
    trainer.train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        resume_from=args.resume_from
    )


def multi_gpu_training(args, config_dict: dict) -> None:
    """Run multi-GPU distributed training."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for multi-GPU training")
    
    num_gpus = torch.cuda.device_count()
    if num_gpus < 2:
        print("Warning: Only 1 GPU available, falling back to single GPU training")
        single_gpu_training(args, config_dict)
        return
    
    print(f"Starting distributed training on {num_gpus} GPUs")
    
    # Convert config
    config = config_dict_to_gridnerf_config(config_dict)
    
    # Scale batch size for multiple GPUs
    config.batch_size = config.batch_size * num_gpus
    
    # Prepare data configuration
    data_config = {
        'train_data_path': config_dict.get('data', {}).get('data_path', args.data_path),
        'val_data_path': config_dict.get('data', {}).get('data_path', args.data_path),
        'test_data_path': config_dict.get('data', {}).get('data_path', args.data_path),
        'train_kwargs': {},
        'val_kwargs': {},
        'test_kwargs': {},
        'resume_from': args.resume_from
    }
    
    # Launch distributed training
    mp.spawn(
        main_worker,
        args=(num_gpus, config, args.output_dir, data_config),
        nprocs=num_gpus,
        join=True
    )


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Train Grid-NeRF model")
    
    # Data arguments
    parser.add_argument("--data_path", type=str, help="Path to training data")
    parser.add_argument("--output_dir", type=str, default="./outputs", 
                       help="Output directory for results")
    
    # Configuration arguments
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--create_config", type=str, 
                       help="Create template configuration file and exit")
    
    # Training arguments
    parser.add_argument("--resume_from", type=str, help="Resume training from checkpoint")
    parser.add_argument("--distributed", action="store_true", 
                       help="Use distributed training on multiple GPUs")
    
    # Override arguments
    parser.add_argument("--batch_size", type=int, help="Override batch size")
    parser.add_argument("--num_epochs", type=int, help="Override number of epochs")
    parser.add_argument("--grid_lr", type=float, help="Override grid learning rate")
    parser.add_argument("--mlp_lr", type=float, help="Override MLP learning rate")
    
    # Debug arguments
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--dry_run", action="store_true", 
                       help="Dry run - check configuration without training")
    
    args = parser.parse_args()
    
    # Create template configuration file
    if args.create_config:
        create_default_config_file(args.create_config)
        return
    
    # Load configuration
    if args.config:
        if not os.path.exists(args.config):
            print(f"Error: Configuration file not found: {args.config}")
            return
        config_dict = load_config_file(args.config)
    else:
        # Use default configuration
        config_dict = {
            'data': {'data_path': args.data_path},
            'training': {},
            'logging': {}
        }
        
        # Merge with defaults
        default_config = get_default_config()
        config_dict = merge_configs(default_config, config_dict)
    
    # Apply command-line overrides
    overrides = {}
    if args.batch_size is not None:
        overrides['batch_size'] = args.batch_size
    if args.num_epochs is not None:
        overrides['num_epochs'] = args.num_epochs
    if args.grid_lr is not None:
        overrides['grid_lr'] = args.grid_lr
    if args.mlp_lr is not None:
        overrides['mlp_lr'] = args.mlp_lr
    
    if overrides:
        config_dict = merge_configs(config_dict, overrides)
    
    # Validate configuration
    if not args.data_path and not config_dict.get('data', {}).get('data_path'):
        print("Error: Data path must be specified either via --data_path or in config file")
        return
    
    # Set debug mode
    if args.debug:
        import logging
        logging.basicConfig(level=logging.DEBUG)
        config_dict['logging']['save_debug_images'] = True
    
    # Save final configuration
    os.makedirs(args.output_dir, exist_ok=True)
    save_config_file(config_dict, os.path.join(args.output_dir, 'config.yaml'))
    
    # Dry run - just check configuration
    if args.dry_run:
        print("=== Configuration Check ===")
        print(f"Data path: {config_dict.get('data', {}).get('data_path', args.data_path)}")
        print(f"Output directory: {args.output_dir}")
        print(f"Distributed training: {args.distributed}")
        
        try:
            config = config_dict_to_gridnerf_config(config_dict)
            print("✓ Configuration is valid")
            
            # Check if data path exists
            data_path = config_dict.get('data', {}).get('data_path', args.data_path)
            if data_path and os.path.exists(data_path):
                print("✓ Data path exists")
            else:
                print("✗ Data path does not exist")
                
        except Exception as e:
            print(f"✗ Configuration error: {e}")
        
        return
    
    # Start training
    try:
        if args.distributed:
            multi_gpu_training(args, config_dict)
        else:
            single_gpu_training(args, config_dict)
        
        print("Training completed successfully!")
        
    except KeyboardInterrupt:
        print("Training interrupted by user")
    except Exception as e:
        print(f"Training failed with error: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main() 