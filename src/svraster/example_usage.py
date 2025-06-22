"""
Example usage script for SVRaster.

This script demonstrates how to use the SVRaster package for training
and inference with sparse voxel rasterization.
"""

import os
import argparse
import torch
import logging
from pathlib import Path

from .core import SVRasterConfig, SVRasterModel
from .dataset import SVRasterDatasetConfig, create_svraster_dataset
from .trainer import SVRasterTrainerConfig, create_svraster_trainer

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_sample_config():
    """Create sample configuration for SVRaster."""
    
    # Model configuration
    model_config = SVRasterConfig(
        max_octree_levels=12,
        base_resolution=64,
        scene_bounds=(-2.0, -2.0, -2.0, 2.0, 2.0, 2.0),
        density_activation="exp",
        color_activation="sigmoid",
        sh_degree=2,
        subdivision_threshold=0.01,
        pruning_threshold=0.001,
        ray_samples_per_voxel=8,
        background_color=(0.0, 0.0, 0.0),
        use_view_dependent_color=True,
        use_opacity_regularization=True,
        opacity_reg_weight=0.01
    )
    
    # Dataset configuration
    dataset_config = SVRasterDatasetConfig(
        data_dir="./data/nerf_synthetic/lego",
        images_dir="images",
        dataset_type="blender",
        image_height=800,
        image_width=800,
        downscale_factor=2.0,
        train_split=0.8,
        val_split=0.1,
        test_split=0.1,
        num_rays_train=1024,
        num_rays_val=512,
        white_background=True
    )
    
    # Trainer configuration
    trainer_config = SVRasterTrainerConfig(
        num_epochs=100,
        batch_size=1,
        learning_rate=1e-3,
        weight_decay=1e-4,
        optimizer_type="adam",
        scheduler_type="cosine",
        rgb_loss_weight=1.0,
        enable_subdivision=True,
        subdivision_start_epoch=10,
        subdivision_interval=5,
        enable_pruning=True,
        pruning_start_epoch=20,
        pruning_interval=10,
        val_interval=5,
        log_interval=100,
        save_interval=1000,
        checkpoint_dir="./checkpoints/svraster",
        log_dir="./logs/svraster",
        device="cuda" if torch.cuda.is_available() else "cpu",
        use_mixed_precision=True,
        gradient_clip_norm=1.0,
        render_chunk_size=1024
    )
    
    return model_config, dataset_config, trainer_config


def train_svraster(data_dir: str, output_dir: str):
    """
    Train SVRaster model on a dataset.
    
    Args:
        data_dir: Path to dataset directory
        output_dir: Path to output directory for checkpoints and logs
    """
    logger.info("Starting SVRaster training...")
    
    # Create configurations
    model_config, dataset_config, trainer_config = create_sample_config()
    
    # Update paths
    dataset_config.data_dir = data_dir
    trainer_config.checkpoint_dir = os.path.join(output_dir, "checkpoints")
    trainer_config.log_dir = os.path.join(output_dir, "logs")
    
    # Create output directories
    os.makedirs(trainer_config.checkpoint_dir, exist_ok=True)
    os.makedirs(trainer_config.log_dir, exist_ok=True)
    
    # Create datasets
    logger.info("Loading datasets...")
    train_dataset = create_svraster_dataset(dataset_config, split="train")
    val_dataset = create_svraster_dataset(dataset_config, split="val")
    
    # Create trainer
    logger.info("Creating trainer...")
    trainer = create_svraster_trainer(
        model_config, trainer_config, train_dataset, val_dataset
    )
    
    # Start training
    logger.info("Starting training loop...")
    trainer.train()
    
    logger.info("Training completed!")


def render_svraster(checkpoint_path: str, output_dir: str, dataset_config: SVRasterDatasetConfig):
    """
    Render images using trained SVRaster model.
    
    Args:
        checkpoint_path: Path to model checkpoint
        output_dir: Output directory for rendered images
        dataset_config: Dataset configuration
    """
    logger.info("Loading model for rendering...")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model_config = checkpoint['model_config']
    
    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SVRasterModel(model_config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Create test dataset
    test_dataset = create_svraster_dataset(dataset_config, split="test")
    
    # Render images
    os.makedirs(output_dir, exist_ok=True)
    
    with torch.no_grad():
        for i, batch in enumerate(test_dataset):
            logger.info(f"Rendering image {i+1}/{len(test_dataset)}")
            
            rays_o = batch['rays_o'].to(device)
            rays_d = batch['rays_d'].to(device)
            
            # Render in chunks
            chunk_size = 1024
            num_rays = rays_o.shape[0]
            
            rendered_colors = []
            for j in range(0, num_rays, chunk_size):
                end_j = min(j + chunk_size, num_rays)
                chunk_rays_o = rays_o[j:end_j]
                chunk_rays_d = rays_d[j:end_j]
                
                outputs = model(chunk_rays_o, chunk_rays_d)
                rendered_colors.append(outputs['rgb'].cpu())
            
            # Combine chunks
            rendered_image = torch.cat(rendered_colors, dim=0)
            
            # Reshape to image
            H, W = dataset_config.image_height, dataset_config.image_width
            rendered_image = rendered_image.reshape(H, W, 3)
            
            # Save image
            import numpy as np
            from PIL import Image
            
            image_np = (rendered_image.numpy() * 255).astype(np.uint8)
            image_pil = Image.fromarray(image_np)
            image_pil.save(os.path.join(output_dir, f"rendered_{i:04d}.png"))
    
    logger.info("Rendering completed!")


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="SVRaster Example Usage")
    parser.add_argument("--mode", choices=["train", "render"], required=True,
                       help="Mode: train or render")
    parser.add_argument("--data_dir", required=True,
                       help="Path to dataset directory")
    parser.add_argument("--output_dir", required=True,
                       help="Path to output directory")
    parser.add_argument("--checkpoint", 
                       help="Path to checkpoint file (for rendering)")
    
    args = parser.parse_args()
    
    if args.mode == "train":
        train_svraster(args.data_dir, args.output_dir)
    elif args.mode == "render":
        if not args.checkpoint:
            raise ValueError("Checkpoint path required for rendering")
        
        # Create dataset config for rendering
        _, dataset_config, _ = create_sample_config()
        dataset_config.data_dir = args.data_dir
        
        render_svraster(args.checkpoint, args.output_dir, dataset_config)


if __name__ == "__main__":
    main() 