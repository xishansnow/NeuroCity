"""
Plenoxels Example Usage

This script demonstrates how to use the Plenoxels implementation for training
and inference on various datasets.
"""

import os
import argparse
import torch
import numpy as np
import logging
from pathlib import Path

from .core import PlenoxelConfig, PlenoxelModel, PlenoxelLoss
from .dataset import PlenoxelDatasetConfig, create_plenoxel_dataset
from .trainer import PlenoxelTrainerConfig, create_plenoxel_trainer

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_plenoxel(args):
    """Train a Plenoxel model."""
    logger.info("Starting Plenoxel training...")
    
    # Model configuration
    model_config = PlenoxelConfig(
        grid_resolution=(
            args.resolution,
            args.resolution,
            args.resolution,
        )
    )
    
    # Dataset configuration
    dataset_config = create_plenoxel_dataset(
        data_dir=args.data_dir, dataset_type=args.dataset_type, downsample_factor=args.downsample, white_background=args.white_background, num_rays_train=args.num_rays, near=args.near, far=args.far
    )
    
    # Trainer configuration
    trainer_config = PlenoxelTrainerConfig(
        max_epochs=args.max_epochs, learning_rate=args.learning_rate, weight_decay=args.weight_decay, color_loss_weight=1.0, tv_loss_weight=args.tv_lambda, l1_loss_weight=args.l1_lambda, pruning_threshold=args.sparsity_threshold, pruning_interval=args.pruning_interval, eval_interval=args.eval_interval, save_interval=args.save_interval, log_interval=args.log_interval, experiment_name=args.experiment_name, output_dir=args.output_dir, resume_from=args.resume_from
    )
    
    # Create trainer and start training
    trainer = create_plenoxel_trainer(model_config, trainer_config, dataset_config)
    trainer.train()
    
    logger.info("Training completed!")


def render_plenoxel(args):
    """Render novel views using a trained Plenoxel model."""
    logger.info(f"Rendering novel views from {args.checkpoint}")
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    model_config = checkpoint['model_config']
    model = PlenoxelModel(model_config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Create dataset for rendering
    dataset_config = create_plenoxel_dataset(
        data_dir=args.data_dir, dataset_type=args.dataset_type, downsample_factor=args.downsample
    )
    
    # Generate novel view poses
    if args.render_poses is None:
        # Generate spiral poses
        from .dataset import PlenoxelDataset
        dataset = PlenoxelDataset(dataset_config, split='test')
        poses = dataset.get_test_poses(n_poses=args.num_renders)
    else:
        poses = np.load(args.render_poses)
    
    # Render novel views
    output_dir = args.render_output_dir or os.path.join(args.output_dir, 'renders')
    os.makedirs(output_dir, exist_ok=True)
    
    rendered_images = []
    
    with torch.no_grad():
        for i, pose in enumerate(poses):
            logger.info(f"Rendering view {i+1}/{len(poses)}")
            
            # Generate rays for this pose
            H, W = 400, 400  # Default resolution
            focal = 555.0    # Default focal length
            
            pose_tensor = torch.from_numpy(pose).float().to(device)
            
            # Generate rays
            i_coords, j_coords = np.meshgrid(
                np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy'
            )
            
            dirs = np.stack([
                (i_coords - W * 0.5) / focal, -(j_coords - H * 0.5) / focal, -np.ones_like(i_coords)
            ], -1)
            
            rays_d = torch.from_numpy(dirs @ pose[:3, :3].T).float().to(device)
            rays_o = torch.from_numpy(
                np.broadcast_to,
            )
            
            # Render in chunks
            chunk_size = 1024
            rgb_chunks = []
            
            rays_o_flat = rays_o.view(-1, 3)
            rays_d_flat = rays_d.view(-1, 3)
            
            for j in range(0, len(rays_o_flat), chunk_size):
                rays_o_chunk = rays_o_flat[j:j+chunk_size]
                rays_d_chunk = rays_d_flat[j:j+chunk_size]
                
                outputs = model(rays_o_chunk, rays_d_chunk)
                rgb_chunks.append(outputs['rgb'].cpu())
            
            # Combine and reshape
            rgb = torch.cat(rgb_chunks, dim=0).view(H, W, 3)
            rgb_np = rgb.numpy()
            rgb_np = np.clip(rgb_np, 0, 1)
            
            rendered_images.append(rgb_np)
            
            # Save image
            rgb_uint8 = (rgb_np * 255).astype(np.uint8)
            import imageio
            imageio.imwrite(os.path.join(output_dir, f'render_{i:03d}.png'), rgb_uint8)
    
    logger.info(f"Rendered {len(rendered_images)} images to {output_dir}")


def evaluate_plenoxel(args):
    """Evaluate a trained Plenoxel model."""
    logger.info(f"Evaluating model from {args.checkpoint}")
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    model_config = checkpoint['model_config']
    model = PlenoxelModel(model_config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Create dataset
    dataset_config = create_plenoxel_dataset(
        data_dir=args.data_dir, dataset_type=args.dataset_type, downsample_factor=args.downsample
    )
    
    from .dataset import PlenoxelDataset, create_plenoxel_dataloader
    test_dataset = PlenoxelDataset(dataset_config, split='test')
    test_loader = create_plenoxel_dataloader(dataset_config, split='test', shuffle=False)
    
    # Evaluation metrics
    total_psnr = 0.0
    total_ssim = 0.0
    num_images = 0
    
    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            rays_o = batch['rays_o'].squeeze(0)
            rays_d = batch['rays_d'].squeeze(0)
            target_colors = batch['colors'].squeeze(0)
            
            H, W = rays_o.shape[:2]
            
            # Render in chunks
            chunk_size = 1024
            rgb_chunks = []
            
            for i in range(0, H * W, chunk_size):
                rays_o_chunk = rays_o.view(-1, 3)[i:i+chunk_size]
                rays_d_chunk = rays_d.view(-1, 3)[i:i+chunk_size]
                
                outputs = model(rays_o_chunk, rays_d_chunk)
                rgb_chunks.append(outputs['rgb'])
            
            # Combine chunks
            rgb_pred = torch.cat(rgb_chunks, dim=0).view(H, W, 3)
            
            # Compute PSNR
            mse = torch.mean((rgb_pred - target_colors) ** 2)
            psnr = -10.0 * torch.log10(mse)
            
            total_psnr += psnr.item()
            num_images += 1
            
            logger.info(f"Image {num_images}: PSNR = {psnr.item():.2f}")
    
    avg_psnr = total_psnr / num_images
    logger.info(f"Average PSNR: {avg_psnr:.2f}")


def demo_plenoxels() -> None:
    """Demonstrate Plenoxel on a simple scene."""
    logger.info("Running Plenoxel demo...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Model configuration
    model_config = PlenoxelConfig(
        grid_resolution=(64, 64, 64), scene_bounds=(-1.0, -1.0, -1.0, 1.0, 1.0, 1.0), sh_degree=1
    )
    
    # Create model
    model = PlenoxelModel(model_config).to(device)
    
    # Create test rays
    num_rays = 100
    rays_o = torch.randn(num_rays, 3, device=device) * 0.1
    rays_d = torch.randn(num_rays, 3, device=device)
    rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
    
    # Forward pass
    with torch.no_grad():
        outputs = model(rays_o, rays_d, num_samples=64)
    
    logger.info(f"Demo completed successfully!")
    logger.info(f"Output RGB shape: {outputs['rgb'].shape}")
    logger.info(f"Output depth shape: {outputs['depth'].shape}")


def train_plenoxels():
    """Train a simple Plenoxel model."""
    logger.info("Training Plenoxel model...")
    
    # Configuration
    model_config = PlenoxelConfig(
        grid_resolution=(128, 128, 128), sh_degree=2
    )
    
    dataset_config = PlenoxelDatasetConfig(
        data_dir="data/nerf_synthetic/lego", dataset_type="blender", num_rays_train=1024
    )
    
    trainer_config = PlenoxelTrainerConfig(
        max_epochs=1000, learning_rate=0.1, experiment_name="plenoxel_demo"
    )
    
    # Create and run trainer
    trainer = create_plenoxel_trainer(model_config, trainer_config, dataset_config)
    trainer.train()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Plenoxels Example')
    parser.add_argument(
        '--mode',
        type=str,
        choices=['demo',
        'train'],
        default='demo',
        help='Mode to run',
    )
    
    args = parser.parse_args()
    
    if args.mode == 'demo':
        demo_plenoxels()
    elif args.mode == 'train':
        train_plenoxels()


if __name__ == '__main__':
    main() 