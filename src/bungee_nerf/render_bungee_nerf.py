"""
Rendering script for BungeeNeRF
"""

import argparse
import os
import sys
import logging
from pathlib import Path

import torch
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm

from .core import BungeeNeRF, BungeeNeRFConfig
from .dataset import BungeeNeRFDataset, GoogleEarthDataset
from .utils import load_bungee_model, compute_psnr, compute_ssim

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Render with BungeeNeRF model")
    
    # Model arguments
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--config", type=str, default=None,
                       help="Path to config file (optional)")
    
    # Data arguments
    parser.add_argument("--data_dir", type=str, required=True,
                       help="Path to dataset directory")
    parser.add_argument("--dataset_type", type=str, default="auto",
                       choices=["auto", "nerf_synthetic", "llff", "google_earth"],
                       help="Dataset type")
    parser.add_argument("--split", type=str, default="test",
                       choices=["train", "val", "test"],
                       help="Dataset split to render")
    parser.add_argument("--img_downscale", type=int, default=1,
                       help="Image downscale factor")
    
    # Rendering arguments
    parser.add_argument("--render_type", type=str, default="test",
                       choices=["test", "train", "video", "spiral"],
                       help="Type of rendering")
    parser.add_argument("--output_dir", type=str, default="./renders",
                       help="Output directory for rendered images")
    parser.add_argument("--save_depth", action="store_true",
                       help="Save depth maps")
    parser.add_argument("--save_weights", action="store_true",
                       help="Save attention weights")
    
    # Video rendering arguments
    parser.add_argument("--video_fps", type=int, default=30,
                       help="FPS for video rendering")
    parser.add_argument("--video_frames", type=int, default=120,
                       help="Number of frames for video")
    
    # Spiral rendering arguments
    parser.add_argument("--spiral_radius", type=float, default=1.0,
                       help="Radius for spiral camera path")
    parser.add_argument("--spiral_height_variation", type=float, default=0.5,
                       help="Height variation for spiral path")
    
    # Evaluation arguments
    parser.add_argument("--compute_metrics", action="store_true",
                       help="Compute evaluation metrics")
    parser.add_argument("--save_metrics", action="store_true",
                       help="Save metrics to file")
    
    # Device arguments
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use for rendering")
    parser.add_argument("--chunk_size", type=int, default=1024,
                       help="Chunk size for rendering (to avoid OOM)")
    
    return parser.parse_args()


def load_model(checkpoint_path: str, device: str):
    """Load BungeeNeRF model from checkpoint"""
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get config
    if 'config' in checkpoint:
        config_dict = checkpoint['config']
        config = BungeeNeRFConfig(**config_dict)
    else:
        # Use default config
        logger.warning("No config found in checkpoint, using default")
        config = BungeeNeRFConfig()
    
    # Create model
    model = BungeeNeRF(config)
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Set progressive stage
    if 'stage' in checkpoint:
        model.set_current_stage(checkpoint['stage'])
        logger.info(f"Set model to stage {checkpoint['stage']}")
    
    logger.info(f"Loaded model from {checkpoint_path}")
    
    return model, config


def create_dataset(args, split: str):
    """Create dataset for rendering"""
    
    if args.dataset_type == "google_earth":
        dataset_class = GoogleEarthDataset
    else:
        dataset_class = BungeeNeRFDataset
    
    dataset = dataset_class(
        data_dir=args.data_dir,
        split=split,
        img_downscale=args.img_downscale
    )
    
    return dataset


def render_image(
    model: BungeeNeRF,
    rays_o: torch.Tensor,
    rays_d: torch.Tensor,
    bounds: torch.Tensor,
    distances: torch.Tensor,
    chunk_size: int = 1024,
    device: str = "cuda"
) -> dict:
    """
    Render a single image
    
    Args:
        model: BungeeNeRF model
        rays_o: Ray origins [H*W, 3]
        rays_d: Ray directions [H*W, 3]
        bounds: Near/far bounds [H*W, 2]
        distances: Distance to camera [H*W]
        chunk_size: Chunk size for processing
        device: Device to use
        
    Returns:
        Dictionary of rendered outputs
    """
    model.eval()
    
    with torch.no_grad():
        # Move to device
        rays_o = rays_o.to(device)
        rays_d = rays_d.to(device)
        bounds = bounds.to(device)
        distances = distances.to(device)
        
        # Initialize outputs
        all_outputs = {}
        
        # Process in chunks to avoid OOM
        for i in range(0, rays_o.shape[0], chunk_size):
            end_i = min(i + chunk_size, rays_o.shape[0])
            
            # Get chunk
            chunk_rays_o = rays_o[i:end_i]
            chunk_rays_d = rays_d[i:end_i]
            chunk_bounds = bounds[i:end_i]
            chunk_distances = distances[i:end_i]
            
            # Forward pass
            chunk_outputs = model(chunk_rays_o, chunk_rays_d, chunk_bounds, chunk_distances)
            
            # Store outputs
            for key, value in chunk_outputs.items():
                if key not in all_outputs:
                    all_outputs[key] = []
                all_outputs[key].append(value.cpu())
        
        # Concatenate all chunks
        for key in all_outputs:
            all_outputs[key] = torch.cat(all_outputs[key], dim=0)
    
    return all_outputs


def render_test_images(model, dataset, args):
    """Render test images"""
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    metrics = {"psnr": [], "ssim": []}
    
    logger.info(f"Rendering {len(dataset)} test images...")
    
    for i in tqdm(range(len(dataset))):
        # Get data
        data = dataset[i]
        
        # Get rays
        rays_o = data["rays_o"].reshape(-1, 3)
        rays_d = data["rays_d"].reshape(-1, 3)
        bounds = data["bounds"].reshape(-1, 2)
        distances = data["distance"].expand(rays_o.shape[0])
        
        # Render
        outputs = render_image(
            model, rays_o, rays_d, bounds, distances, 
            args.chunk_size, args.device
        )
        
        # Reshape to image
        H, W = data["image"].shape[:2]
        rgb_pred = outputs["rgb"].reshape(H, W, 3).numpy()
        rgb_pred = np.clip(rgb_pred, 0, 1)
        
        # Save image
        img_path = os.path.join(args.output_dir, f"render_{i:04d}.png")
        Image.fromarray((rgb_pred * 255).astype(np.uint8)).save(img_path)
        
        # Save depth if requested
        if args.save_depth and "depth" in outputs:
            depth = outputs["depth"].reshape(H, W).numpy()
            depth_normalized = (depth - depth.min()) / (depth.max() - depth.min())
            depth_path = os.path.join(args.output_dir, f"depth_{i:04d}.png")
            Image.fromarray((depth_normalized * 255).astype(np.uint8)).save(depth_path)
        
        # Compute metrics if requested
        if args.compute_metrics:
            rgb_gt = data["image"].numpy()
            
            psnr = compute_psnr(torch.from_numpy(rgb_pred), torch.from_numpy(rgb_gt))
            ssim = compute_ssim(torch.from_numpy(rgb_pred), torch.from_numpy(rgb_gt))
            
            metrics["psnr"].append(psnr)
            metrics["ssim"].append(ssim)
            
            logger.info(f"Image {i}: PSNR={psnr:.2f}, SSIM={ssim:.4f}")
    
    # Save metrics
    if args.compute_metrics:
        avg_psnr = np.mean(metrics["psnr"])
        avg_ssim = np.mean(metrics["ssim"])
        
        logger.info(f"Average PSNR: {avg_psnr:.2f}")
        logger.info(f"Average SSIM: {avg_ssim:.4f}")
        
        if args.save_metrics:
            metrics_path = os.path.join(args.output_dir, "metrics.txt")
            with open(metrics_path, 'w') as f:
                f.write(f"Average PSNR: {avg_psnr:.2f}\n")
                f.write(f"Average SSIM: {avg_ssim:.4f}\n")
                f.write(f"Per-image PSNR: {metrics['psnr']}\n")
                f.write(f"Per-image SSIM: {metrics['ssim']}\n")


def create_spiral_path(dataset, num_frames: int, radius: float, height_variation: float):
    """Create spiral camera path"""
    
    # Get dataset bounds
    poses = np.stack([dataset.poses[i] for i in range(len(dataset))], axis=0)
    
    # Compute scene center and radius
    positions = poses[:, :3, 3]
    center = positions.mean(axis=0)
    scene_radius = np.linalg.norm(positions - center, axis=1).max()
    
    # Create spiral path
    spiral_poses = []
    
    for i in range(num_frames):
        # Angle for spiral
        angle = 2 * np.pi * i / num_frames
        
        # Position on spiral
        x = center[0] + radius * scene_radius * np.cos(angle)
        y = center[1] + radius * scene_radius * np.sin(angle)
        z = center[2] + height_variation * scene_radius * np.sin(2 * angle)
        
        # Look at center
        position = np.array([x, y, z])
        forward = center - position
        forward = forward / np.linalg.norm(forward)
        
        # Create up vector
        up = np.array([0, 0, 1])
        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)
        up = np.cross(right, forward)
        
        # Create pose matrix
        pose = np.eye(4)
        pose[:3, 0] = right
        pose[:3, 1] = up
        pose[:3, 2] = -forward
        pose[:3, 3] = position
        
        spiral_poses.append(pose)
    
    return spiral_poses


def render_spiral_video(model, dataset, args):
    """Render spiral video"""
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create spiral path
    spiral_poses = create_spiral_path(
        dataset, args.video_frames, args.spiral_radius, args.spiral_height_variation
    )
    
    logger.info(f"Rendering {len(spiral_poses)} frames for spiral video...")
    
    # Render frames
    frames = []
    
    for i, pose in enumerate(tqdm(spiral_poses)):
        # Create rays for this pose
        H, W = dataset.H, dataset.W
        
        # Create pixel coordinates
        i_coords, j_coords = np.meshgrid(
            np.arange(W, dtype=np.float32),
            np.arange(H, dtype=np.float32),
            indexing='xy'
        )
        
        # Convert to camera coordinates
        dirs = np.stack([
            (i_coords - W * 0.5) / dataset.focal,
            -(j_coords - H * 0.5) / dataset.focal,
            -np.ones_like(i_coords)
        ], axis=-1)
        
        # Transform to world coordinates
        rays_d = np.sum(dirs[..., None, :] * pose[:3, :3], axis=-1)
        rays_o = np.broadcast_to(pose[:3, -1], rays_d.shape)
        
        # Flatten
        rays_o = rays_o.reshape(-1, 3)
        rays_d = rays_d.reshape(-1, 3)
        
        # Create bounds
        near, far = dataset.near_far
        bounds = np.full((rays_o.shape[0], 2), [near, far], dtype=np.float32)
        
        # Compute distances
        distances = np.linalg.norm(pose[:3, 3]) * np.ones(rays_o.shape[0])
        
        # Convert to tensors
        rays_o = torch.from_numpy(rays_o)
        rays_d = torch.from_numpy(rays_d)
        bounds = torch.from_numpy(bounds)
        distances = torch.from_numpy(distances)
        
        # Render
        outputs = render_image(
            model, rays_o, rays_d, bounds, distances,
            args.chunk_size, args.device
        )
        
        # Reshape to image
        rgb = outputs["rgb"].reshape(H, W, 3).numpy()
        rgb = np.clip(rgb, 0, 1)
        
        # Convert to uint8
        frame = (rgb * 255).astype(np.uint8)
        frames.append(frame)
        
        # Save individual frame
        frame_path = os.path.join(args.output_dir, f"frame_{i:04d}.png")
        Image.fromarray(frame).save(frame_path)
    
    # Create video
    video_path = os.path.join(args.output_dir, "spiral_video.mp4")
    
    # Use OpenCV to create video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(video_path, fourcc, args.video_fps, (W, H))
    
    for frame in frames:
        # Convert RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        video_writer.write(frame_bgr)
    
    video_writer.release()
    logger.info(f"Spiral video saved to {video_path}")


def main():
    """Main rendering function"""
    args = parse_args()
    
    # Check device
    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU")
        args.device = "cpu"
    
    logger.info(f"Using device: {args.device}")
    
    # Load model
    logger.info("Loading model...")
    model, config = load_model(args.checkpoint, args.device)
    
    # Create dataset
    logger.info("Loading dataset...")
    dataset = create_dataset(args, args.split)
    logger.info(f"Dataset: {len(dataset)} images")
    
    # Render based on type
    if args.render_type == "test":
        render_test_images(model, dataset, args)
    elif args.render_type == "spiral":
        render_spiral_video(model, dataset, args)
    elif args.render_type == "video":
        # Use dataset poses for video
        logger.info("Rendering video from dataset poses...")
        render_test_images(model, dataset, args)
        
        # Create video from rendered images
        video_path = os.path.join(args.output_dir, "dataset_video.mp4")
        
        # Get all rendered images
        img_files = sorted([f for f in os.listdir(args.output_dir) if f.startswith("render_")])
        
        if img_files:
            # Read first image to get dimensions
            first_img = cv2.imread(os.path.join(args.output_dir, img_files[0]))
            H, W = first_img.shape[:2]
            
            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(video_path, fourcc, args.video_fps, (W, H))
            
            for img_file in img_files:
                img_path = os.path.join(args.output_dir, img_file)
                frame = cv2.imread(img_path)
                video_writer.write(frame)
            
            video_writer.release()
            logger.info(f"Dataset video saved to {video_path}")
    
    logger.info("Rendering completed!")


if __name__ == "__main__":
    main()
