#!/usr/bin/env python3
"""
Rendering script for PyNeRF
"""

import argparse
import os
import sys
import logging
import torch
import numpy as np
from PIL import Image
import cv2
from pathlib import Path
from tqdm import tqdm

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from . import (
    PyNeRF, PyNeRFConfig, PyNeRFDataset, load_pyramid_model, compute_psnr, compute_ssim
)

# Setup logging
logging.basicConfig(
    level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Render PyNeRF model")
    
    # Model arguments
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config file,
    )
    
    # Data arguments
    parser.add_argument("--data_dir", type=str, required=True, help="Path to dataset directory")
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train",
        "val",
        "test"],
        help="Dataset split to render",
    )
    parser.add_argument("--img_downscale", type=int, default=1, help="Image downscale factor")
    parser.add_argument("--white_background", action="store_true", help="Use white background")
    
    # Rendering arguments
    parser.add_argument(
        "--render_mode",
        type=str,
        default="images",
        choices=["images",
        "video",
        "spiral",
        "interpolate"],
        help="Rendering mode",
    )
    parser.add_argument("--output_dir", type=str, default="./renders", help="Output directory")
    parser.add_argument("--chunk_size", type=int, default=1024, help="Chunk size for rendering")
    
    # Video arguments
    parser.add_argument("--fps", type=int, default=30, help="FPS for video rendering")
    parser.add_argument("--video_format", type=str, default="mp4", help="Video format")
    
    # Spiral path arguments
    parser.add_argument("--spiral_radius", type=float, default=3.0, help="Spiral path radius")
    parser.add_argument(
        "--spiral_height",
        type=float,
        default=0.5,
        help="Spiral path height variation",
    )
    parser.add_argument(
        "--num_spiral_frames",
        type=int,
        default=120,
        help="Number of frames in spiral path",
    )
    
    # Evaluation arguments
    parser.add_argument("--compute_metrics", action="store_true", help="Compute evaluation metrics")
    parser.add_argument("--save_depth", action="store_true", help="Save depth maps")
    
    # Device arguments
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    
    return parser.parse_args()


def load_model_and_config(checkpoint_path, device):
    """Load model and configuration from checkpoint"""
    logger.info(f"Loading model from {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config_dict = checkpoint.get('config', {})
    
    # Create config object
    config = PyNeRFConfig(**config_dict)
    
    # Create model
    model = PyNeRF(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    logger.info(f"Model loaded successfully")
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):, }")
    
    return model, config


def render_image(
    model: PyNeRF, rays_o: torch.Tensor, rays_d: torch.Tensor, bounds: torch.Tensor, chunk_size: int = 1024, device: str = "cuda"
) -> dict:
    """
    Render a single image
    
    Args:
        model: PyNeRF model
        rays_o: Ray origins [H, W, 3]
        rays_d: Ray directions [H, W, 3]
        bounds: Near/far bounds [H, W, 2]
        chunk_size: Chunk size for rendering
        device: Device to use
        
    Returns:
        Dictionary containing rendered outputs
    """
    H, W = rays_o.shape[:2]
    
    # Flatten rays
    rays_o_flat = rays_o.view(-1, 3).to(device)
    rays_d_flat = rays_d.view(-1, 3).to(device)
    bounds_flat = bounds.view(-1, 2).to(device)
    
    # Initialize output tensors
    rgb_map = torch.zeros(H * W, 3, device=device)
    depth_map = torch.zeros(H * W, device=device)
    acc_map = torch.zeros(H * W, device=device)
    
    # Render in chunks
    with torch.no_grad():
        for i in tqdm(range(0, H * W, chunk_size), desc="Rendering"):
            end_i = min(i + chunk_size, H * W)
            
            # Get chunk rays
            chunk_rays_o = rays_o_flat[i:end_i]
            chunk_rays_d = rays_d_flat[i:end_i]
            chunk_bounds = bounds_flat[i:end_i]
            
            # Forward pass
            outputs = model(chunk_rays_o, chunk_rays_d, chunk_bounds)
            
            # Store outputs
            rgb_map[i:end_i] = outputs["rgb"]
            if "depth" in outputs:
                depth_map[i:end_i] = outputs["depth"]
            if "acc" in outputs:
                acc_map[i:end_i] = outputs["acc"]
    
    # Reshape to image format
    rgb_image = rgb_map.view(H, W, 3).cpu().numpy()
    depth_image = depth_map.view(H, W).cpu().numpy()
    acc_image = acc_map.view(H, W).cpu().numpy()
    
    return {
        "rgb": rgb_image, "depth": depth_image, "acc": acc_image
    }


def generate_spiral_path(
    poses: np.ndarray, radius: float = 3.0, height: float = 0.5, num_frames: int = 120
) -> np.ndarray:
    """
    Generate spiral camera path
    
    Args:
        poses: Camera poses [N, 4, 4]
        radius: Spiral radius
        height: Height variation
        num_frames: Number of frames
        
    Returns:
        Spiral poses [num_frames, 4, 4]
    """
    # Get scene center and up vector
    center = poses[:, :3, 3].mean(axis=0)
    up = poses[:, :3, 1].mean(axis=0)
    up = up / np.linalg.norm(up)
    
    # Generate spiral path
    spiral_poses = []
    for i in range(num_frames):
        angle = 2 * np.pi * i / num_frames
        
        # Position on spiral
        x = radius * np.cos(angle)
        z = radius * np.sin(angle)
        y = height * np.sin(2 * angle)
        
        pos = center + np.array([x, y, z])
        
        # Look at center
        forward = center - pos
        forward = forward / np.linalg.norm(forward)
        
        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)
        
        up_new = np.cross(right, forward)
        
        # Create pose matrix
        pose = np.eye(4)
        pose[:3, 0] = right
        pose[:3, 1] = up_new
        pose[:3, 2] = -forward
        pose[:3, 3] = pos
        
        spiral_poses.append(pose)
    
    return np.stack(spiral_poses, axis=0)


def render_dataset(args, model, config, dataset):
    """Render dataset images"""
    output_dir = os.path.join(args.output_dir, f"{args.split}_images")
    os.makedirs(output_dir, exist_ok=True)
    
    metrics = {"psnr": [], "ssim": []} if args.compute_metrics else None
    
    for idx in tqdm(range(len(dataset)), desc=f"Rendering {args.split} images"):
        # Get data
        data = dataset[idx]
        rays_o = data["rays_o"]
        rays_d = data["rays_d"]
        bounds = data["bounds"]
        
        if args.compute_metrics:
            target_image = data["image"].numpy()
        
        # Render image
        outputs = render_image(
            model, rays_o, rays_d, bounds, chunk_size=args.chunk_size, device=args.device
        )
        
        # Save RGB image
        rgb_image = outputs["rgb"]
        rgb_image = np.clip(rgb_image, 0, 1)
        rgb_pil = Image.fromarray((rgb_image * 255).astype(np.uint8))
        rgb_pil.save(os.path.join(output_dir, f"{idx:04d}.png"))
        
        # Save depth map if requested
        if args.save_depth and "depth" in outputs:
            depth_image = outputs["depth"]
            depth_normalized = (depth_image - depth_image.min()) / (depth_image.max() - depth_image.min())
            depth_pil = Image.fromarray((depth_normalized * 255).astype(np.uint8))
            depth_pil.save(os.path.join(output_dir, f"{idx:04d}_depth.png"))
        
        # Compute metrics if requested
        if args.compute_metrics:
            psnr = compute_psnr(torch.from_numpy(rgb_image), torch.from_numpy(target_image))
            ssim = compute_ssim(torch.from_numpy(rgb_image), torch.from_numpy(target_image))
            
            metrics["psnr"].append(psnr)
            metrics["ssim"].append(ssim)
    
    # Log metrics
    if args.compute_metrics:
        avg_psnr = np.mean(metrics["psnr"])
        avg_ssim = np.mean(metrics["ssim"])
        
        logger.info(f"Average PSNR: {avg_psnr:.4f}")
        logger.info(f"Average SSIM: {avg_ssim:.4f}")
        
        # Save metrics
        with open(os.path.join(output_dir, "metrics.txt"), 'w') as f:
            f.write(f"Average PSNR: {avg_psnr:.4f}\n")
            f.write(f"Average SSIM: {avg_ssim:.4f}\n")
            f.write(f"Per-image PSNR: {metrics['psnr']}\n")
            f.write(f"Per-image SSIM: {metrics['ssim']}\n")


def render_spiral(args, model, config, dataset):
    """Render spiral video"""
    output_dir = os.path.join(args.output_dir, "spiral")
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate spiral path
    poses = np.stack([dataset.poses[i] for i in range(len(dataset))], axis=0)
    spiral_poses = generate_spiral_path(
        poses, radius=args.spiral_radius, height=args.spiral_height, num_frames=args.num_spiral_frames
    )
    
    # Get camera parameters
    focal = dataset.focal
    H, W = dataset.H, dataset.W
    
    frames = []
    
    for idx, pose in enumerate(tqdm(spiral_poses, desc="Rendering spiral")):
        # Generate rays for this pose
        i, j = np.meshgrid(
            np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy'
        )
        
        dirs = np.stack([
            (i - W * 0.5) / focal, -(j - H * 0.5) / focal, -np.ones_like(i)
        ], axis=-1)
        
        rays_d = np.sum(dirs[..., None, :] * pose[:3, :3], axis=-1)
        rays_o = np.broadcast_to(pose[:3, -1], rays_d.shape)
        
        rays_o = torch.from_numpy(rays_o)
        rays_d = torch.from_numpy(rays_d)
        
        # Create bounds
        near, far = dataset.near_far
        bounds = torch.tensor([near, far], dtype=torch.float32)
        bounds = bounds.expand(H, W, 2)
        
        # Render frame
        outputs = render_image(
            model, rays_o, rays_d, bounds, chunk_size=args.chunk_size, device=args.device
        )
        
        # Convert to image
        rgb_image = outputs["rgb"]
        rgb_image = np.clip(rgb_image, 0, 1)
        frame = (rgb_image * 255).astype(np.uint8)
        frames.append(frame)
        
        # Save individual frame
        frame_pil = Image.fromarray(frame)
        frame_pil.save(os.path.join(output_dir, f"{idx:04d}.png"))
    
    # Create video
    if args.render_mode == "video":
        video_path = os.path.join(output_dir, f"spiral.{args.video_format}")
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(video_path, fourcc, args.fps, (W, H))
        
        for frame in frames:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            video_writer.write(frame_bgr)
        
        video_writer.release()
        logger.info(f"Video saved to {video_path}")


def main():
    """Main rendering function"""
    args = parse_args()
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load model and config
    model, config = load_model_and_config(args.checkpoint, device)
    
    # Create dataset
    logger.info("Loading dataset...")
    dataset = PyNeRFDataset(
        data_dir=args.data_dir, split=args.split, img_downscale=args.img_downscale, white_background=args.white_background
    )
    logger.info(f"Dataset: {len(dataset)} images")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Render based on mode
    if args.render_mode == "images":
        render_dataset(args, model, config, dataset)
    elif args.render_mode in ["video", "spiral"]:
        render_spiral(args, model, config, dataset)
    else:
        raise ValueError(f"Unknown render mode: {args.render_mode}")
    
    logger.info("Rendering completed!")


if __name__ == "__main__":
    main()
