"""
Visualization utilities for Mip-NeRF

This module contains functions for visualizing training progress, rendering results, and debugging Mip-NeRF models.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Any
import cv2
import imageio
from pathlib import Path


def visualize_rays(
    origins: torch.Tensor,
    directions: torch.Tensor,
    t_vals: torch.Tensor,
    num_rays: int = 10,
)
    """
    Visualize ray sampling for debugging
    
    Args:
        origins: [..., 3] ray origins
        directions: [..., 3] ray directions
        t_vals: [..., num_samples] t values along rays
        num_rays: Number of rays to visualize
    """
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Select subset of rays
    indices = torch.randperm(origins.shape[0])[:num_rays]
    origins_sub = origins[indices].cpu().numpy()
    directions_sub = directions[indices].cpu().numpy()
    t_vals_sub = t_vals[indices].cpu().numpy()
    
    colors = plt.cm.viridis(np.linspace(0, 1, num_rays))
    
    for i in range(num_rays):
        origin = origins_sub[i]
        direction = directions_sub[i]
        t_values = t_vals_sub[i]
        
        # Compute sample points along ray
        points = origin[None, :] + direction[None, :] * t_values[:, None]
        
        # Plot ray
        ax.plot(points[:, 0], points[:, 1], points[:, 2], color=colors[i], alpha=0.7, linewidth=2)
        
        # Plot origin
        ax.scatter(origin[0], origin[1], origin[2], color=colors[i], s=50, marker='o')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Ray Sampling Visualization')
    plt.show()


def plot_training_curves(
    losses: dict[str,
    list[float]],
    metrics: dict[str,
    list[float]],
    save_path: Optional[str] = None,
)
    """
    Plot training curves for losses and metrics
    
    Args:
        losses: Dictionary of loss curves
        metrics: Dictionary of metric curves
        save_path: Optional path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot losses
    ax = axes[0, 0]
    for name, values in losses.items():
        ax.plot(values, label=name)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss')
    ax.set_title('Training Losses')
    ax.legend()
    ax.grid(True)
    
    # Plot PSNR
    ax = axes[0, 1]
    if 'psnr' in metrics:
        ax.plot(metrics['psnr'], label='PSNR')
    if 'val_psnr' in metrics:
        ax.plot(metrics['val_psnr'], label='Val PSNR')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('PSNR (dB)')
    ax.set_title('PSNR Progress')
    ax.legend()
    ax.grid(True)
    
    # Plot learning rate
    ax = axes[1, 0]
    if 'lr' in metrics:
        ax.plot(metrics['lr'])
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate Schedule')
    ax.grid(True)
    
    # Plot other metrics
    ax = axes[1, 1]
    for name, values in metrics.items():
        if name not in ['psnr', 'val_psnr', 'lr']:
            ax.plot(values, label=name)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Value')
    ax.set_title('Other Metrics')
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def render_video(
    model: torch.nn.Module,
    poses: torch.Tensor,
    focal_length: float,
    image_width: int,
    image_height: int,
    output_path: str,
    fps: int = 30,
)
    """
    Render a video from camera poses
    
    Args:
        model: Trained Mip-NeRF model
        poses: [num_frames, 4, 4] camera poses
        focal_length: Camera focal length
        image_width: Output image width
        image_height: Output image height
        output_path: Path to save the video
        fps: Frames per second
    """
    model.eval()
    frames = []
    
    with torch.no_grad():
        for i, pose in enumerate(poses):
            print(f"Rendering frame {i+1}/{len(poses)}")
            
            # Generate rays for this pose
            origins, directions, radii = generate_rays_from_pose(
                pose, focal_length, image_width, image_height
            )
            
            # Render image
            rendered = render_image(model, origins, directions, radii)
            
            # Convert to numpy and scale to [0, 255]
            image = rendered['rgb'].cpu().numpy()
            image = (np.clip(image, 0, 1) * 255).astype(np.uint8)
            frames.append(image)
    
    # Save video
    imageio.mimsave(output_path, frames, fps=fps)
    print(f"Video saved to {output_path}")


def save_rendered_images(
    images: dict[str,
    torch.Tensor],
    output_dir: str,
    prefix: str = "render",
)
    """
    Save rendered images to disk
    
    Args:
        images: Dictionary containing rendered images
        output_dir: Output directory
        prefix: Filename prefix
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for name, image in images.items():
        if isinstance(image, torch.Tensor):
            image_np = image.cpu().numpy()
        else:
            image_np = image
        
        # Handle different image formats
        if len(image_np.shape) == 3 and image_np.shape[-1] == 3:
            # RGB image
            image_np = np.clip(image_np, 0, 1)
            image_np = (image_np * 255).astype(np.uint8)
            cv2.imwrite(
                str,
            )
        elif len(image_np.shape) == 2:
            # Grayscale image (depth, etc.)
            image_np = np.clip(image_np, 0, 1)
            image_np = (image_np * 255).astype(np.uint8)
            cv2.imwrite(str(output_path / f"{prefix}_{name}.png"), image_np)


def visualize_depth_map(depth: torch.Tensor, title: str = "Depth Map") -> None:
    """
    Visualize depth map with colormap
    
    Args:
        depth: [..., H, W] depth map
        title: Plot title
    """
    if isinstance(depth, torch.Tensor):
        depth_np = depth.cpu().numpy()
    else:
        depth_np = depth
    
    # Squeeze if necessary
    if len(depth_np.shape) > 2:
        depth_np = depth_np.squeeze()
    
    plt.figure(figsize=(10, 8))
    plt.imshow(depth_np, cmap='plasma')
    plt.colorbar(label='Depth')
    plt.title(title)
    plt.axis('off')
    plt.show()


def compare_images(pred: torch.Tensor, target: torch.Tensor, title: str = "Comparison") -> None:
    """
    Compare predicted and target images side by side
    
    Args:
        pred: Predicted image
        target: Target image
        title: Plot title
    """
    if isinstance(pred, torch.Tensor):
        pred_np = pred.cpu().numpy()
    else:
        pred_np = pred
        
    if isinstance(target, torch.Tensor):
        target_np = target.cpu().numpy()
    else:
        target_np = target
    
    # Compute difference
    diff = np.abs(pred_np - target_np)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(np.clip(pred_np, 0, 1))
    axes[0].set_title('Predicted')
    axes[0].axis('off')
    
    axes[1].imshow(np.clip(target_np, 0, 1))
    axes[1].set_title('Target')
    axes[1].axis('off')
    
    axes[2].imshow(diff, cmap='hot')
    axes[2].set_title('Absolute Difference')
    axes[2].axis('off')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def visualize_weights(
    weights: torch.Tensor,
    t_vals: torch.Tensor,
    num_rays: int = 5,
    title: str = "Ray Weights",
)
    """
    Visualize ray weights along depth
    
    Args:
        weights: [..., num_samples] ray weights
        t_vals: [..., num_samples] t values
        num_rays: Number of rays to visualize
        title: Plot title
    """
    if isinstance(weights, torch.Tensor):
        weights_np = weights.cpu().numpy()
        t_vals_np = t_vals.cpu().numpy()
    else:
        weights_np = weights
        t_vals_np = t_vals
    
    # Select random subset of rays
    indices = np.random.choice(weights_np.shape[0], num_rays, replace=False)
    
    plt.figure(figsize=(12, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, num_rays))
    
    for i, idx in enumerate(indices):
        plt.plot(t_vals_np[idx], weights_np[idx], color=colors[i], alpha=0.7, linewidth=2)
    
    plt.xlabel('Depth (t)')
    plt.ylabel('Weight')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_frustum_visualization(
    origins: torch.Tensor,
    directions: torch.Tensor,
    t_vals: torch.Tensor,
    radii: torch.Tensor,
    sample_idx: int = 0,
)
    """
    Visualize conical frustums for a single ray
    
    Args:
        origins: [..., 3] ray origins
        directions: [..., 3] ray directions
        t_vals: [..., num_samples] t values
        radii: [...] pixel radii
        sample_idx: Index of ray to visualize
    """
    origin = origins[sample_idx].cpu().numpy()
    direction = directions[sample_idx].cpu().numpy()
    t_values = t_vals[sample_idx].cpu().numpy()
    radius = radii[sample_idx].cpu().numpy()
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Compute sample points
    points = origin[None, :] + direction[None, :] * t_values[:, None]
    
    # Plot ray
    ax.plot(points[:, 0], points[:, 1], points[:, 2], 'b-', linewidth=2, label='Ray')
    
    # Plot frustum boundaries
    for i, (point, t) in enumerate(zip(points, t_values)):
        # Compute frustum radius at this distance
        frustum_radius = radius * t
        
        # Create circle around the point
        theta = np.linspace(0, 2*np.pi, 20)
        
        # Find perpendicular vectors
        if np.abs(direction[2]) < 0.9:
            v1 = np.cross(direction, [0, 0, 1])
        else:
            v1 = np.cross(direction, [1, 0, 0])
        v1 = v1 / np.linalg.norm(v1)
        v2 = np.cross(direction, v1)
        v2 = v2 / np.linalg.norm(v2)
        
        # Create circle
        circle = point[None, :] + frustum_radius * (
            v1[None, :] * np.cos(theta)[:, None] + 
            v2[None, :] * np.sin(theta)[:, None]
        )
        
        ax.plot(circle[:, 0], circle[:, 1], circle[:, 2], 'r-', alpha=0.5, linewidth=1)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Conical Frustum Visualization')
    ax.legend()
    plt.show()


def generate_rays_from_pose(
    pose: torch.Tensor,
    focal_length: float,
    image_width: int,
    image_height: int,
)
    """
    Generate rays from camera pose (helper function for rendering)
    
    Args:
        pose: [4, 4] camera pose matrix
        focal_length: Camera focal length
        image_width: Image width
        image_height: Image height
        
    Returns:
        Tuple of (origins, directions, radii)
    """
    # Create pixel coordinates
    i, j = torch.meshgrid(
        torch.arange(
            image_width,
            dtype=torch.float32,
        )
    )
    
    # Convert to camera coordinates (assuming centered principal point)
    cx, cy = image_width / 2, image_height / 2
    dirs = torch.stack([
        (i - cx) / focal_length, -(j - cy) / focal_length, -torch.ones_like(i)
    ], dim=-1)
    
    # Transform to world coordinates
    dirs = torch.sum(dirs[..., None, :] * pose[:3, :3], dim=-1)
    origins = pose[:3, 3].expand(dirs.shape)
    
    # Compute pixel radii
    pixel_radius = 1.0 / focal_length * np.sqrt(2) / 2
    radii = torch.full(dirs.shape[:-1], pixel_radius)
    
    return origins, dirs, radii


def render_image(
    model: torch.nn.Module,
    origins: torch.Tensor,
    directions: torch.Tensor,
    radii: torch.Tensor,
    chunk_size: int = 1024,
)
    """
    Render a full image using the model (helper function)
    
    Args:
        model: Trained Mip-NeRF model
        origins: [..., 3] ray origins
        directions: [..., 3] ray directions
        radii: [...] pixel radii
        chunk_size: Number of rays to process at once
        
    Returns:
        Dictionary containing rendered results
    """
    # Flatten spatial dimensions
    orig_shape = origins.shape[:-1]
    origins_flat = origins.reshape(-1, 3)
    directions_flat = directions.reshape(-1, 3)
    radii_flat = radii.reshape(-1)
    
    # Process in chunks
    all_results = []
    for i in range(0, origins_flat.shape[0], chunk_size):
        chunk_origins = origins_flat[i:i+chunk_size]
        chunk_directions = directions_flat[i:i+chunk_size]
        chunk_radii = radii_flat[i:i+chunk_size]
        
        # Render chunk
        with torch.no_grad():
            chunk_results = model(
                chunk_origins,
                chunk_directions,
                chunk_directions,
                near=2.0,
                far=6.0,
                pixel_radius=chunk_radii,
            )
        
        all_results.append(chunk_results)
    
    # Combine results
    combined_results = {}
    for key in all_results[0].keys():
        if key in ['coarse', 'fine']:
            combined_results[key] = {}
            for subkey in all_results[0][key].keys():
                tensors = [result[key][subkey] for result in all_results]
                combined_results[key][subkey] = torch.cat(tensors, dim=0)
        else:
            tensors = [result[key] for result in all_results]
            combined_results[key] = torch.cat(tensors, dim=0)
    
    # Reshape back to image dimensions
    for key in combined_results:
        if key in ['coarse', 'fine']:
            for subkey in combined_results[key]:
                tensor = combined_results[key][subkey]
                if len(tensor.shape) > 1:
                    combined_results[key][subkey] = tensor.reshape(*orig_shape, -1)
                else:
                    combined_results[key][subkey] = tensor.reshape(*orig_shape)
    
    # Return final rendering (prefer fine if available)
    final_key = 'fine' if 'fine' in combined_results else 'coarse'
    return combined_results[final_key] 