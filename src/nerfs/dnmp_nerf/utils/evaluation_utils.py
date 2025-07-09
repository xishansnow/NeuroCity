from __future__ import annotations

from typing import Optional, Union
"""
Evaluation utilities for DNMP.

This module provides functions for model evaluation, metrics computation, and result visualization.
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path

def compute_image_metrics(
    pred_images: torch.Tensor,
    target_images: torch.Tensor
):
    """
    Compute image quality metrics.
    
    Args:
        pred_images: Predicted images [B, H, W, 3] or [H, W, 3]
        target_images: Target images [B, H, W, 3] or [H, W, 3]
        
    Returns:
        Dictionary containing various metrics
    """
    if pred_images.dim() == 3:
        pred_images = pred_images.unsqueeze(0)
        target_images = target_images.unsqueeze(0)
    
    # Ensure values are in [0, 1]
    pred_images = torch.clamp(pred_images, 0, 1)
    target_images = torch.clamp(target_images, 0, 1)
    
    # MSE and PSNR
    mse = F.mse_loss(pred_images, target_images)
    psnr = -10.0 * torch.log10(mse)
    
    # SSIM (using the function from rendering_utils)
    from .rendering_utils import compute_ssim
    ssim = compute_ssim(pred_images, target_images)
    
    # LPIPS (would need external library, placeholder for now)
    lpips = torch.tensor(0.0)  # Placeholder
    
    return {
        'mse': mse.item(), 'psnr': psnr.item(), 'ssim': ssim.item(), 'lpips': lpips.item()
    }

def evaluate_novel_view_synthesis(
    model,
    test_dataloader,
    device: torch.device,
    num_views: int = 10,
    save_dir: Optional[str] = None
):
    """
    Evaluate novel view synthesis quality.
    
    Args:
        model: Trained DNMP model
        test_dataloader: Test data loader
        device: Device for computation
        num_views: Number of views to evaluate
        save_dir: Directory to save rendered images (optional)
        
    Returns:
        Dictionary containing evaluation metrics
    """
    model.eval()
    
    all_metrics = {'mse': [], 'psnr': [], 'ssim': [], 'lpips': []}
    
    with torch.no_grad():
        for i, batch in enumerate(test_dataloader):
            if i >= num_views:
                break
                
            # Move batch to device
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            
            # Render image
            rays_o = batch['rays_o'].view(-1, 3)
            rays_d = batch['rays_d'].view(-1, 3)
            target_image = batch['image']
            
            # Render in chunks to avoid memory issues
            chunk_size = 4096
            rendered_colors = []
            
            for j in range(0, len(rays_o), chunk_size):
                rays_o_chunk = rays_o[j:j+chunk_size]
                rays_d_chunk = rays_d[j:j+chunk_size]
                
                outputs = model(rays_o_chunk, rays_d_chunk)
                rendered_colors.append(outputs['rgb'])
            
            rendered_colors = torch.cat(rendered_colors, dim=0)
            rendered_image = rendered_colors.view(target_image.shape)
            
            # Compute metrics
            metrics = compute_image_metrics(rendered_image, target_image)
            
            for key, value in metrics.items():
                all_metrics[key].append(value)
            
            # Save images if requested
            if save_dir is not None:
                save_path = Path(save_dir)
                save_path.mkdir(parents=True, exist_ok=True)
                
                # Save rendered and target images
                save_image(rendered_image, save_path / f'rendered_{i:03d}.png')
                save_image(target_image, save_path / f'target_{i:03d}.png')
    
    # Average metrics
    avg_metrics = {key: np.mean(values) for key, values in all_metrics.items()}
    std_metrics = {f'{key}_std': np.std(values) for key, values in all_metrics.items()}
    
    return {**avg_metrics, **std_metrics}

def save_image(image: torch.Tensor, filename: str):
    """Save image tensor to file."""
    if image.dim() == 4:
        image = image.squeeze(0)
    
    # Convert to numpy and scale to [0, 255]
    image_np = (torch.clamp(image, 0, 1) * 255).cpu().numpy().astype(np.uint8)
    
    # Convert RGB to BGR for OpenCV
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(filename), image_bgr)

def create_video_from_images(image_dir: str, output_path: str, fps: int = 30):
    """Create video from sequence of images."""
    image_files = sorted(Path(image_dir).glob('*.png'))
    
    if not image_files:
        print(f"No images found in {image_dir}")
        return
    
    # Read first image to get dimensions
    first_image = cv2.imread(str(image_files[0]))
    height, width, _ = first_image.shape
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for image_file in image_files:
        image = cv2.imread(str(image_file))
        video_writer.write(image)
    
    video_writer.release()
    print(f"Video saved to {output_path}")

def visualize_training_progress(log_dict: dict[str, list[float]], save_path: str):
    """Visualize training progress."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Plot training loss
    if 'train_loss' in log_dict:
        axes[0, 0].plot(log_dict['train_loss'])
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].set_xlabel('Iteration')
        axes[0, 0].set_ylabel('Loss')
    
    # Plot validation metrics
    if 'val_psnr' in log_dict:
        axes[0, 1].plot(log_dict['val_psnr'])
        axes[0, 1].set_title('Validation PSNR')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('PSNR (dB)')
    
    if 'val_ssim' in log_dict:
        axes[1, 0].plot(log_dict['val_ssim'])
        axes[1, 0].set_title('Validation SSIM')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('SSIM')
    
    # Plot learning rate
    if 'learning_rate' in log_dict:
        axes[1, 1].plot(log_dict['learning_rate'])
        axes[1, 1].set_title('Learning Rate')
        axes[1, 1].set_xlabel('Iteration')
        axes[1, 1].set_ylabel('LR')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def compute_geometry_metrics(
    pred_mesh_vertices: torch.Tensor,
    target_mesh_vertices: torch.Tensor
):
    """
    Compute geometry reconstruction metrics.
    
    Args:
        pred_mesh_vertices: Predicted mesh vertices [N, 3]
        target_mesh_vertices: Target mesh vertices [M, 3]
        
    Returns:
        Dictionary containing geometry metrics
    """
    from .geometry_utils import chamfer_distance
    
    # Chamfer distance
    cd = chamfer_distance(pred_mesh_vertices, target_mesh_vertices)
    
    # Hausdorff distance (simplified version)
    dist_matrix = torch.cdist(pred_mesh_vertices, target_mesh_vertices)
    hausdorff_dist = max(dist_matrix.min(dim=1)[0].max(), dist_matrix.min(dim=0)[0].max())
    
    return {
        'chamfer_distance': cd.item(), 'hausdorff_distance': hausdorff_dist.item()
    }

def render_depth_map(depth: torch.Tensor, near: float = 0.1, far: float = 100.0) -> torch.Tensor:
    """
    Render depth map as RGB image.
    
    Args:
        depth: Depth values [H, W]
        near: Near clipping plane
        far: Far clipping plane
        
    Returns:
        depth_rgb: Depth visualization [H, W, 3]
    """
    # Normalize depth to [0, 1]
    depth_normalized = (depth - near) / (far - near)
    depth_normalized = torch.clamp(depth_normalized, 0, 1)
    
    # Convert to color using colormap
    depth_colored = plt.cm.viridis(depth_normalized.cpu().numpy())[:, :, :3]
    
    return torch.from_numpy(depth_colored).float()

def create_comparison_grid(
    images_dict: dict[str,
    torch.Tensor],
    titles: Optional[list[str]] = None,
    save_path: Optional[str] = None
):
    """
    Create comparison grid of images.
    
    Args:
        images_dict: Dictionary of images {name: image_tensor}
        titles: list of titles for images
        save_path: Path to save the grid (optional)
        
    Returns:
        grid_image: Combined grid image
    """
    import torchvision.utils as vutils
    
    images = list(images_dict.values())
    if titles is None:
        titles = list(images_dict.keys())
    
    # Ensure all images have same dimensions
    min_h = min(img.shape[-2] for img in images)
    min_w = min(img.shape[-1] for img in images)
    
    resized_images = []
    for img in images:
        if img.dim() == 3:
            img = img.unsqueeze(0)
        resized = F.interpolate(img.permute(0, 3, 1, 2), size=(min_h, min_w), mode='bilinear')
        resized_images.append(resized.permute(0, 2, 3, 1).squeeze(0))
    
    # Create grid
    grid = vutils.make_grid(
        [img.permute(1, 2, 0) for img in resized_images]
    )
    
    grid_image = grid.permute(1, 2, 0)
    
    if save_path is not None:
        save_image(grid_image, save_path)
    
    return grid_image

def benchmark_rendering_speed(
    model,
    test_rays: tuple[torch.Tensor,
    torch.Tensor],
    device: torch.device,
    num_runs: int = 10
):
    """
    Benchmark rendering speed.
    
    Args:
        model: DNMP model
        test_rays: tuple of (ray_origins, ray_directions)
        device: Device for computation
        num_runs: Number of runs for timing
        
    Returns:
        Dictionary containing timing statistics
    """
    model.eval()
    rays_o, rays_d = test_rays
    rays_o, rays_d = rays_o.to(device), rays_d.to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(3):
            _ = model(rays_o, rays_d)
    
    # Timing runs
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            torch.cuda.synchronize() if device.type == 'cuda' else None
            start_time = torch.cuda.Event(enable_timing=True) if device.type == 'cuda' else None
            end_time = torch.cuda.Event(enable_timing=True) if device.type == 'cuda' else None
            
            if device.type == 'cuda':
                start_time.record()
            else:
                import time
                start_time = time.time()
            
            _ = model(rays_o, rays_d)
            
            if device.type == 'cuda':
                end_time.record()
                torch.cuda.synchronize()
                elapsed_time = start_time.elapsed_time(end_time) / 1000.0  # Convert to seconds
            else:
                elapsed_time = time.time() - start_time
            
            times.append(elapsed_time)
    
    # Compute statistics
    times = np.array(times)
    return {
        'mean_time': times.mean(
        )
    }

def evaluate_view_dependent_effects(
    model,
    dataloader,
    device: torch.device,
    fixed_position: torch.Tensor,
    num_views: int = 36
): 
    """
    Evaluate view-dependent rendering effects.
    
    Args:
        model: DNMP model
        dataloader: Data loader
        device: Device for computation
        fixed_position: Fixed 3D position to render from different views
        num_views: Number of different viewing angles
        
    Returns:
        Dictionary containing rendered images from different views
    """
    model.eval()
    
    # Generate camera poses around fixed position
    angles = torch.linspace(0, 2 * np.pi, num_views)
    radius = 5.0  # Distance from fixed position
    
    rendered_views = []
    
    with torch.no_grad():
        for angle in angles:
            # Compute camera position
            cam_x = fixed_position[0] + radius * torch.cos(angle)
            cam_y = fixed_position[1] + radius * torch.sin(angle)
            cam_z = fixed_position[2] + 2.0  # Slightly elevated
            
            camera_pos = torch.tensor([cam_x, cam_y, cam_z], device=device)
            
            # Generate rays looking at fixed position
            look_at = fixed_position
            up = torch.tensor([0, 0, 1], device=device, dtype=torch.float32)
            
            # Create view matrix (simplified)
            view_dir = F.normalize(look_at - camera_pos, dim=0)
            right = F.normalize(torch.cross(view_dir, up), dim=0)
            up_new = torch.cross(right, view_dir)
            
            # Generate rays for small patch
            patch_size = 64
            rays_o = camera_pos.expand(patch_size * patch_size, 3)
            
            # Create ray directions
            i, j = torch.meshgrid(
                torch.linspace(
                    -0.5,
                    0.5,
                    patch_size,
                    device=device,
                )
            )
            
            dirs = torch.stack([
                i.flatten(
                )
            ], dim=1)
            
            rays_d = F.normalize(dirs, dim=1)
            
            # Render
            outputs = model(rays_o, rays_d)
            rendered_image = outputs['rgb'].view(patch_size, patch_size, 3)
            rendered_views.append(rendered_image)
    
    return {
        'views': torch.stack(rendered_views), 'angles': angles
    } 