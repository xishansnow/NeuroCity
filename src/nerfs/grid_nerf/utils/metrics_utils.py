from __future__ import annotations

from typing import Any, Optional, Union, Sequence, Mapping
"""
Metrics and evaluation utilities for Grid-NeRF.

This module provides utility functions for computing evaluation metrics, visualization functions, and other evaluation-related operations.
"""

import torch
import torch.nn.functional as F
import numpy as np
import math
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

def compute_psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> float:
    """
    Compute Peak Signal-to-Noise Ratio (PSNR) between predicted and target images.
    
    Args:
        pred: Predicted image tensor [H, W, 3] or [B, H, W, 3]
        target: Target image tensor [H, W, 3] or [B, H, W, 3]  
        max_val: Maximum pixel value (1.0 for normalized images)
        
    Returns:
        PSNR value in dB
    """
    mse = F.mse_loss(pred, target)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(max_val / torch.sqrt(mse)).item()

def create_gaussian_window(window_size: int, sigma: float) -> torch.Tensor:
    """Create a 2D Gaussian window for SSIM computation."""
    gauss = torch.Tensor([
        np.exp(-(x - window_size//2)**2 / float(2*sigma**2)) 
        for x in range(window_size)
    ])
    gauss = gauss / gauss.sum()
    window_2d = gauss.unsqueeze(1) @ gauss.unsqueeze(0)
    return window_2d

def compute_ssim(
    pred: torch.Tensor, target: torch.Tensor, window_size: int = 11, data_range: float = 1.0
) -> float:
    """
    Compute Structural Similarity Index (SSIM) between predicted and target images.
    
    Args:
        pred: Predicted image tensor [H, W, 3] or [B, H, W, 3]
        target: Target image tensor [H, W, 3] or [B, H, W, 3]
        window_size: Size of the sliding window
        data_range: Dynamic range of the images
        
    Returns:
        SSIM value
    """
    # Ensure images are in [B, C, H, W] format
    if pred.dim() == 3:
        pred = pred.permute(2, 0, 1).unsqueeze(0)  # [H, W, C] -> [1, C, H, W]
        target = target.permute(2, 0, 1).unsqueeze(0)
    elif pred.dim() == 4 and pred.shape[-1] == 3:
        pred = pred.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]
        target = target.permute(0, 3, 1, 2)
    
    # Convert to grayscale for SSIM computation
    pred_gray = 0.299 * pred[:, 0] + 0.587 * pred[:, 1] + 0.114 * pred[:, 2]
    target_gray = 0.299 * target[:, 0] + 0.587 * target[:, 1] + 0.114 * target[:, 2]
    
    pred_gray = pred_gray.unsqueeze(1)  # [B, 1, H, W]
    target_gray = target_gray.unsqueeze(1)
    
    # SSIM constants
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    
    # Create Gaussian window
    sigma = 1.5
    window = create_gaussian_window(window_size, sigma).to(pred.device)
    window = window.expand(1, 1, window_size, window_size)
    
    mu1 = F.conv2d(pred_gray, window, padding=window_size//2)
    mu2 = F.conv2d(target_gray, window, padding=window_size//2)
    
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = F.conv2d(pred_gray * pred_gray, window, padding=window_size//2) - mu1_sq
    sigma2_sq = F.conv2d(target_gray * target_gray, window, padding=window_size//2) - mu2_sq
    sigma12 = F.conv2d(pred_gray * target_gray, window, padding=window_size//2) - mu1_mu2
    
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return ssim_map.mean().item()

def compute_lpips(pred: torch.Tensor, target: torch.Tensor, net: str = 'alex') -> float:
    """
    Compute LPIPS (Learned Perceptual Image Patch Similarity) metric.
    Note: This requires the lpips package to be installed.
    
    Args:
        pred: Predicted image tensor [H, W, 3] or [B, H, W, 3]
        target: Target image tensor [H, W, 3] or [B, H, W, 3]
        net: Network to use ('alex', 'vgg', 'squeeze')
        
    Returns:
        LPIPS value
    """
    try:
        import lpips
    except ImportError:
        print("Warning: lpips package not installed. Returning 0.0")
        return 0.0
    
    # Initialize LPIPS model (cache it to avoid reloading)
    if not hasattr(compute_lpips, '_lpips_model'):
        compute_lpips._lpips_model = lpips.LPIPS(net=net)
        if pred.is_cuda:
            compute_lpips._lpips_model = compute_lpips._lpips_model.cuda()
    
    lpips_model = compute_lpips._lpips_model
    
    # Ensure images are in [B, C, H, W] format and in range [-1, 1]
    if pred.dim() == 3:
        pred = pred.permute(2, 0, 1).unsqueeze(0)  # [H, W, C] -> [1, C, H, W]
        target = target.permute(2, 0, 1).unsqueeze(0)
    elif pred.dim() == 4 and pred.shape[-1] == 3:
        pred = pred.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]
        target = target.permute(0, 3, 1, 2)
    
    # Convert from [0, 1] to [-1, 1]
    pred = pred * 2.0 - 1.0
    target = target * 2.0 - 1.0
    
    with torch.no_grad():
        lpips_val = lpips_model(pred, target)
    
    return lpips_val.mean().item()

def compute_depth_metrics(
    pred_depth: torch.Tensor,
    target_depth: torch.Tensor,
    mask: torch.Tensor | None = None
) -> dict[str, torch.Tensor]:
    """
    Compute depth estimation metrics.
    
    Args:
        pred_depth: Predicted depth [H, W] or [B, H, W]
        target_depth: Target depth [H, W] or [B, H, W]
        mask: Valid depth mask (optional)
        
    Returns:
        Dictionary of depth metrics
    """
    if mask is not None:
        pred_depth = pred_depth[mask]
        target_depth = target_depth[mask]
    else:
        pred_depth = pred_depth.flatten()
        target_depth = target_depth.flatten()
    
    # Remove invalid depths
    valid_mask = (target_depth > 0) & (pred_depth > 0)
    pred_depth = pred_depth[valid_mask]
    target_depth = target_depth[valid_mask]
    
    if len(pred_depth) == 0:
        return {'abs_rel': torch.tensor(float('inf'))}
    
    abs_rel = torch.mean(torch.abs(pred_depth - target_depth) / target_depth)
    rmse = torch.sqrt(torch.mean((pred_depth - target_depth) ** 2))
    
    return {'abs_rel': abs_rel, 'rmse': rmse}

def compute_novel_view_metrics(
    pred_images: torch.Tensor,
    target_images: torch.Tensor
) -> dict[str, torch.Tensor]:
    """
    Compute novel view synthesis metrics.
    
    Args:
        pred_images: Predicted images [N, H, W, C]
        target_images: Target images [N, H, W, C]
        
    Returns:
        Dictionary of metrics
    """
    metrics = {
        'psnr': [], 'ssim': [], 'lpips': []
    }
    
    for i in range(len(pred_images)):
        # PSNR
        psnr = compute_psnr(pred_images[i], target_images[i])
        metrics['psnr'].append(psnr)
        
        # SSIM
        ssim = compute_ssim(pred_images[i], target_images[i])
        metrics['ssim'].append(ssim)
        
        # LPIPS (simplified)
        lpips = compute_lpips(pred_images[i], target_images[i])
        metrics['lpips'].append(lpips)
    
    # Compute mean metrics
    result = {}
    for key, values in metrics.items():
        result[f'{key}_mean'] = torch.mean(torch.stack(values))
        result[f'{key}_std'] = torch.std(torch.stack(values))
    
    return result

def visualize_grid(
    grid: torch.Tensor,
    threshold: float = 0.01,
    slice_axis: int = 2,
    slice_idx: int | None = None
) -> np.ndarray:
    """
    Visualize a 3D grid by taking a slice along a specified axis.
    
    Args:
        grid: Grid tensor [D, H, W] or [D, H, W, C]
        threshold: Threshold for binary visualization
        slice_axis: Axis to slice along (0=depth, 1=height, 2=width)
        slice_idx: Index to slice at (None for middle slice)
        
    Returns:
        Visualization array [H, W] or [H, W, C]
    """
    if slice_idx is None:
        slice_idx = grid.shape[slice_axis] // 2
    
    if grid.dim() == 3:
        if slice_axis == 0:
            vis = grid[slice_idx]
        elif slice_axis == 1:
            vis = grid[:, slice_idx]
        else:
            vis = grid[:, :, slice_idx]
    else:
        if slice_axis == 0:
            vis = grid[slice_idx]
        elif slice_axis == 1:
            vis = grid[:, slice_idx]
        else:
            vis = grid[:, :, slice_idx]
    
    vis = vis.detach().cpu().numpy()
    if vis.ndim == 2:
        vis = (vis > threshold).astype(np.float32)
    else:
        vis = np.clip(vis, 0, 1)
    
    return vis

def create_error_map(pred: torch.Tensor, target: torch.Tensor, colormap: str = 'hot') -> np.ndarray:
    """
    Create an error map visualization between predicted and target images.
    
    Args:
        pred: Predicted image tensor [H, W, 3]
        target: Target image tensor [H, W, 3]
        colormap: Matplotlib colormap name
        
    Returns:
        Error map visualization [H, W, 3]
    """
    error = torch.abs(pred - target).mean(dim=-1)
    error = error.detach().cpu().numpy()
    
    cmap = plt.get_cmap(colormap)
    error_vis = cmap(error / error.max())[:, :, :3]
    
    return error_vis

def save_rendering_comparison(
    pred_images: Sequence[torch.Tensor],
    target_images: Sequence[torch.Tensor],
    save_path: str | Path,
    num_samples: int = 8
) -> None:
    """
    Save a comparison grid of predicted and target images.
    
    Args:
        pred_images: List of predicted image tensors [H, W, 3]
        target_images: List of target image tensors [H, W, 3]
        save_path: Path to save the comparison grid
        num_samples: Number of image pairs to include
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    n = min(len(pred_images), len(target_images), num_samples)
    fig, axes = plt.subplots(2, n, figsize=(3*n, 6))
    
    for i in range(n):
        # Predicted image
        pred = pred_images[i].detach().cpu().numpy()
        axes[0, i].imshow(np.clip(pred, 0, 1))
        axes[0, i].axis('off')
        axes[0, i].set_title('Predicted')
        
        # Target image
        target = target_images[i].detach().cpu().numpy()
        axes[1, i].imshow(np.clip(target, 0, 1))
        axes[1, i].axis('off')
        axes[1, i].set_title('Target')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def compute_rendering_statistics(images: torch.Tensor) -> dict[str, float]:
    """
    Compute statistics of rendered images.
    
    Args:
        images: Image tensor [N, H, W, 3]
        
    Returns:
        Dictionary of statistics
    """
    stats = {}
    
    # Mean brightness
    stats['mean_brightness'] = images.mean().item()
    
    # Contrast (standard deviation)
    stats['contrast'] = images.std().item()
    
    # Color distribution
    stats['mean_r'] = images[..., 0].mean().item()
    stats['mean_g'] = images[..., 1].mean().item()
    stats['mean_b'] = images[..., 2].mean().item()
    
    return stats

def evaluate_model_performance(
    model: torch.nn.Module,
    test_dataloader: torch.utils.data.DataLoader,
    device: str = 'cuda',
    save_dir: str | Path | None = None
) -> dict[str, float]:
    """
    Evaluate model performance on test dataset.
    
    Args:
        model: Neural network model
        test_dataloader: Test data loader
        device: Device to run evaluation on
        save_dir: Optional directory to save visualizations
        
    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()
    metrics = {
        'psnr': [], 'ssim': [], 'lpips': []
    }
    
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
    
    with torch.no_grad():
        for i, batch in enumerate(test_dataloader):
            # Move data to device
            rays = batch['rays'].to(device)
            target = batch['target'].to(device)
            
            # Forward pass
            output = model(rays)
            pred = output['rgb']
            
            # Compute metrics
            metrics['psnr'].append(compute_psnr(pred, target))
            metrics['ssim'].append(compute_ssim(pred, target))
            metrics['lpips'].append(compute_lpips(pred, target))
            
            # Save visualizations
            if save_dir is not None and i < 10:
                save_path = save_dir / f'comparison_{i:03d}.png'
                save_rendering_comparison([pred], [target], save_path)
    
    # Average metrics
    return {k: sum(v) / len(v) for k, v in metrics.items()}

def plot_training_curves(
    train_losses: Sequence[float],
    val_losses: Sequence[float],
    save_path: str | Path
) -> None:
    """
    Plot training and validation loss curves.
    
    Args:
        train_losses: List of training losses
        val_losses: List of validation losses
        save_path: Path to save the plot
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def save_image(image: torch.Tensor, path: str | Path, normalize: bool = True) -> None:
    """
    Save a tensor image to file.
    
    Args:
        image: Image tensor [H, W, 3] or [C, H, W]
        path: Output file path
        normalize: Whether to normalize to [0, 1] range
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    if isinstance(image, torch.Tensor):
        if image.dim() == 3 and image.shape[0] == 3:  # [C, H, W]
            image = image.permute(1, 2, 0)  # [H, W, C]
        image = image.detach().cpu().numpy()
    
    if normalize:
        image = np.clip(image, 0, 1)
    
    image = (image * 255).astype(np.uint8)
    Image.fromarray(image).save(str(path))

def load_image(path: str | Path, to_tensor: bool = True) -> torch.Tensor | np.ndarray:
    """
    Load an image from file.
    
    Args:
        path: Image file path
        to_tensor: Whether to convert to PyTorch tensor
        
    Returns:
        Image array/tensor [H, W, C] in range [0, 1]
    """
    path = Path(path)
    image = np.array(Image.open(str(path)))
    
    # Convert to float32 and normalize
    image = image.astype(np.float32) / 255.0
    
    if to_tensor:
        return torch.from_numpy(image)
    return image

def create_video_from_images(
    image_dir: str | Path,
    output_path: str | Path,
    fps: int = 30,
    pattern: str = "*.png"
) -> None:
    """
    Create a video from a directory of images.
    
    Args:
        image_dir: Directory containing images
        output_path: Output video file path
        fps: Frames per second
        pattern: Glob pattern for image files
    """
    image_dir = Path(image_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Get sorted list of images
    image_paths = sorted(image_dir.glob(pattern))
    if not image_paths:
        raise ValueError(f"No images found in {image_dir} with pattern {pattern}")
    
    # Read first image to get dimensions
    first_image = cv2.imread(str(image_paths[0]))
    if first_image is None:
        raise ValueError(f"Could not read first image: {image_paths[0]}")
    
    height, width = first_image.shape[:2]
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    try:
        for image_path in image_paths:
            image = cv2.imread(str(image_path))
            if image is not None:
                writer.write(image)
    finally:
        writer.release()
    
    print(f"Created video: {output_path}")

def create_comparison_grid(
    images: Sequence[torch.Tensor],
    titles: Sequence[str] | None = None,
    save_path: str | Path | None = None,
    grid_size: tuple[int, int] | None = None
) -> np.ndarray | None:
    """
    Create a comparison grid of images.
    
    Args:
        images: List of image tensors [H, W, C]
        titles: Optional list of titles for each image
        save_path: Optional path to save the grid
        grid_size: Optional grid dimensions (rows, cols)
        
    Returns:
        Grid array if save_path is None, else None
    """
    n = len(images)
    if grid_size is None:
        cols = int(np.ceil(np.sqrt(n)))
        rows = int(np.ceil(n / cols))
    else:
        rows, cols = grid_size
    
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
    if rows == 1 and cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for i in range(n):
        image = images[i]
        if isinstance(image, torch.Tensor):
            image = image.detach().cpu().numpy()
        
        axes[i].imshow(np.clip(image, 0, 1))
        axes[i].axis('off')
        
        if titles and i < len(titles):
            axes[i].set_title(titles[i])
    
    # Hide empty subplots
    for i in range(n, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        return None
    else:
        return plt.gcf() 