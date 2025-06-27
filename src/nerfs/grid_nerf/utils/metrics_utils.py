from typing import Any, Optional, Union
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

def compute_psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> torch.Tensor:
    """
    Compute Peak Signal-to-Noise Ratio (PSNR).
    
    Args:
        pred: Predicted values
        target: Target values
        max_val: Maximum possible value
        
    Returns:
        PSNR value
    """
    mse = torch.mean((pred - target) ** 2)
    
    if mse == 0:
        return torch.tensor(float('inf'))
    
    psnr = 20 * torch.log10(max_val / torch.sqrt(mse))
    
    return psnr

def compute_ssim(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Compute Structural Similarity Index (SSIM) - simplified version.
    
    Args:
        pred: Predicted image [H, W, C] or [B, H, W, C]
        target: Target image [H, W, C] or [B, H, W, C]
        
    Returns:
        SSIM value
    """
    # Convert to grayscale if RGB
    if pred.shape[-1] == 3:
        pred_gray = 0.299 * pred[..., 0] + 0.587 * pred[..., 1] + 0.114 * pred[..., 2]
        target_gray = 0.299 * target[..., 0] + 0.587 * target[..., 1] + 0.114 * target[..., 2]
    else:
        pred_gray = pred.squeeze(-1)
        target_gray = target.squeeze(-1)
    
    # Simple SSIM approximation
    mu1 = torch.mean(pred_gray)
    mu2 = torch.mean(target_gray)
    
    sigma1_sq = torch.var(pred_gray)
    sigma2_sq = torch.var(target_gray)
    sigma12 = torch.mean((pred_gray - mu1) * (target_gray - mu2))
    
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    
    ssim = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / \
           ((mu1**2 + mu2**2 + c1) * (sigma1_sq + sigma2_sq + c2))
    
    return ssim

def compute_lpips(pred: torch.Tensor, target: torch.Tensor, net: str = 'alex') -> torch.Tensor:
    """
    Compute Learned Perceptual Image Patch Similarity (LPIPS).
    Note: This is a simplified version. For full LPIPS, use the official implementation.
    
    Args:
        pred: Predicted image [H, W, C] or [B, H, W, C]
        target: Target image [H, W, C] or [B, H, W, C]
        net: Network type ('alex' or 'vgg')
        
    Returns:
        LPIPS value (simplified MSE-based approximation)
    """
    # This is a simplified version - for proper LPIPS, use the official package
    # Here we provide a basic perceptual loss approximation
    
    if pred.dim() == 3:
        pred = pred.unsqueeze(0)
        target = target.unsqueeze(0)
    
    # Simple gradient-based perceptual approximation
    pred_grad_x = torch.diff(pred, dim=2)
    pred_grad_y = torch.diff(pred, dim=1)
    
    target_grad_x = torch.diff(target, dim=2)
    target_grad_y = torch.diff(target, dim=1)
    
    # Pad to match original size
    pred_grad_x = F.pad(pred_grad_x, (0, 0, 0, 1))
    pred_grad_y = F.pad(pred_grad_y, (0, 0, 1, 0))
    target_grad_x = F.pad(target_grad_x, (0, 0, 0, 1))
    target_grad_y = F.pad(target_grad_y, (0, 0, 1, 0))
    
    # Compute gradient loss (perceptual approximation)
    grad_loss = torch.mean((pred_grad_x - target_grad_x) ** 2) + \
                torch.mean((pred_grad_y - target_grad_y) ** 2)
    
    return grad_loss

def compute_depth_metrics(
    pred_depth: torch.Tensor,
    target_depth: torch.Tensor,
    mask: Optional[torch.Tensor] = None
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
    slice_idx: Optional[int] = None
) -> np.ndarray:
    """
    Visualize 3D grid as 2D slice.
    
    Args:
        grid: Grid tensor [D, H, W, C]
        threshold: Visualization threshold
        slice_axis: Axis to slice (0, 1, or 2)
        slice_idx: Slice index (middle if None)
        
    Returns:
        Visualization as numpy array
    """
    # Compute magnitude
    magnitude = torch.norm(grid, dim=-1)
    
    # Get slice
    if slice_idx is None:
        slice_idx = magnitude.shape[slice_axis] // 2
    
    if slice_axis == 0:
        slice_data = magnitude[slice_idx, :, :]
    elif slice_axis == 1:
        slice_data = magnitude[:, slice_idx, :]
    else:
        slice_data = magnitude[:, :, slice_idx]
    
    # Apply threshold
    slice_data = torch.clamp(slice_data, min=threshold)
    
    # Normalize
    slice_data = (slice_data - slice_data.min()) / (slice_data.max() - slice_data.min() + 1e-8)
    
    return slice_data.cpu().numpy()

def create_error_map(pred: torch.Tensor, target: torch.Tensor, colormap: str = 'hot') -> np.ndarray:
    """
    Create error visualization map.
    
    Args:
        pred: Predicted values [H, W, C]
        target: Target values [H, W, C]
        colormap: Matplotlib colormap name
        
    Returns:
        Error map as numpy array
    """
    # Compute error
    error = torch.mean((pred - target) ** 2, dim=-1)
    
    # Normalize error
    error_norm = (error - error.min()) / (error.max() - error.min() + 1e-8)
    
    # Apply colormap
    cmap = plt.get_cmap(colormap)
    error_map = cmap(error_norm.cpu().numpy())
    
    return error_map

def save_rendering_comparison(
    pred_images: list[torch.Tensor],
    target_images: list[torch.Tensor],
    save_path: str,
    num_samples: int = 8
) -> None:
    """
    Save rendering comparison visualization.
    
    Args:
        pred_images: list of predicted images [H, W, C]
        target_images: list of target images [H, W, C]
        save_path: Path to save visualization
        num_samples: Number of samples to visualize
    """
    num_samples = min(num_samples, len(pred_images))
    
    # Create figure
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4*num_samples))
    
    for i in range(num_samples):
        # Predicted image
        axes[i, 0].imshow(pred_images[i].cpu().numpy())
        axes[i, 0].set_title('Predicted')
        axes[i, 0].axis('off')
        
        # Target image
        axes[i, 1].imshow(target_images[i].cpu().numpy())
        axes[i, 1].set_title('Target')
        axes[i, 1].axis('off')
        
        # Error map
        error_map = create_error_map(pred_images[i], target_images[i])
        axes[i, 2].imshow(error_map)
        axes[i, 2].set_title('Error')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def compute_rendering_statistics(images: torch.Tensor) -> dict[str, float]:
    """
    Compute statistics for rendered images.
    
    Args:
        images: Image tensor [N, H, W, C]
        
    Returns:
        Dictionary of statistics
    """
    stats = {
        'mean': float(torch.mean(images).item()),
        'std': float(torch.std(images).item()),
        'min': float(torch.min(images).item()),
        'max': float(torch.max(images).item()),
        'num_black_pixels': int(torch.sum(torch.all(images < 0.02, dim=-1)).item()),
        'num_white_pixels': int(torch.sum(torch.all(images > 0.98, dim=-1)).item())
    }
    
    return stats

def evaluate_model_performance(
    model: Any,
    test_dataloader: Any,
    device: str = 'cuda',
    save_dir: Optional[str] = None
) -> dict[str, Any]:
    """
    Evaluate model performance on test dataset.
    
    Args:
        model: Neural network model
        test_dataloader: Test data loader
        device: Device to use
        save_dir: Directory to save visualizations (optional)
        
    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()
    metrics = {
        'psnr': [], 'ssim': [], 'lpips': []
    }
    
    with torch.no_grad():
        for batch in test_dataloader:
            # Forward pass
            pred = model(batch)
            
            # Compute metrics
            batch_metrics = compute_novel_view_metrics(pred, batch['target_images'])
            
            # Accumulate metrics
            for key, value in batch_metrics.items():
                metrics[key].append(value)
    
    # Compute mean metrics
    result = {}
    for key, values in metrics.items():
        result[f'{key}_mean'] = float(torch.mean(torch.stack(values)).item())
        result[f'{key}_std'] = float(torch.std(torch.stack(values)).item())
    
    return result

def plot_training_curves(train_losses: list[float], val_losses: list[float], save_path: str) -> None:
    """
    Plot training and validation loss curves.
    
    Args:
        train_losses: list of training losses
        val_losses: list of validation losses
        save_path: Path to save plot
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Progress')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def save_image(image: torch.Tensor, path: str, normalize: bool = True) -> None:
    """
    Save image tensor as file.
    
    Args:
        image: Image tensor [H, W, C]
        path: Save path
        normalize: Whether to normalize values to [0, 1]
    """
    if normalize:
        image = (image - image.min()) / (image.max() - image.min())
    
    image = (image * 255).byte().cpu().numpy()
    Image.fromarray(image).save(path)

def load_image(path: str, to_tensor: bool = True) -> Union[torch.Tensor, np.ndarray]:
    """
    Load image from file.
    
    Args:
        path: Image file path
        to_tensor: Whether to convert to tensor
        
    Returns:
        Image as tensor or numpy array
    """
    image = np.array(Image.open(path))
    
    if to_tensor:
        image = torch.from_numpy(image).float() / 255.0
    
    return image

def create_video_from_images(
    image_dir: str,
    output_path: str,
    fps: int = 30,
    pattern: str = "*.png"
) -> None:
    """
    Create video from sequence of images.
    
    Args:
        image_dir: Directory containing images
        output_path: Output video path
        fps: Frames per second
        pattern: Image filename pattern
    """
    import cv2
    import glob
    
    # Get image files
    image_files = sorted(glob.glob(f"{image_dir}/{pattern}"))
    
    if not image_files:
        raise ValueError(f"No images found in {image_dir} matching pattern {pattern}")
    
    # Read first image to get dimensions
    frame = cv2.imread(image_files[0])
    height, width, channels = frame.shape
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Write frames
    for image_file in image_files:
        frame = cv2.imread(image_file)
        out.write(frame)
    
    out.release()

def create_comparison_grid(
    images: list[torch.Tensor],
    titles: Optional[list[str]] = None,
    save_path: Optional[str] = None,
    grid_size: Optional[tuple[int, int]] = None
) -> Optional[np.ndarray]:
    """
    Create comparison grid of images.
    
    Args:
        images: list of images [H, W, C]
        titles: Optional list of titles
        save_path: Path to save visualization
        grid_size: Grid dimensions (rows, cols)
        
    Returns:
        Grid visualization as numpy array if save_path is None
    """
    if grid_size is None:
        n = len(images)
        cols = int(math.sqrt(n))
        rows = (n + cols - 1) // cols
        grid_size = (rows, cols)
    
    # Create figure
    fig, axes = plt.subplots(grid_size[0], grid_size[1], figsize=(4*grid_size[1], 4*grid_size[0]))
    axes = axes.flatten()
    
    for i, image in enumerate(images):
        if i >= len(axes):
            break
            
        axes[i].imshow(image.cpu().numpy())
        if titles and i < len(titles):
            axes[i].set_title(titles[i])
        axes[i].axis('off')
    
    # Hide empty subplots
    for i in range(len(images), len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
        return None
    else:
        # Convert to numpy array
        fig.canvas.draw()
        grid = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        grid = grid.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close()
        return grid 