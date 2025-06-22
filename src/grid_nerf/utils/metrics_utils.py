"""
Metrics and evaluation utilities for Grid-NeRF.

This module provides utility functions for computing evaluation metrics,
visualization functions, and other evaluation-related operations.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Optional, Union, Dict
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


def compute_ssim(pred: torch.Tensor, 
                target: torch.Tensor) -> torch.Tensor:
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


def compute_lpips(pred: torch.Tensor, 
                 target: torch.Tensor,
                 net: str = 'alex') -> torch.Tensor:
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


def compute_depth_metrics(pred_depth: torch.Tensor, 
                         target_depth: torch.Tensor,
                         mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
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


def compute_novel_view_metrics(pred_images: torch.Tensor,
                              target_images: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    Compute novel view synthesis metrics.
    
    Args:
        pred_images: Predicted images [N, H, W, C]
        target_images: Target images [N, H, W, C]
        
    Returns:
        Dictionary of metrics
    """
    metrics = {
        'psnr': [],
        'ssim': [],
        'lpips': []
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


def visualize_grid(grid: torch.Tensor,
                  threshold: float = 0.01,
                  slice_axis: int = 2,
                  slice_idx: Optional[int] = None) -> np.ndarray:
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


def create_error_map(pred: torch.Tensor, 
                    target: torch.Tensor,
                    colormap: str = 'hot') -> np.ndarray:
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
    error_colored = cmap(error_norm.cpu().numpy())
    
    return error_colored


def save_rendering_comparison(pred_images: List[torch.Tensor],
                            target_images: List[torch.Tensor],
                            save_path: str,
                            num_samples: int = 8):
    """
    Save comparison of rendered images.
    
    Args:
        pred_images: List of predicted images
        target_images: List of target images
        save_path: Path to save comparison
        num_samples: Number of samples to show
    """
    import matplotlib.pyplot as plt
    
    num_samples = min(num_samples, len(pred_images))
    
    fig, axes = plt.subplots(3, num_samples, figsize=(num_samples * 3, 9))
    
    for i in range(num_samples):
        # Predicted image
        pred_img = pred_images[i].cpu().numpy()
        axes[0, i].imshow(pred_img)
        axes[0, i].set_title(f'Predicted {i+1}')
        axes[0, i].axis('off')
        
        # Target image
        target_img = target_images[i].cpu().numpy()
        axes[1, i].imshow(target_img)
        axes[1, i].set_title(f'Target {i+1}')
        axes[1, i].axis('off')
        
        # Error map
        error_map = create_error_map(pred_images[i], target_images[i])
        axes[2, i].imshow(error_map)
        axes[2, i].set_title(f'Error {i+1}')
        axes[2, i].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def compute_rendering_statistics(images: torch.Tensor) -> Dict[str, float]:
    """
    Compute statistics for rendered images.
    
    Args:
        images: Rendered images [N, H, W, C]
        
    Returns:
        Dictionary of statistics
    """
    stats = {}
    
    # Basic statistics
    stats['mean'] = torch.mean(images).item()
    stats['std'] = torch.std(images).item()
    stats['min'] = torch.min(images).item()
    stats['max'] = torch.max(images).item()
    
    # Per-channel statistics
    for c in range(images.shape[-1]):
        channel_data = images[..., c]
        stats[f'mean_ch{c}'] = torch.mean(channel_data).item()
        stats[f'std_ch{c}'] = torch.std(channel_data).item()
    
    # Dynamic range
    stats['dynamic_range'] = stats['max'] - stats['min']
    
    # Histogram statistics
    hist, _ = torch.histogram(images.flatten(), bins=100)
    stats['entropy'] = -torch.sum(hist * torch.log(hist + 1e-8)).item()
    
    return stats


def evaluate_model_performance(model,
                             test_dataloader,
                             device: str = 'cuda',
                             save_dir: Optional[str] = None) -> Dict[str, float]:
    """
    Evaluate model performance on test set.
    
    Args:
        model: Grid-NeRF model
        test_dataloader: Test data loader
        device: Device to use
        save_dir: Directory to save results (optional)
        
    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()
    
    all_metrics = {
        'psnr': [],
        'ssim': [],
        'lpips': []
    }
    
    pred_images = []
    target_images = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_dataloader):
            # Move to device
            ray_origins = batch['ray_origins'].to(device)
            ray_directions = batch['ray_directions'].to(device)
            target_colors = batch['colors'].to(device)
            
            # Forward pass
            outputs = model(ray_origins, ray_directions)
            pred_colors = outputs['colors']
            
            # Reshape for metric computation
            H, W = batch['image_height'], batch['image_width']
            pred_img = pred_colors.view(H, W, 3)
            target_img = target_colors.view(H, W, 3)
            
            # Compute metrics
            psnr = compute_psnr(pred_img, target_img)
            ssim = compute_ssim(pred_img, target_img)
            lpips = compute_lpips(pred_img, target_img)
            
            all_metrics['psnr'].append(psnr)
            all_metrics['ssim'].append(ssim)
            all_metrics['lpips'].append(lpips)
            
            pred_images.append(pred_img)
            target_images.append(target_img)
    
    # Compute final metrics
    final_metrics = {}
    for key, values in all_metrics.items():
        final_metrics[f'{key}_mean'] = torch.mean(torch.stack(values)).item()
        final_metrics[f'{key}_std'] = torch.std(torch.stack(values)).item()
    
    # Save comparison if requested
    if save_dir is not None:
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        comparison_path = os.path.join(save_dir, 'rendering_comparison.png')
        save_rendering_comparison(pred_images, target_images, comparison_path)
        
        # Save metrics
        metrics_path = os.path.join(save_dir, 'metrics.txt')
        with open(metrics_path, 'w') as f:
            for key, value in final_metrics.items():
                f.write(f'{key}: {value:.6f}\n')
    
    return final_metrics


def plot_training_curves(train_losses: List[float],
                        val_losses: List[float],
                        save_path: str):
    """
    Plot training and validation curves.
    
    Args:
        train_losses: Training losses
        val_losses: Validation losses
        save_path: Path to save plot
    """
    plt.figure(figsize=(10, 6))
    
    plt.plot(train_losses, label='Training Loss', alpha=0.7)
    plt.plot(val_losses, label='Validation Loss', alpha=0.7)
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close() 