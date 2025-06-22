"""
Utility functions for PyNeRF
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import logging
import os
import json

logger = logging.getLogger(__name__)


def compute_sample_area(
    rays_o: torch.Tensor,
    rays_d: torch.Tensor,
    z_vals: torch.Tensor,
    cone_angle: float = 0.00628
) -> torch.Tensor:
    """
    Compute sample areas for pyramid level selection
    
    Args:
        rays_o: Ray origins [N_rays, 3]
        rays_d: Ray directions [N_rays, 3]
        z_vals: Sample depths [N_rays, N_samples]
        cone_angle: Cone angle for area computation
        
    Returns:
        Sample areas [N_rays, N_samples]
    """
    # Compute cone radius at each sample point
    # Area grows linearly with distance from camera
    cone_radius = z_vals * cone_angle + 1.0
    
    # Sample area is proportional to radius squared
    sample_areas = np.pi * cone_radius ** 2
    
    return sample_areas


def get_pyramid_level(
    sample_areas: torch.Tensor,
    pyramid_levels: List[int],
    scale_factor: float = 2.0
) -> torch.Tensor:
    """
    Determine pyramid level for each sample based on area
    
    Args:
        sample_areas: Sample areas [N]
        pyramid_levels: List of pyramid level resolutions
        scale_factor: Scale factor between levels
        
    Returns:
        Pyramid levels [N]
    """
    device = sample_areas.device
    
    # Convert areas to level indices
    # Use log scale to determine appropriate level
    log_areas = torch.log2(sample_areas + 1e-8)
    base_log_area = torch.log2(torch.tensor(pyramid_levels[0], device=device))
    
    # Calculate level indices
    level_indices = (log_areas - base_log_area) / np.log2(scale_factor)
    level_indices = torch.clamp(level_indices, 0, len(pyramid_levels) - 1)
    
    return level_indices.long()


def interpolate_pyramid_outputs(
    rgb_outputs: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    sigma_outputs: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    pyramid_levels: torch.Tensor,
    num_points: int,
    device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Interpolate outputs from different pyramid levels
    
    Args:
        rgb_outputs: List of (rgb, mask, weight) tuples
        sigma_outputs: List of (sigma, mask, weight) tuples
        pyramid_levels: Level indices for each point
        num_points: Total number of points
        device: Device for computations
        
    Returns:
        Tuple of (final_rgb, final_sigma)
    """
    # Initialize output tensors
    final_rgb = torch.zeros(num_points, 3, device=device)
    final_sigma = torch.zeros(num_points, device=device)
    
    # Aggregate outputs from all levels
    for (rgb, mask, weight), (sigma, sigma_mask, sigma_weight) in zip(rgb_outputs, sigma_outputs):
        final_rgb[mask] += rgb * weight.unsqueeze(-1)
        final_sigma[mask] += sigma * sigma_weight
    
    return final_rgb, final_sigma


def create_pyramid_hierarchy(
    num_levels: int = 8,
    base_resolution: int = 16,
    scale_factor: float = 2.0,
    max_resolution: int = 2048
) -> List[int]:
    """
    Create pyramid hierarchy of resolutions
    
    Args:
        num_levels: Number of pyramid levels
        base_resolution: Base resolution
        scale_factor: Scale factor between levels
        max_resolution: Maximum resolution
        
    Returns:
        List of resolutions for each level
    """
    resolutions = []
    for level in range(num_levels):
        resolution = min(
            base_resolution * (scale_factor ** level),
            max_resolution
        )
        resolutions.append(int(resolution))
    
    return resolutions


def save_pyramid_model(
    model: torch.nn.Module,
    config: dict,
    save_path: str,
    epoch: Optional[int] = None,
    optimizer_state: Optional[dict] = None
) -> None:
    """
    Save PyNeRF model and configuration
    
    Args:
        model: PyNeRF model
        config: Model configuration
        save_path: Path to save model
        epoch: Current epoch
        optimizer_state: Optimizer state dict
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'config': config,
        'pyramid_info': model.get_pyramid_info() if hasattr(model, 'get_pyramid_info') else None
    }
    
    if epoch is not None:
        checkpoint['epoch'] = epoch
    
    if optimizer_state is not None:
        checkpoint['optimizer_state_dict'] = optimizer_state
    
    torch.save(checkpoint, save_path)
    logger.info(f"Model saved to {save_path}")


def load_pyramid_model(
    model: torch.nn.Module,
    load_path: str,
    device: str = "cuda"
) -> Tuple[torch.nn.Module, dict]:
    """
    Load PyNeRF model and configuration
    
    Args:
        model: PyNeRF model instance
        load_path: Path to load model from
        device: Device to load model on
        
    Returns:
        Tuple of (loaded_model, config)
    """
    checkpoint = torch.load(load_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    config = checkpoint.get('config', {})
    
    logger.info(f"Model loaded from {load_path}")
    
    return model, config


def compute_psnr(
    img_pred: torch.Tensor,
    img_gt: torch.Tensor,
    mask: Optional[torch.Tensor] = None
) -> float:
    """
    Compute Peak Signal-to-Noise Ratio (PSNR)
    
    Args:
        img_pred: Predicted image [H, W, 3]
        img_gt: Ground truth image [H, W, 3]
        mask: Optional mask [H, W]
        
    Returns:
        PSNR value
    """
    if mask is not None:
        img_pred = img_pred[mask]
        img_gt = img_gt[mask]
    
    mse = torch.mean((img_pred - img_gt) ** 2)
    
    if mse == 0:
        return float('inf')
    
    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
    return psnr.item()


def compute_ssim(
    img_pred: torch.Tensor,
    img_gt: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5
) -> float:
    """
    Compute Structural Similarity Index (SSIM)
    
    Args:
        img_pred: Predicted image [H, W, 3]
        img_gt: Ground truth image [H, W, 3]
        window_size: Window size for SSIM computation
        sigma: Gaussian sigma for window
        
    Returns:
        SSIM value
    """
    # Convert to grayscale if needed
    if img_pred.shape[-1] == 3:
        img_pred = torch.mean(img_pred, dim=-1, keepdim=True)
        img_gt = torch.mean(img_gt, dim=-1, keepdim=True)
    
    # Add batch and channel dimensions
    img_pred = img_pred.permute(2, 0, 1).unsqueeze(0)
    img_gt = img_gt.permute(2, 0, 1).unsqueeze(0)
    
    # Create Gaussian window
    coords = torch.arange(window_size, dtype=torch.float32, device=img_pred.device)
    coords -= window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()
    
    window = g.outer(g).unsqueeze(0).unsqueeze(0)
    
    # Constants for SSIM
    C1 = (0.01) ** 2
    C2 = (0.03) ** 2
    
    # Compute means
    mu1 = F.conv2d(img_pred, window, padding=window_size//2)
    mu2 = F.conv2d(img_gt, window, padding=window_size//2)
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    # Compute variances and covariance
    sigma1_sq = F.conv2d(img_pred ** 2, window, padding=window_size//2) - mu1_sq
    sigma2_sq = F.conv2d(img_gt ** 2, window, padding=window_size//2) - mu2_sq
    sigma12 = F.conv2d(img_pred * img_gt, window, padding=window_size//2) - mu1_mu2
    
    # Compute SSIM
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return ssim_map.mean().item()


def log_pyramid_stats(
    model: torch.nn.Module,
    step: int,
    losses: Dict[str, float]
) -> None:
    """
    Log pyramid model statistics
    
    Args:
        model: PyNeRF model
        step: Current training step
        losses: Dictionary of loss values
    """
    if hasattr(model, 'get_pyramid_info'):
        pyramid_info = model.get_pyramid_info()
        
        logger.info(f"Step {step}:")
        logger.info(f"  Total Loss: {losses.get('total_loss', 0.0):.6f}")
        logger.info(f"  Color Loss: {losses.get('color_loss', 0.0):.6f}")
        logger.info(f"  Pyramid Levels: {pyramid_info['num_levels']}")
        logger.info(f"  Total Parameters: {pyramid_info['total_parameters']:,}")


def create_training_schedule(
    max_steps: int,
    warmup_steps: int = 1000,
    decay_steps: int = 5000,
    decay_rate: float = 0.1
) -> Dict[str, any]:
    """
    Create learning rate schedule for PyNeRF training
    
    Args:
        max_steps: Maximum training steps
        warmup_steps: Number of warmup steps
        decay_steps: Steps between learning rate decay
        decay_rate: Learning rate decay factor
        
    Returns:
        Training schedule configuration
    """
    return {
        "max_steps": max_steps,
        "warmup_steps": warmup_steps,
        "decay_steps": decay_steps,
        "decay_rate": decay_rate,
        "schedule_type": "exponential_decay_with_warmup"
    }


def apply_learning_rate_schedule(
    optimizer: torch.optim.Optimizer,
    step: int,
    schedule: Dict[str, any],
    base_lr: float
) -> float:
    """
    Apply learning rate schedule
    
    Args:
        optimizer: PyTorch optimizer
        step: Current training step
        schedule: Schedule configuration
        base_lr: Base learning rate
        
    Returns:
        Current learning rate
    """
    if step < schedule["warmup_steps"]:
        # Warmup phase
        lr = base_lr * (step / schedule["warmup_steps"])
    else:
        # Exponential decay
        decay_steps = schedule["decay_steps"]
        decay_rate = schedule["decay_rate"]
        lr = base_lr * (decay_rate ** (step // decay_steps))
    
    # Update optimizer
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    return lr
