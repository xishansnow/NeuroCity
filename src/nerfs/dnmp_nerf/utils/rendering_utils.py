from __future__ import annotations

from typing import Optional, Union
"""
Rendering utilities for DNMP.

This module provides functions for camera operations, ray generation, and rendering-related computations.
"""

import torch
import torch.nn.functional as F
import numpy as np

def generate_camera_rays(
    camera_matrix: torch.Tensor,
    view_matrix: torch.Tensor,
    image_size: Tuple[int, int]
):
    """
    Generate camera rays for given camera parameters.
    
    Args:
        camera_matrix: Camera intrinsic matrix [3, 3]
        view_matrix: View transformation matrix [4, 4]
        image_size: Image size (height, width)
        
    Returns:
        ray_origins: Ray origins [H, W, 3]
        ray_directions: Ray directions [H, W, 3]
    """
    device = camera_matrix.device
    height, width = image_size
    
    # Generate pixel coordinates
    i, j = torch.meshgrid(
        torch.arange(
            width,
            device=device,
            dtype=torch.float32,
        )
    )
    
    # Convert to normalized device coordinates
    dirs = torch.stack([
        (
            i - camera_matrix[0,
            2],
        )
    ], dim=-1)
    
    # Transform ray directions to world coordinates
    view_matrix_inv = torch.linalg.inv(view_matrix)
    rotation_matrix = view_matrix_inv[:3, :3]
    translation = view_matrix_inv[:3, 3]
    
    # Apply rotation to ray directions
    ray_directions = torch.sum(dirs[..., None, :] * rotation_matrix.T, dim=-1)
    ray_directions = ray_directions / torch.norm(ray_directions, dim=-1, keepdim=True)
    
    # Ray origins are at camera center
    ray_origins = translation.expand(height, width, 3)
    
    return ray_origins, ray_directions

def sample_rays_from_image(
    ray_origins: torch.Tensor,
    ray_directions: torch.Tensor,
    image: torch.Tensor,
    num_rays: int,
    device: torch.device = None
):
    """
    Sample rays from image for training.
    
    Args:
        ray_origins: Ray origins [H, W, 3]
        ray_directions: Ray directions [H, W, 3]
        image: RGB image [H, W, 3]
        num_rays: Number of rays to sample
        device: Target device
        
    Returns:
        Dictionary containing sampled rays and colors
    """
    device = device or ray_origins.device
    height, width = ray_origins.shape[:2]
    
    # Generate random pixel indices
    pixel_indices = torch.randint(0, height * width, (num_rays, ), device=device)
    
    # Convert to 2D coordinates
    pixel_y = pixel_indices // width
    pixel_x = pixel_indices % width
    
    # Sample rays and colors
    sampled_origins = ray_origins[pixel_y, pixel_x]  # [num_rays, 3]
    sampled_directions = ray_directions[pixel_y, pixel_x]  # [num_rays, 3]
    sampled_colors = image[pixel_y, pixel_x]  # [num_rays, 3]
    
    return {
        'ray_origins': sampled_origins, 'ray_directions': sampled_directions, 'target_colors': sampled_colors, 'pixel_indices': pixel_indices, 'pixel_coords': torch.stack(
            [pixel_x,
            pixel_y],
            dim=1,
        )
    }

def volume_rendering(
    colors: torch.Tensor,
    densities: torch.Tensor,
    distances: torch.Tensor,
    background_color: Optional[torch.Tensor] = None
):
    """
    Perform volume rendering using alpha compositing.
    
    Args:
        colors: RGB colors along rays [N_rays, N_samples, 3]
        densities: Volume densities [N_rays, N_samples]
        distances: Sample distances [N_rays, N_samples]
        background_color: Background color [3] (optional)
        
    Returns:
        Dictionary containing rendered colors and weights
    """
    device = colors.device
    
    # Compute delta distances
    deltas = distances[..., 1:] - distances[..., :-1]
    deltas = torch.cat([deltas, torch.full_like(deltas[..., :1], 1e10)], dim=-1)
    
    # Compute alpha values
    alphas = 1.0 - torch.exp(-densities * deltas)
    
    # Compute transmittance
    transmittance = torch.cumprod(1.0 - alphas + 1e-10, dim=-1)
    transmittance = torch.roll(transmittance, shifts=1, dims=-1)
    transmittance[..., 0] = 1.0
    
    # Compute weights
    weights = alphas * transmittance
    
    # Composite colors
    rendered_colors = torch.sum(weights.unsqueeze(-1) * colors, dim=-2)
    
    # Add background color if specified
    if background_color is not None:
        background_weight = 1.0 - weights.sum(dim=-1, keepdim=True)
        rendered_colors = rendered_colors + background_weight * background_color.unsqueeze(0)
    
    # Compute depth
    rendered_depth = torch.sum(weights * distances, dim=-1)
    
    # Compute accumulated opacity
    opacity = weights.sum(dim=-1)
    
    return {
        'rgb': rendered_colors, 'depth': rendered_depth, 'opacity': opacity, 'weights': weights
    }

def hierarchical_sampling(
    ray_origins: torch.Tensor,
    ray_directions: torch.Tensor,
    near: float,
    far: float,
    num_coarse: int,
    num_fine: int,
    coarse_weights: Optional[torch.Tensor] = None
):
    """
    Perform hierarchical sampling along rays.
    
    Args:
        ray_origins: Ray origins [N_rays, 3]
        ray_directions: Ray directions [N_rays, 3]
        near: Near clipping distance
        far: Far clipping distance
        num_coarse: Number of coarse samples
        num_fine: Number of fine samples
        coarse_weights: Weights from coarse network [N_rays, num_coarse] (optional)
        
    Returns:
        Dictionary containing sample points and distances
    """
    device = ray_origins.device
    num_rays = ray_origins.shape[0]
    
    # Coarse sampling
    t_coarse = torch.linspace(near, far, num_coarse, device=device)
    t_coarse = t_coarse.expand(num_rays, num_coarse)
    
    # Add noise for training
    if coarse_weights is None:
        noise = torch.rand_like(t_coarse) * (far - near) / num_coarse
        t_coarse = t_coarse + noise
    
    # Fine sampling based on coarse weights
    if coarse_weights is not None and num_fine > 0:
        # Normalize weights
        weights = coarse_weights / (coarse_weights.sum(dim=-1, keepdim=True) + 1e-8)
        
        # Create CDF
        cdf = torch.cumsum(weights, dim=-1)
        cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], dim=-1)
        
        # Sample from CDF
        u = torch.rand(num_rays, num_fine, device=device)
        indices = torch.searchsorted(cdf, u, right=True)
        indices = torch.clamp(indices - 1, 0, num_coarse - 1)
        
        # Interpolate sample positions
        t_left = t_coarse.gather(-1, indices)
        t_right = t_coarse.gather(-1, torch.clamp(indices + 1, 0, num_coarse - 1))
        
        cdf_left = cdf.gather(-1, indices)
        cdf_right = cdf.gather(-1, indices + 1)
        
        # Linear interpolation
        denom = cdf_right - cdf_left
        denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
        t_fine = t_left + (u - cdf_left) / denom * (t_right - t_left)
        
        # Combine coarse and fine samples
        t_combined, _ = torch.sort(torch.cat([t_coarse, t_fine], dim=-1), dim=-1)
    else:
        t_combined = t_coarse
    
    # Compute sample points
    sample_points = ray_origins.unsqueeze(-2) + ray_directions.unsqueeze(-2) * t_combined.unsqueeze(-1)
    
    return {
        'sample_points': sample_points, 'sample_distances': t_combined, 't_coarse': t_coarse, 't_fine': t_fine if coarse_weights is not None and num_fine > 0 else None
    }

def compute_psnr(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Compute Peak Signal-to-Noise Ratio (PSNR).
    
    Args:
        pred: Predicted values [...]
        target: Target values [...]
        
    Returns:
        psnr: PSNR value
    """
    mse = F.mse_loss(pred, target)
    psnr = -10.0 * torch.log10(mse)
    return psnr

def compute_ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5
): 
    """
    Compute Structural Similarity Index (SSIM).
    
    Args:
        pred: Predicted image [B, C, H, W] or [H, W, C]
        target: Target image [B, C, H, W] or [H, W, C]
        window_size: Window size for SSIM computation
        sigma: Gaussian window standard deviation
        
    Returns:
        ssim: SSIM value
    """
    if pred.dim() == 3:
        pred = pred.permute(2, 0, 1).unsqueeze(0)
        target = target.permute(2, 0, 1).unsqueeze(0)
    
    # Create Gaussian window
    coords = torch.arange(window_size, dtype=torch.float32, device=pred.device)
    coords -= window_size // 2
    
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()
    
    window = g.outer(g).unsqueeze(0).unsqueeze(0)
    window = window.expand(pred.shape[1], 1, -1, -1)
    
    # SSIM computation
    mu1 = F.conv2d(pred, window, groups=pred.shape[1], padding=window_size//2)
    mu2 = F.conv2d(target, window, groups=target.shape[1], padding=window_size//2)
    
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = F.conv2d(pred * pred, window, groups=pred.shape[1], padding=window_size//2) - mu1_sq
    sigma2_sq = F.conv2d(
        target * target,
        window,
        groups=target.shape[1],
        padding=window_size//2,
    )
    sigma12 = F.conv2d(
        pred * target,
        window,
        groups=pred.shape[1],
        padding=window_size//2,
    )
    
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return ssim_map.mean()

def create_spherical_coordinates(phi: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """
    Convert spherical coordinates to Cartesian coordinates.
    
    Args:
        phi: Azimuthal angle [N]
        theta: Polar angle [N]
        
    Returns:
        cartesian: Cartesian coordinates [N, 3]
    """
    x = torch.sin(theta) * torch.cos(phi)
    y = torch.sin(theta) * torch.sin(phi)
    z = torch.cos(theta)
    
    return torch.stack([x, y, z], dim=-1)

def fibonacci_sphere_sampling(num_samples: int, device: torch.device = None) -> torch.Tensor:
    """
    Generate uniformly distributed points on sphere using Fibonacci sampling.
    
    Args:
        num_samples: Number of samples
        device: Target device
        
    Returns:
        points: Points on unit sphere [N, 3]
    """
    device = device or torch.device('cpu')
    
    indices = torch.arange(0, num_samples, dtype=torch.float32, device=device)
    
    # Golden angle
    golden_angle = torch.pi * (3.0 - torch.sqrt(torch.tensor(5.0)))
    
    # Spherical coordinates
    theta = golden_angle * indices
    y = 1 - (indices / (num_samples - 1)) * 2  # y goes from 1 to -1
    radius = torch.sqrt(1 - y * y)
    
    x = torch.cos(theta) * radius
    z = torch.sin(theta) * radius
    
    return torch.stack([x, y, z], dim=-1) 