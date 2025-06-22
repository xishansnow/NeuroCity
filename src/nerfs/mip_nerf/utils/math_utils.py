"""
Mathematical utility functions for Mip-NeRF

This module contains mathematical utilities specifically for Mip-NeRF implementation,
including safe mathematical operations and specialized functions for integrated
positional encoding.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


def safe_exp(x: torch.Tensor, threshold: float = 10.0) -> torch.Tensor:
    """Safe exponential function to prevent overflow"""
    return torch.exp(torch.clamp(x, max=threshold))


def safe_log(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Safe logarithm function to prevent log(0)"""
    return torch.log(torch.clamp(x, min=eps))


def safe_sqrt(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Safe square root function to prevent sqrt of negative numbers"""
    return torch.sqrt(torch.clamp(x, min=eps))


def expected_sin(x: torch.Tensor, x_var: torch.Tensor) -> torch.Tensor:
    """
    Compute E[sin(x)] where x ~ N(x, x_var)
    
    Uses the identity: E[sin(x)] = exp(-var/2) * sin(mean)
    
    Args:
        x: Mean values [..., dim]
        x_var: Variance values [..., dim]
        
    Returns:
        Expected sin values [..., dim]
    """
    return torch.exp(-0.5 * x_var) * torch.sin(x)


def expected_cos(x: torch.Tensor, x_var: torch.Tensor) -> torch.Tensor:
    """
    Compute E[cos(x)] where x ~ N(x, x_var)
    
    Uses the identity: E[cos(x)] = exp(-var/2) * cos(mean)
    
    Args:
        x: Mean values [..., dim]
        x_var: Variance values [..., dim]
        
    Returns:
        Expected cos values [..., dim]
    """
    return torch.exp(-0.5 * x_var) * torch.cos(x)


def integrated_pos_enc(x_coord: torch.Tensor, x_cov: torch.Tensor, 
                      min_deg: int, max_deg: int) -> torch.Tensor:
    """
    Compute integrated positional encoding for multivariate Gaussians
    
    Args:
        x_coord: [..., 3] coordinates (means)
        x_cov: [..., 3] or [..., 3, 3] covariances
        min_deg: Minimum degree
        max_deg: Maximum degree
        
    Returns:
        [..., 2*3*(max_deg-min_deg)] encoded features
    """
    # Handle both diagonal and full covariance matrices
    if x_cov.dim() == x_coord.dim():  # Diagonal covariance
        x_var = x_cov
    else:  # Full covariance matrix
        x_var = torch.diagonal(x_cov, dim1=-2, dim2=-1)
    
    # Generate frequency bands
    scales = torch.pow(2.0, torch.arange(min_deg, max_deg, 
                                       device=x_coord.device, dtype=x_coord.dtype))
    
    # Scale coordinates and variances
    scaled_x = x_coord[..., None, :] * scales[None, :, None]  # [..., num_freqs, 3]
    scaled_var = x_var[..., None, :] * (scales[None, :, None] ** 2)  # [..., num_freqs, 3]
    
    # Compute expected sin and cos
    sin_vals = expected_sin(scaled_x, scaled_var)
    cos_vals = expected_cos(scaled_x, scaled_var)
    
    # Concatenate and flatten
    encoding = torch.cat([sin_vals, cos_vals], dim=-1)  # [..., num_freqs, 6]
    return encoding.reshape(*encoding.shape[:-2], -1)  # [..., 2*3*num_freqs]


def lift_gaussian(directions: torch.Tensor, t_mean: torch.Tensor, 
                 t_var: torch.Tensor, r_var: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Lift a Gaussian defined along a ray to 3D coordinates
    
    Args:
        directions: [..., 3] ray directions
        t_mean: [...] mean along ray
        t_var: [...] variance along ray
        r_var: [...] radial variance
        
    Returns:
        Tuple of (mean, covariance) in 3D coordinates
    """
    # Compute mean positions
    mean = directions * t_mean[..., None]
    
    # Compute covariance matrices
    # This is a simplified version - full version would account for pixel footprint
    d_outer = directions[..., :, None] * directions[..., None, :]  # [..., 3, 3]
    
    # Identity matrix minus outer product gives perpendicular space
    eye = torch.eye(3, device=directions.device, dtype=directions.dtype)
    perp_outer = eye - d_outer
    
    # Combine axial and radial components
    cov = (t_var[..., None, None] * d_outer + 
           r_var[..., None, None] * perp_outer)
    
    return mean, cov


def compute_depth_variance(t_samples: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """
    Compute variance of depth from samples and weights
    
    Args:
        t_samples: [..., num_samples] sample positions along rays
        weights: [..., num_samples] sample weights
        
    Returns:
        [...] depth variance
    """
    # Compute expected depth
    depth_mean = torch.sum(weights * t_samples, dim=-1)
    
    # Compute second moment
    depth_second_moment = torch.sum(weights * t_samples**2, dim=-1)
    
    # Variance = E[X^2] - E[X]^2
    depth_var = depth_second_moment - depth_mean**2
    
    return depth_var


def compute_alpha_weights(density: torch.Tensor, dists: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute alpha values and transmittance weights from density and distances
    
    Args:
        density: [..., num_samples] density values
        dists: [..., num_samples] distances between samples
        
    Returns:
        Tuple of (alpha, weights)
    """
    # Compute alpha from density and distance
    alpha = 1.0 - torch.exp(-F.relu(density) * dists)
    
    # Compute transmittance (cumulative product of (1-alpha))
    transmittance = torch.cumprod(1.0 - alpha + 1e-10, dim=-1)
    transmittance = torch.cat([
        torch.ones_like(transmittance[..., :1]), 
        transmittance[..., :-1]
    ], dim=-1)
    
    # Compute weights
    weights = alpha * transmittance
    
    return alpha, weights


def piecewise_constant_pdf(t_vals: torch.Tensor, weights: torch.Tensor, 
                          num_samples: int, randomized: bool = True) -> torch.Tensor:
    """
    Sample from piecewise constant PDF defined by t_vals and weights
    
    Args:
        t_vals: [..., num_bins+1] bin edges
        weights: [..., num_bins] bin weights
        num_samples: Number of samples to draw
        randomized: Whether to add random jitter
        
    Returns:
        [..., num_samples] sampled t values
    """
    # Add small epsilon to weights to prevent issues with zero weights
    weights = weights + 1e-5
    
    # Normalize weights to get PDF
    pdf = weights / torch.sum(weights, dim=-1, keepdim=True)
    
    # Compute CDF
    cdf = torch.cumsum(pdf, dim=-1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], dim=-1)
    
    # Sample uniformly
    if randomized:
        u = torch.rand(*weights.shape[:-1], num_samples, device=weights.device)
    else:
        u = torch.linspace(0., 1., num_samples, device=weights.device)
        u = u.expand(*weights.shape[:-1], num_samples)
    
    # Invert CDF
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.clamp(inds - 1, 0, cdf.shape[-1] - 1)
    above = torch.clamp(inds, 0, cdf.shape[-1] - 1)
    
    # Linear interpolation
    inds_g = torch.stack([below, above], dim=-1)
    cdf_g = torch.gather(cdf[..., None], -2, inds_g)
    t_vals_g = torch.gather(t_vals[..., None], -2, inds_g)
    
    denom = cdf_g[..., 1] - cdf_g[..., 0]
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = t_vals_g[..., 0] + t * (t_vals_g[..., 1] - t_vals_g[..., 0])
    
    return samples


def sample_along_rays(origins: torch.Tensor, directions: torch.Tensor,
                     num_samples: int, near: float, far: float, 
                     randomized: bool = True, lindisp: bool = False) -> torch.Tensor:
    """
    Sample points along rays
    
    Args:
        origins: [..., 3] ray origins
        directions: [..., 3] ray directions
        num_samples: Number of samples per ray
        near: Near plane distance
        far: Far plane distance
        randomized: Whether to add random jitter
        lindisp: Whether to sample linearly in disparity space
        
    Returns:
        [..., num_samples] t values along rays
    """
    # Create linearly spaced samples
    t_vals = torch.linspace(0., 1., num_samples, device=origins.device)
    
    if lindisp:
        # Sample linearly in disparity space
        t_vals = 1.0 / (1.0 / near * (1.0 - t_vals) + 1.0 / far * t_vals)
    else:
        # Sample linearly in depth
        t_vals = near * (1.0 - t_vals) + far * t_vals
    
    if randomized:
        # Add stratified random jitter
        mids = 0.5 * (t_vals[:-1] + t_vals[1:])
        upper = torch.cat([mids, t_vals[-1:]])
        lower = torch.cat([t_vals[:1], mids])
        t_rand = torch.rand_like(t_vals)
        t_vals = lower + (upper - lower) * t_rand
    
    return t_vals.expand(*origins.shape[:-1], num_samples)


def compute_multiscale_loss(pred_rgb: torch.Tensor, target_rgb: torch.Tensor,
                           resolutions: list = [1, 2, 4, 8]) -> torch.Tensor:
    """
    Compute multiscale loss by downsampling images at different resolutions
    
    Args:
        pred_rgb: [H, W, 3] predicted RGB
        target_rgb: [H, W, 3] target RGB
        resolutions: List of downsampling factors
        
    Returns:
        Scalar multiscale loss
    """
    total_loss = 0.0
    
    for res in resolutions:
        if res == 1:
            # Full resolution
            pred_down = pred_rgb
            target_down = target_rgb
        else:
            # Downsample
            pred_down = F.avg_pool2d(
                pred_rgb.permute(2, 0, 1).unsqueeze(0), 
                kernel_size=res, stride=res
            ).squeeze(0).permute(1, 2, 0)
            
            target_down = F.avg_pool2d(
                target_rgb.permute(2, 0, 1).unsqueeze(0),
                kernel_size=res, stride=res
            ).squeeze(0).permute(1, 2, 0)
        
        # Compute MSE loss at this resolution
        loss = F.mse_loss(pred_down, target_down)
        total_loss += loss / res  # Weight by resolution
    
    return total_loss / len(resolutions) 