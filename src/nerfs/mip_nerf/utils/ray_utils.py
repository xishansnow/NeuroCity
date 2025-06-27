from typing import Optional
"""
Ray utility functions for Mip-NeRF

This module contains utilities for ray casting, conical frustum operations, and volumetric rendering specific to Mip-NeRF.
"""

import torch
import torch.nn.functional as F
import numpy as np
import math

def cast_rays(
    origins: torch.Tensor,
    directions: torch.Tensor,
    radii: torch.Tensor,
    near: float,
    far: float,
) -> dict[str, torch.Tensor]:
    """
    Cast rays and compute pixel radii for Mip-NeRF
    
    Args:
        origins: [batch_size, 3] ray origins
        directions: [batch_size, 3] ray directions (normalized)
        radii: [batch_size] pixel radii
        near: Near plane distance
        far: Far plane distance
        
    Returns:
        Dictionary containing ray information
    """
    # Ensure directions are normalized
    directions = F.normalize(directions, dim=-1)
    
    return {
        'origins': origins, 'directions': directions, 'radii': radii, 'near': near, 'far': far
    }

def conical_frustum_to_gaussian(
    origins: torch.Tensor,
    directions: torch.Tensor,
    t_vals: torch.Tensor,
    radii: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Convert conical frustums to Gaussian representation
    
    Args:
        origins: [..., 3] ray origins
        directions: [..., 3] ray directions
        t_vals: [..., num_samples] t values along rays
        radii: [...] pixel radii
        
    Returns:
        tuple of (means, covariances) representing Gaussians
    """
    # Compute frustum centers
    mu = origins[..., None, :] + directions[..., None, :] * t_vals[..., :, None]
    
    # Compute distances between consecutive samples
    d = torch.diff(t_vals, dim=-1)
    d = torch.cat([d, d[..., -1:]], dim=-1)  # Extend last distance
    
    # Compute variance along ray (axial)
    t_mean = t_vals
    t_var = (d**2) / 12.0  # Uniform distribution variance
    
    # Compute radial variance (perpendicular to ray)
    r_var = radii[..., None]**2 * (t_mean**2 + t_var)
    
    # Build covariance matrices
    batch_shape = mu.shape[:-1]
    cov = torch.zeros(*batch_shape, 3, 3, device=mu.device, dtype=mu.dtype)
    
    # Normalize directions
    directions_norm = F.normalize(directions, dim=-1)
    
    # Create orthogonal basis
    # Find a vector not parallel to the ray direction
    temp = torch.tensor([1., 0., 0.], device=directions.device, dtype=directions.dtype)
    if torch.allclose(torch.abs(torch.dot(directions_norm[0], temp)), torch.tensor(1.0)):
        temp = torch.tensor([0., 1., 0.], device=directions.device, dtype=directions.dtype)
    
    # Gram-Schmidt to create orthonormal basis
    u = torch.cross(directions_norm, temp)
    u = F.normalize(u, dim=-1)
    v = torch.cross(directions_norm, u)
    v = F.normalize(v, dim=-1)
    
    # Expand to match batch dimensions
    u = u[..., None, :].expand(*batch_shape, 3)
    v = v[..., None, :].expand(*batch_shape, 3)
    directions_norm = directions_norm[..., None, :].expand(*batch_shape, 3)
    
    # Construct covariance matrix
    for i in range(3):
        for j in range(3):
            cov[..., i, j] = (
                t_var[..., None] * directions_norm[..., i] * directions_norm[..., j] +
                r_var[..., None] * (u[..., i] * u[..., j] + v[..., i] * v[..., j])
            )
    
    return mu, cov

def sample_along_rays(
    origins: torch.Tensor,
    directions: torch.Tensor,
    radii: torch.Tensor,
    num_samples: int,
    near: float,
    far: float,
    randomized: bool = True,
    lindisp: bool = False,
)
    """
    Sample points along rays for Mip-NeRF
    
    Args:
        origins: [..., 3] ray origins
        directions: [..., 3] ray directions
        radii: [...] pixel radii
        num_samples: Number of samples per ray
        near: Near plane distance
        far: Far plane distance
        randomized: Whether to add random jitter
        lindisp: Whether to sample linearly in disparity space
        
    Returns:
        Dictionary containing sampling results
    """
    # Sample t values
    t_vals = torch.linspace(0., 1., num_samples, device=origins.device)
    
    if lindisp:
        # Sample linearly in disparity space
        t_vals = 1.0 / (1.0 / near * (1.0 - t_vals) + 1.0 / far * t_vals)
    else:
        # Sample linearly in depth
        t_vals = near * (1.0 - t_vals) + far * t_vals
    
    if randomized:
        # Stratified sampling
        mids = 0.5 * (t_vals[:-1] + t_vals[1:])
        upper = torch.cat([mids, t_vals[-1:]])
        lower = torch.cat([t_vals[:1], mids])
        t_rand = torch.rand_like(t_vals)
        t_vals = lower + (upper - lower) * t_rand
    
    # Expand to match batch dimensions
    t_vals = t_vals.expand(*origins.shape[:-1], num_samples)
    
    # Convert to Gaussian representation
    means, covs = conical_frustum_to_gaussian(origins, directions, t_vals, radii)
    
    return {
        't_vals': t_vals, 'means': means, 'covs': covs
    }

def hierarchical_sample(
    origins: torch.Tensor,
    directions: torch.Tensor,
    radii: torch.Tensor,
    t_vals: torch.Tensor,
    weights: torch.Tensor,
    num_samples: int,
    randomized: bool = True,
)
    """
    Hierarchical sampling based on coarse weights
    
    Args:
        origins: [..., 3] ray origins
        directions: [..., 3] ray directions
        radii: [...] pixel radii
        t_vals: [..., num_coarse] coarse t values
        weights: [..., num_coarse] coarse weights
        num_samples: Number of fine samples
        randomized: Whether to add random jitter
        
    Returns:
        Dictionary containing fine sampling results
    """
    # Convert weights to PDF
    weights = weights[..., 1:-1]  # Remove first and last (no contribution)
    weights = weights + 1e-5  # Prevent zero weights
    pdf = weights / torch.sum(weights, dim=-1, keepdim=True)
    
    # Compute CDF
    cdf = torch.cumsum(pdf, dim=-1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], dim=-1)
    
    # Sample from CDF
    if randomized:
        u = torch.rand(*cdf.shape[:-1], num_samples, device=origins.device)
    else:
        u = torch.linspace(0., 1., num_samples, device=origins.device)
        u = u.expand(*cdf.shape[:-1], num_samples)
    
    # Invert CDF to get sample positions
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
    t_fine = t_vals_g[..., 0] + t * (t_vals_g[..., 1] - t_vals_g[..., 0])
    
    # Convert to Gaussian representation
    means, covs = conical_frustum_to_gaussian(origins, directions, t_fine, radii)
    
    return {
        't_vals': t_fine, 'means': means, 'covs': covs
    }

def volumetric_rendering(
    densities: torch.Tensor,
    colors: torch.Tensor,
    t_vals: torch.Tensor,
    white_bkgd: bool = False,
)
    """
    Volumetric rendering for Mip-NeRF
    
    Args:
        densities: [..., num_samples] density values
        colors: [..., num_samples, 3] color values
        t_vals: [..., num_samples] t values along rays
        white_bkgd: Whether to use white background
        
    Returns:
        Dictionary containing rendered results
    """
    # Compute distances between samples
    dists = torch.diff(t_vals, dim=-1)
    dists = torch.cat([dists, torch.full_like(dists[..., :1], 1e10)], dim=-1)
    
    # Compute alpha values
    alpha = 1.0 - torch.exp(-F.relu(densities[..., 0]) * dists)
    
    # Compute transmittance
    transmittance = torch.cumprod(1.0 - alpha + 1e-10, dim=-1)
    transmittance = torch.cat([
        torch.ones_like(transmittance[..., :1]), transmittance[..., :-1]
    ], dim=-1)
    
    # Compute weights
    weights = alpha * transmittance
    
    # Render color
    rgb = torch.sum(weights[..., None] * colors, dim=-2)
    
    # Add white background if specified
    if white_bkgd:
        acc_alpha = torch.sum(weights, dim=-1, keepdim=True)
        rgb = rgb + (1.0 - acc_alpha)
    
    # Compute depth
    depth = torch.sum(weights * t_vals, dim=-1)
    
    # Compute accumulated alpha (opacity)
    acc_alpha = torch.sum(weights, dim=-1)
    
    # Compute depth variance
    depth_var = torch.sum(weights * t_vals**2, dim=-1) - depth**2
    
    return {
        'rgb': rgb, 'depth': depth, 'acc_alpha': acc_alpha, 'weights': weights, 'depth_var': depth_var
    }

def compute_pixel_radii(focal_length: float, image_width: int, image_height: int) -> torch.Tensor:
    """
    Compute pixel radii for anti-aliasing
    
    Args:
        focal_length: Camera focal length
        image_width: Image width in pixels
        image_height: Image height in pixels
        
    Returns:
        Pixel radius value
    """
    # Compute pixel size in camera coordinates
    pixel_size = 1.0 / focal_length
    
    # For square pixels, radius is half the diagonal
    pixel_radius = pixel_size * math.sqrt(2) / 2
    
    return torch.tensor(pixel_radius)

def generate_rays(
    camera_matrix: torch.Tensor,
    image_width: int,
    image_height: int,
    c2w: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate rays from camera parameters
    
    Args:
        camera_matrix: [3, 3] camera intrinsic matrix
        image_width: Image width
        image_height: Image height
        c2w: [4, 4] camera-to-world transformation matrix
        
    Returns:
        tuple of (origins, directions, radii)
    """
    # Extract focal lengths
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]
    
    # Create pixel coordinates
    i, j = torch.meshgrid(
        torch.arange(
            image_width,
            dtype=torch.float32,
        )
    )
    
    # Convert to camera coordinates
    dirs = torch.stack([
        (i - cx) / fx, -(j - cy) / fy, -torch.ones_like(i)
    ], dim=-1)
    
    # Transform to world coordinates
    dirs = torch.sum(dirs[..., None, :] * c2w[:3, :3], dim=-1)
    origins = c2w[:3, 3].expand(dirs.shape)
    
    # Compute pixel radii
    pixel_radius = compute_pixel_radii(fx, image_width, image_height)
    radii = pixel_radius.expand(dirs.shape[:-1])
    
    return origins, dirs, radii

def compute_multiscale_weights(
    t_vals: torch.Tensor,
    weights: torch.Tensor,
    num_levels: int = 4,
)
    """
    Compute multiscale weights for hierarchical sampling
    
    Args:
        t_vals: [..., num_samples] t values
        weights: [..., num_samples] weights
        num_levels: Number of levels for multiscale
        
    Returns:
        list of weights at different scales
    """
    multiscale_weights = []
    
    for level in range(num_levels):
        # Downsample by factor of 2^level
        scale = 2 ** level
        if scale == 1:
            multiscale_weights.append(weights)
        else:
            # Average pool weights
            num_samples = weights.shape[-1]
            new_size = max(num_samples // scale, 1)
            
            # Reshape and average
            weights_reshaped = weights[..., :new_size * scale].reshape(
                *weights.shape[:-1], new_size, scale
            )
            weights_downsampled = torch.mean(weights_reshaped, dim=-1)
            multiscale_weights.append(weights_downsampled)
    
    return multiscale_weights

def resample_along_rays(
    origins: torch.Tensor,
    directions: torch.Tensor,
    radii: torch.Tensor,
    t_vals: torch.Tensor,
    weights: torch.Tensor,
    num_samples: int,
    randomized: bool = True,
    resample_padding: float = 0.01,
)
    """
    Resample along rays using Dirichlet/alpha padding
    
    Args:
        origins: [..., 3] ray origins
        directions: [..., 3] ray directions
        radii: [...] pixel radii
        t_vals: [..., num_old_samples] old t values
        weights: [..., num_old_samples] old weights
        num_samples: Number of new samples
        randomized: Whether to add random jitter
        resample_padding: Dirichlet/alpha padding parameter
        
    Returns:
        Dictionary containing resampled results
    """
    # Add padding to weights
    weights_padded = weights + resample_padding
    
    # Normalize to get PDF
    pdf = weights_padded / torch.sum(weights_padded, dim=-1, keepdim=True)
    
    # Compute CDF
    cdf = torch.cumsum(pdf, dim=-1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], dim=-1)
    
    # Sample from CDF
    if randomized:
        u = torch.rand(*cdf.shape[:-1], num_samples, device=origins.device)
    else:
        u = torch.linspace(0., 1., num_samples, device=origins.device)
        u = u.expand(*cdf.shape[:-1], num_samples)
    
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
    t_new = t_vals_g[..., 0] + t * (t_vals_g[..., 1] - t_vals_g[..., 0])
    
    # Convert to Gaussian representation
    means, covs = conical_frustum_to_gaussian(origins, directions, t_new, radii)
    
    return {
        't_vals': t_new, 'means': means, 'covs': covs
    } 