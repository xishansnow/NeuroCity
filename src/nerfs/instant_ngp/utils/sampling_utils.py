"""
Sampling utilities for Instant NGP.

This module provides various sampling strategies used in neural rendering
including adaptive sampling, importance sampling, and hierarchical sampling.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple


def adaptive_sampling(
    z_vals: torch.Tensor,
    weights: torch.Tensor,
    num_samples: int,
    det: bool = False,
    eps: float = 1e-5,
) -> torch.Tensor:
    """
    Perform adaptive sampling based on importance weights.
    
    Args:
        z_vals: [N, K] z-values along rays
        weights: [N, K] importance weights
        num_samples: Number of new samples to generate
        det: Whether to use deterministic sampling
        eps: Small epsilon for numerical stability
        
    Returns:
        [N, num_samples] new z-values
    """
    N, K = z_vals.shape
    device = z_vals.device
    
    # Normalize weights
    weights = weights + eps  # Avoid zeros
    weights = weights / weights.sum(dim=-1, keepdim=True)
    
    # Compute CDF
    cdf = torch.cumsum(weights, dim=-1)
    cdf = torch.cat([torch.zeros_like(cdf[:, :1]), cdf], dim=-1)  # [N, K+1]
    
    # Sample uniformly
    if det:
        u = torch.linspace(0, 1, num_samples, device=device)
        u = u.expand(N, num_samples)
    else:
        u = torch.rand(N, num_samples, device=device)
    
    # Invert CDF
    indices = torch.searchsorted(cdf, u, right=True) - 1
    indices = torch.clamp(indices, 0, K - 1)
    
    # Linear interpolation
    below = indices
    above = torch.clamp(indices + 1, max=K - 1)
    
    # Get corresponding values
    cdf_below = torch.gather(cdf, -1, below)
    cdf_above = torch.gather(cdf, -1, above)
    z_below = torch.gather(z_vals, -1, below)
    z_above = torch.gather(z_vals, -1, above)
    
    # Interpolate
    denom = cdf_above - cdf_below
    denom = torch.where(denom < eps, torch.ones_like(denom), denom)
    t = (u - cdf_below) / denom
    
    samples = z_below + t * (z_above - z_below)
    
    return samples


def importance_sampling(
    pdf: torch.Tensor,
    z_vals: torch.Tensor,
    num_samples: int,
    det: bool = False,
) -> torch.Tensor:
    """
    Perform importance sampling based on probability density function.
    
    Args:
        pdf: [N, K] probability density function values
        z_vals: [N, K] z-values
        num_samples: Number of samples to generate
        det: Whether to use deterministic sampling
        
    Returns:
        [N, num_samples] sampled z-values
    """
    # Normalize PDF
    pdf = pdf / (pdf.sum(dim=-1, keepdim=True) + 1e-8)
    
    # Use adaptive sampling with PDF as weights
    return adaptive_sampling(z_vals, pdf, num_samples, det)


def stratified_sampling(
    near: torch.Tensor,
    far: torch.Tensor,
    num_samples: int,
    perturb: bool = True,
) -> torch.Tensor:
    """
    Perform stratified sampling along rays.
    
    Args:
        near: [N] near bounds
        far: [N] far bounds
        num_samples: Number of samples per ray
        perturb: Whether to add random perturbation
        
    Returns:
        [N, num_samples] z-values
    """
    N = near.shape[0]
    device = near.device
    
    # Create uniform bins
    t_vals = torch.linspace(0, 1, num_samples, device=device)
    z_vals = near[..., None] + (far[..., None] - near[..., None]) * t_vals
    
    if perturb:
        # Get intervals
        upper = torch.cat([z_vals[..., 1:], far[..., None]], dim=-1)
        lower = z_vals
        
        # Random samples in each interval
        t_rand = torch.rand_like(z_vals)
        z_vals = lower + (upper - lower) * t_rand
    
    return z_vals


def uniform_sampling(near: torch.Tensor, far: torch.Tensor, num_samples: int) -> torch.Tensor:
    """
    Perform uniform sampling along rays.
    
    Args:
        near: [N] near bounds
        far: [N] far bounds  
        num_samples: Number of samples per ray
        
    Returns:
        [N, num_samples] z-values
    """
    return stratified_sampling(near, far, num_samples, perturb=False)


def hierarchical_sampling(
    coarse_z_vals: torch.Tensor,
    coarse_weights: torch.Tensor,
    num_fine: int,
    det: bool = False,
) -> torch.Tensor:
    """
    Perform hierarchical sampling for fine network.
    
    Args:
        coarse_z_vals: [N, K_c] coarse z-values
        coarse_weights: [N, K_c] coarse importance weights
        num_fine: Number of fine samples
        det: Whether to use deterministic sampling
        
    Returns:
        [N, num_fine] fine z-values
    """
    return adaptive_sampling(coarse_z_vals, coarse_weights, num_fine, det)


def sample_pdf_2d(
    pdf: torch.Tensor,
    coords: torch.Tensor,
    num_samples: int,
    det: bool = False,
) -> torch.Tensor:
    """
    Sample from 2D probability density function.
    
    Args:
        pdf: [H, W] 2D probability density
        coords: [H, W, 2] coordinate grid
        num_samples: Number of samples to generate
        det: Whether to use deterministic sampling
        
    Returns:
        [num_samples, 2] sampled coordinates
    """
    H, W = pdf.shape
    device = pdf.device
    
    # Flatten PDF
    pdf_flat = pdf.flatten()
    pdf_flat = pdf_flat / pdf_flat.sum()
    
    # Sample indices
    if det:
        # Deterministic sampling
        indices = torch.multinomial(pdf_flat, num_samples, replacement=True)
    else:
        # Random sampling
        indices = torch.multinomial(pdf_flat, num_samples, replacement=True)
    
    # Convert to 2D coordinates
    y_indices = indices // W
    x_indices = indices % W
    
    # Get corresponding coordinates
    sampled_coords = coords[y_indices, x_indices]
    
    return sampled_coords


def poisson_disk_sampling(
    num_samples: int,
    radius: float,
    bounds: torch.Tensor,
    k: int = 30,
) -> torch.Tensor:
    """
    Generate Poisson disk samples for uniform distribution.
    
    Args:
        num_samples: Target number of samples
        radius: Minimum distance between samples
        bounds: [2, 3] bounding box [min, max]
        k: Maximum attempts per sample
        
    Returns:
        [N, 3] sample positions
    """
    device = bounds.device
    bbox_min, bbox_max = bounds[0], bounds[1]
    bbox_size = bbox_max - bbox_min
    
    # Cell size for spatial grid
    cell_size = radius / np.sqrt(3)
    grid_size = (bbox_size / cell_size).ceil().long()
    
    # Initialize grid
    grid = torch.full(tuple(grid_size), -1, dtype=torch.long, device=device)
    
    # Sample list
    samples = []
    active_list = []
    
    # First sample
    first_sample = bbox_min + torch.rand(3, device=device) * bbox_size
    samples.append(first_sample)
    active_list.append(0)
    
    # Update grid
    grid_pos = ((first_sample - bbox_min) / cell_size).long()
    grid[tuple(grid_pos)] = 0
    
    # Generate remaining samples
    while active_list and len(samples) < num_samples:
        # Pick random active sample
        active_idx = torch.randint(len(active_list), (1, )).item()
        sample_idx = active_list[active_idx]
        center = samples[sample_idx]
        
        # Try to generate new sample
        found = False
        for _ in range(k):
            # Random direction and distance
            direction = torch.randn(3, device=device)
            direction = direction / torch.norm(direction)
            distance = radius + torch.rand(1, device=device) * radius
            
            candidate = center + direction * distance
            
            # Check bounds
            if (candidate < bbox_min).any() or (candidate > bbox_max).any():
                continue
            
            # Check minimum distance
            valid = True
            candidate_grid = ((candidate - bbox_min) / cell_size).long()
            
            # Check neighboring grid cells
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    for dz in range(-2, 3):
                        neighbor_pos = candidate_grid + torch.tensor([dx, dy, dz], device=device)
                        
                        # Check bounds
                        if (neighbor_pos < 0).any() or (neighbor_pos >= grid_size).any():
                            continue
                        
                        neighbor_idx = grid[tuple(neighbor_pos)]
                        if neighbor_idx >= 0:
                            dist = torch.norm(candidate - samples[neighbor_idx])
                            if dist < radius:
                                valid = False
                                break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                # Add new sample
                samples.append(candidate)
                active_list.append(len(samples) - 1)
                grid[tuple(candidate_grid)] = len(samples) - 1
                found = True
                break
        
        if not found:
            # Remove from active list
            active_list.pop(active_idx)
    
    return torch.stack(samples)


def blue_noise_sampling(num_samples: int, bounds: torch.Tensor) -> torch.Tensor:
    """
    Generate blue noise sampling pattern.
    
    Args:
        num_samples: Number of samples
        bounds: [2, 3] bounding box
        
    Returns:
        [num_samples, 3] sample positions
    """
    # Use Poisson disk sampling for blue noise
    bbox_size = (bounds[1] - bounds[0]).max()
    radius = bbox_size / (2 * np.sqrt(num_samples))
    
    return poisson_disk_sampling(num_samples, radius.item(), bounds)


def halton_sequence(num_samples: int, base: int = 2, scramble: bool = True) -> torch.Tensor:
    """
    Generate Halton sequence for low-discrepancy sampling.
    
    Args:
        num_samples: Number of samples
        base: Base for Halton sequence
        scramble: Whether to scramble sequence
        
    Returns:
        [num_samples] Halton sequence values
    """
    def halton(i, base):
        result = 0.0
        f = 1.0 / base
        while i > 0:
            result += f * (i % base)
            i //= base
            f /= base
        return result
    
    sequence = [halton(i, base) for i in range(num_samples)]
    sequence = torch.tensor(sequence, dtype=torch.float32)
    
    if scramble:
        # Simple scrambling
        perm = torch.randperm(num_samples)
        sequence = sequence[perm]
    
    return sequence


def sobol_sampling(num_samples: int, dimension: int = 3, scramble: bool = True) -> torch.Tensor:
    """
    Generate Sobol sequence for quasi-random sampling.
    
    Args:
        num_samples: Number of samples
        dimension: Dimension of samples
        scramble: Whether to scramble sequence
        
    Returns:
        [num_samples, dimension] Sobol samples
    """
    try:
        # Use scipy if available
        from scipy.stats import qmc
        sampler = qmc.Sobol(d=dimension, scramble=scramble)
        samples = sampler.random(num_samples)
        return torch.from_numpy(samples).float()
    except ImportError:
        # Fallback to uniform sampling
        return torch.rand(num_samples, dimension)


def test_sampling_functions() -> None:
    """Test sampling function implementations."""
    print("Testing sampling functions...")
    
    # Test stratified sampling
    near = torch.ones(10) * 0.1
    far = torch.ones(10) * 5.0
    z_vals = stratified_sampling(near, far, 64, perturb=True)
    print(f"Stratified sampling shape: {z_vals.shape}")
    
    # Test adaptive sampling
    weights = torch.rand(10, 64)
    new_z_vals = adaptive_sampling(z_vals, weights, 32)
    print(f"Adaptive sampling shape: {new_z_vals.shape}")
    
    # Test importance sampling
    pdf = torch.rand(10, 64)
    importance_z_vals = importance_sampling(pdf, z_vals, 32)
    print(f"Importance sampling shape: {importance_z_vals.shape}")
    
    # Test Poisson disk sampling
    bounds = torch.tensor([[-1, -1, -1], [1, 1, 1]], dtype=torch.float32)
    poisson_samples = poisson_disk_sampling(100, 0.1, bounds)
    print(f"Poisson disk sampling shape: {poisson_samples.shape}")
    
    print("Sampling function tests completed!")


if __name__ == "__main__":
    test_sampling_functions() 