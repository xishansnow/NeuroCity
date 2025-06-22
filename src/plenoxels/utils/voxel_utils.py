"""
Voxel Utilities for Plenoxels

This module provides utilities for voxel grid operations, including creation,
pruning, coordinate transformations, and boundary computations.
"""

import torch
import numpy as np
from typing import Tuple, Optional, List, Dict, Any
import logging

logger = logging.getLogger(__name__)


def create_voxel_grid(resolution: Tuple[int, int, int],
                     scene_bounds: Tuple[float, float, float, float, float, float],
                     device: torch.device = None) -> Dict[str, torch.Tensor]:
    """
    Create a voxel grid with specified resolution and scene bounds.
    
    Args:
        resolution: (D, H, W) voxel grid resolution
        scene_bounds: (x_min, y_min, z_min, x_max, y_max, z_max) scene bounds
        device: Device to create tensors on
        
    Returns:
        Dictionary containing voxel grid information
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    D, H, W = resolution
    x_min, y_min, z_min, x_max, y_max, z_max = scene_bounds
    
    # Create coordinate grids
    x_coords = torch.linspace(x_min, x_max, W, device=device)
    y_coords = torch.linspace(y_min, y_max, H, device=device)
    z_coords = torch.linspace(z_min, z_max, D, device=device)
    
    # Create meshgrid
    Z, Y, X = torch.meshgrid(z_coords, y_coords, x_coords, indexing='ij')
    
    # Stack coordinates
    coords = torch.stack([X, Y, Z], dim=-1)  # [D, H, W, 3]
    
    # Compute voxel size
    voxel_size = torch.tensor([
        (x_max - x_min) / W,
        (y_max - y_min) / H,
        (z_max - z_min) / D
    ], device=device)
    
    return {
        'coords': coords,
        'resolution': torch.tensor(resolution, device=device),
        'scene_bounds': torch.tensor(scene_bounds, device=device),
        'voxel_size': voxel_size
    }


def prune_voxel_grid(density: torch.Tensor,
                    sh_coeffs: torch.Tensor,
                    threshold: float = 0.01) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Prune voxel grid by removing low-density voxels.
    
    Args:
        density: Density grid [D, H, W]
        sh_coeffs: SH coefficients [D, H, W, 3, n_coeffs]
        threshold: Density threshold for pruning
        
    Returns:
        Tuple of (pruned_density, pruned_sh_coeffs)
    """
    # Compute occupancy mask
    occupancy_mask = torch.exp(density) > threshold
    
    # Apply mask to density
    pruned_density = density * occupancy_mask.float()
    
    # Apply mask to SH coefficients
    mask_expanded = occupancy_mask.unsqueeze(-1).unsqueeze(-1)  # [D, H, W, 1, 1]
    pruned_sh_coeffs = sh_coeffs * mask_expanded.float()
    
    return pruned_density, pruned_sh_coeffs


def compute_voxel_bounds(coords: torch.Tensor,
                        scene_bounds: Tuple[float, float, float, float, float, float]) -> torch.Tensor:
    """
    Compute which voxels contain the given coordinates.
    
    Args:
        coords: World coordinates [N, 3]
        scene_bounds: Scene boundary box
        
    Returns:
        Voxel indices [N, 3] (can be out of bounds)
    """
    x_min, y_min, z_min, x_max, y_max, z_max = scene_bounds
    
    # Normalize coordinates to [0, 1]
    normalized_coords = torch.stack([
        (coords[:, 0] - x_min) / (x_max - x_min),
        (coords[:, 1] - y_min) / (y_max - y_min),
        (coords[:, 2] - z_min) / (z_max - z_min)
    ], dim=-1)
    
    return normalized_coords


def voxel_to_world_coords(voxel_indices: torch.Tensor,
                         resolution: Tuple[int, int, int],
                         scene_bounds: Tuple[float, float, float, float, float, float]) -> torch.Tensor:
    """
    Convert voxel indices to world coordinates.
    
    Args:
        voxel_indices: Voxel indices [N, 3]
        resolution: Voxel grid resolution
        scene_bounds: Scene boundary box
        
    Returns:
        World coordinates [N, 3]
    """
    D, H, W = resolution
    x_min, y_min, z_min, x_max, y_max, z_max = scene_bounds
    
    # Normalize voxel indices
    normalized_coords = torch.stack([
        voxel_indices[:, 0] / (W - 1),
        voxel_indices[:, 1] / (H - 1),
        voxel_indices[:, 2] / (D - 1)
    ], dim=-1)
    
    # Scale to world coordinates
    world_coords = torch.stack([
        normalized_coords[:, 0] * (x_max - x_min) + x_min,
        normalized_coords[:, 1] * (y_max - y_min) + y_min,
        normalized_coords[:, 2] * (z_max - z_min) + z_min
    ], dim=-1)
    
    return world_coords


def world_to_voxel_coords(world_coords: torch.Tensor,
                         resolution: Tuple[int, int, int],
                         scene_bounds: Tuple[float, float, float, float, float, float]) -> torch.Tensor:
    """
    Convert world coordinates to voxel indices.
    
    Args:
        world_coords: World coordinates [N, 3]
        resolution: Voxel grid resolution
        scene_bounds: Scene boundary box
        
    Returns:
        Voxel indices [N, 3]
    """
    D, H, W = resolution
    x_min, y_min, z_min, x_max, y_max, z_max = scene_bounds
    
    # Normalize to [0, 1]
    normalized_coords = torch.stack([
        (world_coords[:, 0] - x_min) / (x_max - x_min),
        (world_coords[:, 1] - y_min) / (y_max - y_min),
        (world_coords[:, 2] - z_min) / (z_max - z_min)
    ], dim=-1)
    
    # Scale to voxel indices
    voxel_indices = torch.stack([
        normalized_coords[:, 0] * (W - 1),
        normalized_coords[:, 1] * (H - 1),
        normalized_coords[:, 2] * (D - 1)
    ], dim=-1)
    
    return voxel_indices


def compute_voxel_occupancy_stats(density: torch.Tensor,
                                 threshold: float = 0.01) -> Dict[str, float]:
    """
    Compute statistics about voxel occupancy.
    
    Args:
        density: Density grid [D, H, W]
        threshold: Density threshold for occupancy
        
    Returns:
        Dictionary with occupancy statistics
    """
    # Convert density to occupancy
    occupancy = torch.exp(density) > threshold
    
    total_voxels = density.numel()
    occupied_voxels = occupancy.sum().item()
    
    return {
        'total_voxels': total_voxels,
        'occupied_voxels': occupied_voxels,
        'occupancy_ratio': occupied_voxels / total_voxels,
        'sparsity_ratio': 1.0 - (occupied_voxels / total_voxels),
        'density_mean': density.mean().item(),
        'density_std': density.std().item(),
        'density_min': density.min().item(),
        'density_max': density.max().item()
    }


def interpolate_voxel_grid(grid: torch.Tensor,
                          coords: torch.Tensor,
                          mode: str = 'trilinear') -> torch.Tensor:
    """
    Interpolate values from a voxel grid at given coordinates.
    
    Args:
        grid: Voxel grid [D, H, W, ...] or [1, C, D, H, W]
        coords: Normalized coordinates [N, 3] in range [0, 1]
        mode: Interpolation mode ('trilinear', 'nearest')
        
    Returns:
        Interpolated values [N, ...]
    """
    if grid.dim() == 3:
        # Add batch and channel dimensions
        grid = grid.unsqueeze(0).unsqueeze(0)  # [1, 1, D, H, W]
    elif grid.dim() == 4:
        # Add batch dimension
        grid = grid.unsqueeze(0)  # [1, C, D, H, W]
    
    # Convert coordinates to grid_sample format
    # PyTorch expects coordinates in range [-1, 1]
    coords_normalized = coords * 2.0 - 1.0
    
    # Reshape coordinates for grid_sample
    coords_grid = coords_normalized.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # [1, 1, 1, N, 3]
    
    # Perform interpolation
    if mode == 'trilinear':
        interpolated = torch.nn.functional.grid_sample(
            grid, coords_grid, 
            mode='bilinear', 
            padding_mode='border', 
            align_corners=True
        )
    else:  # nearest
        interpolated = torch.nn.functional.grid_sample(
            grid, coords_grid, 
            mode='nearest', 
            padding_mode='border', 
            align_corners=True
        )
    
    # Reshape output
    interpolated = interpolated.squeeze(0).squeeze(2).squeeze(2)  # [C, N]
    interpolated = interpolated.transpose(0, 1)  # [N, C]
    
    return interpolated


def subdivide_voxel_grid(density: torch.Tensor,
                        sh_coeffs: torch.Tensor,
                        subdivision_factor: int = 2) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Subdivide voxel grid by increasing resolution.
    
    Args:
        density: Density grid [D, H, W]
        sh_coeffs: SH coefficients [D, H, W, 3, n_coeffs]
        subdivision_factor: Factor by which to increase resolution
        
    Returns:
        Tuple of (subdivided_density, subdivided_sh_coeffs)
    """
    # Subdivide density
    density_expanded = density.unsqueeze(0).unsqueeze(0)  # [1, 1, D, H, W]
    new_size = [d * subdivision_factor for d in density.shape]
    density_subdivided = torch.nn.functional.interpolate(
        density_expanded, size=new_size, mode='trilinear', align_corners=True
    )
    density_subdivided = density_subdivided.squeeze(0).squeeze(0)
    
    # Subdivide SH coefficients
    D, H, W, _, n_coeffs = sh_coeffs.shape
    sh_coeffs_flat = sh_coeffs.reshape(D, H, W, -1).permute(3, 0, 1, 2).unsqueeze(0)  # [1, 3*n_coeffs, D, H, W]
    sh_coeffs_subdivided = torch.nn.functional.interpolate(
        sh_coeffs_flat, size=new_size, mode='trilinear', align_corners=True
    )
    sh_coeffs_subdivided = sh_coeffs_subdivided.squeeze(0).permute(1, 2, 3, 0)  # [D', H', W', 3*n_coeffs]
    sh_coeffs_subdivided = sh_coeffs_subdivided.reshape(*new_size, 3, n_coeffs)
    
    return density_subdivided, sh_coeffs_subdivided


def compute_voxel_gradient(density: torch.Tensor) -> torch.Tensor:
    """
    Compute gradient of density field for total variation loss.
    
    Args:
        density: Density grid [D, H, W]
        
    Returns:
        Gradient magnitudes [D, H, W]
    """
    # Compute gradients in each dimension
    grad_x = torch.abs(density[:, :, 1:] - density[:, :, :-1])
    grad_y = torch.abs(density[:, 1:, :] - density[:, :-1, :])
    grad_z = torch.abs(density[1:, :, :] - density[:-1, :, :])
    
    # Pad gradients to match original size
    grad_x = torch.nn.functional.pad(grad_x, (0, 1), mode='replicate')
    grad_y = torch.nn.functional.pad(grad_y, (0, 0, 0, 1), mode='replicate')
    grad_z = torch.nn.functional.pad(grad_z, (0, 0, 0, 0, 0, 1), mode='replicate')
    
    # Compute gradient magnitude
    gradient_magnitude = torch.sqrt(grad_x**2 + grad_y**2 + grad_z**2 + 1e-8)
    
    return gradient_magnitude


def apply_voxel_mask(density: torch.Tensor,
                    sh_coeffs: torch.Tensor,
                    mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply a binary mask to voxel grid.
    
    Args:
        density: Density grid [D, H, W]
        sh_coeffs: SH coefficients [D, H, W, 3, n_coeffs]
        mask: Binary mask [D, H, W]
        
    Returns:
        Tuple of (masked_density, masked_sh_coeffs)
    """
    # Apply mask to density
    masked_density = density * mask.float()
    
    # Apply mask to SH coefficients
    mask_expanded = mask.unsqueeze(-1).unsqueeze(-1)  # [D, H, W, 1, 1]
    masked_sh_coeffs = sh_coeffs * mask_expanded.float()
    
    return masked_density, masked_sh_coeffs 