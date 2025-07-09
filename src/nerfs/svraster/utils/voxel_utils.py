from __future__ import annotations

from typing import Optional
"""
Voxel utilities for SVRaster.
"""

import torch
import numpy as np

def voxel_pruning(
    densities: Optional[torch.Tensor] = None,
    colors: Optional[torch.Tensor] = None,
    positions: Optional[torch.Tensor] = None,
    threshold: float = 0.001,
    **kwargs
) -> dict[str, torch.Tensor]:
    """
    Prune voxels with low density.
    
    Args:
        densities: Voxel density values [N, 1] (optional)
        colors: Voxel colors [N, 3] (optional)
        positions: Voxel positions [N, 3] (optional)
        threshold: Pruning threshold
        **kwargs: Additional voxel data to prune
        
    Returns:
        Dictionary of pruned voxel data
    """
    # Determine the number of voxels and which ones to keep
    if densities is not None:
        N = densities.shape[0]
        # Apply activation and create keep mask
        if densities.dim() == 2 and densities.shape[1] == 1:
            density_values = torch.exp(densities.squeeze(-1))
        else:
            density_values = torch.exp(densities)
        keep_mask = density_values > threshold
    elif colors is not None:
        N = colors.shape[0]
        # Use color magnitude for pruning
        color_magnitudes = torch.norm(colors, dim=1)
        keep_mask = color_magnitudes > threshold
    elif positions is not None:
        N = positions.shape[0]
        # Use position magnitude for pruning
        position_magnitudes = torch.norm(positions, dim=1)
        keep_mask = position_magnitudes > threshold
    else:
        raise ValueError("At least one of densities, colors, or positions must be provided")
    
    # Apply pruning to all provided data
    result = {}
    
    if densities is not None:
        result['densities'] = densities[keep_mask]
    if colors is not None:
        result['colors'] = colors[keep_mask]
    if positions is not None:
        result['positions'] = positions[keep_mask]
    
    # Handle additional keyword arguments
    for key, value in kwargs.items():
        if isinstance(value, torch.Tensor) and value.shape[0] == N:
            result[key] = value[keep_mask]
    
    return result

def compute_voxel_bounds(
    voxel_positions: torch.Tensor,
    voxel_sizes: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute bounding boxes for voxels.
    
    Args:
        voxel_positions: Voxel center positions [N, 3]
        voxel_sizes: Voxel sizes [N]
        
    Returns:
        tuple of (box_min, box_max) each [N, 3]
    """
    half_sizes = voxel_sizes.unsqueeze(-1) * 0.5
    box_min = voxel_positions - half_sizes
    box_max = voxel_positions + half_sizes
    
    return box_min, box_max

def voxel_to_world_coords(
    voxel_coords: torch.Tensor,
    scene_bounds: tuple[float,
    float,
    float,
    float,
    float,
    float],
    grid_resolution: int,
) -> torch.Tensor:
    """
    Convert voxel coordinates to world coordinates.
    
    Args:
        voxel_coords: Voxel coordinates [N, 3]
        scene_bounds: Scene bounds (min_x, min_y, min_z, max_x, max_y, max_z)
        grid_resolution: Grid resolution
        
    Returns:
        World coordinates [N, 3]
    """
    scene_min = torch.tensor(scene_bounds[:3], device=voxel_coords.device)
    scene_max = torch.tensor(scene_bounds[3:], device=voxel_coords.device)
    scene_size = scene_max - scene_min
    
    # Normalize voxel coordinates to [0, 1]
    normalized_coords = voxel_coords.float() / grid_resolution
    
    # Convert to world coordinates
    world_coords = normalized_coords * scene_size + scene_min
    
    return world_coords

def world_to_voxel_coords(
    world_coords: torch.Tensor,
    scene_bounds: tuple[float,
    float,
    float,
    float,
    float,
    float],
    grid_resolution: int,
) -> torch.Tensor:
    """
    Convert world coordinates to voxel coordinates.
    
    Args:
        world_coords: World coordinates [N, 3]
        scene_bounds: Scene bounds (min_x, min_y, min_z, max_x, max_y, max_z)
        grid_resolution: Grid resolution
        
    Returns:
        Voxel coordinates [N, 3]
    """
    scene_min = torch.tensor(scene_bounds[:3], device=world_coords.device)
    scene_max = torch.tensor(scene_bounds[3:], device=world_coords.device)
    scene_size = scene_max - scene_min
    
    # Normalize to [0, 1]
    normalized_coords = (world_coords - scene_min) / scene_size
    
    # Convert to voxel coordinates
    voxel_coords = (normalized_coords * grid_resolution).long()
    voxel_coords = torch.clamp(voxel_coords, 0, grid_resolution - 1)
    
    return voxel_coords 