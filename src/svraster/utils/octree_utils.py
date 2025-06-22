"""
Octree utilities for SVRaster.
"""

import torch
import numpy as np
from typing import List, Tuple, Optional


def octree_subdivision(voxel_positions: torch.Tensor,
                      voxel_sizes: torch.Tensor,
                      subdivision_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Subdivide voxels into 8 child voxels.
    
    Args:
        voxel_positions: Voxel center positions [N, 3]
        voxel_sizes: Voxel sizes [N]
        subdivision_mask: Boolean mask for voxels to subdivide [N]
        
    Returns:
        Tuple of (child_positions, child_sizes)
    """
    if not subdivision_mask.any():
        return torch.empty(0, 3, device=voxel_positions.device), torch.empty(0, device=voxel_sizes.device)
    
    # Get voxels to subdivide
    parent_positions = voxel_positions[subdivision_mask]
    parent_sizes = voxel_sizes[subdivision_mask]
    
    num_parents = parent_positions.shape[0]
    child_size = parent_sizes / 2
    
    # Child offsets (8 corners of cube)
    offsets = torch.tensor([
        [-1, -1, -1], [1, -1, -1], [-1, 1, -1], [1, 1, -1],
        [-1, -1, 1], [1, -1, 1], [-1, 1, 1], [1, 1, 1]
    ], dtype=torch.float32, device=voxel_positions.device) * 0.25
    
    # Generate child positions
    child_positions = []
    child_sizes_list = []
    
    for i in range(num_parents):
        parent_pos = parent_positions[i]
        size = child_size[i]
        
        for offset in offsets:
            child_pos = parent_pos + offset * parent_sizes[i]
            child_positions.append(child_pos)
            child_sizes_list.append(size)
    
    # Convert to tensors
    child_positions = torch.stack(child_positions)
    child_sizes_tensor = torch.stack(child_sizes_list)
    
    return child_positions, child_sizes_tensor


def octree_pruning(voxel_densities: torch.Tensor,
                  threshold: float = 0.001) -> torch.Tensor:
    """
    Create pruning mask for low-density voxels.
    
    Args:
        voxel_densities: Voxel density values [N]
        threshold: Pruning threshold
        
    Returns:
        Boolean mask for voxels to keep [N]
    """
    # Apply activation (assume exp activation)
    densities = torch.exp(voxel_densities)
    
    # Create keep mask
    keep_mask = densities > threshold
    
    return keep_mask


def compute_octree_level(voxel_size: float, base_size: float) -> int:
    """
    Compute octree level based on voxel size.
    
    Args:
        voxel_size: Size of the voxel
        base_size: Base voxel size at level 0
        
    Returns:
        Octree level
    """
    if voxel_size >= base_size:
        return 0
    
    level = int(np.log2(base_size / voxel_size))
    return level


def get_octree_neighbors(voxel_position: torch.Tensor,
                        voxel_size: float,
                        all_positions: torch.Tensor,
                        all_sizes: torch.Tensor) -> torch.Tensor:
    """
    Find neighboring voxels in the octree.
    
    Args:
        voxel_position: Position of the query voxel [3]
        voxel_size: Size of the query voxel
        all_positions: All voxel positions [N, 3]
        all_sizes: All voxel sizes [N]
        
    Returns:
        Indices of neighboring voxels
    """
    # Compute distances
    distances = torch.norm(all_positions - voxel_position, dim=1)
    
    # Find neighbors within a reasonable distance
    max_distance = voxel_size * 2  # Adjust as needed
    neighbor_mask = distances <= max_distance
    
    # Exclude self
    self_mask = distances > 1e-6
    neighbor_mask = neighbor_mask & self_mask
    
    neighbor_indices = torch.where(neighbor_mask)[0]
    
    return neighbor_indices 