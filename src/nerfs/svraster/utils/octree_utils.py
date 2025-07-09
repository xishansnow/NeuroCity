from __future__ import annotations

from typing import Optional

"""
Octree utilities for SVRaster.
"""

import torch
import numpy as np


def octree_subdivision(
    voxel_data: torch.Tensor,
    voxel_sizes: Optional[torch.Tensor] = None,
    subdivision_mask: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Subdivide voxels into 8 child voxels.

    Args:
        voxel_data: Voxel data [N, D] where D is feature dimension
        voxel_sizes: Voxel sizes [N] (optional, defaults to uniform)
        subdivision_mask: Boolean mask for voxels to subdivide [N] (optional, defaults to all)

    Returns:
        tuple of (child_positions, child_sizes) or just child_data if input is data
    """
    N = voxel_data.shape[0]
    device = voxel_data.device

    # Handle defaults
    if voxel_sizes is None:
        voxel_sizes = torch.ones(N, device=device)
    if subdivision_mask is None:
        subdivision_mask = torch.ones(N, dtype=torch.bool, device=device)

    # If input looks like feature data rather than positions, create more data
    if voxel_data.shape[1] != 3:
        # This is feature data, create 8x more data (subdivision)
        num_keep = subdivision_mask.sum().item()
        if num_keep == 0:
            return torch.empty(0, voxel_data.shape[1], device=device), torch.empty(0, device=device)

        # Create 8 children for each subdivided voxel
        child_data = voxel_data[subdivision_mask].repeat_interleave(8, dim=0)
        # Add some variation to child data
        child_data += torch.randn_like(child_data) * 0.1

        return child_data, child_data[:, :1]

    # Original position-based subdivision
    if not subdivision_mask.any():
        empty_positions = torch.empty(0, 3, device=device)
        empty_sizes = torch.empty(0, device=device)
        return empty_positions, empty_sizes

    # Get voxels to subdivide
    parent_positions = voxel_data[subdivision_mask]
    parent_sizes = voxel_sizes[subdivision_mask]

    num_parents = parent_positions.shape[0]
    child_size = parent_sizes / 2

    # Child offsets (8 corners of cube)
    offsets = (
        torch.tensor(
            [
                [-1, -1, -1],
                [1, -1, -1],
                [-1, 1, -1],
                [1, 1, -1],
                [-1, -1, 1],
                [1, -1, 1],
                [-1, 1, 1],
                [1, 1, 1],
            ],
            dtype=torch.float32,
            device=device,
        )
        * 0.25
    )

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


def octree_pruning(voxel_data: torch.Tensor, threshold: float = 0.001) -> torch.Tensor:
    """
    Create pruning mask for low-density voxels or prune voxel data directly.

    Args:
        voxel_data: Voxel data [N, D] where D is feature dimension
        threshold: Pruning threshold

    Returns:
        Pruned voxel data [M, D] where M <= N
    """
    # If this looks like density data (1D), use density-based pruning
    if voxel_data.dim() == 1:
        densities = torch.exp(voxel_data)
        keep_mask = densities > threshold
        return voxel_data[keep_mask]

    # For multi-dimensional data, use magnitude-based pruning
    # Calculate magnitude of each voxel's features
    magnitudes = torch.norm(voxel_data, dim=1)

    # Create keep mask based on magnitude
    keep_mask = magnitudes > threshold

    # Return pruned data
    return voxel_data[keep_mask]


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


def get_octree_neighbors(
    voxel_position: torch.Tensor,
    voxel_size: float,
    all_positions: torch.Tensor,
    all_sizes: torch.Tensor,
) -> torch.Tensor:
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
