"""
Voxel processing utilities for DNMP.

This module provides functions for voxel grid operations, point cloud voxelization,
and voxel-based scene management.
"""

import torch
import numpy as np
from typing import Tuple, List, Optional, Dict, Union
import open3d as o3d


def voxelize_point_cloud(points: torch.Tensor, 
                        voxel_size: float = 0.1,
                        scene_bounds: Optional[Tuple[float, ...]] = None) -> Dict[str, torch.Tensor]:
    """
    Voxelize point cloud into regular grid.
    
    Args:
        points: Point cloud [N, 3]
        voxel_size: Size of each voxel
        scene_bounds: Optional scene bounds [x_min, y_min, z_min, x_max, y_max, z_max]
        
    Returns:
        Dictionary containing voxel information
    """
    device = points.device
    
    # Apply scene bounds if specified
    if scene_bounds is not None:
        min_bound = torch.tensor(scene_bounds[:3], device=device)
        max_bound = torch.tensor(scene_bounds[3:], device=device)
        
        mask = torch.all(points >= min_bound, dim=1) & torch.all(points <= max_bound, dim=1)
        points = points[mask]
    
    if len(points) == 0:
        return {
            'voxel_centers': torch.empty(0, 3, device=device),
            'voxel_coords': torch.empty(0, 3, device=device, dtype=torch.long),
            'point_voxel_ids': torch.empty(0, device=device, dtype=torch.long),
            'points_per_voxel': torch.empty(0, device=device, dtype=torch.long),
            'voxel_size': voxel_size
        }
    
    # Compute voxel coordinates
    min_coords = points.min(dim=0)[0]
    voxel_coords = torch.floor((points - min_coords) / voxel_size).long()
    
    # Find unique voxels
    unique_coords, inverse_indices, counts = torch.unique(
        voxel_coords, dim=0, return_inverse=True, return_counts=True)
    
    # Compute voxel centers
    voxel_centers = unique_coords.float() * voxel_size + min_coords + voxel_size / 2
    
    return {
        'voxel_centers': voxel_centers,
        'voxel_coords': unique_coords,
        'point_voxel_ids': inverse_indices,
        'points_per_voxel': counts,
        'voxel_size': voxel_size,
        'origin': min_coords
    }


def create_voxel_grid(scene_bounds: Tuple[float, ...],
                     voxel_size: float) -> Dict[str, torch.Tensor]:
    """
    Create regular voxel grid for given scene bounds.
    
    Args:
        scene_bounds: Scene bounds [x_min, y_min, z_min, x_max, y_max, z_max]
        voxel_size: Size of each voxel
        
    Returns:
        Dictionary containing voxel grid information
    """
    min_bound = np.array(scene_bounds[:3])
    max_bound = np.array(scene_bounds[3:])
    
    # Compute grid dimensions
    grid_size = np.ceil((max_bound - min_bound) / voxel_size).astype(int)
    
    # Create voxel coordinates
    x_coords = torch.arange(grid_size[0], dtype=torch.long)
    y_coords = torch.arange(grid_size[1], dtype=torch.long)
    z_coords = torch.arange(grid_size[2], dtype=torch.long)
    
    xx, yy, zz = torch.meshgrid(x_coords, y_coords, z_coords, indexing='ij')
    voxel_coords = torch.stack([xx.flatten(), yy.flatten(), zz.flatten()], dim=1)
    
    # Compute voxel centers
    voxel_centers = voxel_coords.float() * voxel_size + torch.tensor(min_bound) + voxel_size / 2
    
    return {
        'voxel_centers': voxel_centers,
        'voxel_coords': voxel_coords,
        'grid_size': grid_size,
        'voxel_size': voxel_size,
        'scene_bounds': scene_bounds
    }


def adaptive_voxel_subdivision(points: torch.Tensor,
                              initial_voxel_size: float = 1.0,
                              max_points_per_voxel: int = 1000,
                              min_voxel_size: float = 0.1) -> Dict[str, torch.Tensor]:
    """
    Perform adaptive voxel subdivision based on point density.
    
    Args:
        points: Point cloud [N, 3]
        initial_voxel_size: Initial voxel size
        max_points_per_voxel: Maximum points per voxel before subdivision
        min_voxel_size: Minimum allowed voxel size
        
    Returns:
        Dictionary containing adaptive voxel information
    """
    device = points.device
    
    # Start with initial voxelization
    voxel_info = voxelize_point_cloud(points, initial_voxel_size)
    
    if len(voxel_info['voxel_centers']) == 0:
        return voxel_info
    
    # Find voxels that need subdivision
    needs_subdivision = voxel_info['points_per_voxel'] > max_points_per_voxel
    
    if not needs_subdivision.any() or initial_voxel_size <= min_voxel_size:
        return voxel_info
    
    # Separate voxels that don't need subdivision
    keep_mask = ~needs_subdivision
    final_centers = voxel_info['voxel_centers'][keep_mask]
    final_coords = voxel_info['voxel_coords'][keep_mask]
    final_sizes = torch.full((keep_mask.sum(),), initial_voxel_size, device=device)
    
    # Recursively subdivide dense voxels
    subdivision_centers = []
    subdivision_coords = []
    subdivision_sizes = []
    
    for voxel_idx in torch.where(needs_subdivision)[0]:
        # Get points in this voxel
        point_mask = voxel_info['point_voxel_ids'] == voxel_idx
        voxel_points = points[point_mask]
        
        # Recursively subdivide
        sub_voxel_info = adaptive_voxel_subdivision(
            voxel_points, initial_voxel_size / 2, max_points_per_voxel, min_voxel_size)
        
        if len(sub_voxel_info['voxel_centers']) > 0:
            subdivision_centers.append(sub_voxel_info['voxel_centers'])
            subdivision_coords.append(sub_voxel_info['voxel_coords'])
            subdivision_sizes.append(
                torch.full((len(sub_voxel_info['voxel_centers']),), 
                          initial_voxel_size / 2, device=device))
    
    # Combine results
    if subdivision_centers:
        all_centers = torch.cat([final_centers] + subdivision_centers, dim=0)
        all_coords = torch.cat([final_coords] + subdivision_coords, dim=0)
        all_sizes = torch.cat([final_sizes] + subdivision_sizes, dim=0)
    else:
        all_centers = final_centers
        all_coords = final_coords
        all_sizes = final_sizes
    
    return {
        'voxel_centers': all_centers,
        'voxel_coords': all_coords,
        'voxel_sizes': all_sizes,
        'adaptive': True
    }


def compute_voxel_features(points: torch.Tensor,
                          point_features: Optional[torch.Tensor],
                          voxel_info: Dict[str, torch.Tensor],
                          aggregation: str = 'mean') -> torch.Tensor:
    """
    Compute aggregated features for each voxel.
    
    Args:
        points: Point cloud [N, 3]
        point_features: Point features [N, F] (optional)
        voxel_info: Voxel information from voxelize_point_cloud
        aggregation: Aggregation method ('mean', 'max', 'sum')
        
    Returns:
        voxel_features: Aggregated features per voxel [V, F]
    """
    device = points.device
    num_voxels = len(voxel_info['voxel_centers'])
    
    if point_features is None:
        # Use point coordinates as features
        point_features = points
    
    feature_dim = point_features.shape[1]
    voxel_features = torch.zeros(num_voxels, feature_dim, device=device)
    
    # Aggregate features for each voxel
    for voxel_idx in range(num_voxels):
        point_mask = voxel_info['point_voxel_ids'] == voxel_idx
        if point_mask.any():
            voxel_point_features = point_features[point_mask]
            
            if aggregation == 'mean':
                voxel_features[voxel_idx] = voxel_point_features.mean(dim=0)
            elif aggregation == 'max':
                voxel_features[voxel_idx] = voxel_point_features.max(dim=0)[0]
            elif aggregation == 'sum':
                voxel_features[voxel_idx] = voxel_point_features.sum(dim=0)
            else:
                raise ValueError(f"Unknown aggregation method: {aggregation}")
    
    return voxel_features


def voxel_to_mesh(voxel_centers: torch.Tensor,
                 voxel_size: float,
                 voxel_features: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert voxel grid to mesh representation.
    
    Args:
        voxel_centers: Voxel centers [V, 3]
        voxel_size: Size of each voxel
        voxel_features: Optional voxel features [V, F]
        
    Returns:
        vertices: Mesh vertices [N, 3]
        faces: Mesh faces [M, 3]
    """
    device = voxel_centers.device
    num_voxels = len(voxel_centers)
    
    if num_voxels == 0:
        return torch.empty(0, 3, device=device), torch.empty(0, 3, device=device, dtype=torch.long)
    
    # Create cube vertices for each voxel
    half_size = voxel_size / 2
    cube_vertices = torch.tensor([
        [-half_size, -half_size, -half_size],
        [half_size, -half_size, -half_size],
        [half_size, half_size, -half_size],
        [-half_size, half_size, -half_size],
        [-half_size, -half_size, half_size],
        [half_size, -half_size, half_size],
        [half_size, half_size, half_size],
        [-half_size, half_size, half_size]
    ], device=device, dtype=torch.float32)
    
    # Create cube faces
    cube_faces = torch.tensor([
        [0, 1, 2], [0, 2, 3],  # bottom
        [4, 7, 6], [4, 6, 5],  # top
        [0, 4, 5], [0, 5, 1],  # front
        [2, 6, 7], [2, 7, 3],  # back
        [0, 3, 7], [0, 7, 4],  # left
        [1, 5, 6], [1, 6, 2],  # right
    ], device=device, dtype=torch.long)
    
    # Generate vertices for all voxels
    all_vertices = []
    all_faces = []
    
    for voxel_idx, center in enumerate(voxel_centers):
        # Translate cube vertices to voxel center
        voxel_vertices = cube_vertices + center.unsqueeze(0)
        all_vertices.append(voxel_vertices)
        
        # Update face indices
        voxel_faces = cube_faces + voxel_idx * 8
        all_faces.append(voxel_faces)
    
    vertices = torch.cat(all_vertices, dim=0)
    faces = torch.cat(all_faces, dim=0)
    
    return vertices, faces


def sparse_voxel_grid(points: torch.Tensor,
                     voxel_size: float,
                     feature_threshold: float = 0.1) -> Dict[str, torch.Tensor]:
    """
    Create sparse voxel grid representation.
    
    Args:
        points: Point cloud [N, 3]
        voxel_size: Size of each voxel
        feature_threshold: Minimum feature value to keep voxel
        
    Returns:
        Dictionary containing sparse voxel grid
    """
    voxel_info = voxelize_point_cloud(points, voxel_size)
    
    if len(voxel_info['voxel_centers']) == 0:
        return voxel_info
    
    # Compute density features
    density_features = voxel_info['points_per_voxel'].float()
    density_features = density_features / density_features.max()
    
    # Filter voxels based on threshold
    keep_mask = density_features >= feature_threshold
    
    return {
        'voxel_centers': voxel_info['voxel_centers'][keep_mask],
        'voxel_coords': voxel_info['voxel_coords'][keep_mask],
        'density_features': density_features[keep_mask],
        'voxel_size': voxel_size,
        'sparse': True
    } 