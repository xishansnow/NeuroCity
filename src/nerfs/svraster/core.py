from typing import Any, ByteString, Callable, Collection, Container, Deque, Generator, Generic, Iterable, Iterator, Mapping, Match, Optional, Pattern, Reversible, Sequence, SupportsAbs, SupportsBytes, SupportsComplex, SupportsFloat, SupportsIndex, SupportsInt, SupportsRound, Text, TypeVar, Union
"""
Core components for SVRaster: Adaptive sparse voxels and rasterization.

This module implements the main components of the SVRaster method including
adaptive sparse voxel representation, custom rasterizer, and the main model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
import logging
import os

logger = logging.getLogger(__name__)

@dataclass
class SVRasterConfig:
    """Configuration for SVRaster model."""
    
    # Scene representation
    max_octree_levels: int = 16  # Maximum octree levels (65536^3 resolution)
    base_resolution: int = 64    # Base grid resolution
    scene_bounds: tuple[float, float, float, float, float, float] = (
        -1.0,
        -1.0,
        -1.0,
        1.0,
        1.0,
        1.0,
    )
    
    # Voxel properties
    density_activation: str = "exp"     # Density activation function
    color_activation: str = "sigmoid"   # Color activation function
    sh_degree: int = 2                  # Spherical harmonics degree
    
    # Adaptive allocation
    subdivision_threshold: float = 0.01  # Threshold for voxel subdivision
    pruning_threshold: float = 0.001     # Threshold for voxel pruning
    max_voxels_per_level: int = 1000000  # Maximum voxels per octree level
    
    # Rasterization
    ray_samples_per_voxel: int = 8       # Number of samples per voxel along ray
    depth_peeling_layers: int = 4        # Number of depth peeling layers
    morton_ordering: bool = True         # Use Morton ordering for depth sorting
    
    # Rendering
    background_color: tuple[float, float, float] = (0.0, 0.0, 0.0)
    near_plane: float = 0.1
    far_plane: float = 100.0
    
    # Optimization
    use_view_dependent_color: bool = True
    use_opacity_regularization: bool = True
    opacity_reg_weight: float = 0.01

class AdaptiveSparseVoxels(nn.Module):
    """
    Adaptive sparse voxel representation with octree-based level-of-detail.
    
    This class manages sparse voxels at different octree levels without
    storing the full octree structure, keeping only leaf nodes.
    """
    
    def __init__(self, config: SVRasterConfig):
        super().__init__()
        self.config = config
        
        # Initialize voxel storage
        self.voxel_positions = nn.ParameterList()  # Voxel center positions
        self.voxel_sizes = nn.ParameterList()      # Voxel sizes (level-dependent)
        self.voxel_densities = nn.ParameterList()  # Density values
        self.voxel_colors = nn.ParameterList()     # Color/SH coefficients
        self.voxel_levels = []                     # Octree levels
        self.voxel_morton_codes = []               # Morton codes for sorting
        
        # Scene bounds
        self.register_buffer('scene_min', torch.tensor(config.scene_bounds[:3]))
        self.register_buffer('scene_max', torch.tensor(config.scene_bounds[3:]))
        self.scene_size = self.scene_max - self.scene_min
        
        # SH coefficients count
        self.num_sh_coeffs = (config.sh_degree + 1) ** 2
        
        # Initialize with base level voxels
        self._initialize_base_voxels()
    
    def _initialize_base_voxels(self):
        """Initialize base level voxels covering the scene."""
        base_res = self.config.base_resolution
        
        # Create regular grid at base level
        x = torch.linspace(0, 1, base_res + 1)[:-1] + 0.5 / base_res
        y = torch.linspace(0, 1, base_res + 1)[:-1] + 0.5 / base_res
        z = torch.linspace(0, 1, base_res + 1)[:-1] + 0.5 / base_res
        
        xx, yy, zz = torch.meshgrid(x, y, z, indexing='ij')
        positions = torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3)
        
        # Convert to world coordinates
        positions = positions * self.scene_size + self.scene_min
        
        # Initialize parameters
        num_voxels = positions.shape[0]
        voxel_size = self.scene_size.max() / base_res
        
        self.voxel_positions.append(nn.Parameter(positions))
        self.voxel_sizes.append(nn.Parameter(torch.full((num_voxels, ), voxel_size)))
        self.voxel_densities.append(nn.Parameter(torch.randn(num_voxels) * 0.1))
        
        # Initialize SH coefficients (RGB + SH)
        color_dim = 3 * self.num_sh_coeffs
        self.voxel_colors.append(nn.Parameter(torch.randn(num_voxels, color_dim) * 0.1))
        
        # set levels and Morton codes
        self.voxel_levels.append(torch.zeros(num_voxels, dtype=torch.int))
        self.voxel_morton_codes.append(self._compute_morton_codes(positions, 0))
    
    def _compute_morton_codes(self, positions: torch.Tensor, level: int) -> torch.Tensor:
        """Compute Morton codes for voxel positions at given level."""
        # Normalize positions to [0, 1]
        normalized_pos = (positions - self.scene_min) / self.scene_size
        
        # Discretize to grid coordinates
        grid_res = self.config.base_resolution * (2 ** level)
        grid_coords = (normalized_pos * grid_res).long()
        grid_coords = torch.clamp(grid_coords, 0, grid_res - 1)
        
        # Compute Morton codes
        morton_codes = []
        for i in range(positions.shape[0]):
            x, y, z = grid_coords[i]
            morton_code = self._morton_encode_3d(x.item(), y.item(), z.item())
            morton_codes.append(morton_code)
        
        return torch.tensor(morton_codes, dtype=torch.long)
    
    def _morton_encode_3d(self, x: int, y: int, z: int) -> int:
        """Encode 3D coordinates into Morton code."""
        def part1by2(n):
            n &= 0x000003ff
            n = (n ^ (n << 16)) & 0xff0000ff
            n = (n ^ (n << 8)) & 0x0300f00f
            n = (n ^ (n << 4)) & 0x030c30c3
            n = (n ^ (n << 2)) & 0x09249249
            return n
        
        return (part1by2(z) << 2) + (part1by2(y) << 1) + part1by2(x)
    
    def subdivide_voxels(self, subdivision_mask: torch.Tensor, level_idx: int):
        """Subdivide voxels based on subdivision mask."""
        if not subdivision_mask.any():
            return
        
        # Get voxels to subdivide
        parent_positions = self.voxel_positions[level_idx][subdivision_mask]
        parent_sizes = self.voxel_sizes[level_idx][subdivision_mask]
        parent_densities = self.voxel_densities[level_idx][subdivision_mask]
        parent_colors = self.voxel_colors[level_idx][subdivision_mask]
        
        # Create 8 child voxels for each parent
        num_parents = parent_positions.shape[0]
        child_size = parent_sizes / 2
        
        # Child offsets (8 corners of cube)
        offsets = torch.tensor([
            [-1, -1, -1], [1, -1, -1], [-1, 1, -1], [1, 1, -1], [-1, -1, 1], [1, -1, 1], [-1, 1, 1], [1, 1, 1]
        ], dtype=torch.float32, device=parent_positions.device) * 0.25
        
        # Generate child positions
        child_positions = []
        child_sizes_list = []
        child_densities_list = []
        child_colors_list = []
        
        for i in range(num_parents):
            parent_pos = parent_positions[i]
            size = child_size[i]
            
            for offset in offsets:
                child_pos = parent_pos + offset * parent_sizes[i]
                child_positions.append(child_pos)
                child_sizes_list.append(size)
                child_densities_list.append(parent_densities[i])
                child_colors_list.append(parent_colors[i])
        
        # Convert to tensors
        child_positions = torch.stack(child_positions)
        child_sizes_tensor = torch.stack(child_sizes_list)
        child_densities_tensor = torch.stack(child_densities_list)
        child_colors_tensor = torch.stack(child_colors_list)
        
        # Add to next level
        new_level = level_idx + 1
        if new_level >= len(self.voxel_positions):
            # Create new level
            self.voxel_positions.append(nn.Parameter(child_positions))
            self.voxel_sizes.append(nn.Parameter(child_sizes_tensor))
            self.voxel_densities.append(nn.Parameter(child_densities_tensor))
            self.voxel_colors.append(nn.Parameter(child_colors_tensor))
            self.voxel_levels.append(torch.full((child_positions.shape[0], ), new_level))
            self.voxel_morton_codes.append(self._compute_morton_codes(child_positions, new_level))
        else:
            # Append to existing level
            old_positions = self.voxel_positions[new_level].data
            old_sizes = self.voxel_sizes[new_level].data
            old_densities = self.voxel_densities[new_level].data
            old_colors = self.voxel_colors[new_level].data
            
            self.voxel_positions[new_level].data = torch.cat([old_positions, child_positions])
            self.voxel_sizes[new_level].data = torch.cat([old_sizes, child_sizes_tensor])
            self.voxel_densities[new_level].data = torch.cat(
                [old_densities,
                child_densities_tensor],
            )
            self.voxel_colors[new_level].data = torch.cat([old_colors, child_colors_tensor])
            
            # Update levels and Morton codes
            new_levels = torch.full((child_positions.shape[0], ), new_level)
            self.voxel_levels[new_level] = torch.cat([self.voxel_levels[new_level], new_levels])
            new_morton = self._compute_morton_codes(child_positions, new_level)
            self.voxel_morton_codes[new_level] = torch.cat(
                [self.voxel_morton_codes[new_level],
                new_morton],
            )
        
        # Remove subdivided parent voxels
        keep_mask = ~subdivision_mask
        self.voxel_positions[level_idx].data = self.voxel_positions[level_idx].data[keep_mask]
        self.voxel_sizes[level_idx].data = self.voxel_sizes[level_idx].data[keep_mask]
        self.voxel_densities[level_idx].data = self.voxel_densities[level_idx].data[keep_mask]
        self.voxel_colors[level_idx].data = self.voxel_colors[level_idx].data[keep_mask]
        self.voxel_levels[level_idx] = self.voxel_levels[level_idx][keep_mask]
        self.voxel_morton_codes[level_idx] = self.voxel_morton_codes[level_idx][keep_mask]
    
    def prune_voxels(self, threshold: float = None):
        """Remove voxels with low density."""
        if threshold is None:
            threshold = self.config.pruning_threshold
        
        for level_idx in range(len(self.voxel_densities)):
            if self.config.density_activation == "exp":
                densities = torch.exp(self.voxel_densities[level_idx])
            else:
                densities = torch.relu(self.voxel_densities[level_idx])
            
            keep_mask = densities > threshold
            
            if keep_mask.any() and not keep_mask.all():
                self.voxel_positions[level_idx].data = self.voxel_positions[level_idx].data[keep_mask]
                self.voxel_sizes[level_idx].data = self.voxel_sizes[level_idx].data[keep_mask]
                self.voxel_densities[level_idx].data = self.voxel_densities[level_idx].data[keep_mask]
                self.voxel_colors[level_idx].data = self.voxel_colors[level_idx].data[keep_mask]
                self.voxel_levels[level_idx] = self.voxel_levels[level_idx][keep_mask]
                self.voxel_morton_codes[level_idx] = self.voxel_morton_codes[level_idx][keep_mask]
    
    def get_all_voxels(self) -> dict[str, torch.Tensor]:
        """Get all voxels across all levels."""
        all_positions = []
        all_sizes = []
        all_densities = []
        all_colors = []
        all_levels = []
        all_morton_codes = []
        
        for level_idx in range(len(self.voxel_positions)):
            all_positions.append(self.voxel_positions[level_idx].data)
            all_sizes.append(self.voxel_sizes[level_idx].data)
            all_densities.append(self.voxel_densities[level_idx].data)
            all_colors.append(self.voxel_colors[level_idx].data)
            all_levels.append(self.voxel_levels[level_idx])
            all_morton_codes.append(self.voxel_morton_codes[level_idx])
        
        return {
            'positions': torch.cat(all_positions),
            'sizes': torch.cat(all_sizes),
            'densities': torch.cat(all_densities),
            'colors': torch.cat(all_colors),
            'levels': torch.cat(all_levels),
            'morton_codes': torch.cat(all_morton_codes)
        }
    
    def get_total_voxel_count(self) -> int:
        """Get total number of voxels across all levels."""
        return sum(pos.shape[0] for pos in self.voxel_positions)

class VoxelRasterizer(nn.Module):
    """
    Custom rasterizer for efficient sparse voxel rendering.
    
    Uses ray direction-dependent Morton ordering for correct depth sorting
    and avoids popping artifacts found in Gaussian splatting.
    """
    
    def __init__(self, config: SVRasterConfig):
        super().__init__()
        self.config = config
    
    def forward(
        self,
        voxels: dict[str, torch.Tensor],
        ray_origins: torch.Tensor,
        ray_directions: torch.Tensor,
        camera_params: Optional[dict[str, torch.Tensor]] = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass for voxel rasterization."""
        batch_size = ray_origins.shape[0]
        device = ray_origins.device
        
        # Sort voxels by mean ray direction for better depth ordering
        mean_ray_direction = ray_directions.mean(dim=0)
        sorted_voxels = self._sort_voxels_by_ray_direction(voxels, mean_ray_direction)
        
        # Process each ray
        rgb_list = []
        depth_list = []
        weights_list = []
        
        for i in range(batch_size):
            ray_o = ray_origins[i]
            ray_d = ray_directions[i]
            
            # Find ray-voxel intersections
            intersections = self._ray_voxel_intersections(ray_o, ray_d, sorted_voxels)
            
            # Render ray through intersected voxels
            rgb, depth, weights = self._render_ray(ray_o, ray_d, intersections, sorted_voxels)
            
            rgb_list.append(rgb)
            depth_list.append(depth)
            weights_list.append(weights)
        
        # Stack results
        rgb = torch.stack(rgb_list)
        depth = torch.stack(depth_list)
        weights = torch.stack(weights_list)
        
        return {
            'rgb': rgb,
            'depth': depth,
            'weights': weights
        }
    
    def _sort_voxels_by_ray_direction(
        self,
        voxels: dict[str, torch.Tensor],
        mean_ray_direction: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Sort voxels based on mean ray direction for better depth ordering."""
        if not self.config.morton_ordering:
            return voxels
        
        # Project voxel centers onto mean ray direction
        positions = voxels['positions']
        proj_dist = torch.sum(positions * mean_ray_direction, dim=-1)
        
        # Sort by projection distance
        sorted_indices = torch.argsort(proj_dist)
        
        # Sort all voxel attributes
        sorted_voxels = {}
        for key, value in voxels.items():
            sorted_voxels[key] = value[sorted_indices]
        
        return sorted_voxels
    
    def _ray_voxel_intersections(
        self,
        ray_o: torch.Tensor,
        ray_d: torch.Tensor,
        voxels: dict[str, torch.Tensor],
    ) -> list[tuple[int, float, float]]:
        """Find intersections between ray and voxels."""
        positions = voxels['positions']
        sizes = voxels['sizes']
        
        intersections = []
        for i in range(positions.shape[0]):
            pos = positions[i]
            size = sizes[i]
            
            # Ray-AABB intersection
            half_size = size / 2
            box_min = pos - half_size
            box_max = pos + half_size
            
            t_min = (box_min - ray_o) / ray_d
            t_max = (box_max - ray_o) / ray_d
            
            t1 = torch.min(t_min, t_max)
            t2 = torch.max(t_min, t_max)
            
            t_near = torch.max(t1)
            t_far = torch.min(t2)
            
            if t_far > t_near and t_far > 0:
                intersections.append((i, t_near.item(), t_far.item()))
        
        # Sort by t_near
        intersections.sort(key=lambda x: x[1])
        return intersections
    
    def _render_ray(
        self,
        ray_o: torch.Tensor,
        ray_d: torch.Tensor,
        intersections: list[tuple[int, float, float]],
        voxels: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Render a single ray through intersected voxels."""
        if not intersections:
            return (
                torch.tensor(self.config.background_color, device=ray_o.device),
                torch.tensor(self.config.far_plane, device=ray_o.device),
                torch.zeros(1, device=ray_o.device)
            )
        
        # Sample points along ray within each voxel
        rgb_acc = torch.zeros(3, device=ray_o.device)
        depth_acc = torch.zeros(1, device=ray_o.device)
        weight_acc = torch.zeros(1, device=ray_o.device)
        transmittance = 1.0
        
        for voxel_idx, t_near, t_far in intersections:
            # Sample points within voxel
            t_samples = torch.linspace(
                t_near,
                t_far,
                self.config.ray_samples_per_voxel,
                device=ray_o.device
            )
            sample_points = ray_o + ray_d * t_samples.unsqueeze(-1)
            
            # Get voxel properties
            density = voxels['densities'][voxel_idx]
            color = voxels['colors'][voxel_idx]
            
            # Compute opacity
            delta_t = (t_far - t_near) / (self.config.ray_samples_per_voxel - 1)
            if self.config.density_activation == "exp":
                opacity = 1 - torch.exp(-torch.exp(density) * delta_t)
            else:
                opacity = 1 - torch.exp(-F.relu(density) * delta_t)
            
            # Accumulate color and depth
            weight = opacity * transmittance
            rgb_acc += weight * color[:3]  # Use only base color for now
            depth_acc += weight * t_near
            weight_acc += weight
            
            transmittance *= (1 - opacity)
            
            if transmittance < 0.01:
                break
        
        # Add background contribution
        if transmittance > 0:
            bg_color = torch.tensor(self.config.background_color, device=ray_o.device)
            rgb_acc += transmittance * bg_color
        
        return rgb_acc, depth_acc, weight_acc

class SVRasterModel(nn.Module):
    """
    Main SVRaster model combining adaptive sparse voxels and rasterization.
    """
    
    def __init__(self, config: SVRasterConfig):
        super().__init__()
        self.config = config
        
        # Core components
        self.sparse_voxels = AdaptiveSparseVoxels(config)
        self.rasterizer = VoxelRasterizer(config)
        
        # Background color
        self.register_buffer('background_color', torch.tensor(config.background_color))
    
    def forward(
        self,
        ray_origins: torch.Tensor,
        ray_directions: torch.Tensor,
        camera_params: Optional[dict[str, torch.Tensor]] = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass for SVRaster model."""
        # Get current voxel representation
        voxels = self.sparse_voxels.get_all_voxels()
        
        # Rasterize voxels
        render_output = self.rasterizer(
            voxels,
            ray_origins,
            ray_directions,
            camera_params
        )
        
        return render_output
    
    def adaptive_subdivision(self, subdivision_criteria: torch.Tensor) -> None:
        """Adaptively subdivide voxels based on criteria."""
        for level_idx in range(len(self.sparse_voxels.voxel_positions)):
            if level_idx >= self.config.max_octree_levels - 1:
                break
            
            # Apply subdivision
            self.sparse_voxels.subdivide_voxels(subdivision_criteria[level_idx], level_idx)
    
    def get_voxel_statistics(self) -> dict[str, int]:
        """Get statistics about voxel distribution."""
        stats = {
            'total_voxels': self.sparse_voxels.get_total_voxel_count(),
            'num_levels': len(self.sparse_voxels.voxel_positions)
        }
        
        for level_idx in range(len(self.sparse_voxels.voxel_positions)):
            stats[f'level_{level_idx}_voxels'] = self.sparse_voxels.voxel_positions[level_idx].shape[0]
        
        return stats

    def visualize_structure(self, output_dir: str) -> None:
        """Visualize octree structure."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save voxel positions and sizes for each level
        for level_idx in range(len(self.sparse_voxels.voxel_positions)):
            positions = self.sparse_voxels.voxel_positions[level_idx].data.cpu().numpy()
            sizes = self.sparse_voxels.voxel_sizes[level_idx].data.cpu().numpy()
            
            np.save(
                os.path.join(output_dir, f'level_{level_idx}_positions.npy'),
                positions
            )
            np.save(
                os.path.join(output_dir, f'level_{level_idx}_sizes.npy'),
                sizes
            )

class SVRasterLoss(nn.Module):
    """Loss functions for SVRaster training."""
    
    def __init__(self, config: SVRasterConfig):
        super().__init__()
        self.config = config
    
    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
        model: SVRasterModel,
    ) -> dict[str, torch.Tensor]:
        """Forward pass for SVRaster loss."""
        loss_dict = {}
        
        # RGB reconstruction loss
        rgb_loss = F.mse_loss(outputs['rgb'], targets['rgb'])
        loss_dict['rgb_loss'] = rgb_loss
        
        # Optional opacity regularization
        if self.config.use_opacity_regularization:
            opacity_reg = 0
            for level_idx in range(len(model.sparse_voxels.voxel_densities)):
                densities = model.sparse_voxels.voxel_densities[level_idx]
                if self.config.density_activation == "exp":
                    opacity = torch.exp(densities)
                else:
                    opacity = F.relu(densities)
                opacity_reg += torch.mean(opacity)
            
            loss_dict['opacity_reg'] = opacity_reg * self.config.opacity_reg_weight
            loss_dict['total_loss'] = rgb_loss + loss_dict['opacity_reg']
        else:
            loss_dict['total_loss'] = rgb_loss
        
        return loss_dict 