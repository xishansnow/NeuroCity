"""
Core components for SVRaster: Adaptive sparse voxels and rasterization.

This module implements the main components of the SVRaster method including
adaptive sparse voxel representation, custom rasterizer, and the main model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class SVRasterConfig:
    """Configuration for SVRaster model."""
    
    # Scene representation
    max_octree_levels: int = 16  # Maximum octree levels (65536^3 resolution)
    base_resolution: int = 64    # Base grid resolution
    scene_bounds: Tuple[float, float, float, float, float, float] = (-1.0, -1.0, -1.0, 1.0, 1.0, 1.0)
    
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
    background_color: Tuple[float, float, float] = (0.0, 0.0, 0.0)
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
        self.voxel_sizes.append(nn.Parameter(torch.full((num_voxels,), voxel_size)))
        self.voxel_densities.append(nn.Parameter(torch.randn(num_voxels) * 0.1))
        
        # Initialize SH coefficients (RGB + SH)
        color_dim = 3 * self.num_sh_coeffs
        self.voxel_colors.append(nn.Parameter(torch.randn(num_voxels, color_dim) * 0.1))
        
        # Set levels and Morton codes
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
            [-1, -1, -1], [1, -1, -1], [-1, 1, -1], [1, 1, -1],
            [-1, -1, 1], [1, -1, 1], [-1, 1, 1], [1, 1, 1]
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
            self.voxel_levels.append(torch.full((child_positions.shape[0],), new_level))
            self.voxel_morton_codes.append(self._compute_morton_codes(child_positions, new_level))
        else:
            # Append to existing level
            old_positions = self.voxel_positions[new_level].data
            old_sizes = self.voxel_sizes[new_level].data
            old_densities = self.voxel_densities[new_level].data
            old_colors = self.voxel_colors[new_level].data
            
            self.voxel_positions[new_level].data = torch.cat([old_positions, child_positions])
            self.voxel_sizes[new_level].data = torch.cat([old_sizes, child_sizes_tensor])
            self.voxel_densities[new_level].data = torch.cat([old_densities, child_densities_tensor])
            self.voxel_colors[new_level].data = torch.cat([old_colors, child_colors_tensor])
            
            # Update levels and Morton codes
            new_levels = torch.full((child_positions.shape[0],), new_level)
            self.voxel_levels[new_level] = torch.cat([self.voxel_levels[new_level], new_levels])
            new_morton = self._compute_morton_codes(child_positions, new_level)
            self.voxel_morton_codes[new_level] = torch.cat([self.voxel_morton_codes[new_level], new_morton])
        
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
    
    def get_all_voxels(self) -> Dict[str, torch.Tensor]:
        """Get all voxels from all levels."""
        all_positions = []
        all_sizes = []
        all_densities = []
        all_colors = []
        all_levels = []
        all_morton_codes = []
        
        for level_idx in range(len(self.voxel_positions)):
            if self.voxel_positions[level_idx].shape[0] > 0:
                all_positions.append(self.voxel_positions[level_idx])
                all_sizes.append(self.voxel_sizes[level_idx])
                all_densities.append(self.voxel_densities[level_idx])
                all_colors.append(self.voxel_colors[level_idx])
                all_levels.append(self.voxel_levels[level_idx])
                all_morton_codes.append(self.voxel_morton_codes[level_idx])
        
        if not all_positions:
            # Return empty tensors
            device = next(self.parameters()).device
            return {
                'positions': torch.empty(0, 3, device=device),
                'sizes': torch.empty(0, device=device),
                'densities': torch.empty(0, device=device),
                'colors': torch.empty(0, 3 * self.num_sh_coeffs, device=device),
                'levels': torch.empty(0, dtype=torch.int, device=device),
                'morton_codes': torch.empty(0, dtype=torch.long, device=device)
            }
        
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
        total = 0
        for level_idx in range(len(self.voxel_positions)):
            total += self.voxel_positions[level_idx].shape[0]
        return total


class VoxelRasterizer(nn.Module):
    """
    Custom rasterizer for efficient sparse voxel rendering.
    
    Uses ray direction-dependent Morton ordering for correct depth sorting
    and avoids popping artifacts found in Gaussian splatting.
    """
    
    def __init__(self, config: SVRasterConfig):
        super().__init__()
        self.config = config
    
    def forward(self, 
                voxels: Dict[str, torch.Tensor],
                ray_origins: torch.Tensor,
                ray_directions: torch.Tensor,
                camera_params: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        """
        Rasterize sparse voxels along rays.
        
        Args:
            voxels: Dictionary containing voxel data
            ray_origins: Ray origin points [N, 3]
            ray_directions: Ray direction vectors [N, 3]
            camera_params: Optional camera parameters
            
        Returns:
            Dictionary with rendered outputs
        """
        num_rays = ray_origins.shape[0]
        device = ray_origins.device
        
        # Initialize output
        rgb = torch.zeros(num_rays, 3, device=device)
        depth = torch.zeros(num_rays, device=device)
        alpha = torch.zeros(num_rays, device=device)
        
        if voxels['positions'].shape[0] == 0:
            return {'rgb': rgb, 'depth': depth, 'alpha': alpha}
        
        # Sort voxels by ray direction-dependent Morton order
        sorted_voxels = self._sort_voxels_by_ray_direction(voxels, ray_directions.mean(dim=0))
        
        # Perform ray-voxel intersection and rendering
        for ray_idx in range(num_rays):
            ray_o = ray_origins[ray_idx]
            ray_d = ray_directions[ray_idx]
            
            # Find intersecting voxels
            intersections = self._ray_voxel_intersections(
                ray_o, ray_d, sorted_voxels
            )
            
            if len(intersections) == 0:
                continue
            
            # Render along ray
            ray_rgb, ray_depth, ray_alpha = self._render_ray(
                ray_o, ray_d, intersections, sorted_voxels
            )
            
            rgb[ray_idx] = ray_rgb
            depth[ray_idx] = ray_depth
            alpha[ray_idx] = ray_alpha
        
        return {'rgb': rgb, 'depth': depth, 'alpha': alpha}
    
    def _sort_voxels_by_ray_direction(self, 
                                     voxels: Dict[str, torch.Tensor],
                                     mean_ray_direction: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Sort voxels using ray direction-dependent Morton ordering."""
        positions = voxels['positions']
        morton_codes = voxels['morton_codes']
        
        # Compute dot product with mean ray direction for secondary sorting
        dots = torch.sum(positions * mean_ray_direction, dim=1)
        
        # Primary sort by Morton code, secondary by ray direction
        # This ensures correct depth order while maintaining spatial coherence
        sort_keys = morton_codes.float() + dots * 1e-6
        sort_indices = torch.argsort(sort_keys)
        
        # Apply sorting to all voxel data
        sorted_voxels = {}
        for key, value in voxels.items():
            sorted_voxels[key] = value[sort_indices]
        
        return sorted_voxels
    
    def _ray_voxel_intersections(self,
                               ray_o: torch.Tensor,
                               ray_d: torch.Tensor,
                               voxels: Dict[str, torch.Tensor]) -> List[Tuple[int, float, float]]:
        """Find ray-voxel intersections and return (voxel_idx, t_near, t_far)."""
        positions = voxels['positions']
        sizes = voxels['sizes']
        
        # Compute ray-box intersections for all voxels
        half_sizes = sizes.unsqueeze(1) * 0.5  # [N, 1]
        box_min = positions - half_sizes
        box_max = positions + half_sizes
        
        # Ray-box intersection
        inv_dir = 1.0 / (ray_d + 1e-8)
        t1 = (box_min - ray_o) * inv_dir
        t2 = (box_max - ray_o) * inv_dir
        
        t_near = torch.max(torch.min(t1, t2), dim=1)[0]
        t_far = torch.min(torch.max(t1, t2), dim=1)[0]
        
        # Valid intersections
        valid_mask = (t_near <= t_far) & (t_far > 0)
        
        intersections = []
        valid_indices = torch.where(valid_mask)[0]
        
        for idx in valid_indices:
            intersections.append((
                idx.item(),
                max(t_near[idx].item(), 0.0),
                t_far[idx].item()
            ))
        
        # Sort by t_near for front-to-back rendering
        intersections.sort(key=lambda x: x[1])
        
        return intersections
    
    def _render_ray(self,
                   ray_o: torch.Tensor,
                   ray_d: torch.Tensor,
                   intersections: List[Tuple[int, float, float]],
                   voxels: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Render a single ray through intersecting voxels."""
        device = ray_o.device
        rgb = torch.zeros(3, device=device)
        depth = torch.tensor(0.0, device=device)
        transmittance = torch.tensor(1.0, device=device)
        
        for voxel_idx, t_near, t_far in intersections:
            if transmittance < 1e-4:  # Early termination
                break
            
            # Sample points along ray within voxel
            t_samples = torch.linspace(t_near, t_far, self.config.ray_samples_per_voxel, device=device)
            dt = (t_far - t_near) / self.config.ray_samples_per_voxel
            
            # Get voxel properties
            voxel_density = voxels['densities'][voxel_idx]
            voxel_color = voxels['colors'][voxel_idx]
            
            # Apply density activation
            if self.config.density_activation == "exp":
                density = torch.exp(voxel_density)
            else:
                density = torch.relu(voxel_density)
            
            # Compute opacity
            opacity = 1.0 - torch.exp(-density * dt)
            
            # Extract RGB from SH coefficients (DC component)
            color_dim = 3 * self.num_sh_coeffs if hasattr(self, 'num_sh_coeffs') else voxel_color.shape[0]
            dc_coeffs = voxel_color[:3]  # DC component
            
            if self.config.color_activation == "sigmoid":
                color = torch.sigmoid(dc_coeffs)
            else:
                color = torch.clamp(dc_coeffs, 0, 1)
            
            # Volume rendering
            weight = transmittance * opacity
            rgb += weight * color
            depth += weight * t_samples.mean()
            transmittance *= (1.0 - opacity)
        
        alpha = 1.0 - transmittance
        
        return rgb, depth, alpha


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
    
    def forward(self,
                ray_origins: torch.Tensor,
                ray_directions: torch.Tensor,
                camera_params: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass of SVRaster model.
        
        Args:
            ray_origins: Ray origin points [N, 3]
            ray_directions: Ray direction vectors [N, 3]
            camera_params: Optional camera parameters
            
        Returns:
            Dictionary with rendered outputs
        """
        # Get all voxels
        voxels = self.sparse_voxels.get_all_voxels()
        
        # Rasterize voxels
        outputs = self.rasterizer(voxels, ray_origins, ray_directions, camera_params)
        
        # Apply background color
        alpha = outputs['alpha'].unsqueeze(-1)
        rgb = outputs['rgb'] * alpha + self.background_color * (1 - alpha)
        
        outputs['rgb'] = rgb
        return outputs
    
    def adaptive_subdivision(self, subdivision_criteria: torch.Tensor):
        """Perform adaptive subdivision based on criteria."""
        # This would be called during training based on gradient information
        # or other subdivision criteria
        pass
    
    def get_voxel_statistics(self) -> Dict[str, int]:
        """Get statistics about voxel distribution."""
        stats = {}
        total_voxels = 0
        
        for level_idx in range(len(self.sparse_voxels.voxel_positions)):
            count = self.sparse_voxels.voxel_positions[level_idx].shape[0]
            stats[f'level_{level_idx}'] = count
            total_voxels += count
        
        stats['total_voxels'] = total_voxels
        return stats


class SVRasterLoss(nn.Module):
    """Loss functions for SVRaster training."""
    
    def __init__(self, config: SVRasterConfig):
        super().__init__()
        self.config = config
    
    def forward(self,
                outputs: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor],
                model: SVRasterModel) -> Dict[str, torch.Tensor]:
        """
        Compute SVRaster losses.
        
        Args:
            outputs: Model outputs
            targets: Ground truth targets
            model: SVRaster model for regularization
            
        Returns:
            Dictionary of losses
        """
        losses = {}
        
        # RGB reconstruction loss
        if 'rgb' in outputs and 'colors' in targets:
            losses['rgb_loss'] = F.mse_loss(outputs['rgb'], targets['colors'])
        
        # Depth loss (if available)
        if 'depth' in outputs and 'depth' in targets:
            losses['depth_loss'] = F.l1_loss(outputs['depth'], targets['depth'])
        
        # Opacity regularization
        if self.config.use_opacity_regularization:
            voxels = model.sparse_voxels.get_all_voxels()
            if voxels['densities'].numel() > 0:
                if self.config.density_activation == "exp":
                    densities = torch.exp(voxels['densities'])
                else:
                    densities = torch.relu(voxels['densities'])
                
                # Encourage sparsity
                opacity_reg = torch.mean(densities)
                losses['opacity_reg'] = self.config.opacity_reg_weight * opacity_reg
        
        # Total loss
        total_loss = sum(losses.values())
        losses['total_loss'] = total_loss
        
        return losses 