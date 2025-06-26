"""
Grid-guided Neural Radiance Fields for Large Urban Scenes.

This module implements the core components of Grid-NeRF as described in:
"Grid-guided neural radiance fields for large urban scenes"

Key features:
- Hierarchical grid structure for large-scale scenes
- Multi-resolution grid representation
- Grid-guided neural networks
- Efficient rendering for urban environments
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Any, Union, Dict
from dataclasses import dataclass
import numpy as np
import math
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class GridNeRFConfig:
    """Configuration for Grid-NeRF model."""
    
    # Scene bounds
    scene_bounds: tuple[float, float, float, float, float, float] = (-100, -100, -10, 100, 100, 50)
    
    # Grid configuration
    base_grid_resolution: int = 64  # Base resolution for coarsest level
    max_grid_resolution: int = 512  # Maximum resolution for finest level
    num_grid_levels: int = 4        # Number of hierarchical levels
    grid_feature_dim: int = 32      # Feature dimension per grid cell
    
    # Network architecture
    mlp_hidden_dim: int = 256
    mlp_num_layers: int = 4
    view_dependent: bool = True
    view_embed_dim: int = 27
    
    # Training settings
    learning_rate: float = 5e-4
    weight_decay: float = 1e-6
    
    # Rendering settings
    near_plane: float = 0.1
    far_plane: float = 1000.0
    num_samples_coarse: int = 64
    num_samples_fine: int = 128
    
    # Grid update settings
    grid_update_freq: int = 100
    density_threshold: float = 0.01
    
    # Loss weights
    color_loss_weight: float = 1.0
    depth_loss_weight: float = 0.1
    grid_regularization_weight: float = 0.001


class HierarchicalGrid(nn.Module):
    """Hierarchical grid structure for multi-resolution scene representation."""
    
    def __init__(self, config: GridNeRFConfig):
        super().__init__()
        self.config = config
        self.scene_bounds = torch.tensor(config.scene_bounds)
        self.scene_size = self.scene_bounds[3:] - self.scene_bounds[:3]
        
        # Create multi-level grids
        self.grids = nn.ParameterList()
        self.grid_resolutions = []
        
        for level in range(config.num_grid_levels):
            # Exponentially increase resolution
            resolution = config.base_grid_resolution * (2 ** level)
            resolution = min(resolution, config.max_grid_resolution)
            self.grid_resolutions.append(resolution)
            
            # Create grid features for this level
            grid_features = nn.Parameter(
                torch.randn(resolution, resolution, resolution, config.grid_feature_dim) * 0.1
            )
            self.grids.append(grid_features)
    
    def world_to_grid_coords(self, world_coords: torch.Tensor, level: int) -> torch.Tensor:
        """Convert world coordinates to grid coordinates for a specific level."""
        resolution = self.grid_resolutions[level]
        
        # Normalize to [0, 1]
        normalized = (world_coords - self.scene_bounds[:3]) / self.scene_size
        
        # Scale to grid resolution
        grid_coords = normalized * (resolution - 1)
        
        return grid_coords
    
    def sample_grid_features(self, world_coords: torch.Tensor, level: int) -> torch.Tensor:
        """Sample grid features at world coordinates using trilinear interpolation."""
        grid_coords = self.world_to_grid_coords(world_coords, level)
        
        # Trilinear interpolation
        grid_features = self.grids[level]
        features = F.grid_sample(
            grid_features.permute(3, 0, 1, 2).unsqueeze(0), # [1, C, D, H, W]
            grid_coords.unsqueeze(0).unsqueeze(0).unsqueeze(0), # [1, 1, 1, N, 3]
            mode='bilinear', padding_mode='border', align_corners=True
        )
        
        # Remove extra dimensions and transpose
        features = features.squeeze(0).squeeze(1).squeeze(1).transpose(0, 1)  # [N, C]
        
        return features
    
    def get_multi_level_features(self, world_coords: torch.Tensor) -> torch.Tensor:
        """Get features from all grid levels and concatenate them."""
        all_features = []
        
        for level in range(self.config.num_grid_levels):
            features = self.sample_grid_features(world_coords, level)
            all_features.append(features)
        
        # Concatenate features from all levels
        multi_level_features = torch.cat(all_features, dim=-1)
        
        return multi_level_features
    
    def get_occupied_cells(self, level: int, threshold: float = 0.01) -> torch.Tensor:
        """Get mask of occupied cells based on feature magnitude."""
        grid_features = self.grids[level]
        feature_magnitude = torch.norm(grid_features, dim=-1)
        occupied_mask = feature_magnitude > threshold
        
        return occupied_mask


class GridGuidedMLP(nn.Module):
    """MLP network guided by hierarchical grid features."""
    
    def __init__(self, config: GridNeRFConfig):
        super().__init__()
        self.config = config
        
        # Input dimension: grid features from all levels + positional encoding
        input_dim = config.grid_feature_dim * config.num_grid_levels
        
        # Add positional encoding dimension
        pos_encode_dim = 3 * 2 * 10  # 3 coords * 2 (sin/cos) * 10 frequencies
        input_dim += pos_encode_dim
        
        # Density network
        density_layers = []
        for i in range(config.mlp_num_layers):
            if i == 0:
                density_layers.append(nn.Linear(input_dim, config.mlp_hidden_dim))
            else:
                density_layers.append(nn.Linear(config.mlp_hidden_dim, config.mlp_hidden_dim))
            
            if i < config.mlp_num_layers - 1:
                density_layers.append(nn.ReLU(inplace=True))
        
        # Final density layer
        density_layers.append(nn.Linear(config.mlp_hidden_dim, 1))
        self.density_net = nn.Sequential(*density_layers)
        
        # Color network
        if config.view_dependent:
            view_embed_dim = 3 * 2 * 4  # 3 coords * 2 (sin/cos) * 4 frequencies
            color_input_dim = config.mlp_hidden_dim + view_embed_dim
        else:
            color_input_dim = config.mlp_hidden_dim
        
        self.color_net = nn.Sequential(
            nn.Linear(
                color_input_dim,
                config.mlp_hidden_dim // 2,
            )
        )
        
        # Feature extraction layer for color network
        self.feature_layer = nn.Linear(config.mlp_hidden_dim, config.mlp_hidden_dim)
    
    def positional_encoding(self, coords: torch.Tensor, num_freqs: int = 10) -> torch.Tensor:
        """Apply positional encoding to coordinates."""
        encoded = []
        
        for freq in range(num_freqs):
            for func in [torch.sin, torch.cos]:
                encoded.append(func(coords * (2.0 ** freq) * math.pi))
        
        return torch.cat(encoded, dim=-1)
    
    def forward(
        self,
        grid_features: torch.Tensor,
        view_dirs: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the MLP network.
        
        Args:
            grid_features: Multi-level grid features [N, D]
            view_dirs: Optional view directions [N, 3]
            
        Returns:
            tuple[torch.Tensor, torch.Tensor]: (rgb_colors, densities)
                - rgb_colors: RGB colors [N, 3]
                - densities: Volume densities [N, 1]
        """
        # Process through density network
        density_features = self.density_net(grid_features)
        
        # Extract features for color network
        color_features = self.feature_layer(density_features)
        
        if self.config.view_dependent and view_dirs is not None:
            # Encode view directions
            view_embed = self.positional_encoding(view_dirs, num_freqs=4)
            color_input = torch.cat([color_features, view_embed], dim=-1)
        else:
            color_input = color_features
        
        # Process through color network
        rgb = torch.sigmoid(self.color_net(color_input))
        
        return rgb, density_features


class GridNeRFRenderer(nn.Module):
    """Neural renderer for Grid-NeRF."""
    
    def __init__(self, config: GridNeRFConfig) -> None:
        """Initialize the renderer.
        
        Args:
            config: Configuration object for Grid-NeRF
        """
        super().__init__()
        self.config = config
    
    def sample_points_along_rays(
        self,
        ray_origins: torch.Tensor,
        ray_directions: torch.Tensor,
        near: float,
        far: float,
        num_samples: int,
        stratified: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample points along rays using stratified sampling.
        
        Args:
            ray_origins: Ray origin points [N, 3]
            ray_directions: Ray direction vectors [N, 3]
            near: Near plane distance
            far: Far plane distance
            num_samples: Number of samples per ray
            stratified: Whether to use stratified sampling
            
        Returns:
            tuple[torch.Tensor, torch.Tensor]: (sample_points, t_vals)
                - sample_points: Sampled 3D points [N, num_samples, 3]
                - t_vals: Distance values along rays [N, num_samples]
        """
        # Generate sampling distances
        t_vals = torch.linspace(0., 1., num_samples, device=ray_origins.device)
        t_vals = near + t_vals * (far - near)
        
        if stratified and self.training:
            # Add random offsets for stratified sampling
            noise = torch.rand_like(t_vals) * (far - near) / num_samples
            t_vals = t_vals + noise
        
        # Expand t_vals for batch dimension
        t_vals = t_vals.expand(ray_origins.shape[0], num_samples)
        
        # Compute sample points
        sample_points = (
            ray_origins.unsqueeze(-2) +  # [N, 1, 3]
            ray_directions.unsqueeze(-2) * t_vals.unsqueeze(-1)  # [N, S, 3]
        )
        
        return sample_points, t_vals
    
    def hierarchical_sampling(
        self,
        ray_origins: torch.Tensor,
        ray_directions: torch.Tensor,
        coarse_weights: torch.Tensor,
        coarse_t_vals: torch.Tensor,
        num_fine_samples: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample points hierarchically based on coarse network predictions.
        
        Args:
            ray_origins: Ray origin points [N, 3]
            ray_directions: Ray direction vectors [N, 3]
            coarse_weights: Weights from coarse network [N, num_coarse_samples]
            coarse_t_vals: Distance values from coarse sampling [N, num_coarse_samples]
            num_fine_samples: Number of fine samples to generate
            
        Returns:
            tuple[torch.Tensor, torch.Tensor]: (fine_points, fine_t_vals)
                - fine_points: Sampled 3D points [N, num_fine_samples, 3]
                - fine_t_vals: Distance values along rays [N, num_fine_samples]
        """
        # Add small epsilon to weights to prevent division by zero
        weights = coarse_weights + 1e-5
        
        # Get PDF and CDF for importance sampling
        pdf = weights / torch.sum(weights, dim=-1, keepdim=True)
        cdf = torch.cumsum(pdf, dim=-1)
        cdf = torch.cat([torch.zeros_like(cdf[...,:1]), cdf], dim=-1)
        
        # Draw uniform samples
        if self.training:
            u = torch.rand(
                ray_origins.shape[0], num_fine_samples, device=ray_origins.device
            )
        else:
            u = torch.linspace(0., 1., num_fine_samples, device=ray_origins.device)
            u = u.expand(ray_origins.shape[0], num_fine_samples)
        
        # Invert CDF to get samples
        u = u.contiguous()
        inds = torch.searchsorted(cdf, u, right=True)
        below = torch.clamp(inds-1, 0, cdf.shape[-1]-1)
        above = torch.clamp(inds, 0, cdf.shape[-1]-1)
        
        cdf_below = torch.gather(cdf, -1, below)
        cdf_above = torch.gather(cdf, -1, above)
        t_below = torch.gather(coarse_t_vals, -1, below)
        t_above = torch.gather(coarse_t_vals, -1, above)
        
        denom = cdf_above - cdf_below
        denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
        t = (u - cdf_below) / denom
        fine_t_vals = t_below + t * (t_above - t_below)
        
        # Compute fine sample points
        fine_points = (
            ray_origins.unsqueeze(-2) +
            ray_directions.unsqueeze(-2) * fine_t_vals.unsqueeze(-1)
        )
        
        return fine_points, fine_t_vals
    
    def volume_rendering(
        self,
        colors: torch.Tensor,
        densities: torch.Tensor,
        t_vals: torch.Tensor,
        ray_directions: torch.Tensor,
        background_color: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Render colors and depths using volume rendering equation.
        
        Args:
            colors: RGB colors for each sample [N, num_samples, 3]
            densities: Volume density values [N, num_samples, 1]
            t_vals: Distance values along rays [N, num_samples]
            ray_directions: Ray direction vectors [N, 3]
            background_color: Optional background color [3]
            
        Returns:
            Dict[str, torch.Tensor]: Rendering outputs containing:
                - 'color': Rendered RGB colors [N, 3]
                - 'depth': Rendered depth values [N, 1]
                - 'weights': Sample weights [N, num_samples]
                - 'transmittance': Ray transmittance values [N, num_samples]
        """
        # Convert densities to alpha values
        deltas = t_vals[..., 1:] - t_vals[..., :-1]
        delta_inf = 1e10 * torch.ones_like(deltas[...,:1])
        deltas = torch.cat([deltas, delta_inf], dim=-1)
        
        # Account for viewing direction
        deltas = deltas * torch.norm(ray_directions.unsqueeze(-1), dim=-1)
        
        # Compute alpha values
        alpha = 1. - torch.exp(-densities * deltas)
        
        # Compute weights and transmittance
        transmittance = torch.cumprod(
            torch.cat([torch.ones_like(alpha[:,:1]), 1. - alpha + 1e-10], dim=-1),
            dim=-1
        )[..., :-1]
        weights = alpha * transmittance
        
        # Compute color and depth
        color = torch.sum(weights.unsqueeze(-1) * colors, dim=-2)
        depth = torch.sum(weights * t_vals, dim=-1, keepdim=True)
        
        # Add background color if provided
        if background_color is not None:
            background_color = background_color.to(color.device)
            color = color + (1. - weights.sum(dim=-1, keepdim=True)) * background_color
        
        outputs = {
            'color': color,
            'depth': depth,
            'weights': weights,
            'transmittance': transmittance
        }
        
        return outputs


class GridNeRF(nn.Module):
    """Grid-guided Neural Radiance Fields model."""
    
    def __init__(self, config: GridNeRFConfig) -> None:
        """Initialize the Grid-NeRF model.
        
        Args:
            config: Configuration object for Grid-NeRF
        """
        super().__init__()
        self.config = config
        
        # Initialize components
        self.grid = HierarchicalGrid(config)
        self.mlp = GridGuidedMLP(config)
        self.renderer = GridNeRFRenderer(config)
        
        self._reset_parameters()
    
    def query_grid_features(self, points: torch.Tensor) -> torch.Tensor:
        """Query multi-level grid features at given points.
        
        Args:
            points: 3D points to query [N, 3]
            
        Returns:
            torch.Tensor: Concatenated grid features from all levels [N, D]
        """
        return self.grid.get_multi_level_features(points)
    
    def forward(
        self,
        ray_origins: torch.Tensor,
        ray_directions: torch.Tensor,
        background_color: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Forward pass of Grid-NeRF model.
        
        Args:
            ray_origins: Ray origin points [N, 3]
            ray_directions: Ray direction vectors [N, 3]
            background_color: Optional background color [3]
            
        Returns:
            Dict[str, torch.Tensor]: Rendering outputs containing:
                - 'coarse': Coarse rendering outputs
                - 'fine': Fine rendering outputs (if using hierarchical sampling)
                - 'combined': Combined rendering outputs
        """
        outputs = {}
        
        # Sample points along rays
        coarse_points, coarse_t_vals = self.renderer.sample_points_along_rays(
            ray_origins,
            ray_directions,
            self.config.near_plane,
            self.config.far_plane,
            self.config.num_samples_coarse
        )
        
        # Query grid features and run through MLP
        coarse_features = self.query_grid_features(coarse_points)
        coarse_rgb, coarse_density = self.mlp(
            coarse_features,
            ray_directions.unsqueeze(-2).expand_as(coarse_points)
        )
        
        # Perform coarse rendering
        coarse_outputs = self.renderer.volume_rendering(
            coarse_rgb,
            coarse_density,
            coarse_t_vals,
            ray_directions,
            background_color
        )
        outputs['coarse'] = coarse_outputs
        
        # Perform fine sampling if needed
        if self.config.num_samples_fine > 0:
            # Sample points based on coarse weights
            fine_points, fine_t_vals = self.renderer.hierarchical_sampling(
                ray_origins,
                ray_directions,
                coarse_outputs['weights'],
                coarse_t_vals,
                self.config.num_samples_fine
            )
            
            # Query grid features and run through MLP
            fine_features = self.query_grid_features(fine_points)
            fine_rgb, fine_density = self.mlp(
                fine_features,
                ray_directions.unsqueeze(-2).expand_as(fine_points)
            )
            
            # Perform fine rendering
            fine_outputs = self.renderer.volume_rendering(
                fine_rgb,
                fine_density,
                fine_t_vals,
                ray_directions,
                background_color
            )
            outputs['fine'] = fine_outputs
            
            # Combine coarse and fine outputs
            outputs['combined'] = {
                'color': (coarse_outputs['color'] + fine_outputs['color']) / 2,
                'depth': (coarse_outputs['depth'] + fine_outputs['depth']) / 2
            }
        else:
            outputs['combined'] = outputs['coarse']
        
        return outputs
    
    def update_grid_features(self, threshold: Optional[float] = None) -> None:
        """Update grid features based on density threshold.
        
        Args:
            threshold: Optional density threshold value. If None, uses config value.
        """
        if threshold is None:
            threshold = self.config.density_threshold
        
        # Update grid features based on density threshold
        for level in range(self.config.num_grid_levels):
            occupied = self.grid.get_occupied_cells(level, threshold)
            self.grid.grids[level].data[~occupied] = 0
    
    def get_grid_statistics(self) -> Dict[str, Any]:
        """Get statistics about the grid representation.
        
        Returns:
            Dict[str, Any]: Dictionary containing grid statistics:
                - 'num_occupied': Number of occupied cells per level
                - 'occupancy_ratio': Ratio of occupied cells per level
                - 'memory_usage': Approximate memory usage in MB
        """
        stats = {
            'num_occupied': [],
            'occupancy_ratio': [],
            'memory_usage': 0
        }
        
        for level in range(self.config.num_grid_levels):
            occupied = self.grid.get_occupied_cells(level)
            num_occupied = torch.sum(occupied).item()
            total_cells = occupied.numel()
            
            stats['num_occupied'].append(num_occupied)
            stats['occupancy_ratio'].append(num_occupied / total_cells)
            stats['memory_usage'] += (
                num_occupied * self.config.grid_feature_dim * 4 / (1024 * 1024)
            )  # Approximate memory in MB
        
        return stats
    
    def _reset_parameters(self) -> None:
        """Reset model parameters to initial values."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


class GridNeRFLoss(nn.Module):
    """Loss function for Grid-NeRF model."""
    
    def __init__(self, config: GridNeRFConfig) -> None:
        """Initialize the loss function.
        
        Args:
            config: Configuration object for Grid-NeRF
        """
        super().__init__()
        self.config = config
    
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Compute the loss for Grid-NeRF predictions.
        
        Args:
            predictions: Dictionary containing model predictions:
                - 'coarse': Coarse rendering outputs
                - 'fine': Fine rendering outputs (optional)
                - 'combined': Combined rendering outputs
            targets: Dictionary containing ground truth values:
                - 'color': Target RGB colors [N, 3]
                - 'depth': Target depth values [N, 1] (optional)
            
        Returns:
            Dict[str, torch.Tensor]: Dictionary containing loss terms:
                - 'loss': Total combined loss
                - 'color_loss': Color reconstruction loss
                - 'depth_loss': Depth supervision loss (if depth targets provided)
                - 'grid_loss': Grid regularization loss
        """
        losses = {}
        
        # Color loss
        color_loss = 0.0
        if 'fine' in predictions:
            color_loss += F.mse_loss(predictions['fine']['color'], targets['color'])
        color_loss += F.mse_loss(predictions['coarse']['color'], targets['color'])
        losses['color_loss'] = self.config.color_loss_weight * color_loss
        
        # Depth loss (if provided)
        if 'depth' in targets:
            depth_loss = 0.0
            if 'fine' in predictions:
                depth_loss += F.mse_loss(predictions['fine']['depth'], targets['depth'])
            depth_loss += F.mse_loss(predictions['coarse']['depth'], targets['depth'])
            losses['depth_loss'] = self.config.depth_loss_weight * depth_loss
        
        # Grid regularization
        grid_loss = 0.0
        for grid in self.grid.grids:
            grid_loss += torch.mean(torch.abs(grid))
        losses['grid_loss'] = self.config.grid_regularization_weight * grid_loss
        
        # Total loss
        losses['loss'] = sum(
            loss for loss in losses.values()
            if isinstance(loss, torch.Tensor)
        )
        
        return losses 