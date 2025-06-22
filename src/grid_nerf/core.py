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
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import numpy as np
import math


@dataclass
class GridNeRFConfig:
    """Configuration for Grid-NeRF model."""
    
    # Scene bounds
    scene_bounds: Tuple[float, float, float, float, float, float] = (-100, -100, -10, 100, 100, 50)
    
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
            grid_features.permute(3, 0, 1, 2).unsqueeze(0),  # [1, C, D, H, W]
            grid_coords.unsqueeze(0).unsqueeze(0).unsqueeze(0),  # [1, 1, 1, N, 3]
            mode='bilinear',
            padding_mode='border',
            align_corners=True
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
            nn.Linear(color_input_dim, config.mlp_hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(config.mlp_hidden_dim // 2, 3),
            nn.Sigmoid()
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
    
    def forward(self, grid_features: torch.Tensor, 
                coords: torch.Tensor,
                view_dirs: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of grid-guided MLP.
        
        Args:
            grid_features: Multi-level grid features [N, F]
            coords: 3D coordinates [N, 3]
            view_dirs: View directions [N, 3] (optional)
            
        Returns:
            density: Volume density [N, 1]
            color: RGB color [N, 3]
        """
        # Apply positional encoding to coordinates
        pos_encoded = self.positional_encoding(coords)
        
        # Concatenate grid features and positional encoding
        x = torch.cat([grid_features, pos_encoded], dim=-1)
        
        # Pass through density network
        density_features = x
        for layer in self.density_net[:-1]:
            density_features = layer(density_features)
        
        # Extract density
        density = self.density_net[-1](density_features)
        density = F.relu(density)
        
        # Extract features for color network
        color_features = self.feature_layer(density_features)
        
        # Add view direction if view-dependent
        if self.config.view_dependent and view_dirs is not None:
            view_encoded = self.positional_encoding(view_dirs, num_freqs=4)
            color_features = torch.cat([color_features, view_encoded], dim=-1)
        
        # Predict color
        color = self.color_net(color_features)
        
        return density, color


class GridNeRFRenderer(nn.Module):
    """Renderer for Grid-NeRF with hierarchical sampling."""
    
    def __init__(self, config: GridNeRFConfig):
        super().__init__()
        self.config = config
        
    def sample_points_along_rays(self, ray_origins: torch.Tensor,
                                ray_directions: torch.Tensor,
                                near: float, far: float,
                                num_samples: int,
                                stratified: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample points along rays."""
        device = ray_origins.device
        batch_size = ray_origins.shape[0]
        
        # Create sample distances
        t_vals = torch.linspace(0.0, 1.0, num_samples, device=device)
        t_vals = near + (far - near) * t_vals
        t_vals = t_vals.expand(batch_size, num_samples)
        
        if stratified:
            # Add noise for stratified sampling
            mids = 0.5 * (t_vals[..., 1:] + t_vals[..., :-1])
            upper = torch.cat([mids, t_vals[..., -1:]], dim=-1)
            lower = torch.cat([t_vals[..., :1], mids], dim=-1)
            t_rand = torch.rand_like(t_vals)
            t_vals = lower + (upper - lower) * t_rand
        
        # Compute sample points
        sample_points = ray_origins.unsqueeze(-2) + ray_directions.unsqueeze(-2) * t_vals.unsqueeze(-1)
        
        return sample_points, t_vals
    
    def hierarchical_sampling(self, ray_origins: torch.Tensor,
                            ray_directions: torch.Tensor,
                            coarse_weights: torch.Tensor,
                            coarse_t_vals: torch.Tensor,
                            num_fine_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform hierarchical sampling based on coarse weights."""
        device = ray_origins.device
        batch_size = ray_origins.shape[0]
        
        # Get PDF from weights
        weights = coarse_weights + 1e-5  # Prevent division by zero
        pdf = weights / torch.sum(weights, dim=-1, keepdim=True)
        cdf = torch.cumsum(pdf, dim=-1)
        cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], dim=-1)
        
        # Sample from CDF
        u = torch.rand(batch_size, num_fine_samples, device=device)
        indices = torch.searchsorted(cdf, u, right=True)
        below = torch.clamp(indices - 1, 0, coarse_t_vals.shape[-1] - 1)
        above = torch.clamp(indices, 0, coarse_t_vals.shape[-1] - 1)
        
        # Linear interpolation
        indices_g = torch.stack([below, above], dim=-1)  # [batch_size, num_fine_samples, 2]
        
        # Expand cdf and t_vals to match indices_g shape for gathering
        cdf_expanded = cdf.unsqueeze(-1).expand(-1, -1, 2)  # [batch_size, num_coarse+1, 2]
        t_vals_expanded = coarse_t_vals.unsqueeze(-1).expand(-1, -1, 2)  # [batch_size, num_coarse, 2]
        
        cdf_g = torch.gather(cdf_expanded, 1, indices_g)  # [batch_size, num_fine_samples, 2]
        t_vals_g = torch.gather(t_vals_expanded, 1, indices_g)  # [batch_size, num_fine_samples, 2]
        
        denom = cdf_g[..., 1] - cdf_g[..., 0]
        denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
        t = (u - cdf_g[..., 0]) / denom
        fine_t_vals = t_vals_g[..., 0] + t * (t_vals_g[..., 1] - t_vals_g[..., 0])
        
        # Combine coarse and fine samples
        combined_t_vals, _ = torch.sort(torch.cat([coarse_t_vals, fine_t_vals], dim=-1), dim=-1)
        
        # Compute fine sample points
        fine_points = ray_origins.unsqueeze(-2) + ray_directions.unsqueeze(-2) * combined_t_vals.unsqueeze(-1)
        
        return fine_points, combined_t_vals
    
    def volume_rendering(self, colors: torch.Tensor,
                        densities: torch.Tensor,
                        t_vals: torch.Tensor,
                        ray_directions: torch.Tensor,
                        background_color: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Perform volume rendering."""
        # Compute distances between adjacent samples
        dists = t_vals[..., 1:] - t_vals[..., :-1]
        dists = torch.cat([dists, torch.full_like(dists[..., :1], 1e10)], dim=-1)
        
        # Apply ray direction norm to distances
        dists = dists * torch.norm(ray_directions.unsqueeze(-2), dim=-1)
        
        # Compute alpha values
        alpha = 1.0 - torch.exp(-densities.squeeze(-1) * dists)
        
        # Compute transmittance
        transmittance = torch.cumprod(1.0 - alpha + 1e-10, dim=-1)
        transmittance = torch.roll(transmittance, 1, dims=-1)
        transmittance[..., 0] = 1.0
        
        # Compute weights
        weights = alpha * transmittance
        
        # Composite colors
        rgb = torch.sum(weights.unsqueeze(-1) * colors, dim=-2)
        
        # Add background color
        if background_color is not None:
            accumulated_alpha = torch.sum(weights, dim=-1, keepdim=True)
            rgb = rgb + (1.0 - accumulated_alpha) * background_color
        
        # Compute depth
        depth = torch.sum(weights * t_vals, dim=-1)
        
        # Compute accumulated opacity
        opacity = torch.sum(weights, dim=-1)
        
        return {
            'rgb': rgb,
            'depth': depth,
            'opacity': opacity,
            'weights': weights
        }


class GridNeRF(nn.Module):
    """Grid-guided Neural Radiance Fields for large urban scenes."""
    
    def __init__(self, config: GridNeRFConfig):
        super().__init__()
        self.config = config
        
        # Hierarchical grid
        self.hierarchical_grid = HierarchicalGrid(config)
        
        # Grid-guided MLP
        self.grid_mlp = GridGuidedMLP(config)
        
        # Renderer
        self.renderer = GridNeRFRenderer(config)
    
    def query_grid_features(self, points: torch.Tensor) -> torch.Tensor:
        """Query grid features at given 3D points."""
        return self.hierarchical_grid.get_multi_level_features(points)
    
    def forward(self, ray_origins: torch.Tensor,
                ray_directions: torch.Tensor,
                background_color: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Render rays using Grid-NeRF.
        
        Args:
            ray_origins: Ray origins [N, 3]
            ray_directions: Ray directions [N, 3]
            background_color: Background color [3] (optional)
            
        Returns:
            Dictionary containing rendered outputs
        """
        device = ray_origins.device
        
        # Coarse sampling
        coarse_points, coarse_t_vals = self.renderer.sample_points_along_rays(
            ray_origins, ray_directions,
            self.config.near_plane, self.config.far_plane,
            self.config.num_samples_coarse
        )
        
        # Query grid features for coarse points
        coarse_points_flat = coarse_points.view(-1, 3)
        coarse_grid_features = self.query_grid_features(coarse_points_flat)
        
        # Prepare view directions for coarse points
        view_dirs = ray_directions.unsqueeze(-2).expand(-1, self.config.num_samples_coarse, -1)
        view_dirs_flat = view_dirs.reshape(-1, 3)
        
        # Forward pass through MLP for coarse points
        coarse_densities, coarse_colors = self.grid_mlp(
            coarse_grid_features, coarse_points_flat, view_dirs_flat
        )
        
        # Reshape outputs
        coarse_densities = coarse_densities.view(ray_origins.shape[0], -1, 1)
        coarse_colors = coarse_colors.view(ray_origins.shape[0], -1, 3)
        
        # Volume rendering for coarse samples
        coarse_outputs = self.renderer.volume_rendering(
            coarse_colors, coarse_densities, coarse_t_vals, ray_directions, background_color
        )
        
        # Fine sampling using hierarchical sampling
        fine_points, fine_t_vals = self.renderer.hierarchical_sampling(
            ray_origins, ray_directions,
            coarse_outputs['weights'], coarse_t_vals,
            self.config.num_samples_fine
        )
        
        # Query grid features for fine points
        fine_points_flat = fine_points.view(-1, 3)
        fine_grid_features = self.query_grid_features(fine_points_flat)
        
        # Prepare view directions for fine points
        total_samples = self.config.num_samples_coarse + self.config.num_samples_fine
        view_dirs_fine = ray_directions.unsqueeze(-2).expand(-1, total_samples, -1)
        view_dirs_fine_flat = view_dirs_fine.reshape(-1, 3)
        
        # Forward pass through MLP for fine points
        fine_densities, fine_colors = self.grid_mlp(
            fine_grid_features, fine_points_flat, view_dirs_fine_flat
        )
        
        # Reshape outputs
        fine_densities = fine_densities.view(ray_origins.shape[0], -1, 1)
        fine_colors = fine_colors.view(ray_origins.shape[0], -1, 3)
        
        # Volume rendering for fine samples
        fine_outputs = self.renderer.volume_rendering(
            fine_colors, fine_densities, fine_t_vals, ray_directions, background_color
        )
        
        return {
            'rgb_coarse': coarse_outputs['rgb'],
            'depth_coarse': coarse_outputs['depth'],
            'opacity_coarse': coarse_outputs['opacity'],
            'rgb_fine': fine_outputs['rgb'],
            'depth_fine': fine_outputs['depth'],
            'opacity_fine': fine_outputs['opacity'],
            'weights_coarse': coarse_outputs['weights'],
            'weights_fine': fine_outputs['weights']
        }
    
    def update_grid_features(self, threshold: float = None):
        """Update grid features by pruning inactive cells."""
        if threshold is None:
            threshold = self.config.density_threshold
        
        with torch.no_grad():
            for level in range(self.config.num_grid_levels):
                occupied_mask = self.hierarchical_grid.get_occupied_cells(level, threshold)
                
                # Zero out features for unoccupied cells
                grid_features = self.hierarchical_grid.grids[level]
                grid_features.data[~occupied_mask] = 0.0


class GridNeRFLoss(nn.Module):
    """Loss function for Grid-NeRF training."""
    
    def __init__(self, config: GridNeRFConfig):
        super().__init__()
        self.config = config
    
    def forward(self, predictions: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute Grid-NeRF losses.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            
        Returns:
            Dictionary containing loss components
        """
        losses = {}
        
        # Color loss for coarse and fine networks
        color_loss_coarse = F.mse_loss(predictions['rgb_coarse'], targets['rgb'])
        color_loss_fine = F.mse_loss(predictions['rgb_fine'], targets['rgb'])
        color_loss = color_loss_coarse + color_loss_fine
        
        losses['color_loss'] = color_loss * self.config.color_loss_weight
        
        # Depth loss if available
        if 'depth' in targets:
            depth_loss_coarse = F.mse_loss(predictions['depth_coarse'], targets['depth'])
            depth_loss_fine = F.mse_loss(predictions['depth_fine'], targets['depth'])
            depth_loss = depth_loss_coarse + depth_loss_fine
            losses['depth_loss'] = depth_loss * self.config.depth_loss_weight
        
        # Grid regularization loss
        grid_reg_loss = 0.0
        for level, grid_features in enumerate(predictions.get('grid_features', [])):
            grid_reg_loss += torch.mean(torch.norm(grid_features, dim=-1))
        
        losses['grid_regularization'] = grid_reg_loss * self.config.grid_regularization_weight
        
        # Total loss
        total_loss = sum(losses.values())
        losses['total_loss'] = total_loss
        
        return losses 