"""
PyNeRF Core Module
Implements the main PyNeRF model with pyramidal representation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import numpy as np
import logging

from .pyramid_encoder import PyramidEncoder, HashEncoder
from .pyramid_renderer import PyramidRenderer
from .utils import compute_sample_area, get_pyramid_level, interpolate_pyramid_outputs

logger = logging.getLogger(__name__)


@dataclass
class PyNeRFConfig:
    """Configuration for PyNeRF model"""
    
    # Pyramid structure
    num_levels: int = 8
    base_resolution: int = 16
    scale_factor: float = 2.0
    max_resolution: int = 2048
    
    # Hash encoding
    hash_table_size: int = 2**20
    features_per_level: int = 4
    
    # MLP architecture
    hidden_dim: int = 64
    num_layers: int = 2
    
    # Rendering
    num_samples: int = 64
    num_importance_samples: int = 128
    use_viewdirs: bool = True
    
    # Training
    learning_rate: float = 5e-4
    batch_size: int = 8192
    max_steps: int = 20000
    
    # Anti-aliasing
    use_integrated_encoding: bool = True
    cone_angle: float = 0.00628  # 1/159.15
    
    # Device
    device: str = "cuda"


class PyNeRF(nn.Module):
    """
    PyNeRF: Pyramidal Neural Radiance Fields
    
    Implements a pyramid of grid-based NeRF models at different resolutions.
    Uses multi-resolution hash encoding with separate MLP heads for each level.
    """
    
    def __init__(self, config: PyNeRFConfig):
        super().__init__()
        self.config = config
        
        # Initialize pyramid levels
        self.pyramid_levels = self._create_pyramid_levels()
        
        # Create encoders for each level
        self.encoders = nn.ModuleDict()
        self.mlps = nn.ModuleDict()
        
        for level in range(config.num_levels):
            resolution = min(
                config.base_resolution * (config.scale_factor ** level),
                config.max_resolution
            )
            
            # Hash encoder for this level
            encoder = HashEncoder(
                resolution=int(resolution),
                hash_table_size=config.hash_table_size,
                features_per_level=config.features_per_level,
                num_levels=1  # Single level per encoder
            )
            self.encoders[f"level_{level}"] = encoder
            
            # MLP head for this level
            mlp = PyNeRFMLP(
                input_dim=config.features_per_level,
                hidden_dim=config.hidden_dim,
                num_layers=config.num_layers,
                use_viewdirs=config.use_viewdirs
            )
            self.mlps[f"level_{level}"] = mlp
        
        # Pyramid renderer
        self.renderer = PyramidRenderer(config)
        
        logger.info(f"Initialized PyNeRF with {config.num_levels} pyramid levels")
    
    def _create_pyramid_levels(self) -> List[int]:
        """Create list of pyramid level resolutions"""
        levels = []
        for level in range(self.config.num_levels):
            resolution = min(
                self.config.base_resolution * (self.config.scale_factor ** level),
                self.config.max_resolution
            )
            levels.append(int(resolution))
        return levels
    
    def forward(
        self,
        rays_o: torch.Tensor,
        rays_d: torch.Tensor,
        bounds: torch.Tensor,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through PyNeRF
        
        Args:
            rays_o: Ray origins [N, 3]
            rays_d: Ray directions [N, 3]
            bounds: Near/far bounds [N, 2]
            
        Returns:
            Dictionary containing rendered outputs
        """
        batch_size = rays_o.shape[0]
        device = rays_o.device
        
        # Sample points along rays
        t_vals = torch.linspace(0.0, 1.0, self.config.num_samples, device=device)
        z_vals = bounds[..., 0:1] * (1.0 - t_vals) + bounds[..., 1:2] * t_vals
        z_vals = z_vals.expand(batch_size, self.config.num_samples)
        
        # Add noise to z_vals during training
        if self.training:
            mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
            upper = torch.cat([mids, z_vals[..., -1:]], -1)
            lower = torch.cat([z_vals[..., :1], mids], -1)
            t_rand = torch.rand(z_vals.shape, device=device)
            z_vals = lower + (upper - lower) * t_rand
        
        # Compute sample positions
        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
        pts_flat = pts.reshape(-1, 3)
        
        # Compute sample areas for pyramid level selection
        sample_areas = compute_sample_area(rays_o, rays_d, z_vals)
        
        # Query pyramid levels
        pyramid_outputs = self._query_pyramid(pts_flat, rays_d, sample_areas)
        
        # Reshape outputs
        rgb = pyramid_outputs["rgb"].reshape(batch_size, self.config.num_samples, 3)
        sigma = pyramid_outputs["sigma"].reshape(batch_size, self.config.num_samples)
        
        # Volume rendering
        outputs = self.renderer.render(rgb, sigma, z_vals, rays_d)
        
        return outputs
    
    def _query_pyramid(
        self,
        pts: torch.Tensor,
        rays_d: torch.Tensor,
        sample_areas: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Query pyramid levels based on sample areas
        
        Args:
            pts: Sample points [N, 3]
            rays_d: Ray directions [N_rays, 3]
            sample_areas: Sample areas [N_rays, N_samples]
            
        Returns:
            Dictionary with aggregated rgb and sigma values
        """
        device = pts.device
        num_points = pts.shape[0]
        
        # Determine pyramid levels for each sample
        pyramid_levels = get_pyramid_level(
            sample_areas.flatten(),
            self.pyramid_levels,
            self.config.scale_factor
        )
        
        # Initialize outputs
        rgb_outputs = []
        sigma_outputs = []
        weights = []
        
        # Query each level
        for level in range(self.config.num_levels):
            level_mask = (pyramid_levels == level)
            if not level_mask.any():
                continue
            
            level_pts = pts[level_mask]
            level_dirs = rays_d.repeat_interleave(
                self.config.num_samples, dim=0
            )[level_mask]
            
            # Encode positions
            encoder = self.encoders[f"level_{level}"]
            encoded_pts = encoder(level_pts)
            
            # MLP forward pass
            mlp = self.mlps[f"level_{level}"]
            level_rgb, level_sigma = mlp(encoded_pts, level_dirs)
            
            # Store outputs with level weights
            level_weight = torch.ones(level_pts.shape[0], device=device)
            
            rgb_outputs.append((level_rgb, level_mask, level_weight))
            sigma_outputs.append((level_sigma, level_mask, level_weight))
        
        # Interpolate between pyramid levels
        final_rgb, final_sigma = interpolate_pyramid_outputs(
            rgb_outputs, sigma_outputs, pyramid_levels, num_points, device
        )
        
        return {
            "rgb": final_rgb,
            "sigma": final_sigma
        }
    
    def get_pyramid_info(self) -> Dict[str, any]:
        """Get information about pyramid structure"""
        return {
            "num_levels": self.config.num_levels,
            "resolutions": self.pyramid_levels,
            "scale_factor": self.config.scale_factor,
            "total_parameters": sum(p.numel() for p in self.parameters())
        }


class PyNeRFMLP(nn.Module):
    """MLP head for PyNeRF pyramid level"""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        use_viewdirs: bool = True
    ):
        super().__init__()
        self.use_viewdirs = use_viewdirs
        
        # Density network
        density_layers = []
        for i in range(num_layers):
            if i == 0:
                density_layers.append(nn.Linear(input_dim, hidden_dim))
            else:
                density_layers.append(nn.Linear(hidden_dim, hidden_dim))
            density_layers.append(nn.ReLU(inplace=True))
        
        density_layers.append(nn.Linear(hidden_dim, 1))
        self.density_net = nn.Sequential(*density_layers)
        
        # Color network
        color_input_dim = hidden_dim
        if use_viewdirs:
            color_input_dim += 3  # Add viewing direction
        
        color_layers = []
        for i in range(num_layers):
            if i == 0:
                color_layers.append(nn.Linear(color_input_dim, hidden_dim))
            else:
                color_layers.append(nn.Linear(hidden_dim, hidden_dim))
            color_layers.append(nn.ReLU(inplace=True))
        
        color_layers.append(nn.Linear(hidden_dim, 3))
        color_layers.append(nn.Sigmoid())
        self.color_net = nn.Sequential(*color_layers)
        
        # Feature extraction for color network
        self.feature_linear = nn.Linear(input_dim, hidden_dim)
    
    def forward(
        self,
        encoded_pts: torch.Tensor,
        viewdirs: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through MLP
        
        Args:
            encoded_pts: Encoded position features [N, input_dim]
            viewdirs: Viewing directions [N, 3]
            
        Returns:
            Tuple of (rgb, sigma)
        """
        # Density prediction
        sigma = self.density_net(encoded_pts)
        sigma = F.relu(sigma)
        
        # Color prediction
        features = F.relu(self.feature_linear(encoded_pts))
        
        if self.use_viewdirs and viewdirs is not None:
            # Normalize viewing directions
            viewdirs = F.normalize(viewdirs, dim=-1)
            color_input = torch.cat([features, viewdirs], dim=-1)
        else:
            color_input = features
        
        rgb = self.color_net(color_input)
        
        return rgb, sigma.squeeze(-1)


class PyNeRFLoss(nn.Module):
    """Loss function for PyNeRF training"""
    
    def __init__(self, config: PyNeRFConfig):
        super().__init__()
        self.config = config
        self.mse_loss = nn.MSELoss()
    
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute PyNeRF loss
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            
        Returns:
            Dictionary of loss components
        """
        losses = {}
        
        # Main color loss
        if "rgb" in predictions and "rgb" in targets:
            losses["color_loss"] = self.mse_loss(predictions["rgb"], targets["rgb"])
        
        # Coarse color loss (if using hierarchical sampling)
        if "rgb_coarse" in predictions and "rgb" in targets:
            losses["color_loss_coarse"] = self.mse_loss(
                predictions["rgb_coarse"], targets["rgb"]
            )
        
        # Total loss
        total_loss = sum(losses.values())
        losses["total_loss"] = total_loss
        
        return losses 