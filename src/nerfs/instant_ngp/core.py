"""
Instant NGP: Instant Neural Graphics Primitives with Multiresolution Hash Encoding.

This module implements the Instant NGP model as described in:
"Instant Neural Graphics Primitives with a Multiresolution Hash Encoding"
by Thomas Müller et al. (SIGGRAPH 2022)

Key components:
- Multiresolution hash encoding for efficient feature lookup
- Small MLP networks for fast inference
- CUDA-optimized hash table operations (fallback to PyTorch for compatibility)
- Support for NeRF and other neural graphics primitives
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass


@dataclass 
class InstantNGPConfig:
    """Configuration for Instant NGP model."""
    
    # Hash encoding parameters
    num_levels: int = 16                 # Number of resolution levels
    level_dim: int = 2                   # Features per level
    per_level_scale: float = 2.0         # Scale factor between levels
    base_resolution: int = 16            # Base grid resolution
    log2_hashmap_size: int = 19          # Hash table size (2^19)
    desired_resolution: int = 2048       # Finest resolution
    
    # Network architecture
    geo_feat_dim: int = 15               # Geometry feature dimension
    hidden_dim: int = 64                 # Hidden layer dimension
    hidden_dim_color: int = 64           # Color network hidden dimension
    num_layers: int = 2                  # Number of hidden layers
    num_layers_color: int = 3            # Color network layers
    
    # Positional encoding for directions
    dir_pe: int = 4                      # Direction PE levels
    
    # Training parameters
    learning_rate: float = 1e-2          # Learning rate
    learning_rate_decay: float = 0.33    # LR decay factor
    decay_step: int = 1000               # Decay step size
    weight_decay: float = 1e-6           # Weight decay
    
    # Rendering parameters
    density_activation: str = 'exp'      # Density activation function
    density_bias: float = -1.0           # Density bias
    rgb_activation: str = 'sigmoid'      # RGB activation function
    
    # Scene bounds
    bound: float = 2.0                   # Scene bound (normalized to [-bound, bound])
    
    # Loss parameters
    lambda_entropy: float = 1e-4         # Entropy regularization weight
    lambda_tv: float = 1e-4              # Total variation loss weight


class HashEncoder(nn.Module):
    """Multiresolution hash encoding."""
    
    def __init__(self, config: InstantNGPConfig):
        super().__init__()
        self.config = config
        
        # Calculate encoding parameters
        self.num_levels = config.num_levels
        self.level_dim = config.level_dim
        self.per_level_scale = config.per_level_scale
        self.base_resolution = config.base_resolution
        self.log2_hashmap_size = config.log2_hashmap_size
        self.desired_resolution = config.desired_resolution
        
        # Compute maximum resolution
        max_res = int(np.ceil(config.base_resolution * (config.per_level_scale ** (config.num_levels - 1))))
        if max_res > config.desired_resolution:
            config.desired_resolution = max_res
        
        # Initialize hash tables for each level
        self.embeddings = nn.ModuleList()
        self.resolutions = []
        
        for i in range(self.num_levels):
            resolution = int(np.ceil(config.base_resolution * (config.per_level_scale ** i)))
            params_in_level = min(resolution ** 3, 2 ** config.log2_hashmap_size)
            
            self.resolutions.append(resolution)
            embedding = nn.Embedding(params_in_level, config.level_dim)
            
            # Xavier uniform initialization
            std = 1e-4
            nn.init.uniform_(embedding.weight, -std, std)
            
            self.embeddings.append(embedding)
        
        self.output_dim = self.num_levels * self.level_dim
    
    def hash_function(self, coords, resolution):
        """Simple hash function for grid coordinates."""
        # Use a simple hash function - in practice this would be optimized CUDA code
        primes = torch.tensor([1, 2654435761, 805459861], device=coords.device, dtype=torch.long)
        
        # Clamp coordinates to valid range
        coords = coords.clamp(0, resolution - 1).long()
        
        # Compute hash
        hash_val = torch.zeros(coords.shape[0], device=coords.device, dtype=torch.long)
        for i in range(3):
            hash_val ^= coords[:, i] * primes[i]
        
        return hash_val % self.embeddings[0].num_embeddings
    
    def get_grid_coordinates(self, positions, resolution):
        """Get grid coordinates and interpolation weights."""
        # Scale positions to grid resolution
        scaled_positions = (positions + 1) / 2 * (resolution - 1)  # [-1, 1] -> [0, res-1]
        
        # Get integer and fractional parts
        grid_coords = torch.floor(scaled_positions).long()
        weights = scaled_positions - grid_coords.float()
        
        return grid_coords, weights
    
    def trilinear_interpolation(self, features, weights):
        """Perform trilinear interpolation."""
        # features: [N, 8, D] - 8 corner features
        # weights: [N, 3] - interpolation weights
        
        # Unpack weights
        wx, wy, wz = weights.unbind(-1)
        
        # Expand weights for broadcasting with features
        wx = wx.unsqueeze(-1)  # [N, 1]
        wy = wy.unsqueeze(-1)  # [N, 1]
        wz = wz.unsqueeze(-1)  # [N, 1]
        
        # Bilinear interpolation in xy plane for z=0 and z=1
        c00 = features[:, 0] * (1 - wx) + features[:, 1] * wx  # (0, 0, 0) and (1, 0, 0)
        c01 = features[:, 2] * (1 - wx) + features[:, 3] * wx  # (0, 1, 0) and (1, 1, 0)
        c10 = features[:, 4] * (1 - wx) + features[:, 5] * wx  # (0, 0, 1) and (1, 0, 1)
        c11 = features[:, 6] * (1 - wx) + features[:, 7] * wx  # (0, 1, 1) and (1, 1, 1)
        
        # Interpolate in y direction
        c0 = c00 * (1 - wy) + c01 * wy  # z=0 plane
        c1 = c10 * (1 - wy) + c11 * wy  # z=1 plane
        
        # Interpolate in z direction
        result = c0 * (1 - wz) + c1 * wz
        
        return result
    
    def forward(self, positions):
        """
        Encode positions using multiresolution hash encoding.
        
        Args:
            positions: [N, 3] input positions in [-1, 1]
            
        Returns:
            encoded: [N, output_dim] encoded features
        """
        batch_size = positions.shape[0]
        device = positions.device
        
        # Collect features from all levels
        encoded_features = []
        
        for level_idx in range(self.num_levels):
            resolution = self.resolutions[level_idx]
            embedding = self.embeddings[level_idx]
            
            # Get grid coordinates and weights
            grid_coords, weights = self.get_grid_coordinates(positions, resolution)
            
            # Get 8 corner coordinates for trilinear interpolation
            corner_coords = []
            for dx in [0, 1]:
                for dy in [0, 1]:
                    for dz in [0, 1]:
                        corner = grid_coords + torch.tensor([dx, dy, dz], device=device)
                        corner = torch.clamp(corner, 0, resolution - 1)
                        corner_coords.append(corner)
            
            corner_coords = torch.stack(corner_coords, dim=1)  # [N, 8, 3]
            
            # Hash coordinates to get indices
            corner_indices = []
            for i in range(8):
                indices = self.hash_function(corner_coords[:, i], resolution)
                corner_indices.append(indices)
            
            corner_indices = torch.stack(corner_indices, dim=1)  # [N, 8]
            
            # Look up embeddings
            corner_features = embedding(corner_indices)  # [N, 8, level_dim]
            
            # Trilinear interpolation
            interpolated = self.trilinear_interpolation(corner_features, weights)
            
            encoded_features.append(interpolated)
        
        # Concatenate features from all levels
        encoded = torch.cat(encoded_features, dim=-1)
        return encoded


class SHEncoder(nn.Module):
    """Spherical harmonics encoder for view directions."""
    
    def __init__(self, degree: int = 4):
        super().__init__()
        self.degree = degree
        self.output_dim = degree ** 2
    
    def forward(self, directions):
        """Encode directions using spherical harmonics."""
        # Normalize directions
        directions = F.normalize(directions, dim=-1)
        
        x, y, z = directions.unbind(-1)
        
        # Compute spherical harmonics up to specified degree
        sh_features = []
        
        # Degree 0
        sh_features.append(0.28209479177387814 * torch.ones_like(x))  # Y_0^0
        
        if self.degree > 1:
            # Degree 1
            sh_features.append(-0.48860251190291987 * y)     # Y_1^{-1}
            sh_features.append(0.48860251190291987 * z)      # Y_1^0
            sh_features.append(-0.48860251190291987 * x)     # Y_1^1
        
        if self.degree > 2:
            # Degree 2
            sh_features.append(1.0925484305920792 * x * y)   # Y_2^{-2}
            sh_features.append(-1.0925484305920792 * y * z)  # Y_2^{-1}
            sh_features.append(0.31539156525252005 * (2 * z**2 - x**2 - y**2))  # Y_2^0
            sh_features.append(-1.0925484305920792 * x * z)  # Y_2^1
            sh_features.append(0.5462742152960396 * (x**2 - y**2))  # Y_2^2
        
        if self.degree > 3:
            # Degree 3
            sh_features.append(-0.5900435899266435 * y * (3 * x**2 - y**2))  # Y_3^{-3}
            sh_features.append(2.890611442640554 * x * y * z)                # Y_3^{-2}
            sh_features.append(-0.4570457994644658 * y * (4 * z**2 - x**2 - y**2))  # Y_3^{-1}
            sh_features.append(0.3731763325901154 * z * (2 * z**2 - 3 * x**2 - 3 * y**2))  # Y_3^0
            sh_features.append(-0.4570457994644658 * x * (4 * z**2 - x**2 - y**2))  # Y_3^1
            sh_features.append(1.445305721320277 * z * (x**2 - y**2))        # Y_3^2
            sh_features.append(-0.5900435899266435 * x * (x**2 - 3 * y**2))  # Y_3^3
        
        return torch.stack(sh_features[:self.output_dim], dim=-1)


class InstantNGP(nn.Module):
    """Instant NGP model with hash encoding."""
    
    def __init__(self, config: InstantNGPConfig):
        super().__init__()
        self.config = config
        
        # Hash encoder for positions
        self.position_encoder = HashEncoder(config)
        
        # Direction encoder
        self.direction_encoder = SHEncoder(config.dir_pe)
        
        # Geometry network
        geo_input_dim = self.position_encoder.output_dim
        
        self.geo_mlp = nn.Sequential(
            nn.Linear(geo_input_dim, config.hidden_dim), nn.ReLU(inplace=True)
        )
        
        for i in range(config.num_layers - 1):
            self.geo_mlp.add_module(
                f"layer_{i+1}", nn.Sequential(
                    nn.Linear(config.hidden_dim, config.hidden_dim), nn.ReLU(inplace=True)
                )
            )
        
        # Density head
        self.density_head = nn.Linear(config.hidden_dim, 1 + config.geo_feat_dim)
        
        # Color network
        color_input_dim = config.geo_feat_dim + self.direction_encoder.output_dim
        
        self.color_mlp = nn.Sequential(
            nn.Linear(color_input_dim, config.hidden_dim_color), nn.ReLU(inplace=True)
        )
        
        for i in range(config.num_layers_color - 1):
            self.color_mlp.add_module(
                f"color_layer_{i+1}", nn.Sequential(
                    nn.Linear(
                        config.hidden_dim_color,
                        config.hidden_dim_color,
                    )
                )
            )
        
        # RGB head
        self.rgb_head = nn.Linear(config.hidden_dim_color, 3)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        """Initialize network weights."""
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    
    def density_activation(self, density):
        """Apply density activation function."""
        if self.config.density_activation == 'exp':
            return torch.exp(density + self.config.density_bias)
        elif self.config.density_activation == 'softplus':
            return F.softplus(density + self.config.density_bias)
        else:
            return F.relu(density)
    
    def rgb_activation(self, rgb):
        """Apply RGB activation function."""
        if self.config.rgb_activation == 'sigmoid':
            return torch.sigmoid(rgb)
        elif self.config.rgb_activation == 'tanh':
            return torch.tanh(rgb) * 0.5 + 0.5
        else:
            return rgb
    
    def forward(self, positions, directions=None):
        """
        Forward pass through Instant NGP.
        
        Args:
            positions: [N, 3] 3D positions in [-bound, bound]
            directions: [N, 3] viewing directions (optional)
            
        Returns:
            density: [N, 1] volume density
            rgb: [N, 3] RGB color (if directions provided)
        """
        # Normalize positions to [-1, 1]
        positions_normalized = positions / self.config.bound
        
        # Encode positions
        pos_encoded = self.position_encoder(positions_normalized)
        
        # Geometry forward pass
        geo_features = self.geo_mlp(pos_encoded)
        geo_output = self.density_head(geo_features)
        
        # Extract density and geometry features
        density = geo_output[:, 0:1]
        geo_feat = geo_output[:, 1:]
        
        # Apply density activation
        density = self.density_activation(density)
        
        if directions is None:
            return density
        
        # Encode directions
        dir_encoded = self.direction_encoder(directions)
        
        # Color forward pass
        color_input = torch.cat([geo_feat, dir_encoded], dim=-1)
        color_features = self.color_mlp(color_input)
        rgb = self.rgb_head(color_features)
        
        # Apply RGB activation
        rgb = self.rgb_activation(rgb)
        
        return density, rgb
    
    def get_density(self, positions):
        """Get density at positions."""
        with torch.no_grad():
            density = self.forward(positions)
            return density


class InstantNGPLoss(nn.Module):
    """Loss function for Instant NGP training."""
    
    def __init__(self, config: InstantNGPConfig):
        super().__init__()
        self.config = config
    
    def forward(self, pred_rgb, target_rgb, pred_density=None, positions=None):
        """
        Compute Instant NGP losses.
        
        Args:
            pred_rgb: [N, 3] predicted RGB
            target_rgb: [N, 3] target RGB
            pred_density: [N, 1] predicted density (optional)
            positions: [N, 3] 3D positions (optional)
            
        Returns:
            Dictionary of losses
        """
        losses = {}
        
        # RGB reconstruction loss
        rgb_loss = F.mse_loss(pred_rgb, target_rgb)
        losses['rgb_loss'] = rgb_loss
        
        # Entropy regularization on density
        if pred_density is not None and self.config.lambda_entropy > 0:
            # Encourage sparsity in density
            entropy_loss = -torch.mean(pred_density * torch.log(pred_density + 1e-10))
            losses['entropy_loss'] = self.config.lambda_entropy * entropy_loss
        
        # Total variation loss on hash grid (simplified)
        if self.config.lambda_tv > 0:
            tv_loss = 0.0
            # In practice, this would compute TV loss on the hash grid
            # For simplicity, we use a placeholder
            losses['tv_loss'] = self.config.lambda_tv * tv_loss
        
        # Total loss
        total_loss = sum(losses.values())
        losses['total_loss'] = total_loss
        
        # PSNR for monitoring
        with torch.no_grad():
            mse = F.mse_loss(pred_rgb, target_rgb)
            psnr = -10. * torch.log10(mse + 1e-8)
            losses['psnr'] = psnr
        
        return losses


class InstantNGPRenderer:
    """Renderer for Instant NGP."""
    
    def __init__(self, config: InstantNGPConfig):
        self.config = config
    
    def sample_rays(self, rays_o, rays_d, near, far, num_samples):
        """Sample points along rays."""
        device = rays_o.device
        num_rays = rays_o.shape[0]
        
        # Linear sampling in depth
        t_vals = torch.linspace(0., 1., steps=num_samples, device=device)
        z_vals = near[..., None] * (1. - t_vals) + far[..., None] * t_vals  # [N_rays, N_samples]
        
        # Add noise for stochastic sampling
        if hasattr(self, 'training') and self.training:
            mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
            upper = torch.cat([mids, z_vals[..., -1:]], -1)
            lower = torch.cat([z_vals[..., :1], mids], -1)
            t_rand = torch.rand(z_vals.shape, device=device)
            z_vals = lower + (upper - lower) * t_rand
        
        # Get sample points
        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]  # [N_rays, N_samples, 3]
        
        return pts.reshape(-1, 3), z_vals
    
    def volume_render(self, rgb, density, z_vals, rays_d):
        """Volume rendering with alpha compositing."""
        device = rgb.device
        
        # Compute distances between adjacent samples
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat(
            [dists,
            torch.tensor([1e10])
        ])
        
        # Multiply by ray direction norm
        dists = dists * torch.norm(rays_d[..., None, :], dim=-1)
        
        # Compute alpha values
        alpha = 1. - torch.exp(-density.squeeze(-1) * dists)  # [N_rays, N_samples]
        
        # Compute transmittance
        transmittance = torch.cumprod(
            torch.cat([torch.ones_like(alpha[..., :1]), 1. - alpha + 1e-10], -1)
        )
        
        # Compute weights
        weights = alpha * transmittance  # [N_rays, N_samples]
        
        # Composite RGB
        rgb_map = torch.sum(weights[..., None] * rgb, -2)  # [N_rays, 3]
        
        # Composite depth
        depth_map = torch.sum(weights * z_vals, -1)  # [N_rays]
        
        # Accumulated opacity
        acc_map = torch.sum(weights, -1)  # [N_rays]
        
        return rgb_map, depth_map, acc_map, weights
    
    def render_rays(self, model, rays_o, rays_d, near, far, num_samples=128):
        """Render rays through the model."""
        # Sample points along rays
        pts, z_vals = self.sample_rays(rays_o, rays_d, near, far, num_samples)
        
        # Get view directions
        viewdirs = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
        viewdirs = viewdirs[:, None].expand(-1, num_samples, -1).reshape(-1, 3)
        
        # Query model
        density, rgb = model(pts, viewdirs)
        
        # Reshape outputs
        num_rays = rays_o.shape[0]
        rgb = rgb.reshape(num_rays, num_samples, 3)
        density = density.reshape(num_rays, num_samples, 1)
        
        # Volume rendering
        rgb_map, depth_map, acc_map, weights = self.volume_render(rgb, density, z_vals, rays_d)
        
        return {
            'rgb': rgb_map, 'depth': depth_map, 'acc': acc_map, 'weights': weights, 'z_vals': z_vals
        }
