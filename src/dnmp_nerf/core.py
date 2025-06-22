"""
Core components of DNMP-NeRF implementation.

This module contains the main classes for Deformable Neural Mesh Primitives,
including the configuration, primitive definition, renderer, and loss functions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import numpy as np


@dataclass
class DNMPConfig:
    """Configuration for DNMP model."""
    
    # Mesh primitive settings
    primitive_resolution: int = 32  # Resolution of each primitive mesh
    latent_dim: int = 128  # Dimension of latent code for mesh shape
    vertex_feature_dim: int = 64  # Dimension of vertex features
    
    # Voxel grid settings
    voxel_size: float = 1.0  # Size of each voxel
    scene_bounds: Tuple[float, float, float, float, float, float] = (-50, 50, -50, 50, -5, 5)
    
    # Network architecture
    mlp_hidden_dim: int = 256
    mlp_num_layers: int = 4
    view_dependent: bool = True
    
    # Rendering settings
    max_ray_samples: int = 64
    near_plane: float = 0.1
    far_plane: float = 100.0
    
    # Training settings
    geometry_lr: float = 1e-3
    radiance_lr: float = 5e-4
    weight_decay: float = 1e-4
    
    # Loss weights
    color_loss_weight: float = 1.0
    depth_loss_weight: float = 0.1
    mesh_regularization_weight: float = 0.01
    latent_regularization_weight: float = 0.001


class DeformableNeuralMeshPrimitive(nn.Module):
    """
    A single Deformable Neural Mesh Primitive (DNMP).
    
    Each DNMP consists of:
    1. A latent code that encodes the mesh shape
    2. Vertex features for radiance information
    3. A mesh decoder that generates vertex positions from latent code
    """
    
    def __init__(self, config: DNMPConfig, mesh_decoder: nn.Module):
        super().__init__()
        self.config = config
        self.mesh_decoder = mesh_decoder
        
        # Learnable latent code for mesh shape
        self.latent_code = nn.Parameter(
            torch.randn(config.latent_dim) * 0.1
        )
        
        # Vertex features for radiance information
        # Number of vertices depends on primitive resolution
        num_vertices = self._get_num_vertices(config.primitive_resolution)
        self.vertex_features = nn.Parameter(
            torch.randn(num_vertices, config.vertex_feature_dim) * 0.1
        )
        
        # Base mesh topology (fixed connectivity)
        self.register_buffer('faces', self._generate_base_faces(config.primitive_resolution))
        
    def _get_num_vertices(self, resolution: int) -> int:
        """Calculate number of vertices for given resolution."""
        return (resolution + 1) ** 3
        
    def _generate_base_faces(self, resolution: int) -> torch.Tensor:
        """Generate base mesh faces for cubic primitive."""
        faces = []
        for i in range(resolution):
            for j in range(resolution):
                for k in range(resolution):
                    # Create 12 triangular faces for each voxel
                    base_idx = i * (resolution + 1) ** 2 + j * (resolution + 1) + k
                    
                    # Define the 8 vertices of the cube
                    v000 = base_idx
                    v001 = base_idx + 1
                    v010 = base_idx + (resolution + 1)
                    v011 = base_idx + (resolution + 1) + 1
                    v100 = base_idx + (resolution + 1) ** 2
                    v101 = base_idx + (resolution + 1) ** 2 + 1
                    v110 = base_idx + (resolution + 1) ** 2 + (resolution + 1)
                    v111 = base_idx + (resolution + 1) ** 2 + (resolution + 1) + 1
                    
                    # Add 12 triangular faces
                    cube_faces = [
                        [v000, v001, v011], [v000, v011, v010],  # bottom
                        [v100, v110, v111], [v100, v111, v101],  # top
                        [v000, v100, v101], [v000, v101, v001],  # front
                        [v010, v011, v111], [v010, v111, v110],  # back
                        [v000, v010, v110], [v000, v110, v100],  # left
                        [v001, v101, v111], [v001, v111, v011],  # right
                    ]
                    faces.extend(cube_faces)
        
        return torch.tensor(faces, dtype=torch.long)
    
    def get_mesh_vertices(self) -> torch.Tensor:
        """Get deformed mesh vertices from latent code."""
        return self.mesh_decoder(self.latent_code.unsqueeze(0)).squeeze(0)
    
    def forward(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of DNMP.
        
        Returns:
            vertices: Deformed mesh vertices [N, 3]
            faces: Mesh faces [M, 3]
            vertex_features: Vertex features [N, F]
        """
        vertices = self.get_mesh_vertices()
        return vertices, self.faces, self.vertex_features


class RadianceMLP(nn.Module):
    """MLP for predicting radiance from interpolated vertex features."""
    
    def __init__(self, config: DNMPConfig):
        super().__init__()
        self.config = config
        
        input_dim = config.vertex_feature_dim
        if config.view_dependent:
            input_dim += 3  # Add view direction
            
        layers = []
        for i in range(config.mlp_num_layers):
            if i == 0:
                layers.append(nn.Linear(input_dim, config.mlp_hidden_dim))
            elif i == config.mlp_num_layers - 1:
                layers.append(nn.Linear(config.mlp_hidden_dim, 4))  # RGB + opacity
            else:
                layers.append(nn.Linear(config.mlp_hidden_dim, config.mlp_hidden_dim))
            
            if i < config.mlp_num_layers - 1:
                layers.append(nn.ReLU(inplace=True))
                
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, vertex_features: torch.Tensor, 
                view_dirs: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Predict radiance from vertex features.
        
        Args:
            vertex_features: Interpolated vertex features [N, F]
            view_dirs: View directions [N, 3] (optional)
            
        Returns:
            radiance: RGB + opacity [N, 4]
        """
        if self.config.view_dependent and view_dirs is not None:
            x = torch.cat([vertex_features, view_dirs], dim=-1)
        else:
            x = vertex_features
            
        radiance = self.mlp(x)
        
        # Apply activations
        rgb = torch.sigmoid(radiance[..., :3])
        opacity = torch.sigmoid(radiance[..., 3:4])
        
        return torch.cat([rgb, opacity], dim=-1)


class DNMPRenderer(nn.Module):
    """Renderer for DNMP-based scene representation."""
    
    def __init__(self, config: DNMPConfig):
        super().__init__()
        self.config = config
        self.radiance_mlp = RadianceMLP(config)
        
    def render_ray(self, ray_origins: torch.Tensor, ray_directions: torch.Tensor,
                   primitives: List[DeformableNeuralMeshPrimitive],
                   rasterizer) -> Dict[str, torch.Tensor]:
        """
        Render a batch of rays through the DNMP scene.
        
        Args:
            ray_origins: Ray origins [N, 3]
            ray_directions: Ray directions [N, 3]
            primitives: List of DNMP primitives
            rasterizer: Rasterization module
            
        Returns:
            Dictionary containing rendered colors, depths, etc.
        """
        batch_size = ray_origins.shape[0]
        device = ray_origins.device
        
        # Initialize outputs
        colors = torch.zeros(batch_size, 3, device=device)
        depths = torch.zeros(batch_size, device=device)
        weights = torch.zeros(batch_size, device=device)
        
        # Sample points along rays
        t_vals = torch.linspace(
            self.config.near_plane, self.config.far_plane,
            self.config.max_ray_samples, device=device
        )
        t_vals = t_vals.expand(batch_size, -1)  # [N, S]
        
        # Add noise to sampling positions during training
        if self.training:
            t_vals = t_vals + torch.rand_like(t_vals) * (
                self.config.far_plane - self.config.near_plane
            ) / self.config.max_ray_samples
            
        # Compute sample positions
        sample_positions = ray_origins.unsqueeze(1) + \
                          ray_directions.unsqueeze(1) * t_vals.unsqueeze(-1)  # [N, S, 3]
        
        # Rasterize all primitives to get interpolated features
        all_features = []
        all_opacities = []
        
        for primitive in primitives:
            vertices, faces, vertex_features = primitive()
            
            # Rasterize primitive to get interpolated features at sample positions
            features, valid_mask = rasterizer.interpolate_features(
                sample_positions.reshape(-1, 3),
                vertices, faces, vertex_features
            )
            
            if features is not None:
                features = features.reshape(batch_size, self.config.max_ray_samples, -1)
                valid_mask = valid_mask.reshape(batch_size, self.config.max_ray_samples)
                
                # Predict radiance
                view_dirs = ray_directions.unsqueeze(1).expand(-1, self.config.max_ray_samples, -1)
                radiance = self.radiance_mlp(
                    features.reshape(-1, features.shape[-1]),
                    view_dirs.reshape(-1, 3) if self.config.view_dependent else None
                )
                radiance = radiance.reshape(batch_size, self.config.max_ray_samples, 4)
                
                # Mask invalid samples
                radiance[~valid_mask] = 0.0
                
                all_features.append(radiance[..., :3])  # RGB
                all_opacities.append(radiance[..., 3])  # Opacity
        
        if all_features:
            # Combine features from all primitives
            combined_features = torch.stack(all_features, dim=0).sum(dim=0)  # [N, S, 3]
            combined_opacities = torch.stack(all_opacities, dim=0).sum(dim=0)  # [N, S]
            
            # Volume rendering
            delta = t_vals[..., 1:] - t_vals[..., :-1]
            delta = torch.cat([delta, torch.full_like(delta[..., :1], 1e10)], dim=-1)
            
            alpha = 1.0 - torch.exp(-combined_opacities * delta)
            transmittance = torch.cumprod(
                torch.cat([torch.ones_like(alpha[..., :1]), 1.0 - alpha + 1e-10], dim=-1),
                dim=-1
            )[..., :-1]
            
            weights_vol = alpha * transmittance
            
            # Composite colors and depths
            colors = (weights_vol.unsqueeze(-1) * combined_features).sum(dim=1)
            depths = (weights_vol * t_vals).sum(dim=1)
            weights = weights_vol.sum(dim=1)
        
        return {
            'rgb': colors,
            'depth': depths,
            'weights': weights,
            'raw_alpha': combined_opacities if all_features else torch.zeros_like(depths).unsqueeze(-1)
        }


class DNMPLoss(nn.Module):
    """Loss function for DNMP training."""
    
    def __init__(self, config: DNMPConfig):
        super().__init__()
        self.config = config
        
    def forward(self, predictions: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor],
                primitives: List[DeformableNeuralMeshPrimitive]) -> Dict[str, torch.Tensor]:
        """
        Compute DNMP loss.
        
        Args:
            predictions: Rendered outputs
            targets: Ground truth targets
            primitives: List of DNMP primitives
            
        Returns:
            Dictionary of loss components
        """
        losses = {}
        
        # Color loss (MSE)
        if 'rgb' in predictions and 'rgb' in targets:
            color_loss = F.mse_loss(predictions['rgb'], targets['rgb'])
            losses['color_loss'] = color_loss * self.config.color_loss_weight
        
        # Depth loss (if available)
        if 'depth' in predictions and 'depth' in targets:
            depth_mask = targets['depth'] > 0
            if depth_mask.sum() > 0:
                depth_loss = F.mse_loss(
                    predictions['depth'][depth_mask],
                    targets['depth'][depth_mask]
                )
                losses['depth_loss'] = depth_loss * self.config.depth_loss_weight
        
        # Mesh regularization
        mesh_reg_loss = 0.0
        for primitive in primitives:
            # Regularize mesh vertices to prevent extreme deformations
            vertices = primitive.get_mesh_vertices()
            mesh_reg_loss += torch.mean(vertices ** 2)
            
        losses['mesh_reg_loss'] = mesh_reg_loss * self.config.mesh_regularization_weight
        
        # Latent code regularization
        latent_reg_loss = 0.0
        for primitive in primitives:
            latent_reg_loss += torch.mean(primitive.latent_code ** 2)
            
        losses['latent_reg_loss'] = latent_reg_loss * self.config.latent_regularization_weight
        
        # Total loss
        total_loss = sum(losses.values())
        losses['total_loss'] = total_loss
        
        return losses


class DNMP(nn.Module):
    """
    Main DNMP model that manages multiple primitives and handles scene representation.
    """
    
    def __init__(self, config: DNMPConfig, mesh_autoencoder):
        super().__init__()
        self.config = config
        self.mesh_autoencoder = mesh_autoencoder
        self.renderer = DNMPRenderer(config)
        self.loss_fn = DNMPLoss(config)
        
        # Will be populated with primitives during scene initialization
        self.primitives = nn.ModuleList()
        
    def initialize_scene(self, point_cloud: torch.Tensor, voxel_size: float = None):
        """
        Initialize DNMP primitives based on point cloud.
        
        Args:
            point_cloud: Input point cloud [N, 3]
            voxel_size: Size of voxels for primitive placement
        """
        if voxel_size is None:
            voxel_size = self.config.voxel_size
            
        # Voxelize point cloud
        min_coords = point_cloud.min(dim=0)[0]
        max_coords = point_cloud.max(dim=0)[0]
        
        # Create voxel grid
        voxel_coords = []
        x_range = torch.arange(min_coords[0], max_coords[0], voxel_size)
        y_range = torch.arange(min_coords[1], max_coords[1], voxel_size)
        z_range = torch.arange(min_coords[2], max_coords[2], voxel_size)
        
        for x in x_range:
            for y in y_range:
                for z in z_range:
                    # Check if voxel contains points
                    voxel_min = torch.tensor([x, y, z])
                    voxel_max = voxel_min + voxel_size
                    
                    mask = torch.all(point_cloud >= voxel_min, dim=1) & \
                           torch.all(point_cloud < voxel_max, dim=1)
                    
                    if mask.sum() > 0:  # Voxel contains points
                        voxel_coords.append([x, y, z])
        
        # Create DNMP for each occupied voxel
        for coord in voxel_coords:
            primitive = DeformableNeuralMeshPrimitive(
                self.config, 
                self.mesh_autoencoder.decoder
            )
            self.primitives.append(primitive)
            
        print(f"Initialized {len(self.primitives)} DNMP primitives")
    
    def forward(self, ray_origins: torch.Tensor, ray_directions: torch.Tensor,
                rasterizer) -> Dict[str, torch.Tensor]:
        """
        Forward pass through DNMP scene.
        
        Args:
            ray_origins: Ray origins [N, 3]
            ray_directions: Ray directions [N, 3]
            rasterizer: Rasterization module
            
        Returns:
            Rendered outputs
        """
        return self.renderer.render_ray(
            ray_origins, ray_directions, list(self.primitives), rasterizer
        )
    
    def compute_loss(self, predictions: Dict[str, torch.Tensor],
                     targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute training loss."""
        return self.loss_fn(predictions, targets, list(self.primitives)) 