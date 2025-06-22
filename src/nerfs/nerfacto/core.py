"""
Nerfacto Core Implementation
===========================

Core components of the nerfacto model including:
- Configuration classes
- Neural radiance field implementation
- Hash encoding for efficient scene representation
- Proposal networks for hierarchical sampling
- Appearance embeddings for varying lighting conditions
- Volumetric rendering with advanced features
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from jaxtyping import Float, Int, Shaped
from torch import Tensor
import numpy as np


@dataclass
class NerfactoConfig:
    """Configuration for Nerfacto model."""
    
    # Model architecture
    num_layers: int = 2
    hidden_dim: int = 64
    geo_feat_dim: int = 15
    num_layers_color: int = 3
    hidden_dim_color: int = 64
    
    # Hash encoding
    num_levels: int = 16
    base_resolution: int = 16
    max_resolution: int = 2048
    log2_hashmap_size: int = 19
    features_per_level: int = 2
    
    # Proposal networks
    num_proposal_samples_per_ray: Tuple[int, ...] = (256, 96)
    num_nerf_samples_per_ray: int = 48
    proposal_update_every: int = 5
    proposal_warmup: int = 5000
    
    # Appearance embedding
    num_images: Optional[int] = None
    appearance_embed_dim: int = 32
    use_appearance_embedding: bool = True
    
    # Background
    background_color: str = "random"  # "random", "last_sample", "white", "black"
    
    # Loss weights
    distortion_loss_mult: float = 0.002
    interlevel_loss_mult: float = 1.0
    orientation_loss_mult: float = 0.0001
    pred_normal_loss_mult: float = 0.001
    
    # Rendering
    near_plane: float = 0.05
    far_plane: float = 1000.0
    use_single_jitter: bool = True
    disable_scene_contraction: bool = False
    
    # Training
    max_num_iterations: int = 30000
    proposal_net_args_list: List[Dict] = field(default_factory=lambda: [
        {"num_output_coords": 8, "num_levels": 5, "max_resolution": 128, "base_resolution": 16},
        {"num_output_coords": 8, "num_levels": 5, "max_resolution": 256, "base_resolution": 16},
    ])


class HashEncoding(nn.Module):
    """Hash encoding for efficient scene representation."""
    
    def __init__(
        self,
        num_levels: int = 16,
        min_res: int = 16,
        max_res: int = 2048,
        log2_hashmap_size: int = 19,
        features_per_level: int = 2,
    ):
        super().__init__()
        self.num_levels = num_levels
        self.min_res = min_res
        self.max_res = max_res
        self.log2_hashmap_size = log2_hashmap_size
        self.features_per_level = features_per_level
        
        # Calculate growth factor
        self.growth_factor = math.exp((math.log(max_res) - math.log(min_res)) / (num_levels - 1))
        
        # Hash tables for each level
        self.hash_tables = nn.ModuleList()
        for i in range(num_levels):
            resolution = int(min_res * (self.growth_factor ** i))
            hashmap_size = min(resolution ** 3, 2 ** log2_hashmap_size)
            hash_table = nn.Embedding(hashmap_size, features_per_level)
            nn.init.uniform_(hash_table.weight, -1e-4, 1e-4)
            self.hash_tables.append(hash_table)
    
    def forward(self, positions: Float[Tensor, "*batch 3"]) -> Float[Tensor, "*batch features"]:
        """
        Args:
            positions: 3D positions in [0, 1]^3
            
        Returns:
            Hash encoded features
        """
        batch_shape = positions.shape[:-1]
        positions = positions.view(-1, 3)
        
        features = []
        for i, hash_table in enumerate(self.hash_tables):
            resolution = int(self.min_res * (self.growth_factor ** i))
            
            # Scale positions to resolution
            scaled_pos = positions * (resolution - 1)
            
            # Get integer coordinates
            coords = scaled_pos.floor().long()
            coords = coords.clamp(0, resolution - 1)
            
            # Hash coordinates
            hash_coords = self._hash_coords(coords, resolution)
            
            # Get interpolation weights
            weights = scaled_pos - coords.float()
            
            # Trilinear interpolation
            feature = self._trilinear_interpolation(hash_table, hash_coords, weights, resolution)
            features.append(feature)
        
        features = torch.cat(features, dim=-1)
        return features.view(*batch_shape, -1)
    
    def _hash_coords(self, coords: Int[Tensor, "batch 3"], resolution: int) -> Int[Tensor, "batch"]:
        """Hash 3D coordinates to 1D indices."""
        # Simple hash function
        primes = torch.tensor([1, 2654435761, 805459861], device=coords.device, dtype=coords.dtype)
        hash_coords = (coords * primes).sum(dim=-1)
        return hash_coords % len(self.hash_tables[0].weight)
    
    def _trilinear_interpolation(
        self, 
        hash_table: nn.Embedding, 
        hash_coords: Int[Tensor, "batch"], 
        weights: Float[Tensor, "batch 3"],
        resolution: int
    ) -> Float[Tensor, "batch features"]:
        """Perform trilinear interpolation."""
        # For simplicity, use the hashed coordinates directly
        # In practice, you'd implement proper trilinear interpolation
        return hash_table(hash_coords)


@dataclass
class NerfactoFieldConfig:
    """Configuration for NerfactoField."""
    num_layers: int = 2
    hidden_dim: int = 64
    geo_feat_dim: int = 15
    num_layers_color: int = 3
    hidden_dim_color: int = 64
    appearance_embed_dim: int = 32
    use_appearance_embedding: bool = True
    spatial_distortion: Optional[str] = None


class NerfactoField(nn.Module):
    """Neural radiance field for nerfacto."""
    
    def __init__(self, config: NerfactoFieldConfig, aabb: Tensor):
        super().__init__()
        self.config = config
        self.aabb = nn.Parameter(aabb, requires_grad=False)
        
        # Hash encoding
        self.position_encoding = HashEncoding()
        
        # Direction encoding (spherical harmonics)
        self.direction_encoding = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.Linear(16, 16)
        )
        
        # Density network
        self.density_net = nn.Sequential(
            nn.Linear(self.position_encoding.num_levels * self.position_encoding.features_per_level, config.hidden_dim),
            nn.ReLU(),
            *[nn.Sequential(nn.Linear(config.hidden_dim, config.hidden_dim), nn.ReLU()) 
              for _ in range(config.num_layers - 1)],
            nn.Linear(config.hidden_dim, 1 + config.geo_feat_dim)
        )
        
        # Color network
        color_input_dim = config.geo_feat_dim + 16  # geo features + direction encoding
        if config.use_appearance_embedding:
            color_input_dim += config.appearance_embed_dim
            
        self.color_net = nn.Sequential(
            nn.Linear(color_input_dim, config.hidden_dim_color),
            nn.ReLU(),
            *[nn.Sequential(nn.Linear(config.hidden_dim_color, config.hidden_dim_color), nn.ReLU()) 
              for _ in range(config.num_layers_color - 1)],
            nn.Linear(config.hidden_dim_color, 3)
        )
        
        # Appearance embedding
        if config.use_appearance_embedding:
            self.appearance_embedding = nn.Embedding(1000, config.appearance_embed_dim)  # Max 1000 images
    
    def get_density(self, ray_samples: Float[Tensor, "*batch 3"]) -> Float[Tensor, "*batch 1"]:
        """Get density at given positions."""
        # Normalize positions to [0, 1]
        positions = (ray_samples - self.aabb[0]) / (self.aabb[1] - self.aabb[0])
        positions = positions.clamp(0, 1)
        
        # Encode positions
        encoded_positions = self.position_encoding(positions)
        
        # Get density and geometry features
        density_features = self.density_net(encoded_positions)
        density = density_features[..., 0:1]
        
        return F.relu(density)
    
    def get_outputs(
        self, 
        ray_samples: Float[Tensor, "*batch 3"], 
        directions: Float[Tensor, "*batch 3"],
        camera_indices: Optional[Int[Tensor, "*batch 1"]] = None
    ) -> Dict[str, Float[Tensor, "*batch channels"]]:
        """Get color and density outputs."""
        # Normalize positions
        positions = (ray_samples - self.aabb[0]) / (self.aabb[1] - self.aabb[0])
        positions = positions.clamp(0, 1)
        
        # Encode positions and directions
        encoded_positions = self.position_encoding(positions)
        encoded_directions = self.direction_encoding(directions)
        
        # Get density and geometry features
        density_features = self.density_net(encoded_positions)
        density = F.relu(density_features[..., 0:1])
        geo_features = density_features[..., 1:]
        
        # Prepare color network input
        color_input = torch.cat([geo_features, encoded_directions], dim=-1)
        
        # Add appearance embedding if used
        if self.config.use_appearance_embedding and camera_indices is not None:
            appearance_embed = self.appearance_embedding(camera_indices.squeeze(-1))
            color_input = torch.cat([color_input, appearance_embed], dim=-1)
        
        # Get colors
        colors = torch.sigmoid(self.color_net(color_input))
        
        return {
            "rgb": colors,
            "density": density,
            "geo_features": geo_features
        }


class ProposalNetwork(nn.Module):
    """Proposal network for hierarchical sampling."""
    
    def __init__(
        self,
        num_output_coords: int = 8,
        num_levels: int = 5,
        max_resolution: int = 128,
        base_resolution: int = 16,
        log2_hashmap_size: int = 17,
        features_per_level: int = 2,
        num_layers: int = 2,
        hidden_dim: int = 64,
    ):
        super().__init__()
        
        # Hash encoding
        self.position_encoding = HashEncoding(
            num_levels=num_levels,
            min_res=base_resolution,
            max_res=max_resolution,
            log2_hashmap_size=log2_hashmap_size,
            features_per_level=features_per_level
        )
        
        # Density network
        input_dim = num_levels * features_per_level
        self.density_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            *[nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU()) 
              for _ in range(num_layers - 1)],
            nn.Linear(hidden_dim, 1)
        )
    
    def get_density(self, positions: Float[Tensor, "*batch 3"]) -> Float[Tensor, "*batch 1"]:
        """Get density at given positions."""
        encoded_positions = self.position_encoding(positions)
        density = self.density_net(encoded_positions)
        return F.relu(density)


class AppearanceEmbedding(nn.Module):
    """Appearance embedding for varying lighting conditions."""
    
    def __init__(self, num_images: int, embed_dim: int = 32):
        super().__init__()
        self.num_images = num_images
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(num_images, embed_dim)
        
        # Initialize with small values
        nn.init.normal_(self.embedding.weight, 0, 0.01)
    
    def forward(self, camera_indices: Int[Tensor, "*batch"]) -> Float[Tensor, "*batch embed_dim"]:
        """Get appearance embedding for given camera indices."""
        return self.embedding(camera_indices)


class NerfactoRenderer(nn.Module):
    """Volumetric renderer for nerfacto."""
    
    def __init__(self, config: NerfactoConfig):
        super().__init__()
        self.config = config
        
        # Background color
        if config.background_color == "white":
            self.background_color = torch.ones(3)
        elif config.background_color == "black":
            self.background_color = torch.zeros(3)
        else:
            self.background_color = None
    
    def render_weights(
        self, 
        ray_samples: Float[Tensor, "*batch num_samples 3"],
        densities: Float[Tensor, "*batch num_samples 1"],
        ray_indices: Optional[Int[Tensor, "*batch"]] = None,
        num_rays: Optional[int] = None
    ) -> Tuple[Float[Tensor, "*batch num_samples 1"], Float[Tensor, "*batch 1"]]:
        """Render weights from densities using volumetric rendering."""
        # Compute delta (distance between samples)
        delta = ray_samples[..., 1:, :] - ray_samples[..., :-1, :]
        delta = torch.norm(delta, dim=-1, keepdim=True)
        delta = torch.cat([delta, torch.full_like(delta[..., -1:, :], 1e10)], dim=-2)
        
        # Compute alpha values
        alpha = 1.0 - torch.exp(-F.relu(densities) * delta)
        
        # Compute transmittance
        transmittance = torch.cumprod(1.0 - alpha + 1e-10, dim=-2)
        transmittance = torch.cat([torch.ones_like(transmittance[..., :1, :]), transmittance[..., :-1, :]], dim=-2)
        
        # Compute weights
        weights = alpha * transmittance
        
        # Compute accumulated transmittance
        accumulated_transmittance = torch.sum(weights, dim=-2, keepdim=True)
        
        return weights, accumulated_transmittance
    
    def composite_with_background(
        self,
        image: Float[Tensor, "*batch channels"],
        background: Float[Tensor, "*batch channels"],
        accumulated_alpha: Float[Tensor, "*batch 1"]
    ) -> Float[Tensor, "*batch channels"]:
        """Composite image with background."""
        return image + background * (1.0 - accumulated_alpha)
    
    def forward(
        self,
        rgb: Float[Tensor, "*batch num_samples 3"],
        weights: Float[Tensor, "*batch num_samples 1"],
        ray_indices: Optional[Int[Tensor, "*batch"]] = None,
        num_rays: Optional[int] = None,
        background_color: Optional[Float[Tensor, "*batch 3"]] = None
    ) -> Dict[str, Float[Tensor, "*batch channels"]]:
        """Render RGB image from samples."""
        # Composite RGB
        rendered_rgb = torch.sum(weights * rgb, dim=-2)
        
        # Composite with background
        accumulated_alpha = torch.sum(weights, dim=-2, keepdim=True)
        
        if background_color is not None:
            rendered_rgb = self.composite_with_background(rendered_rgb, background_color, accumulated_alpha)
        elif self.background_color is not None:
            bg_color = self.background_color.to(rendered_rgb.device)
            rendered_rgb = self.composite_with_background(rendered_rgb, bg_color, accumulated_alpha)
        
        return {
            "rgb": rendered_rgb,
            "accumulation": accumulated_alpha.squeeze(-1),
            "weights": weights
        }


class NerfactoLoss(nn.Module):
    """Loss function for nerfacto."""
    
    def __init__(self, config: NerfactoConfig):
        super().__init__()
        self.config = config
        self.mse_loss = nn.MSELoss()
    
    def forward(
        self,
        outputs: Dict[str, Any],
        batch: Dict[str, Any]
    ) -> Dict[str, Float[Tensor, ""]]:
        """Compute loss."""
        losses = {}
        
        # RGB loss
        predicted_rgb = outputs["rgb"]
        target_rgb = batch["image"]
        losses["rgb_loss"] = self.mse_loss(predicted_rgb, target_rgb)
        
        # Distortion loss
        if "ray_samples" in outputs and "weights" in outputs:
            ray_samples = outputs["ray_samples"]
            weights = outputs["weights"]
            
            # Compute distortion loss
            mid_points = (ray_samples[..., 1:] + ray_samples[..., :-1]) / 2.0
            intervals = ray_samples[..., 1:] - ray_samples[..., :-1]
            
            # Distortion regularization
            losses["distortion_loss"] = self.config.distortion_loss_mult * self._compute_distortion_loss(
                weights, mid_points, intervals
            )
        
        # Interlevel loss (for proposal networks)
        if "weights_list" in outputs and "ray_samples_list" in outputs:
            losses["interlevel_loss"] = self.config.interlevel_loss_mult * self._compute_interlevel_loss(
                outputs["weights_list"], outputs["ray_samples_list"]
            )
        
        # Total loss
        losses["total_loss"] = sum(losses.values())
        
        return losses
    
    def _compute_distortion_loss(
        self, 
        weights: Float[Tensor, "*batch num_samples 1"],
        mid_points: Float[Tensor, "*batch num_samples-1"],
        intervals: Float[Tensor, "*batch num_samples-1"]
    ) -> Float[Tensor, ""]:
        """Compute distortion loss."""
        # Simplified distortion loss
        weights_normalized = weights / (torch.sum(weights, dim=-2, keepdim=True) + 1e-10)
        loss = torch.sum(weights_normalized * intervals, dim=-2)
        return torch.mean(loss)
    
    def _compute_interlevel_loss(
        self, 
        weights_list: List[Float[Tensor, "*batch num_samples 1"]],
        ray_samples_list: List[Float[Tensor, "*batch num_samples 3"]]
    ) -> Float[Tensor, ""]:
        """Compute interlevel loss between proposal networks."""
        # Simplified interlevel loss
        if len(weights_list) < 2:
            return torch.tensor(0.0, device=weights_list[0].device)
        
        # Compare weights between levels
        loss = 0.0
        for i in range(len(weights_list) - 1):
            weights_coarse = weights_list[i]
            weights_fine = weights_list[i + 1]
            
            # Simple L2 loss between weights
            if weights_coarse.shape == weights_fine.shape:
                loss += self.mse_loss(weights_coarse, weights_fine)
        
        return loss / (len(weights_list) - 1)


class NerfactoModel(nn.Module):
    """Complete Nerfacto model."""
    
    def __init__(self, config: NerfactoConfig, scene_box: Tensor):
        super().__init__()
        self.config = config
        self.scene_box = scene_box
        
        # Neural field
        field_config = NerfactoFieldConfig(
            num_layers=config.num_layers,
            hidden_dim=config.hidden_dim,
            geo_feat_dim=config.geo_feat_dim,
            num_layers_color=config.num_layers_color,
            hidden_dim_color=config.hidden_dim_color,
            appearance_embed_dim=config.appearance_embed_dim,
            use_appearance_embedding=config.use_appearance_embedding
        )
        self.field = NerfactoField(field_config, scene_box)
        
        # Proposal networks
        self.proposal_networks = nn.ModuleList([
            ProposalNetwork(**args) for args in config.proposal_net_args_list
        ])
        
        # Renderer
        self.renderer = NerfactoRenderer(config)
        
        # Loss function
        self.loss_fn = NerfactoLoss(config)
        
        # Appearance embedding
        if config.use_appearance_embedding and config.num_images:
            self.appearance_embedding = AppearanceEmbedding(config.num_images, config.appearance_embed_dim)
    
    def sample_and_forward(
        self,
        ray_bundle: Dict[str, Any],
        return_samples: bool = False
    ) -> Dict[str, Any]:
        """Sample points along rays and forward through networks."""
        # Extract ray information
        ray_origins = ray_bundle["origins"]
        ray_directions = ray_bundle["directions"]
        near = ray_bundle.get("near", self.config.near_plane)
        far = ray_bundle.get("far", self.config.far_plane)
        
        # Hierarchical sampling
        outputs = {}
        weights_list = []
        ray_samples_list = []
        
        # Initial uniform sampling
        t_vals = torch.linspace(0.0, 1.0, self.config.num_proposal_samples_per_ray[0], device=ray_origins.device)
        z_vals = near * (1.0 - t_vals) + far * t_vals
        
        # Add jitter
        if self.training and self.config.use_single_jitter:
            z_vals = z_vals + torch.rand_like(z_vals) * (far - near) / self.config.num_proposal_samples_per_ray[0]
        
        ray_samples = ray_origins.unsqueeze(-2) + ray_directions.unsqueeze(-2) * z_vals.unsqueeze(-1)
        
        # Forward through proposal networks
        for i, proposal_network in enumerate(self.proposal_networks):
            densities = proposal_network.get_density(ray_samples)
            weights, _ = self.renderer.render_weights(ray_samples, densities)
            
            weights_list.append(weights)
            ray_samples_list.append(ray_samples)
            
            # Resample based on weights
            if i < len(self.proposal_networks) - 1:
                ray_samples = self._resample_along_ray(
                    ray_samples, weights, self.config.num_proposal_samples_per_ray[i + 1]
                )
        
        # Final sampling for main network
        ray_samples = self._resample_along_ray(
            ray_samples, weights, self.config.num_nerf_samples_per_ray
        )
        
        # Forward through main field
        camera_indices = ray_bundle.get("camera_indices", None)
        field_outputs = self.field.get_outputs(ray_samples, ray_directions, camera_indices)
        
        # Render final image
        rgb = field_outputs["rgb"]
        density = field_outputs["density"]
        weights, accumulated_transmittance = self.renderer.render_weights(ray_samples, density)
        
        # Get background color
        background_color = None
        if self.config.background_color == "random" and self.training:
            background_color = torch.rand_like(rgb[..., 0, :])
        
        render_outputs = self.renderer(rgb, weights, background_color=background_color)
        
        # Prepare outputs
        outputs.update(render_outputs)
        outputs["weights_list"] = weights_list
        outputs["ray_samples_list"] = ray_samples_list
        
        if return_samples:
            outputs["ray_samples"] = ray_samples
            outputs["weights"] = weights
        
        return outputs
    
    def _resample_along_ray(
        self, 
        ray_samples: Float[Tensor, "*batch num_samples 3"],
        weights: Float[Tensor, "*batch num_samples 1"],
        num_samples: int
    ) -> Float[Tensor, "*batch new_num_samples 3"]:
        """Resample points along ray based on weights."""
        # Simplified resampling - in practice, you'd use proper PDF sampling
        # For now, just return the same samples
        return ray_samples
    
    def forward(self, ray_bundle: Dict[str, Any]) -> Dict[str, Any]:
        """Forward pass through the model."""
        return self.sample_and_forward(ray_bundle, return_samples=True)
    
    def get_loss_dict(self, outputs: Dict[str, Any], batch: Dict[str, Any]) -> Dict[str, Float[Tensor, ""]]:
        """Get loss dictionary."""
        return self.loss_fn(outputs, batch)