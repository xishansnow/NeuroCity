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
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from jaxtyping import Float, Int, Shaped
from torch import Tensor
import numpy as np


class SHEncoding(nn.Module):
    """Spherical harmonics encoding for directions."""
    
    def __init__(self, degree: int = 4):
        super().__init__()
        self.degree = degree
        self.num_features = (degree + 1) ** 2
        
    def forward(self, directions: Float[Tensor, "*batch 3"]) -> Float[Tensor, "*batch features"]:
        """Encode directions using spherical harmonics."""
        x, y, z = directions.unbind(-1)
        sh_bands = [
            # L0
            0.28209479177387814 * torch.ones_like(x),  # 1/sqrt(4*pi)
            # L1
            0.4886025119029199 * y,                    # sqrt(3/(4*pi)) * y
            0.4886025119029199 * z,                    # sqrt(3/(4*pi)) * z
            0.4886025119029199 * x,                    # sqrt(3/(4*pi)) * x
            # L2
            1.0925484305920792 * x * y,               # sqrt(15/(4*pi)) * xy
            1.0925484305920792 * y * z,               # sqrt(15/(4*pi)) * yz
            0.9461746957575601 * (2 * z * z - x * x - y * y), # sqrt(5/(16*pi)) * (3z^2 - 1)
            1.0925484305920792 * z * x,               # sqrt(15/(4*pi)) * zx
            0.5462742152960396 * (x * x - y * y),     # sqrt(15/(16*pi)) * (x^2 - y^2)
            # L3
            0.5900435899266435 * y * (3 * x * x - y * y),
            2.890611442640554 * x * y * z,
            0.4570457994644658 * y * (5 * z * z - 1),
            0.3731763325901154 * z * (5 * z * z - 3),
            0.4570457994644658 * x * (5 * z * z - 1),
            1.445305721320277 * z * (x * x - y * y),
            0.5900435899266435 * x * (x * x - 3 * y * y)
        ]
        return torch.stack(sh_bands[:self.num_features], dim=-1)


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
    num_proposal_samples_per_ray: tuple[int, ...] = (256, 96)
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
    proposal_net_args_list: list[dict[str, Any]] = field(default_factory=lambda: [
        {
            "num_output_coords": 8,
            "num_levels": 5,
            "max_resolution": 128,
            "base_resolution": 16,
        }
    ])

class HashEncoding(nn.Module):
    """Hash encoding for efficient scene representation."""
    
    def __init__(
        self, num_levels: int = 16, min_res: int = 16, max_res: int = 2048, log2_hashmap_size: int = 19, features_per_level: int = 2, ):
        super().__init__()
        self.num_levels = num_levels
        self.min_res = min_res
        self.max_res = max_res
        self.log2_hashmap_size = log2_hashmap_size
        self.features_per_level = features_per_level
        self.output_dim = num_levels * features_per_level
        
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
            Hash encoded features [*batch, num_levels * features_per_level]
        """
        batch_shape = positions.shape[:-1]
        positions = positions.reshape(-1, 3)
        
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
        
        features = torch.cat(features, dim=-1)  # [batch, num_levels * features_per_level]
        return features.reshape(*batch_shape, -1)
    
    def _hash_coords(self, coords: Int[Tensor, "batch 3"], resolution: int) -> Int[Tensor, "batch"]:
        """Hash 3D coordinates to 1D indices."""
        # Simple hash function
        primes = torch.tensor([1, 2654435761, 805459861], device=coords.device, dtype=coords.dtype)
        hash_coords = (coords * primes).sum(dim=-1)
        return hash_coords % len(self.hash_tables[0].weight)
    
    def _trilinear_interpolation(
        self, hash_table: nn.Embedding, hash_coords: Int[Tensor, "batch"], weights: Float[Tensor, "batch 3"], resolution: int
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
    """Neural field for Nerfacto."""
    
    def __init__(self, config: NerfactoFieldConfig, aabb: Tensor):
        super().__init__()
        self.config = config
        self.aabb = aabb
        
        # Position encoding
        self.position_encoding = HashEncoding(
            num_levels=16,
            min_res=16,
            max_res=2048,
            log2_hashmap_size=19,
            features_per_level=2
        )
        
        # Direction encoding (spherical harmonics)
        self.direction_encoding = SHEncoding(degree=4)  # 16 features
        
        # Calculate input dimensions
        position_features = self.position_encoding.output_dim  # 32 = 16 * 2
        
        # Density network
        density_layers = []
        for _ in range(config.num_layers - 1):
            density_layers.extend([
                nn.Linear(position_features if _ == 0 else config.hidden_dim, config.hidden_dim),
                nn.ReLU()
            ])
        density_layers.append(nn.Linear(config.hidden_dim, 1 + config.geo_feat_dim))
        self.density_net = nn.Sequential(*density_layers)
        
        # Color network
        # Input: geometry features + encoded directions + (optional) appearance embedding
        color_input_dim = config.geo_feat_dim + 16  # geo_features + encoded_directions
        if config.use_appearance_embedding:
            color_input_dim += config.appearance_embed_dim
            
        # Build color network with proper dimensions
        self.color_net = nn.Sequential(
            nn.Linear(color_input_dim, config.hidden_dim_color),
            nn.ReLU(),
            nn.Linear(config.hidden_dim_color, config.hidden_dim_color),
            nn.ReLU(),
            nn.Linear(config.hidden_dim_color, 3)
        )
        
        # Appearance embedding
        if config.use_appearance_embedding:
            self.appearance_embedding = nn.Embedding(1000, config.appearance_embed_dim)
            
    def _check_shapes(self, **tensors: Dict[str, torch.Tensor]) -> None:
        """Check if tensor shapes are compatible.
        
        Args:
            **tensors: Dictionary of tensors to check
            
        Raises:
            ValueError: If tensor shapes are incompatible
        """
        batch_shape = None
        for name, tensor in tensors.items():
            if tensor is None:
                continue
                
            if batch_shape is None:
                batch_shape = tensor.shape[:-1]
            else:
                current_shape = tensor.shape[:-1]
                if current_shape != batch_shape:
                    raise ValueError(
                        f"Incompatible batch shapes: {name} has shape {tensor.shape}, "
                        f"expected batch shape {batch_shape}"
                    )
    
    def get_outputs(
        self, ray_samples: Float[Tensor, "*batch 3"], directions: Float[Tensor, "*batch 3"], camera_indices: Optional[Int[Tensor, "*batch 1"]] = None
    ) -> dict[str, Float[Tensor, "*batch channels"]]:
        """Get color and density outputs."""
        # Check input shapes
        self._check_shapes(ray_samples=ray_samples, directions=directions)
        
        # Get original batch shape for reshaping later
        batch_shape = ray_samples.shape[:-1]
        
        # Flatten batch dimensions for network processing
        flat_samples = ray_samples.reshape(-1, 3)
        flat_directions = directions.reshape(-1, 3)
        
        # Normalize positions
        positions = (flat_samples - self.aabb[0]) / (self.aabb[1] - self.aabb[0])
        positions = positions.clamp(0, 1)
        
        # Encode positions and directions
        encoded_positions = self.position_encoding(positions)  # [flattened_batch, 32]
        encoded_directions = self.direction_encoding(flat_directions)  # [flattened_batch, 16]
        
        # Get density and geometry features
        density_features = self.density_net(encoded_positions)
        density = F.relu(density_features[..., 0:1])  # [flattened_batch, 1]
        geo_features = density_features[..., 1:self.config.geo_feat_dim+1]  # [flattened_batch, geo_feat_dim]
        
        # Prepare color network input
        color_input = torch.cat([geo_features, encoded_directions], dim=-1)  # [flattened_batch, geo_feat_dim + 16]
        
        # Add appearance embedding if used
        if self.config.use_appearance_embedding and camera_indices is not None:
            # Expand camera indices to match number of samples
            flat_camera_indices = camera_indices.reshape(-1, 1).expand(-1, ray_samples.shape[1]).reshape(-1)
            appearance_embed = self.appearance_embedding(flat_camera_indices)
            color_input = torch.cat([color_input, appearance_embed], dim=-1)
        
        # Get colors
        colors = torch.sigmoid(self.color_net(color_input))  # [flattened_batch, 3]
        
        # Reshape outputs back to original batch shape
        colors = colors.reshape(*batch_shape, 3)
        density = density.reshape(*batch_shape, 1)
        geo_features = geo_features.reshape(*batch_shape, -1)
        
        return {
            "rgb": colors,
            "density": density,
            "geo_features": geo_features
        }


class ProposalNetwork(nn.Module):
    """Proposal network for hierarchical sampling."""
    
    def __init__(
        self, num_output_coords: int = 8, num_levels: int = 5, max_resolution: int = 128, base_resolution: int = 16, log2_hashmap_size: int = 17, features_per_level: int = 2, num_layers: int = 2, hidden_dim: int = 64, ):
        super().__init__()
        
        # Hash encoding
        self.position_encoding = HashEncoding(
            num_levels=num_levels, min_res=base_resolution, max_res=max_resolution, log2_hashmap_size=log2_hashmap_size, features_per_level=features_per_level
        )
        
        # Density network
        input_dim = num_levels * features_per_level
        density_layers = []
        for _ in range(num_layers - 1):
            density_layers.extend([
                nn.Linear(input_dim if _ == 0 else hidden_dim, hidden_dim),
                nn.ReLU()
            ])
        density_layers.append(nn.Linear(hidden_dim, num_output_coords))
        self.density_net = nn.Sequential(*density_layers)
    
    def get_density(self, positions: Float[Tensor, "*batch 3"]) -> Float[Tensor, "*batch 1"]:
        """Get density at given positions."""
        encoded_positions = self.position_encoding(positions)
        density = self.density_net(encoded_positions)
        # Ensure output is [*batch, 1] by taking mean across output coordinates
        density = density.mean(dim=-1, keepdim=True)
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
    
    def _check_shapes(self, **tensors: Dict[str, torch.Tensor]) -> None:
        """Check if tensor shapes are compatible.
        
        Args:
            **tensors: Dictionary of tensors to check
            
        Raises:
            ValueError: If tensor shapes are incompatible
        """
        batch_shape = None
        for name, tensor in tensors.items():
            if tensor is None:
                continue
                
            if batch_shape is None:
                batch_shape = tensor.shape[:-2] if len(tensor.shape) > 2 else tensor.shape[:-1]
            else:
                current_shape = tensor.shape[:-2] if len(tensor.shape) > 2 else tensor.shape[:-1]
                if current_shape != batch_shape:
                    raise ValueError(
                        f"Incompatible batch shapes: {name} has shape {tensor.shape}, "
                        f"expected batch shape {batch_shape}"
                    )
    
    def render_weights(
        self, ray_samples: Float[Tensor, "*batch num_samples 3"], densities: Float[Tensor, "*batch num_samples 1"], ray_indices: Optional[Int[Tensor, "*batch"]] = None, num_rays: Optional[int] = None
    ) -> tuple[Float[Tensor, "*batch num_samples 1"], Float[Tensor, "*batch 1"]]:
        """Render weights from densities using volumetric rendering."""
        # Check shapes
        self._check_shapes(ray_samples=ray_samples, densities=densities)
        if ray_samples.shape[-2] != densities.shape[-2]:
            raise ValueError(
                f"Number of samples mismatch: ray_samples has {ray_samples.shape[-2]} samples, "
                f"but densities has {densities.shape[-2]} samples"
            )
        
        # Compute delta (distance between samples)
        delta = ray_samples[..., 1:, :] - ray_samples[..., :-1, :]
        delta = torch.norm(delta, dim=-1, keepdim=True)
        delta = torch.cat([delta, torch.full_like(delta[..., -1:, :], 1e10)], dim=-2)
        
        # Compute alpha values
        alpha = 1.0 - torch.exp(-F.relu(densities) * delta)
        
        # Compute transmittance
        transmittance = torch.cumprod(torch.cat([torch.ones_like(alpha[..., :1, :]), 1.0 - alpha + 1e-10], dim=-2), dim=-2)
        transmittance = transmittance[..., :-1, :]
        
        # Compute weights
        weights = alpha * transmittance
        
        # Compute accumulated transmittance
        accumulated_transmittance = torch.sum(weights, dim=-2, keepdim=True)
        
        return weights, accumulated_transmittance
    
    def composite_with_background(
        self, image: Float[Tensor, "*batch channels"], background: Float[Tensor, "*batch channels"], accumulated_alpha: Float[Tensor, "*batch 1"]
    ) -> Float[Tensor, "*batch channels"]:
        """Composite image with background."""
        # Check shapes
        self._check_shapes(image=image, background=background, accumulated_alpha=accumulated_alpha)
        if image.shape[-1] != background.shape[-1]:
            raise ValueError(
                f"Channel dimension mismatch: image has {image.shape[-1]} channels, "
                f"but background has {background.shape[-1]} channels"
            )
        
        return image + background * (1.0 - accumulated_alpha)
    
    def forward(
        self, rgb: Float[Tensor, "*batch num_samples 3"], weights: Float[Tensor, "*batch num_samples 1"], ray_indices: Optional[Int[Tensor, "*batch"]] = None, num_rays: Optional[int] = None, background_color: Optional[Float[Tensor, "*batch 3"]] = None
    ) -> dict[str, Float[Tensor, "*batch channels"]]:
        """Render RGB image from samples."""
        # Check shapes
        self._check_shapes(rgb=rgb, weights=weights)
        if rgb.shape[-2] != weights.shape[-2]:
            raise ValueError(
                f"Number of samples mismatch: rgb has {rgb.shape[-2]} samples, "
                f"but weights has {weights.shape[-2]} samples"
            )
        if rgb.shape[-1] != 3:
            raise ValueError(f"RGB tensor must have 3 channels, got {rgb.shape[-1]}")
        if weights.shape[-1] != 1:
            raise ValueError(f"Weights tensor must have 1 channel, got {weights.shape[-1]}")
        
        # Composite RGB
        rendered_rgb = torch.sum(weights * rgb, dim=-2)
        
        # Composite with background
        accumulated_alpha = torch.sum(weights, dim=-2, keepdim=True)
        
        if background_color is not None:
            # Check background color shape
            if background_color.shape[-1] != 3:
                raise ValueError(
                    f"Background color must have 3 channels, got {background_color.shape[-1]}"
                )
            rendered_rgb = self.composite_with_background(
                rendered_rgb,
                background_color,
                accumulated_alpha
            )
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
        self, outputs: dict[str, Any], batch: dict[str, Any]
    ) -> dict[str, Float[Tensor, ""]]:
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
        self, weights: Float[Tensor, "*batch num_samples 1"], mid_points: Float[Tensor, "*batch num_samples-1"], intervals: Float[Tensor, "*batch num_samples-1"]
    ) -> Float[Tensor, ""]:
        """Compute distortion loss."""
        # Simplified distortion loss
        weights_normalized = weights / (torch.sum(weights, dim=-2, keepdim=True) + 1e-10)
        loss = torch.sum(weights_normalized * intervals, dim=-2)
        return torch.mean(loss)
    
    def _compute_interlevel_loss(
        self, weights_list: list[Float[Tensor, "*batch num_samples 1"]], ray_samples_list: list[Float[Tensor, "*batch num_samples 3"]]
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
            num_layers=config.num_layers, hidden_dim=config.hidden_dim, geo_feat_dim=config.geo_feat_dim, num_layers_color=config.num_layers_color, hidden_dim_color=config.hidden_dim_color, appearance_embed_dim=config.appearance_embed_dim, use_appearance_embedding=config.use_appearance_embedding
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
        self, ray_bundle: dict[str, Any], return_samples: bool = False
    ) -> dict[str, Any]:
        """Sample points along rays and forward through networks."""
        # Extract ray information
        ray_origins = ray_bundle["origins"]  # [batch_size, 3]
        ray_directions = ray_bundle["directions"]  # [batch_size, 3]
        near = ray_bundle.get("near", self.config.near_plane)
        far = ray_bundle.get("far", self.config.far_plane)
        
        # Hierarchical sampling
        outputs = {}
        weights_list = []
        ray_samples_list = []
        
        # Initial uniform sampling
        t_vals = torch.linspace(
            0.0,
            1.0,
            self.config.num_proposal_samples_per_ray[0],
            device=ray_origins.device
        )  # [num_samples]
        
        # Expand t_vals to match batch size
        t_vals = t_vals.expand(ray_origins.shape[0], -1)  # [batch_size, num_samples]
        
        # Convert to distances
        z_vals = near * (1.0 - t_vals) + far * t_vals  # [batch_size, num_samples]
        
        # Add jitter
        if self.training and self.config.use_single_jitter:
            z_vals = z_vals + torch.rand_like(z_vals) * (far - near) / self.config.num_proposal_samples_per_ray[0]
        
        # Sample points along rays
        # [batch_size, num_samples, 3] = [batch_size, 1, 3] + [batch_size, 1, 3] * [batch_size, num_samples, 1]
        ray_samples = ray_origins.unsqueeze(1) + ray_directions.unsqueeze(1) * z_vals.unsqueeze(-1)
        
        # Forward through proposal networks
        for i, proposal_network in enumerate(self.proposal_networks):
            # Get densities from proposal network
            densities = proposal_network.get_density(ray_samples)  # [batch_size, num_samples, 1]
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
        
        # Broadcast ray directions to match sampled points
        # [batch_size, num_samples, 3] = [batch_size, 1, 3].expand([batch_size, num_samples, 3])
        ray_directions_expanded = ray_directions.unsqueeze(1).expand_as(ray_samples)
        
        # Forward through main field
        camera_indices = ray_bundle.get("camera_indices", None)
        field_outputs = self.field.get_outputs(ray_samples, ray_directions_expanded, camera_indices)
        
        # Render final image
        rgb = field_outputs["rgb"]  # [batch_size, num_samples, 3]
        density = field_outputs["density"]  # [batch_size, num_samples, 1]
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
        # Get batch shape
        batch_shape = ray_samples.shape[:-2]
        num_rays = np.prod(batch_shape)
        
        # Flatten batch dimensions
        flat_samples = ray_samples.reshape(num_rays, -1, 3)  # [num_rays, num_samples, 3]
        flat_weights = weights.reshape(num_rays, -1, 1)  # [num_rays, num_samples, 1]
        
        # Create CDF for sampling
        weights_sum = flat_weights.sum(dim=1, keepdim=True)  # [num_rays, 1, 1]
        weights_normalized = flat_weights / (weights_sum + 1e-10)  # [num_rays, num_samples, 1]
        cdf = torch.cumsum(weights_normalized.squeeze(-1), dim=-1)  # [num_rays, num_samples]
        cdf = torch.cat([torch.zeros_like(cdf[:, :1]), cdf], dim=-1)  # [num_rays, num_samples+1]
        
        # Sample uniformly
        u = torch.linspace(0.0, 1.0, num_samples + 1, device=ray_samples.device)[:-1]  # [num_samples]
        u = u.expand(num_rays, -1)  # [num_rays, num_samples]
        u = u + torch.rand_like(u) / num_samples  # Add jitter
        u = u.clamp(0.0, 1.0)  # [num_rays, num_samples]
        
        # Invert CDF
        inds = torch.searchsorted(cdf, u, right=True)  # [num_rays, num_samples]
        below = torch.clamp(inds - 1, 0, flat_samples.shape[1] - 1)  # [num_rays, num_samples]
        above = torch.clamp(inds, 0, flat_samples.shape[1] - 1)  # [num_rays, num_samples]
        
        # Get samples
        cdf_g0 = torch.gather(cdf, 1, below)  # [num_rays, num_samples]
        cdf_g1 = torch.gather(cdf, 1, above)  # [num_rays, num_samples]
        samples_g0 = torch.gather(flat_samples, 1, below.unsqueeze(-1).expand(-1, -1, 3))  # [num_rays, num_samples, 3]
        samples_g1 = torch.gather(flat_samples, 1, above.unsqueeze(-1).expand(-1, -1, 3))  # [num_rays, num_samples, 3]
        
        # Linear interpolation
        t = ((u - cdf_g0) / (cdf_g1 - cdf_g0 + 1e-10)).unsqueeze(-1)  # [num_rays, num_samples, 1]
        samples = samples_g0 + t * (samples_g1 - samples_g0)  # [num_rays, num_samples, 3]
        
        # Reshape back to original batch shape
        samples = samples.reshape(*batch_shape, num_samples, 3)
        
        return samples
    
    def forward(self, ray_bundle: dict[str, Any]) -> dict[str, Any]:
        """Forward pass through the model."""
        return self.sample_and_forward(ray_bundle, return_samples=True)
    
    def get_loss_dict(
        self,
        outputs: dict[str,
        Any],
        batch: dict[str,
        Any],
    ) -> dict[str, Float[Tensor, ""]]:
        """Get loss dictionary."""
        return self.loss_fn(outputs, batch)