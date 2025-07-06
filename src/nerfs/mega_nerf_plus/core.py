from __future__ import annotations

from typing import Any, Optional, Union

"""
Core components for Mega-NeRF++

This module implements the main components of Mega-NeRF++, including:
- Scalable neural network architectures
- Hierarchical spatial encoding
- Multi-resolution MLPs
- Photogrammetric rendering optimizations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import numpy as np
import math
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from pathlib import Path
from typing import TypeAlias, Any

# Type aliases for modern Python 3.10
Tensor: TypeAlias = torch.Tensor
Device: TypeAlias = torch.device | str
DType: TypeAlias = torch.dtype
TensorDict: TypeAlias = Dict[str, Tensor]


@dataclass
class MegaNeRFPlusConfig:
    """Configuration for MegaNeRF Plus model with modern features."""

    # Scene decomposition
    num_blocks: int = 8
    block_size: Tuple[int, int, int] = (32, 32, 32)
    overlap: int = 4  # Overlap between blocks

    # Network architecture
    feature_dim: int = 256
    hidden_dim: int = 256
    num_layers: int = 8
    skip_connections: List[int] = (4,)
    activation: str = "relu"
    output_activation: str = "sigmoid"

    # Positional encoding
    pos_encoding_levels: int = 10
    dir_encoding_levels: int = 4

    # Attention mechanism
    num_heads: int = 8
    head_dim: int = 32
    attention_dropout: float = 0.1

    # Rendering settings
    num_samples: int = 192
    num_importance_samples: int = 96
    background_color: str = "white"
    near_plane: float = 0.1
    far_plane: float = 1000.0

    # Training settings
    learning_rate: float = 5e-4
    learning_rate_decay_steps: int = 50000
    learning_rate_decay_mult: float = 0.1
    weight_decay: float = 1e-6

    # Training optimization
    use_amp: bool = True  # Automatic mixed precision
    grad_scaler_init_scale: float = 65536.0
    grad_scaler_growth_factor: float = 2.0
    grad_scaler_backoff_factor: float = 0.5
    grad_scaler_growth_interval: int = 2000

    # Memory optimization
    use_non_blocking: bool = True  # Non-blocking memory transfers
    set_grad_to_none: bool = True  # More efficient gradient clearing
    chunk_size: int = 8192  # Processing chunk size
    cache_grid_samples: bool = True  # Cache grid samples for faster training

    # Device settings
    default_device: str = "cuda"
    pin_memory: bool = True

    # Batch processing
    batch_size: int = 4096
    num_workers: int = 4

    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    save_every_n_steps: int = 1000
    keep_last_n_checkpoints: int = 5

    def __post_init__(self):
        """Post-initialization validation and initialization."""
        # Validate block settings
        assert len(self.block_size) == 3, "Block size must be a 3-tuple"
        assert all(s > 0 for s in self.block_size), "Block dimensions must be positive"
        assert self.overlap >= 0, "Overlap must be non-negative"

        # Initialize device
        self.device = torch.device(self.default_device if torch.cuda.is_available() else "cpu")

        # Initialize grad scaler for AMP
        if self.use_amp:
            self.grad_scaler = GradScaler(
                init_scale=self.grad_scaler_init_scale,
                growth_factor=self.grad_scaler_growth_factor,
                backoff_factor=self.grad_scaler_backoff_factor,
                growth_interval=self.grad_scaler_growth_interval,
            )

        # Create checkpoint directory
        self.checkpoint_dir = Path(self.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)


class HierarchicalSpatialEncoder(nn.Module):
    """
    Hierarchical spatial encoding for large-scale scenes

    This encoder uses multiple resolution levels to efficiently encode
    spatial information across different scales.
    """

    def __init__(self, config: MegaNeRFPlusConfig):
        super().__init__()
        self.config = config
        self.num_levels = config.num_blocks

        # Create encoding levels with different resolutions
        self.encoders = nn.ModuleList()
        for level in range(self.num_levels):
            resolution = config.block_size[0] * (2**level)
            resolution = min(resolution, config.block_size[0])

            # Hash encoding for each level
            self.encoders.append(self._create_hash_encoder(resolution))

        # Feature dimension calculation
        self.feature_dim = sum(enc.feature_dim for enc in self.encoders)

    def _create_hash_encoder(self, resolution: int) -> nn.Module:
        """Create hash encoder for given resolution"""
        # Simplified hash encoding implementation
        # In practice, this would use optimized hash encoding
        return HashEncoder(resolution=resolution, feature_dim=8)

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Encode positions hierarchically

        Args:
            positions: [..., 3] 3D positions

        Returns:
            [..., feature_dim] encoded features
        """
        features = []

        for encoder in self.encoders:
            feat = encoder(positions)
            features.append(feat)

        return torch.cat(features, dim=-1)


class HashEncoder(nn.Module):
    """Simplified hash encoder implementation"""

    def __init__(self, resolution: int, feature_dim: int = 8):
        super().__init__()
        self.resolution = resolution
        self.feature_dim = feature_dim

        # Hash table size (simplified)
        table_size = min(resolution**3, 2**19)  # Limit table size
        self.embedding = nn.Embedding(table_size, feature_dim)

        # Initialize with Xavier uniform
        nn.init.xavier_uniform_(self.embedding.weight)

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """Hash encode positions"""
        # Normalize positions to [0, resolution]
        pos_scaled = (positions + 1.0) * 0.5 * self.resolution
        pos_scaled = torch.clamp(pos_scaled, 0, self.resolution - 1)

        # Convert to grid indices (simplified)
        indices = (
            pos_scaled[..., 0] * self.resolution**2
            + pos_scaled[..., 1] * self.resolution
            + pos_scaled[..., 2]
        ).long()

        # Ensure indices are within bounds
        indices = torch.clamp(indices, 0, self.embedding.num_embeddings - 1)

        return self.embedding(indices)


class MultiResolutionMLP(nn.Module):
    """
    Multi-resolution MLP that can handle different levels of detail
    """

    def __init__(self, config: MegaNeRFPlusConfig, input_dim: int):
        super().__init__()
        self.config = config
        self.input_dim = input_dim

        # Create MLPs for different resolutions
        self.mlps = nn.ModuleList()
        for lod in range(config.num_blocks):
            # Smaller networks for higher LODs (distant views)
            width = config.feature_dim // (2**lod) if lod > 0 else config.feature_dim
            depth = max(config.num_layers - lod, 4)

            mlp = self._create_mlp(input_dim, width, depth)
            self.mlps.append(mlp)

        # Output layers
        self.density_heads = nn.ModuleList(
            [nn.Linear(self.mlps[i][-2].out_features, 1) for i in range(config.num_blocks)]
        )

        self.color_heads = nn.ModuleList(
            [nn.Linear(self.mlps[i][-2].out_features, 3) for i in range(config.num_blocks)]
        )

    def _create_mlp(self, input_dim: int, width: int, depth: int) -> nn.Sequential:
        """Create MLP with given dimensions"""
        layers = []

        # Input layer
        layers.append(nn.Linear(input_dim, width))
        layers.append(nn.ReLU(inplace=True))

        # Hidden layers
        for i in range(depth - 2):
            layers.append(nn.Linear(width, width))
            layers.append(nn.ReLU(inplace=True))

            # Skip connection at middle
            if i == depth // 2 - 1:
                # Note: This is simplified - real implementation would handle skip connections properly
                pass

        # Final hidden layer (no activation for output connection)
        layers.append(nn.Linear(width, width))

        return nn.Sequential(*layers)

    def forward(self, features: torch.Tensor, lod: int = 0) -> Dict[str, torch.Tensor]:
        """
        Forward pass through MLP at specified LOD

        Args:
            features: [..., input_dim] input features
            lod: Level of detail (0 = highest quality)

        Returns:
            Dictionary with density and color predictions
        """
        lod = min(lod, len(self.mlps) - 1)

        # Forward through MLP
        x = self.mlps[lod](features)

        # Predict density and color
        density = self.density_heads[lod](x)
        color = torch.sigmoid(self.color_heads[lod](x))

        return {"density": density, "color": color}


class PhotogrammetricRenderer(nn.Module):
    """
    Specialized renderer for photogrammetric data

    Optimized for high-resolution images with careful memory management
    """

    def __init__(self, config: MegaNeRFPlusConfig):
        super().__init__()
        self.config = config

    def sample_rays(
        self,
        rays_o: torch.Tensor,
        rays_d: torch.Tensor,
        near: float,
        far: float,
        num_samples: int,
        stratified: bool = True,
    ) -> torch.Tensor:
        """Sample points along rays with photogrammetric optimizations"""

        # Use inverse depth sampling for better near-field resolution
        t_vals = torch.linspace(0.0, 1.0, num_samples, device=rays_o.device)

        # Inverse depth sampling
        if stratified:
            # Add stratified sampling
            mids = 0.5 * (t_vals[:-1] + t_vals[1:])
            upper = torch.cat([mids, t_vals[-1:]])
            lower = torch.cat([t_vals[:1], mids])
            t_rand = torch.rand_like(t_vals)
            t_vals = lower + (upper - lower) * t_rand

        # Convert to world coordinates using inverse depth
        z_vals = 1.0 / (1.0 / near * (1.0 - t_vals) + 1.0 / far * t_vals)

        return z_vals.expand(*rays_o.shape[:-1], num_samples)

    def hierarchical_sample(
        self,
        rays_o: torch.Tensor,
        rays_d: torch.Tensor,
        z_vals: torch.Tensor,
        weights: torch.Tensor,
        num_importance: int,
    ) -> torch.Tensor:
        """Hierarchical sampling based on coarse weights"""

        # Get bin centers
        z_vals_mid = 0.5 * (z_vals[..., :-1] + z_vals[..., 1:])

        # Remove last weight (corresponds to infinity)
        weights = weights[..., 1:-1]
        weights = weights + 1e-5  # Prevent nans

        # Create PDF
        pdf = weights / torch.sum(weights, dim=-1, keepdim=True)
        cdf = torch.cumsum(pdf, dim=-1)
        cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], dim=-1)

        # Sample from CDF
        u = torch.rand(*cdf.shape[:-1], num_importance, device=rays_o.device)

        # Invert CDF
        inds = torch.searchsorted(cdf, u, right=True)
        below = torch.clamp(inds - 1, 0, cdf.shape[-1] - 1)
        above = torch.clamp(inds, 0, cdf.shape[-1] - 1)

        # Linear interpolation
        inds_g = torch.stack([below, above], dim=-1)
        cdf_g = torch.gather(cdf.unsqueeze(-1), -2, inds_g).squeeze(-1)
        bins_g = torch.gather(z_vals_mid.unsqueeze(-1), -2, inds_g).squeeze(-1)

        denom = cdf_g[..., 1] - cdf_g[..., 0]
        denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
        t = (u - cdf_g[..., 0]) / denom
        samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

        return samples

    def volume_render(
        self,
        densities: torch.Tensor,
        colors: torch.Tensor,
        z_vals: torch.Tensor,
        rays_d: torch.Tensor,
        white_bkgd: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Volume rendering with photogrammetric optimizations"""

        # Compute distances between samples
        dists = torch.diff(z_vals, dim=-1)
        dists = torch.cat([dists, torch.full_like(dists[..., :1], 1e10)], dim=-1)

        # Multiply by ray direction norm for proper distance
        dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

        # Compute alpha values
        alpha = 1.0 - torch.exp(-F.relu(densities[..., 0]) * dists)

        # Compute transmittance
        transmittance = torch.cumprod(1.0 - alpha + 1e-10, dim=-1)
        transmittance = torch.cat(
            [torch.ones_like(transmittance[..., :1]), transmittance[..., :-1]], dim=-1
        )

        # Compute weights
        weights = alpha * transmittance

        # Accumulate colors
        rgb = torch.sum(weights[..., None] * colors, dim=-2)

        # Add white background if specified
        if white_bkgd:
            acc_alpha = torch.sum(weights, dim=-1, keepdim=True)
            rgb = rgb + (1.0 - acc_alpha)

        # Compute depth
        depth = torch.sum(weights * z_vals, dim=-1)

        # Compute disparity
        disp = 1.0 / torch.max(1e-10 * torch.ones_like(depth), depth / torch.sum(weights, dim=-1))

        # Compute accumulated alpha
        acc_alpha = torch.sum(weights, dim=-1)

        return {
            "rgb": rgb,
            "depth": depth,
            "disp": disp,
            "acc_alpha": acc_alpha,
            "weights": weights,
        }


class ScalableNeRFModel(nn.Module):
    """
    Scalable NeRF model for large scenes

    Combines hierarchical spatial encoding with multi-resolution MLPs
    """

    def __init__(self, config: MegaNeRFPlusConfig):
        super().__init__()
        self.config = config

        # Spatial encoder
        self.spatial_encoder = HierarchicalSpatialEncoder(config)

        # Position encoding for view directions
        self.view_encoder = self._create_positional_encoder(4)  # 4 levels for view dirs

        # Multi-resolution MLPs
        pos_input_dim = self.spatial_encoder.feature_dim
        view_input_dim = self.view_encoder.output_dim if config.use_viewdirs else 0
        total_input_dim = pos_input_dim + view_input_dim

        self.nerf_mlp = MultiResolutionMLP(config, total_input_dim)

        # Renderer
        self.renderer = PhotogrammetricRenderer(config)

    def _create_positional_encoder(self, num_freqs: int) -> nn.Module:
        """Create positional encoder for view directions"""
        return PositionalEncoder(num_freqs)

    def forward(
        self,
        positions: torch.Tensor,
        view_dirs: Optional[torch.Tensor] = None,
        lod: int = 0,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through scalable NeRF model

        Args:
            positions: [..., 3] 3D positions
            view_dirs: [..., 3] viewing directions (optional)
            lod: Level of detail

        Returns:
            Dictionary with density and color predictions
        """
        # Encode positions
        pos_features = self.spatial_encoder(positions)

        # Encode view directions if provided
        if self.config.use_viewdirs and view_dirs is not None:
            view_features = self.view_encoder(view_dirs)
            features = torch.cat([pos_features, view_features], dim=-1)
        else:
            features = pos_features

        # Forward through MLP
        return self.nerf_mlp(features, lod)


class PositionalEncoder(nn.Module):
    """Positional encoder for view directions"""

    def __init__(self, num_freqs: int):
        super().__init__()
        self.num_freqs = num_freqs
        self.output_dim = 3 + 3 * 2 * num_freqs  # identity + sin/cos terms

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input with sinusoidal functions"""
        features = [x]  # Include identity

        for freq in range(self.num_freqs):
            features.append(torch.sin(2**freq * math.pi * x))
            features.append(torch.cos(2**freq * math.pi * x))

        return torch.cat(features, dim=-1)


class MegaNeRFPlus(nn.Module):
    """MegaNeRF Plus model with modern optimizations."""

    def __init__(self, config: MegaNeRFPlusConfig):
        super().__init__()
        self.config = config

        # Initialize blocks
        self.blocks = nn.ModuleList([MegaNeRFPlusBlock(config) for _ in range(config.num_blocks)])

        # Initialize block assignment network with attention
        self.block_assignment = AttentiveBlockAssignment(config)

        # Initialize AMP grad scaler
        self.grad_scaler = config.grad_scaler if config.use_amp else None

        # Initialize cache for grid samples
        self.grid_samples_cache = {} if config.cache_grid_samples else None

        # Move model to device
        self.to(config.device)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity=self.config.activation)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.MultiheadAttention):
                nn.init.xavier_uniform_(m.in_proj_weight)
                if m.in_proj_bias is not None:
                    nn.init.zeros_(m.in_proj_bias)

    def forward(self, coords: Tensor, view_dirs: Tensor | None = None) -> TensorDict:
        """Forward pass with automatic mixed precision support."""
        # Move inputs to device efficiently
        coords = coords.to(self.config.device, non_blocking=self.config.use_non_blocking)
        if view_dirs is not None:
            view_dirs = view_dirs.to(self.config.device, non_blocking=self.config.use_non_blocking)

        # Process points in chunks for memory efficiency
        outputs_list = []
        for i in range(0, coords.shape[0], self.config.chunk_size):
            chunk_coords = coords[i : i + self.config.chunk_size]
            chunk_view_dirs = (
                view_dirs[i : i + self.config.chunk_size] if view_dirs is not None else None
            )

            # Use AMP for forward pass
            with autocast(enabled=self.config.use_amp):
                # Get block assignments with attention
                block_weights, attention_weights = self.block_assignment(chunk_coords)

                # Process through each block
                block_outputs = []
                for block_idx, block in enumerate(self.blocks):
                    block_output = block(chunk_coords, chunk_view_dirs)
                    block_outputs.append(block_output)

                # Combine block outputs with attention weights
                rgb = torch.zeros_like(chunk_coords)
                density = torch.zeros(chunk_coords.shape[0], 1, device=chunk_coords.device)

                for block_idx, block_output in enumerate(block_outputs):
                    weight = block_weights[..., block_idx : block_idx + 1]
                    attn = attention_weights[..., block_idx]
                    rgb += weight * attn.unsqueeze(-1) * block_output["rgb"]
                    density += weight * attn.unsqueeze(-1) * block_output["density"]

                chunk_outputs = {
                    "rgb": rgb,
                    "density": density,
                    "block_weights": block_weights,
                    "attention_weights": attention_weights,
                }

            outputs_list.append(chunk_outputs)

        # Combine chunk outputs
        outputs = {
            k: torch.cat([out[k] for out in outputs_list], dim=0) for k in outputs_list[0].keys()
        }

        return outputs

    def training_step(
        self, batch: TensorDict, optimizer: torch.optim.Optimizer
    ) -> Tuple[Tensor, TensorDict]:
        """Optimized training step with AMP support."""
        # Zero gradients efficiently
        optimizer.zero_grad(set_to_none=self.config.set_grad_to_none)

        # Move batch to device efficiently
        batch = {
            k: v.to(self.config.device, non_blocking=self.config.use_non_blocking)
            for k, v in batch.items()
        }

        # Forward pass with AMP
        with autocast(enabled=self.config.use_amp):
            outputs = self(batch["coords"], batch.get("view_dirs"))
            loss = self.compute_loss(outputs, batch)

        # Backward pass with grad scaling
        if self.config.use_amp:
            self.grad_scaler.scale(loss["total_loss"]).backward()
            self.grad_scaler.step(optimizer)
            self.grad_scaler.update()
        else:
            loss["total_loss"].backward()
            optimizer.step()

        return loss["total_loss"], loss

    @torch.inference_mode()
    def validation_step(self, batch: TensorDict) -> TensorDict:
        """Efficient validation step."""
        batch = {
            k: v.to(self.config.device, non_blocking=self.config.use_non_blocking)
            for k, v in batch.items()
        }

        outputs = self(batch["coords"], batch.get("view_dirs"))
        loss = self.compute_loss(outputs, batch)

        return loss

    def compute_loss(self, predictions: TensorDict, targets: TensorDict) -> TensorDict:
        """Compute training losses."""
        losses = {}

        # RGB loss
        if "rgb" in targets:
            losses["rgb"] = F.mse_loss(predictions["rgb"], targets["rgb"])

        # Density regularization
        if "density" in predictions:
            losses["density_reg"] = torch.mean(predictions["density"])

        # Block assignment regularization
        if "block_weights" in predictions:
            # Encourage sparse block assignments
            entropy = -torch.sum(
                predictions["block_weights"] * torch.log(predictions["block_weights"] + 1e-10),
                dim=-1,
            ).mean()
            losses["block_entropy"] = entropy

        # Attention regularization
        if "attention_weights" in predictions:
            # Encourage focused attention
            attention_entropy = -torch.sum(
                predictions["attention_weights"]
                * torch.log(predictions["attention_weights"] + 1e-10),
                dim=-1,
            ).mean()
            losses["attention_entropy"] = attention_entropy

        # Total loss
        losses["total_loss"] = sum(losses.values())

        return losses

    def update_learning_rate(self, optimizer: torch.optim.Optimizer, step: int):
        """Update learning rate with decay."""
        decay_steps = step // self.config.learning_rate_decay_steps
        decay_factor = self.config.learning_rate_decay_mult**decay_steps

        # Update optimizer learning rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = self.config.learning_rate * decay_factor


class AttentiveBlockAssignment(nn.Module):
    """Attentive block assignment module."""

    def __init__(self, config: MegaNeRFPlusConfig):
        super().__init__()
        self.config = config

        # Initialize positional encoding
        self.pos_encoder = PositionalEncoding(input_dim=3, num_levels=config.pos_encoding_levels)

        # Initialize attention layers
        self.attention = nn.MultiheadAttention(
            embed_dim=config.hidden_dim,
            num_heads=config.num_heads,
            dropout=config.attention_dropout,
        )

        # Initialize MLP
        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.num_blocks),
        )

    def forward(self, coords: Tensor) -> Tuple[Tensor, Tensor]:
        """Forward pass with attention."""
        # Encode coordinates
        encoded_coords = self.pos_encoder(coords)

        # Apply attention
        query = encoded_coords.unsqueeze(0)
        key = value = query
        attn_output, attention_weights = self.attention(query, key, value)

        # Get block weights
        block_weights = F.softmax(self.mlp(attn_output.squeeze(0)), dim=-1)

        return block_weights, attention_weights.squeeze(0)
