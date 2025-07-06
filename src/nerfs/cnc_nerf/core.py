from __future__ import annotations

from typing import Any, Optional

"""
CNC-NeRF Core Module

This module implements the core components of the Context-based NeRF Compression (CNC) framework
with modern PyTorch features and optimizations.

Key components:
- Multi-resolution hash embeddings with binarization
- Level-wise and dimension-wise context models
- Entropy estimation and arithmetic coding
- Hash collision fusion with occupancy grids
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
import math
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from pathlib import Path
from typing import TypeAlias

# Type aliases for modern Python 3.10
Tensor: TypeAlias = torch.Tensor
Device: TypeAlias = torch.device | str
DType: TypeAlias = torch.dtype
TensorDict: TypeAlias = Dict[str, Tensor]

try:
    import tinycudann as tcnn

    TCNN_AVAILABLE = True
except ImportError:
    TCNN_AVAILABLE = False


@dataclass
class CNCNeRFConfig:
    """Configuration for CNC-NeRF model with modern features."""

    # Model architecture
    feature_dim: int = 8
    num_levels: int = 12
    base_resolution: int = 16
    max_resolution: int = 512
    hash_table_size: int = 2**19

    # 2D embeddings for tri-plane features
    num_2d_levels: int = 4
    base_2d_resolution: int = 128
    max_2d_resolution: int = 1024
    hash_table_2d_size: int = 2**17

    # Context models
    context_levels: int = 3  # Lc in paper
    context_fuser_hidden: int = 32

    # Compression
    compression_lambda: float = 2e-3  # Rate-distortion trade-off
    entropy_regularization: bool = True
    arithmetic_coding: bool = True

    # Occupancy grid
    occupancy_grid_resolution: int = 128
    occupancy_threshold: float = 0.01

    # Binarization (following BiRF)
    use_binarization: bool = True
    straight_through_estimator: bool = True

    # Rendering
    background_color: str = "white"
    near_plane: float = 0.2
    far_plane: float = 1000.0

    # Training optimization
    learning_rate: float = 1e-3
    learning_rate_decay_steps: int = 10000
    learning_rate_decay_factor: float = 0.5
    use_amp: bool = True  # Automatic mixed precision
    grad_scaler_init_scale: float = 65536.0
    grad_scaler_growth_factor: float = 2.0
    grad_scaler_backoff_factor: float = 0.5
    grad_scaler_growth_interval: int = 2000

    # Memory optimization
    use_non_blocking: bool = True  # Non-blocking memory transfers
    set_grad_to_none: bool = True  # More efficient gradient clearing
    chunk_size: int = 8192  # Processing chunk size

    # Device settings
    default_device: str = "cuda"
    pin_memory: bool = True

    def __post_init__(self):
        """Post-initialization validation."""
        assert self.num_levels >= self.context_levels, "Context levels cannot exceed total levels"
        assert (
            self.base_resolution < self.max_resolution
        ), "Base resolution must be less than max resolution"

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


class HashEmbeddingEncoder(nn.Module):
    """Multi-resolution hash embedding encoder with binarization support."""

    def __init__(self, config: CNCNeRFConfig):
        super().__init__()
        self.config = config
        self.gradient_checkpointing = False

        # Calculate resolutions for each level
        self.resolutions = []
        for level in range(config.num_levels):
            resolution = int(
                config.base_resolution
                * (config.max_resolution / config.base_resolution)
                ** (level / (config.num_levels - 1))
            )
            self.resolutions.append(resolution)

        # Hash embeddings for each level
        self.hash_embeddings = nn.ModuleList()
        for level in range(config.num_levels):
            embedding = nn.Embedding(config.hash_table_size, config.feature_dim)
            nn.init.uniform_(embedding.weight, -1e-4, 1e-4)
            self.hash_embeddings.append(embedding)

        # 2D embeddings for tri-plane features (following BiRF design)
        self.hash_embeddings_2d = nn.ModuleList()
        self.resolutions_2d = []
        for level in range(config.num_2d_levels):
            resolution = int(
                config.base_2d_resolution
                * (config.max_2d_resolution / config.base_2d_resolution)
                ** (level / (config.num_2d_levels - 1))
            )
            self.resolutions_2d.append(resolution)

            embedding = nn.Embedding(config.hash_table_2d_size, config.feature_dim)
            nn.init.uniform_(embedding.weight, -1e-4, 1e-4)
            self.hash_embeddings_2d.append(embedding)

    def spatial_hash_3d(self, coords: torch.Tensor, resolution: int) -> torch.Tensor:
        """3D spatial hash function."""
        primes = [1, 2654435761, 805459861]
        coords_int = torch.floor(coords * resolution).long()
        coords_int = torch.clamp(coords_int, 0, resolution - 1)

        hash_coords = coords_int[:, 0] * primes[0]
        hash_coords ^= coords_int[:, 1] * primes[1]
        hash_coords ^= coords_int[:, 2] * primes[2]

        return hash_coords % self.config.hash_table_size

    def spatial_hash_2d(self, coords: torch.Tensor, resolution: int) -> torch.Tensor:
        """2D spatial hash function."""
        primes = [1, 2654435761]
        coords_int = torch.floor(coords * resolution).long()
        coords_int = torch.clamp(coords_int, 0, resolution - 1)

        hash_coords = coords_int[:, 0] * primes[0]
        hash_coords ^= coords_int[:, 1] * primes[1]

        return hash_coords % self.config.hash_table_2d_size

    def trilinear_interpolation(self, coords: torch.Tensor, level: int) -> torch.Tensor:
        """Trilinear interpolation for 3D hash grid."""
        resolution = self.resolutions[level]
        coords_scaled = coords * resolution
        coords_floor = torch.floor(coords_scaled)
        coords_frac = coords_scaled - coords_floor

        # Get 8 corner coordinates
        corners = []
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    corner = coords_floor + torch.tensor([i, j, k], device=coords.device)
                    corners.append(corner)

        # Get features for each corner
        corner_features = []
        for corner in corners:
            hash_idx = self.spatial_hash_3d(corner, resolution)
            features = self.hash_embeddings[level](hash_idx)

            # Apply binarization if enabled
            if self.config.use_binarization:
                if self.training and self.config.straight_through_estimator:
                    # Straight-through estimator during training
                    features_binary = torch.sign(features)
                    features = features_binary + features - features.detach()
                else:
                    features = torch.sign(features)

            corner_features.append(features)

        # Trilinear interpolation
        c00 = corner_features[0] * (1 - coords_frac[:, 0:1],)
        c01 = corner_features[2] * (1 - coords_frac[:, 0:1],)
        c10 = corner_features[4] * (1 - coords_frac[:, 0:1],)
        c11 = corner_features[6] * (1 - coords_frac[:, 0:1],)

        c0 = c00 * (1 - coords_frac[:, 1:2]) + c10 * coords_frac[:, 1:2]
        c1 = c01 * (1 - coords_frac[:, 1:2]) + c11 * coords_frac[:, 1:2]

        result = c0 * (1 - coords_frac[:, 2:3]) + c1 * coords_frac[:, 2:3]

        return result

    def bilinear_interpolation(self, coords_2d: torch.Tensor, level: int) -> torch.Tensor:
        """Bilinear interpolation for 2D hash grid."""
        resolution = self.resolutions_2d[level]
        coords_scaled = coords_2d * resolution
        coords_floor = torch.floor(coords_scaled)
        coords_frac = coords_scaled - coords_floor

        # Get 4 corner coordinates
        corners = [
            coords_floor,
            coords_floor
            + torch.tensor(
                [1, 0],
                device=coords_2d.device,
            ),
        ]

        # Get features for each corner
        corner_features = []
        for corner in corners:
            hash_idx = self.spatial_hash_2d(corner, resolution)
            features = self.hash_embeddings_2d[level](hash_idx)

            # Apply binarization if enabled
            if self.config.use_binarization:
                if self.training and self.config.straight_through_estimator:
                    features_binary = torch.sign(features)
                    features = features_binary + features - features.detach()
                else:
                    features = torch.sign(features)

            corner_features.append(features)

        # Bilinear interpolation
        top = corner_features[0] * (1 - coords_frac[:, 0:1],)
        bottom = corner_features[2] * (1 - coords_frac[:, 0:1],)
        result = top * (1 - coords_frac[:, 1:2]) + bottom * coords_frac[:, 1:2]

        return result

    def encode_3d(self, coords: torch.Tensor) -> torch.Tensor:
        """Encode 3D coordinates using multi-resolution hash embeddings."""
        features = []
        for level in range(self.config.num_levels):
            level_features = self.trilinear_interpolation(coords, level)
            features.append(level_features)
        return torch.cat(features, dim=-1)

    def encode_2d(self, coords: torch.Tensor) -> torch.Tensor:
        """Encode coordinates using 2D tri-plane features."""
        # Project 3D coordinates to three 2D planes
        xy_coords = coords[:, [0, 1]]
        xz_coords = coords[:, [0, 2]]
        yz_coords = coords[:, [1, 2]]

        plane_features = []
        for plane_coords in [xy_coords, xz_coords, yz_coords]:
            features = []
            for level in range(self.config.num_2d_levels):
                level_features = self.bilinear_interpolation(plane_coords, level)
                features.append(level_features)
            plane_features.append(torch.cat(features, dim=-1))

        return torch.cat(plane_features, dim=-1)

    def _forward_with_checkpointing(self, coords: Tensor) -> Tensor:
        """Forward pass with gradient checkpointing."""

        def create_custom_forward(level: int):
            def custom_forward(*inputs):
                coords = inputs[0]
                resolution = self.resolutions[level]
                hash_idx = self.spatial_hash_3d(coords, resolution)
                features = self.hash_embeddings[level](hash_idx)
                if self.config.use_binarization:
                    if self.training and self.config.straight_through_estimator:
                        features = (features > 0).float() - features.detach() + features
                    else:
                        features = (features > 0).float()
                return features

            return custom_forward

        # Process each level with checkpointing
        features_list = []
        for level in range(self.config.num_levels):
            features = torch.utils.checkpoint.checkpoint(
                create_custom_forward(level), coords, preserve_rng_state=False
            )
            features_list.append(features)

        return torch.cat(features_list, dim=-1)

    def forward(self, coords: Tensor) -> Tensor:
        """Forward pass."""
        if self.gradient_checkpointing and self.training:
            return self._forward_with_checkpointing(coords)

        # Regular forward pass
        features_list = []
        for level in range(self.config.num_levels):
            resolution = self.resolutions[level]
            hash_idx = self.spatial_hash_3d(coords, resolution)
            features = self.hash_embeddings[level](hash_idx)

            if self.config.use_binarization:
                if self.training and self.config.straight_through_estimator:
                    features = (features > 0).float() - features.detach() + features
                else:
                    features = (features > 0).float()

            features_list.append(features)

        return torch.cat(features_list, dim=-1)


class ContextModel(nn.Module):
    """Base class for context models."""

    def __init__(self, config: CNCNeRFConfig):
        super().__init__()
        self.config = config

    def calculate_frequency(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Calculate occurrence frequency of +1 values in binarized embeddings."""
        if self.config.use_binarization:
            positive_count = (embeddings > 0).float().sum()
            total_count = embeddings.numel()
            frequency = positive_count / (total_count + 1e-8)
        else:
            frequency = embeddings.mean()
        return frequency


class LevelWiseContextModel(ContextModel):
    """Level-wise context model for multi-resolution embeddings."""

    def __init__(self, config: CNCNeRFConfig):
        super().__init__(config)

        # Context fuser MLP - simplified for proper dimension handling
        self.context_fuser = nn.Sequential(
            nn.Linear(1, config.context_fuser_hidden),  # Start with just frequency
            nn.LeakyReLU(
                0.2,
            ),
        )

    def build_context(self, embeddings_list: List[torch.Tensor], level: int) -> torch.Tensor:
        """Build context from previous levels."""
        current_embeddings = embeddings_list[level]

        # For now, just use frequency as a simple context
        frequency = self.calculate_frequency(current_embeddings)
        context = frequency.unsqueeze(0).unsqueeze(0)  # Shape: [1, 1]

        return context

    def forward(self, embeddings_list: List[torch.Tensor], level: int) -> torch.Tensor:
        """Predict probability distribution for current level."""
        context = self.build_context(embeddings_list, level)
        probabilities = self.context_fuser(context)

        # Expand to match embedding dimensions if needed
        current_embeddings = embeddings_list[level]
        if probabilities.shape[0] == 1 and current_embeddings.shape[0] > 1:
            probabilities = probabilities.expand(current_embeddings.shape[0], -1)

        return probabilities


class DimensionWiseContextModel(ContextModel):
    """Dimension-wise context model for 2D-3D cross-dimension dependencies."""

    def __init__(self, config: CNCNeRFConfig):
        super().__init__(config)

        # Lighter context fuser for 2D embeddings - simplified
        self.context_fuser_2d = nn.Sequential(
            nn.Linear(1, config.feature_dim), nn.Sigmoid()  # Start simple with frequency only
        )

    def project_3d_to_2d(self, embeddings_3d: torch.Tensor) -> torch.Tensor:
        """Project 3D voxel features to 2D Projected Voxel Features (PVF)."""
        if self.config.use_binarization:
            # Project along three axes and record frequency of +1s
            pvf_xy = (embeddings_3d > 0).float().mean(dim=2)  # Project along z
            pvf_xz = (embeddings_3d > 0).float().mean(dim=1)  # Project along y
            pvf_yz = (embeddings_3d > 0).float().mean(dim=0)  # Project along x
        else:
            pvf_xy = embeddings_3d.mean(dim=2)
            pvf_xz = embeddings_3d.mean(dim=1)
            pvf_yz = embeddings_3d.mean(dim=0)

        # Combine projected features
        pvf = torch.cat([pvf_xy.flatten(), pvf_xz.flatten(), pvf_yz.flatten()])
        return pvf

    def forward(
        self,
        embeddings_2d_list: List[torch.Tensor],
        embeddings_3d: torch.Tensor,
        level: int,
    ) -> torch.Tensor:
        """Predict probability for 2D embeddings using 3D context."""
        # Simplified: just use frequency of current 2D embeddings
        current_embeddings_2d = embeddings_2d_list[level]
        frequency = self.calculate_frequency(current_embeddings_2d)
        context = frequency.unsqueeze(0).unsqueeze(0)  # Shape: [1, 1]

        probabilities = self.context_fuser_2d(context)

        # Expand to match embedding dimensions if needed
        if probabilities.shape[0] == 1 and current_embeddings_2d.shape[0] > 1:
            probabilities = probabilities.expand(current_embeddings_2d.shape[0], -1)

        return probabilities


class EntropyEstimator(nn.Module):
    """Entropy estimator for rate-distortion optimization."""

    def __init__(self, config: CNCNeRFConfig):
        super().__init__()
        self.config = config

    def bit_estimator(self, probabilities: torch.Tensor, embeddings: torch.Tensor) -> torch.Tensor:
        """Estimate bit consumption using entropy formula from paper."""
        if self.config.use_binarization:
            # For binarized embeddings: θ ∈ {-1, +1}
            # bit(p|θ) = -(1+θ)/2 * log2(p) - (1-θ)/2 * log2(1-p)
            eps = 1e-8
            p_clamped = torch.clamp(probabilities, eps, 1 - eps)

            positive_term = 0.5 * (1 + embeddings) * torch.log2(p_clamped)
            negative_term = 0.5 * (1 - embeddings) * torch.log2(1 - p_clamped)

            bits = -(positive_term + negative_term)
        else:
            # For continuous embeddings
            bits = -torch.log2(probabilities + 1e-8)

        return bits

    def forward(self, probabilities: torch.Tensor, embeddings: torch.Tensor) -> torch.Tensor:
        """Calculate total entropy loss."""
        bits = self.bit_estimator(probabilities, embeddings)
        return bits.sum()


class ArithmeticCoder(nn.Module):
    """Arithmetic coding for actual compression."""

    def __init__(self, config: CNCNeRFConfig):
        super().__init__()
        self.config = config

    def encode(self, embeddings: torch.Tensor, probabilities: torch.Tensor) -> bytes:
        """Encode embeddings using arithmetic coding."""
        # Simplified implementation - in practice would use proper arithmetic coding library
        if self.config.use_binarization:
            # Convert to binary sequence
            binary_seq = (embeddings > 0).int().cpu().numpy().flatten()
            probs = probabilities.detach().cpu().numpy().flatten()

            # Placeholder for actual arithmetic coding
            encoded_size = len(binary_seq) * 0.1  # Assume 10x compression
            return b"compressed" * int(encoded_size)
        else:
            # For continuous values
            quantized = self._quantize(embeddings)
            encoded_size = len(quantized) * 0.1
            return b"compressed" * int(encoded_size)

    def decode(self, encoded_data: bytes, shape: Tuple[int, ...]) -> torch.Tensor:
        """Decode embeddings from compressed data."""
        # Placeholder implementation
        if self.config.use_binarization:
            return torch.sign(torch.randn(shape))
        else:
            return torch.randn(shape)

    def _quantize(self, embeddings: torch.Tensor, bits: int = 8) -> np.ndarray:
        """Quantize continuous embeddings."""
        min_val = embeddings.min().item()
        max_val = embeddings.max().item()

        quantized = ((embeddings - min_val) / (max_val - min_val) * (2**bits - 1)).int()
        return quantized.cpu().numpy().flatten()


class OccupancyGrid(nn.Module):
    """Occupancy grid for spatial pruning and hash fusion."""

    def __init__(self, config: CNCNeRFConfig):
        super().__init__()
        self.config = config
        self.resolution = config.occupancy_grid_resolution

        # Binary occupancy grid
        self.register_buffer("grid", torch.zeros(self.resolution, self.resolution, self.resolution))

    def update_grid(self, coords: torch.Tensor, densities: torch.Tensor):
        """Update occupancy grid based on densities."""
        # Convert coordinates to grid indices
        grid_indices = (coords * (self.resolution - 1)).long()
        grid_indices = torch.clamp(grid_indices, 0, self.resolution - 1)

        # Update occupied cells
        occupied_mask = densities > self.config.occupancy_threshold
        if occupied_mask.any():
            occupied_indices = grid_indices[occupied_mask]
            self.grid[occupied_indices[:, 0], occupied_indices[:, 1], occupied_indices[:, 2]] = 1.0

    def get_occupancy(self, coords: torch.Tensor) -> torch.Tensor:
        """Get occupancy status for given coordinates."""
        grid_indices = (coords * (self.resolution - 1)).long()
        grid_indices = torch.clamp(grid_indices, 0, self.resolution - 1)

        return self.grid[grid_indices[:, 0], grid_indices[:, 1], grid_indices[:, 2]]

    def calculate_aoe(self, coords: torch.Tensor) -> torch.Tensor:
        """Calculate Area of Effect for hash fusion."""
        grid_indices = (coords * (self.resolution - 1)).long()
        grid_indices = torch.clamp(grid_indices, 0, self.resolution - 1)

        aoe_values = []
        for i in range(coords.shape[0]):
            center = grid_indices[i]

            # Check 3x3x3 neighborhood
            occupied_count = 0
            total_count = 0

            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    for dz in [-1, 0, 1]:
                        neighbor = center + torch.tensor([dx, dy, dz], device=coords.device)
                        neighbor = torch.clamp(neighbor, 0, self.resolution - 1)

                        if self.grid[neighbor[0], neighbor[1], neighbor[2]] > 0:
                            occupied_count += 1
                        total_count += 1

            aoe = occupied_count / total_count if total_count > 0 else 0.0
            aoe_values.append(aoe)

        return torch.tensor(aoe_values, device=coords.device)


class CNCRenderer(nn.Module):
    """Neural renderer for CNC-NeRF."""

    def __init__(self, config: CNCNeRFConfig):
        super().__init__()
        self.config = config
        self.gradient_checkpointing = False

        # MLP architecture
        if TCNN_AVAILABLE:
            self.mlp = tcnn.Network(
                n_input_dims=config.feature_dim * config.num_levels + 3,  # +3 for view direction
                n_output_dims=4,  # RGB + density
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": 64,
                    "n_hidden_layers": 2,
                },
            )
        else:
            self.mlp = nn.Sequential(
                nn.Linear(config.feature_dim * config.num_levels + 3, 64),
                nn.ReLU(True),
                nn.Linear(64, 64),
                nn.ReLU(True),
                nn.Linear(64, 4),
            )

    def _forward_with_checkpointing(
        self,
        coords: Tensor,
        view_dirs: Optional[Tensor] = None,
    ) -> TensorDict:
        """Forward pass with gradient checkpointing."""

        def custom_forward(*inputs):
            coords, view_dirs = inputs
            # Get features from encoder
            features = self.encoder(coords)

            # Concatenate with view directions if provided
            if view_dirs is not None:
                features = torch.cat([features, view_dirs], dim=-1)

            # Get raw outputs from MLP
            raw = self.mlp(features)
            rgb = torch.sigmoid(raw[..., :3])
            density = F.relu(raw[..., 3:])

            return {"rgb": rgb, "density": density}

        return torch.utils.checkpoint.checkpoint(
            custom_forward,
            coords,
            view_dirs if view_dirs is not None else coords.new_zeros(*coords.shape[:-1], 3),
            preserve_rng_state=False,
        )

    def forward(
        self,
        coords: Tensor,
        view_dirs: Optional[Tensor] = None,
    ) -> TensorDict:
        """Forward pass."""
        if self.gradient_checkpointing and self.training:
            return self._forward_with_checkpointing(coords, view_dirs)

        # Get features from encoder
        features = self.encoder(coords)

        # Concatenate with view directions if provided
        if view_dirs is not None:
            features = torch.cat([features, view_dirs], dim=-1)

        # Get raw outputs from MLP
        raw = self.mlp(features)
        rgb = torch.sigmoid(raw[..., :3])
        density = F.relu(raw[..., 3:])

        return {"rgb": rgb, "density": density}


class CNCNeRF(nn.Module):
    """CNC-NeRF model with modern features."""

    def __init__(self, config: CNCNeRFConfig):
        super().__init__()
        self.config = config
        self.encoder = HashEmbeddingEncoder(config)
        self.renderer = CNCRenderer(config)
        self.level_context = LevelWiseContextModel(config)
        self.dim_context = DimensionWiseContextModel(config)
        self.entropy_estimator = EntropyEstimator(config)
        self.arithmetic_coder = ArithmeticCoder(config)
        self.occupancy_grid = OccupancyGrid(config)

    def forward(
        self,
        coords: Tensor,
        view_dirs: Optional[Tensor] = None,
    ) -> TensorDict:
        """Forward pass."""
        # Move inputs to device if needed
        if self.config.use_non_blocking:
            coords = coords.to(self.config.device, non_blocking=True)
            if view_dirs is not None:
                view_dirs = view_dirs.to(self.config.device, non_blocking=True)

        # Process in chunks to save memory
        outputs_list = []
        for i in range(0, coords.shape[1], self.config.chunk_size):
            chunk_coords = coords[:, i : i + self.config.chunk_size]
            chunk_view_dirs = (
                view_dirs[:, i : i + self.config.chunk_size] if view_dirs is not None else None
            )

            # Get outputs from renderer
            chunk_outputs = self.renderer(chunk_coords, chunk_view_dirs)
            outputs_list.append(chunk_outputs)

        # Combine chunk outputs
        outputs = {
            k: torch.cat([out[k] for out in outputs_list], dim=1) for k in outputs_list[0].keys()
        }

        return outputs

    def training_step(
        self, batch: TensorDict, optimizer: torch.optim.Optimizer
    ) -> Tuple[Tensor, Dict[str, float]]:
        """Training step."""
        # Move batch to device
        batch = {
            k: v.to(self.config.device, non_blocking=self.config.use_non_blocking)
            for k, v in batch.items()
        }

        # Forward pass with AMP
        with autocast(enabled=self.config.use_amp):
            outputs = self(batch["coords"], batch["view_dirs"])

            # Compute RGB loss
            rgb_loss = F.mse_loss(outputs["rgb"], batch["rgb"])

            # Compute compression loss
            compression_loss = self.compute_compression_loss()

            # Total loss
            total_loss = rgb_loss + self.config.compression_lambda * compression_loss

        # Optimization step with gradient scaling
        if self.config.use_amp:
            self.config.grad_scaler.scale(total_loss).backward()
            self.config.grad_scaler.step(optimizer)
            self.config.grad_scaler.update()
        else:
            total_loss.backward()
            optimizer.step()

        # Clear gradients
        if self.config.set_grad_to_none:
            optimizer.zero_grad(set_to_none=True)
        else:
            optimizer.zero_grad()

        # Compute metrics
        with torch.no_grad():
            psnr = -10 * torch.log10(rgb_loss)
            compression_stats = self.get_compression_stats()

        metrics = {
            "total_loss": total_loss.item(),
            "rgb_loss": rgb_loss.item(),
            "compression_loss": compression_loss.item(),
            "psnr": psnr.item(),
            **compression_stats,
        }

        return total_loss, metrics

    def validation_step(self, batch: TensorDict) -> Dict[str, float]:
        """Validation step."""
        # Move batch to device
        batch = {
            k: v.to(self.config.device, non_blocking=self.config.use_non_blocking)
            for k, v in batch.items()
        }

        # Forward pass with AMP and no gradients
        with torch.no_grad(), autocast(enabled=self.config.use_amp):
            outputs = self(batch["coords"], batch["view_dirs"])

            # Compute RGB loss
            rgb_loss = F.mse_loss(outputs["rgb"], batch["rgb"])

            # Compute compression loss
            compression_loss = self.compute_compression_loss()

            # Compute PSNR
            psnr = -10 * torch.log10(rgb_loss)

            # Get compression stats
            compression_stats = self.get_compression_stats()

        metrics = {
            "total_loss": rgb_loss.item()
            + self.config.compression_lambda * compression_loss.item(),
            "rgb_loss": rgb_loss.item(),
            "compression_loss": compression_loss.item(),
            "psnr": psnr.item(),
            **compression_stats,
        }

        return metrics

    def update_learning_rate(self, optimizer: torch.optim.Optimizer, step: int) -> None:
        """Update learning rate using exponential decay."""
        if step > 0 and step % self.config.learning_rate_decay_steps == 0:
            for param_group in optimizer.param_groups:
                param_group["lr"] *= self.config.learning_rate_decay_factor

    def enable_gradient_checkpointing(self) -> None:
        """Enable gradient checkpointing for memory efficiency."""
        self.encoder.gradient_checkpointing = True
        self.renderer.gradient_checkpointing = True

    def compute_compression_loss(self) -> torch.Tensor:
        """Compute compression loss using context models."""
        total_entropy_loss = 0.0

        # Compress 3D embeddings with level-wise context
        embeddings_3d = []
        for level in range(self.config.num_levels):
            embedding_weights = self.renderer.hash_encoder.hash_embeddings[level].weight
            embeddings_3d.append(embedding_weights)

            # Get probability prediction from context model
            probabilities = self.renderer.level_context_model(embeddings_3d, level)

            # Calculate entropy loss
            entropy_loss = self.renderer.entropy_estimator(probabilities, embedding_weights)
            total_entropy_loss += entropy_loss

        # Compress 2D embeddings with dimension-wise context
        embeddings_2d = []
        finest_3d = embeddings_3d[-1] if embeddings_3d else torch.zeros(1, self.config.feature_dim)

        for level in range(self.config.num_2d_levels):
            embedding_weights = self.renderer.hash_encoder.hash_embeddings_2d[level].weight
            embeddings_2d.append(embedding_weights)

            # Get probability prediction from dimension-wise context
            probabilities = self.renderer.dimension_context_model(embeddings_2d, finest_3d, level)

            # Calculate entropy loss
            entropy_loss = self.renderer.entropy_estimator(probabilities, embedding_weights)
            total_entropy_loss += entropy_loss

        return total_entropy_loss

    def compress_model(self) -> Dict[str, Any]:
        """Compress the model and return compression info."""
        compression_info = {"compressed_data": {}}

        # Calculate original size
        original_size = sum(p.numel() * 4 for p in self.parameters())  # 4 bytes per float32

        # Compress embeddings
        total_compressed_size = 0

        # Compress 3D embeddings
        embeddings_3d = []
        for level in range(self.config.num_levels):
            embedding_weights = self.renderer.hash_encoder.hash_embeddings[level].weight
            embeddings_3d.append(embedding_weights)

            probabilities = self.renderer.level_context_model(embeddings_3d, level)
            compressed_data = self.renderer.arithmetic_coder.encode(
                embedding_weights,
                probabilities,
            )

            compression_info["compressed_data"][f"3d_level_{level}"] = compressed_data
            total_compressed_size += len(compressed_data)

        # Compress 2D embeddings
        embeddings_2d = []
        finest_3d = embeddings_3d[-1] if embeddings_3d else torch.zeros(1, self.config.feature_dim)

        for level in range(self.config.num_2d_levels):
            embedding_weights = self.renderer.hash_encoder.hash_embeddings_2d[level].weight
            embeddings_2d.append(embedding_weights)

            probabilities = self.renderer.dimension_context_model(embeddings_2d, finest_3d, level)
            compressed_data = self.renderer.arithmetic_coder.encode(
                embedding_weights,
                probabilities,
            )

            compression_info["compressed_data"][f"2d_level_{level}"] = compressed_data
            total_compressed_size += len(compressed_data)

        # Calculate compression ratio
        compression_ratio = (
            original_size / total_compressed_size if total_compressed_size > 0 else 1.0
        )

        compression_info.update(
            {
                "compression_ratio": compression_ratio,
                "original_size": original_size,
                "compressed_size": total_compressed_size,
            }
        )

        self.compression_ratio = compression_ratio
        self.original_size = original_size
        self.compressed_size = total_compressed_size

        return compression_info

    def get_compression_stats(self) -> Dict[str, float]:
        """Get compression statistics."""
        return {
            "compression_ratio": self.compression_ratio,
            "original_size_mb": self.original_size / (1024 * 1024,),
        }
