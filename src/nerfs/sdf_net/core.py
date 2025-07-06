from __future__ import annotations

from typing import Any, Optional, Union

"""
DeepSDF Network Core Implementation
基于论文: "DeepSDF: Learning Continuous Signed Distance Functions for Shape Representation"

DeepSDF学习一个连续函数 f: (R^3, Z) -> R，
其中Z是潜在编码，输出是有符号距离值。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from pathlib import Path
from typing import TypeAlias

# Type aliases for modern Python 3.10
Tensor: TypeAlias = torch.Tensor
Device: TypeAlias = torch.device | str
DType: TypeAlias = torch.dtype
TensorDict: TypeAlias = Dict[str, Tensor]


@dataclass
class SDFNetConfig:
    """Configuration for SDF-Net model with modern features."""

    # Network architecture
    hidden_dim: int = 256
    num_layers: int = 8
    skip_connections: List[int] = (4,)
    activation: str = "relu"
    output_activation: str = "sigmoid"

    # SDF settings
    sdf_scale: float = 1.0
    use_sphere_init: bool = True
    sphere_radius: float = 1.0

    # Positional encoding
    pos_encoding_levels: int = 10
    dir_encoding_levels: int = 4

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
        # Validate settings
        assert self.num_samples > 0, "Number of samples must be positive"
        assert self.num_importance_samples > 0, "Number of importance samples must be positive"
        assert self.near_plane < self.far_plane, "Near plane must be less than far plane"

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


class SDFNetwork(nn.Module):
    """Network for predicting SDF values and gradients."""

    def __init__(
        self,
        hidden_dim: int,
        num_layers: int,
        skip_connections: List[int],
        use_sphere_init: bool = True,
        sphere_radius: float = 1.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.skip_connections = skip_connections
        self.use_sphere_init = use_sphere_init
        self.sphere_radius = sphere_radius

        # Initialize network layers
        self.layers = nn.ModuleList()

        # First layer
        self.layers.append(nn.Linear(3, hidden_dim))

        # Hidden layers
        for i in range(num_layers - 1):
            input_dim = hidden_dim + 3 if i in skip_connections else hidden_dim
            self.layers.append(nn.Linear(input_dim, hidden_dim))

        # Output layer
        self.sdf_head = nn.Linear(hidden_dim, 1)

        # Initialize weights for sphere SDF
        if use_sphere_init:
            self._init_sphere_weights()

    def _init_sphere_weights(self):
        """Initialize weights to approximate a sphere SDF."""
        with torch.no_grad():
            # Initialize first layer to compute radius
            self.layers[0].weight.data[0, :3] = 1.0  # Compute radius
            self.layers[0].bias.data[0] = 0.0

            # Initialize output layer to compute sphere SDF
            self.sdf_head.weight.data[0, 0] = 1.0
            self.sdf_head.bias.data[0] = -self.sphere_radius

    def forward(self, coords: Tensor) -> TensorDict:
        """Forward pass with gradient computation."""
        # Enable gradient computation for SDF
        coords.requires_grad_(True)

        # Initial layer
        x = F.relu(self.layers[0](coords))

        # Hidden layers with skip connections
        for i, layer in enumerate(self.layers[1:], 1):
            if i in self.skip_connections:
                x = torch.cat([x, coords], dim=-1)
            x = F.relu(layer(x))

        # SDF prediction
        sdf = self.sdf_head(x)

        # Compute gradients
        gradients = torch.autograd.grad(
            sdf, coords, grad_outputs=torch.ones_like(sdf), create_graph=True
        )[0]

        return {"sdf": sdf, "gradients": gradients}


class ColorNetwork(nn.Module):
    """Network for predicting colors based on position, normals, and view direction."""

    def __init__(self, hidden_dim: int, num_layers: int, skip_connections: List[int]):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.skip_connections = skip_connections

        # Input is position (3) + normal (3) + view direction (3)
        input_dim = 9

        # Initialize network layers
        self.layers = nn.ModuleList()

        # First layer
        self.layers.append(nn.Linear(input_dim, hidden_dim))

        # Hidden layers
        for i in range(num_layers - 1):
            input_dim = hidden_dim + 9 if i in skip_connections else hidden_dim
            self.layers.append(nn.Linear(input_dim, hidden_dim))

        # Output layer
        self.rgb_head = nn.Linear(hidden_dim, 3)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the network."""
        input_features = x

        # Initial layer
        x = F.relu(self.layers[0](x))

        # Hidden layers with skip connections
        for i, layer in enumerate(self.layers[1:], 1):
            if i in self.skip_connections:
                x = torch.cat([x, input_features], dim=-1)
            x = F.relu(layer(x))

        # RGB prediction
        rgb = torch.sigmoid(self.rgb_head(x))

        return rgb


class SDFNet(nn.Module):
    """SDF-Net model with modern optimizations."""

    def __init__(self, config: SDFNetConfig):
        super().__init__()
        self.config = config

        # Initialize SDF network
        self.sdf_network = SDFNetwork(
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            skip_connections=config.skip_connections,
            use_sphere_init=config.use_sphere_init,
            sphere_radius=config.sphere_radius,
        )

        # Initialize color network
        self.color_network = ColorNetwork(
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers // 2,
            skip_connections=config.skip_connections,
        )

        # Initialize AMP grad scaler
        self.grad_scaler = config.grad_scaler if config.use_amp else None

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
                # Get SDF values and gradients
                sdf_output = self.sdf_network(chunk_coords)
                sdf = sdf_output["sdf"]
                sdf_gradients = sdf_output["gradients"]

                # Get colors
                if chunk_view_dirs is not None:
                    color_input = torch.cat([chunk_coords, sdf_gradients, chunk_view_dirs], dim=-1)
                    rgb = self.color_network(color_input)
                else:
                    rgb = torch.zeros_like(chunk_coords)

                chunk_outputs = {"sdf": sdf, "gradients": sdf_gradients, "rgb": rgb}

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

        # SDF loss
        if "sdf" in targets:
            losses["sdf"] = F.mse_loss(predictions["sdf"], targets["sdf"])

        # Gradient loss
        if "gradients" in targets:
            losses["gradients"] = F.mse_loss(predictions["gradients"], targets["gradients"])

        # RGB loss
        if "rgb" in targets:
            losses["rgb"] = F.mse_loss(predictions["rgb"], targets["rgb"])

        # Eikonal loss (gradient norm should be 1)
        if "gradients" in predictions:
            gradient_norm = torch.norm(predictions["gradients"], dim=-1)
            losses["eikonal"] = F.mse_loss(gradient_norm, torch.ones_like(gradient_norm))

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


class LatentSDFNetwork(nn.Module):
    """潜在SDF网络

    包含形状编码器和SDF解码器的完整模型
    """

    def __init__(self, dim_latent: int = 256, num_shapes: Optional[int] = None, **sdf_kwargs):
        super().__init__()

        self.dim_latent = dim_latent
        self.num_shapes = num_shapes

        # SDF解码器
        self.sdf_decoder = SDFNetwork(dim_latent=dim_latent, **sdf_kwargs)

        # 潜在编码（如果指定了形状数量）
        if num_shapes is not None:
            self.latent_codes = nn.Embedding(num_shapes, dim_latent)
            # 初始化潜在编码
            nn.init.normal_(self.latent_codes.weight, 0.0, 1e-4)

    def forward(
        self,
        points: torch.Tensor,
        shape_ids: Optional[torch.Tensor] = None,
        latent_code: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """前向传播

        Args:
            points: 输入点 [B, N, 3]
            shape_ids: 形状ID [B] (可选)
            latent_code: 直接提供的潜在编码 [B, dim_latent] (可选)

        Returns:
            sdf: 有符号距离值 [B, N, 1]
        """
        # 获取潜在编码
        if latent_code is None:
            if shape_ids is None:
                raise ValueError("Either shape_ids or latent_code must be provided")
            latent_code = self.latent_codes(shape_ids)

        return self.sdf_decoder(points, latent_code)

    def encode_shape(self, points: torch.Tensor, sdf: torch.Tensor) -> torch.Tensor:
        """编码形状（简单版本，实际应用中可能需要更复杂的编码器）

        Args:
            points: 输入点 [B, N, 3]
            sdf: SDF值 [B, N, 1]

        Returns:
            latent_code: 潜在编码 [B, dim_latent]
        """
        # 简单的平均池化编码
        features = torch.cat([points, sdf], dim=-1)  # [B, N, 4]
        shape_code = features.mean(dim=1)  # [B, 4]

        # 通过全连接层映射到潜在空间
        if not hasattr(self, "shape_encoder"):
            self.shape_encoder = nn.Linear(4, self.dim_latent)
            if torch.cuda.is_available():
                self.shape_encoder = self.shape_encoder.cuda()

        latent_code = self.shape_encoder(shape_code)
        return latent_code

    def get_latent_code(self, shape_id: int) -> torch.Tensor:
        """获取指定形状的潜在编码"""
        if self.num_shapes is None:
            raise ValueError("num_shapes must be specified to use shape IDs")

        shape_id_tensor = torch.tensor([shape_id], dtype=torch.long)
        if torch.cuda.is_available():
            shape_id_tensor = shape_id_tensor.cuda()

        return self.latent_codes(shape_id_tensor)


class MultiScaleSDFNetwork(SDFNetwork):
    """多尺度SDF网络

    在不同分辨率下进行SDF预测
    """

    def __init__(self, scales: List[float] = [1.0, 0.5, 0.25], **kwargs):
        super().__init__(**kwargs)
        self.scales = scales

    def forward_multiscale(
        self, points: torch.Tensor, latent_code: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """多尺度前向传播

        Args:
            points: 输入点 [B, N, 3]
            latent_code: 潜在编码 [B, dim_latent]

        Returns:
            multi_sdf: 不同尺度的SDF预测
        """
        results = {}

        for scale in self.scales:
            scaled_points = points * scale
            sdf = self.forward(scaled_points, latent_code)
            results[f"scale_{scale}"] = sdf

        return results
