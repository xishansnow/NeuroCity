"""
SVRaster Core Module

This module implements the core components of SVRaster:
- Sparse voxel grids for density and color representation
- Spherical harmonics for view-dependent appearance
- Trilinear interpolation for smooth sampling
- Volume rendering without neural networks

Based on the paper: "SVRaster: Sparse Voxel Radiance Fields"
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
import numpy as np
from dataclasses import dataclass, field
import logging
import os
from typing import Optional, Union, Dict, List, Tuple
from .utils.rendering_utils import ray_direction_dependent_ordering
import math
from contextlib import nullcontext
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class SVRasterConfig:
    """SVRaster configuration.

    Attributes:
        # Scene representation
        max_octree_levels: Maximum octree levels
        base_resolution: Base grid resolution
        scene_bounds: Scene bounds (min_x, min_y, min_z, max_x, max_y, max_z)

        # Voxel properties
        density_activation: Density activation function
        color_activation: Color activation function
        sh_degree: Spherical harmonics degree

        # Training
        learning_rate: Learning rate
        weight_decay: Weight decay
        gradient_clip_val: Gradient clipping value
        use_ema: Whether to use EMA model
        ema_decay: EMA decay rate
        steps_per_epoch: Number of steps per epoch
        num_epochs: Number of epochs
        save_best: Whether to save best model

        # Dataset
        image_width: Image width
        image_height: Image height
        camera_angle_x: Camera horizontal FoV in radians
        data_dir: Data directory
        num_rays_train: Number of rays per training batch
        downscale_factor: Image downscale factor for training

        # Rendering
        ray_samples_per_voxel: Number of samples per voxel along ray
        depth_peeling_layers: Number of depth peeling layers
        morton_ordering: Whether to use Morton ordering for depth sorting
        background_color: Background color (RGB)
        near_plane: Near plane distance
        far_plane: Far plane distance
        use_view_dependent_color: Whether to use view-dependent color
        use_opacity_regularization: Whether to use opacity regularization
        opacity_reg_weight: Opacity regularization weight
        use_ssim_loss: Whether to use SSIM loss
        ssim_loss_weight: SSIM loss weight
        use_distortion_loss: Whether to use distortion loss
        distortion_loss_weight: Distortion loss weight
        use_pointwise_rgb_loss: Whether to use pointwise RGB loss
        pointwise_rgb_loss_weight: Pointwise RGB loss weight

        # New attributes for modern optimizations
        volume_size: Tuple[int, int, int] = (128, 128, 128)
        feature_dim: int = 32
        num_planes: int = 3
        plane_channels: int = 16
        hidden_dim: int = 64
        num_layers: int = 4
        skip_connections: List[int] = (2,)
        activation: str = "relu"
        output_activation: str = "sigmoid"
        num_samples: int = 128
        num_importance_samples: int = 64
        learning_rate_decay_steps: int = 20000
        learning_rate_decay_mult: float = 0.1
        use_amp: bool = True
        grad_scaler_init_scale: float = 65536.0
        grad_scaler_growth_factor: float = 2.0
        grad_scaler_backoff_factor: float = 0.5
        grad_scaler_growth_interval: int = 2000
        use_non_blocking: bool = True
        set_grad_to_none: bool = True
        chunk_size: int = 8192
        cache_volume_samples: bool = True
        default_device: str = "cuda"
        pin_memory: bool = True
        batch_size: int = 4096
        num_workers: int = 4
        checkpoint_dir: str = "checkpoints"
        save_every_n_steps: int = 1000
        keep_last_n_checkpoints: int = 5
    """

    # Scene representation
    max_octree_levels: int = 16
    base_resolution: int = 64
    scene_bounds: Tuple[float, float, float, float, float, float] = (
        -1.0,
        -1.0,
        -1.0,
        1.0,
        1.0,
        1.0,
    )

    # Voxel properties
    density_activation: str = "exp"
    color_activation: str = "sigmoid"
    sh_degree: int = 2

    # Training
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    gradient_clip_val: float = 1.0
    use_ema: bool = True
    ema_decay: float = 0.999
    steps_per_epoch: int = 1000
    num_epochs: int = 100
    save_best: bool = True

    # Dataset
    image_width: int = 800
    image_height: int = 800
    camera_angle_x: float = 0.6911112070083618  # ~40 degrees
    data_dir: str = "data/nerf_synthetic/lego"
    num_rays_train: int = 4096
    downscale_factor: float = 1.0

    # Rendering
    ray_samples_per_voxel: int = 8
    depth_peeling_layers: int = 4
    morton_ordering: bool = True
    background_color: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    near_plane: float = 0.1
    far_plane: float = 100.0
    use_view_dependent_color: bool = True
    use_opacity_regularization: bool = True
    opacity_reg_weight: float = 0.01
    use_ssim_loss: bool = True
    ssim_loss_weight: float = 0.1
    use_distortion_loss: bool = True
    distortion_loss_weight: float = 0.01
    use_pointwise_rgb_loss: bool = True
    pointwise_rgb_loss_weight: float = 1.0

    # New attributes for modern optimizations
    volume_size: Tuple[int, int, int] = (128, 128, 128)
    feature_dim: int = 32
    num_planes: int = 3
    plane_channels: int = 16
    hidden_dim: int = 64
    num_layers: int = 4
    skip_connections: List[int] = field(default_factory=lambda: [2])
    activation: str = "relu"
    output_activation: str = "sigmoid"
    num_samples: int = 128
    num_importance_samples: int = 64
    learning_rate_decay_steps: int = 20000
    learning_rate_decay_mult: float = 0.1
    use_amp: bool = True
    grad_scaler_init_scale: float = 65536.0
    grad_scaler_growth_factor: float = 2.0
    grad_scaler_backoff_factor: float = 0.5
    grad_scaler_growth_interval: int = 2000
    use_non_blocking: bool = True
    set_grad_to_none: bool = True
    chunk_size: int = 8192
    cache_volume_samples: bool = True
    default_device: str = "cuda"
    pin_memory: bool = True
    batch_size: int = 4096
    num_workers: int = 4
    checkpoint_dir: str = "checkpoints"
    save_every_n_steps: int = 1000
    keep_last_n_checkpoints: int = 5
    pruning_threshold: float = 0.01  # Add missing pruning_threshold

    def __post_init__(self):
        """Post-initialization validation and initialization."""
        # Validate volume settings
        assert len(self.volume_size) == 3, "Volume size must be a 3-tuple"
        assert all(s > 0 for s in self.volume_size), "Volume dimensions must be positive"

        # Initialize device
        self.device = torch.device(self.default_device if torch.cuda.is_available() else "cpu")

        # Initialize grad scaler for AMP
        if self.use_amp:
            self.grad_scaler = GradScaler(
                device="cuda" if torch.cuda.is_available() else "cpu",
                init_scale=self.grad_scaler_init_scale,
                growth_factor=self.grad_scaler_growth_factor,
                backoff_factor=self.grad_scaler_backoff_factor,
                growth_interval=self.grad_scaler_growth_interval,
            )

        # Create checkpoint directory - store as Path but convert for property access
        self._checkpoint_path = Path(self.checkpoint_dir)
        self._checkpoint_path.mkdir(parents=True, exist_ok=True)

    @property
    def checkpoint_path(self) -> Path:
        """Get checkpoint directory as Path object."""
        return self._checkpoint_path


class AdaptiveSparseVoxels:
    """
    自适应稀疏体素表示类

    这个类实现了基于八叉树的多分辨率自适应稀疏体素表示，是 SVRaster 的核心组件。
    它管理不同八叉树层级的稀疏体素，只存储叶子节点，不存储完整的八叉树结构。

    主要功能：
    1. 初始化基础体素网格
    2. 根据梯度或密度进行自适应细分
    3. 基于密度阈值进行体素剪枝
    4. 计算 Morton 码用于深度排序
    5. 管理多层级体素的参量值
    """

    def __init__(self, config: SVRasterConfig):
        self.config = config

        # Initialize voxel storage as optimizable tensors
        self.voxel_positions = []  # Voxel center positions
        self.voxel_sizes = []  # Voxel sizes (level-dependent)
        self.voxel_densities = []  # Density values
        self.voxel_colors = []  # Color/SH coefficients
        self.voxel_levels = []  # Octree levels
        self.voxel_morton_codes = []  # Morton codes for sorting

        # Scene bounds
        self.scene_min = torch.tensor(config.scene_bounds[:3])
        self.scene_max = torch.tensor(config.scene_bounds[3:])
        self.scene_size = self.scene_max - self.scene_min

        # SH coefficients count
        self.num_sh_coeffs = (config.sh_degree + 1) ** 2

        # Initialize with base level voxels
        self._initialize_base_voxels()

    def parameters(self) -> List[torch.Tensor]:
        """Get all optimizable parameters."""
        params = []
        for level_idx in range(len(self.voxel_densities)):
            params.append(self.voxel_densities[level_idx])
            params.append(self.voxel_colors[level_idx])
        return params

    def _initialize_base_voxels(self):
        """初始化基础体素网格"""
        base_res = self.config.base_resolution

        # Create regular grid
        x = torch.linspace(self.scene_min[0], self.scene_max[0], base_res)
        y = torch.linspace(self.scene_min[1], self.scene_max[1], base_res)
        z = torch.linspace(self.scene_min[2], self.scene_max[2], base_res)
        grid_x, grid_y, grid_z = torch.meshgrid(x, y, z, indexing="ij")

        # Initialize positions
        positions = torch.stack([grid_x, grid_y, grid_z], dim=-1).reshape(-1, 3)
        self.voxel_positions.append(positions)

        # Initialize sizes
        voxel_size = (self.scene_max - self.scene_min) / base_res
        # Create a tensor of the same shape as positions but filled with voxel size components
        sizes = torch.ones_like(positions) * voxel_size.view(1, 3)
        self.voxel_sizes.append(sizes)

        # Initialize densities and colors with gradients
        densities = torch.zeros(positions.shape[0], requires_grad=True)
        colors = torch.zeros(positions.shape[0], 3, self.num_sh_coeffs, requires_grad=True)

        self.voxel_densities.append(densities)
        self.voxel_colors.append(colors)

        # Initialize levels and Morton codes
        self.voxel_levels.append(torch.zeros(positions.shape[0], dtype=torch.long))
        self.voxel_morton_codes.append(self._compute_morton_codes(positions, 0))

    def _compute_morton_codes(self, positions: torch.Tensor, level: int) -> torch.Tensor:
        """
        计算体素位置的 Morton 码

        Morton 码是一种空间填充曲线，用于将 3D 坐标编码为 1D 索引，
        便于进行深度排序和空间局部性优化。

        Args:
            positions: 体素中心位置 [N, 3]
            level: 八叉树层级

        Returns:
            Morton 码 [N]
        """
        # Normalize positions to [0, 1]
        scene_min_tensor = torch.as_tensor(self.scene_min)
        normalized_pos = (positions - scene_min_tensor) / self.scene_size

        # Discretize to grid coordinates
        grid_res = self.config.base_resolution * (2**level)
        grid_coords = (normalized_pos * grid_res).long()
        grid_coords = torch.clamp(grid_coords, 0, grid_res - 1)

        # Compute Morton codes using vectorized operations for better performance
        morton_codes = self._morton_encode_3d_vectorized(grid_coords)

        return morton_codes

    def _morton_encode_3d_vectorized(self, coords: torch.Tensor) -> torch.Tensor:
        """
        向量化的 3D Morton 码编码

        使用 PyTorch 张量操作进行批量 Morton 码计算，提高性能。

        Args:
            coords: 坐标张量 [N, 3]

        Returns:
            Morton 码张量 [N]
        """
        x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]

        # 确保坐标在合理范围内
        max_coord = 0x1FFFFF  # 21 位最大值
        x = torch.clamp(x, 0, max_coord)
        y = torch.clamp(y, 0, max_coord)
        z = torch.clamp(z, 0, max_coord)

        # 位交错函数
        def part1by2_vectorized(n):
            n = n & 0x1FFFFF
            n = (n ^ (n << 32)) & 0x1F00000000FFFF
            n = (n ^ (n << 16)) & 0x1F0000FF0000FF
            n = (n ^ (n << 8)) & 0x100F00F00F00F00F
            n = (n ^ (n << 4)) & 0x10C30C30C30C30C3
            n = (n ^ (n << 2)) & 0x1249249249249249
            return n

        # 计算 Morton 码
        morton_x = part1by2_vectorized(x)
        morton_y = part1by2_vectorized(y)
        morton_z = part1by2_vectorized(z)

        return (morton_z << 2) + (morton_y << 1) + morton_x

    def _morton_encode_3d(self, x: int, y: int, z: int) -> int:
        """
        将单个 3D 坐标编码为 Morton 码（兼容性方法）

        使用位交错技术将 3 个坐标分量编码为单个整数，
        保持空间局部性。支持更高的分辨率。

        Args:
            x, y, z: 3D 坐标分量

        Returns:
            Morton 码
        """

        def part1by2(n):
            # 支持更大的坐标值，使用更高效的位交错
            # 将 21 位输入扩展为 63 位输出（3 * 21 = 63）
            n &= 0x1FFFFF  # 21 位掩码，支持 0-2097151
            n = (n ^ (n << 32)) & 0x1F00000000FFFF
            n = (n ^ (n << 16)) & 0x1F0000FF0000FF
            n = (n ^ (n << 8)) & 0x100F00F00F00F00F
            n = (n ^ (n << 4)) & 0x10C30C30C30C30C3
            n = (n ^ (n << 2)) & 0x1249249249249249
            return n

        # 确保输入在合理范围内
        x = max(0, min(x, 0x1FFFFF))
        y = max(0, min(y, 0x1FFFFF))
        z = max(0, min(z, 0x1FFFFF))

        return (part1by2(z) << 2) + (part1by2(y) << 1) + part1by2(x)

    def subdivide_voxels(self, subdivision_mask: torch.Tensor, level_idx: int):
        """
        根据细分掩码细分体素

        将选中的体素细分为 8 个子体素，每个子体素的大小是父体素的一半。
        细分后的子体素被添加到下一层级，父体素被移除。

        Args:
            subdivision_mask: 细分掩码，True 表示需要细分的体素
            level_idx: 当前层级索引
        """
        if not subdivision_mask.any():
            return

        # Get voxels to subdivide
        parent_positions = self.voxel_positions[level_idx][subdivision_mask]
        parent_sizes = self.voxel_sizes[level_idx][subdivision_mask]
        parent_densities = self.voxel_densities[level_idx][subdivision_mask]
        parent_colors = self.voxel_colors[level_idx][subdivision_mask]

        # Create 8 child voxels for each parent
        num_parents = parent_positions.shape[0]
        child_size = parent_sizes / 2

        # Child offsets (8 corners of cube)
        offsets = (
            torch.tensor(
                [
                    [-1, -1, -1],
                    [1, -1, -1],
                    [-1, 1, -1],
                    [1, 1, -1],
                    [-1, -1, 1],
                    [1, -1, 1],
                    [-1, 1, 1],
                    [1, 1, 1],
                ],
                dtype=torch.float32,
                device=parent_positions.device,
            )
            * 0.25
        )

        # Generate child positions
        child_positions = []
        child_sizes_list = []
        child_densities_list = []
        child_colors_list = []

        for i in range(num_parents):
            parent_pos = parent_positions[i]
            size = child_size[i]

            for offset in offsets:
                child_pos = parent_pos + offset * parent_sizes[i]
                child_positions.append(child_pos)
                child_sizes_list.append(size)
                child_densities_list.append(parent_densities[i])
                child_colors_list.append(parent_colors[i])

        # Convert to tensors
        child_positions = torch.stack(child_positions)
        child_sizes_tensor = torch.stack(child_sizes_list)
        child_densities_tensor = torch.stack(child_densities_list)
        child_colors_tensor = torch.stack(child_colors_list)

        # Add to next level
        new_level = level_idx + 1
        if new_level >= len(self.voxel_positions):
            # Create new level
            self.voxel_positions.append(child_positions)
            self.voxel_sizes.append(child_sizes_tensor)
            self.voxel_densities.append(child_densities_tensor)
            self.voxel_colors.append(child_colors_tensor)
            self.voxel_levels.append(torch.full((child_positions.shape[0],), new_level))
            self.voxel_morton_codes.append(self._compute_morton_codes(child_positions, new_level))
        else:
            # Append to existing level
            old_positions = self.voxel_positions[new_level].data
            old_sizes = self.voxel_sizes[new_level].data
            old_densities = self.voxel_densities[new_level].data
            old_colors = self.voxel_colors[new_level].data

            self.voxel_positions[new_level].data = torch.cat([old_positions, child_positions])
            self.voxel_sizes[new_level].data = torch.cat([old_sizes, child_sizes_tensor])
            self.voxel_densities[new_level].data = torch.cat(
                [old_densities, child_densities_tensor],
            )
            self.voxel_colors[new_level].data = torch.cat([old_colors, child_colors_tensor])

            # Update levels and Morton codes
            new_levels = torch.full((child_positions.shape[0],), new_level)
            self.voxel_levels[new_level] = torch.cat([self.voxel_levels[new_level], new_levels])
            new_morton = self._compute_morton_codes(child_positions, new_level)
            self.voxel_morton_codes[new_level] = torch.cat(
                [self.voxel_morton_codes[new_level], new_morton],
            )

        # Remove subdivided parent voxels
        keep_mask = ~subdivision_mask
        self.voxel_positions[level_idx].data = self.voxel_positions[level_idx].data[keep_mask]
        self.voxel_sizes[level_idx].data = self.voxel_sizes[level_idx].data[keep_mask]
        self.voxel_densities[level_idx].data = self.voxel_densities[level_idx].data[keep_mask]
        self.voxel_colors[level_idx].data = self.voxel_colors[level_idx].data[keep_mask]
        self.voxel_levels[level_idx] = self.voxel_levels[level_idx][keep_mask]
        self.voxel_morton_codes[level_idx] = self.voxel_morton_codes[level_idx][keep_mask]

    def prune_voxels(self, threshold: Optional[float] = None):
        """
        移除低密度体素

        根据密度阈值移除不重要的体素，减少内存使用并提高渲染效率。
        密度低于阈值的体素被认为是透明的，对最终渲染结果贡献很小。

        Args:
            threshold: 密度阈值，None 时使用配置中的默认值
        """
        if threshold is None:
            threshold = self.config.pruning_threshold

        for level_idx in range(len(self.voxel_densities)):
            if self.config.density_activation == "exp":
                densities = torch.exp(self.voxel_densities[level_idx])
            else:
                densities = torch.relu(self.voxel_densities[level_idx])

            keep_mask = densities > threshold

            if keep_mask.any() and not keep_mask.all():
                self.voxel_positions[level_idx].data = self.voxel_positions[level_idx].data[
                    keep_mask
                ]
                self.voxel_sizes[level_idx].data = self.voxel_sizes[level_idx].data[keep_mask]
                self.voxel_densities[level_idx].data = self.voxel_densities[level_idx].data[
                    keep_mask
                ]
                self.voxel_colors[level_idx].data = self.voxel_colors[level_idx].data[keep_mask]
                self.voxel_levels[level_idx] = self.voxel_levels[level_idx][keep_mask]
                self.voxel_morton_codes[level_idx] = self.voxel_morton_codes[level_idx][keep_mask]

    def get_all_voxels(self) -> Dict[str, torch.Tensor]:
        """
        获取所有层级的体素数据

        将所有层级的体素数据合并为一个字典，包含位置、大小、密度、
        颜色、层级和 Morton 码。

        Returns:
            包含所有体素数据的字典
        """
        all_positions = []
        all_sizes = []
        all_densities = []
        all_colors = []
        all_levels = []
        all_morton_codes = []

        for level_idx in range(len(self.voxel_positions)):
            all_positions.append(self.voxel_positions[level_idx].data)
            all_sizes.append(self.voxel_sizes[level_idx].data)
            all_densities.append(self.voxel_densities[level_idx].data)
            all_colors.append(self.voxel_colors[level_idx].data)
            all_levels.append(self.voxel_levels[level_idx])
            all_morton_codes.append(self.voxel_morton_codes[level_idx])

        return {
            "positions": torch.cat(all_positions),
            "sizes": torch.cat(all_sizes),
            "densities": torch.cat(all_densities),
            "colors": torch.cat(all_colors),
            "levels": torch.cat(all_levels),
            "morton_codes": torch.cat(all_morton_codes),
        }

    def get_total_voxel_count(self) -> int:
        """
        获取所有层级的体素总数

        Returns:
            体素总数
        """
        return sum(pos.shape[0] for pos in self.voxel_positions)

    def add_voxels(self, positions, sizes, densities, colors, level=0):
        """
        向指定层级添加体素。
        Args:
            positions: [N, 3] 体素中心
            sizes: [N] 体素尺寸
            densities: [N] 体素密度
            colors: [N, 3 * (sh_degree+1)^2] 体素颜色/SH系数
            level: int, 层级索引
        """
        self.voxel_positions[level].data = torch.cat([self.voxel_positions[level].data, positions])
        self.voxel_sizes[level].data = torch.cat([self.voxel_sizes[level].data, sizes])
        self.voxel_densities[level].data = torch.cat([self.voxel_densities[level].data, densities])
        self.voxel_colors[level].data = torch.cat([self.voxel_colors[level].data, colors])
        self.voxel_levels[level] = torch.cat(
            [self.voxel_levels[level], torch.full((positions.shape[0],), level)]
        )
        self.voxel_morton_codes[level] = self._compute_morton_codes(positions, level)


class SVRasterModel(nn.Module):
    """SVRaster model with modern PyTorch features.

    Features:
    - Efficient sparse voxel representation
    - Automatic mixed precision training
    - Memory-optimized operations
    - CUDA acceleration with CPU fallback
    - Real-time rendering capabilities
    """

    def __init__(self, config: SVRasterConfig):
        super().__init__()
        self.config = config

        # Initialize components - 延迟导入避免循环依赖
        from .volume_renderer import VolumeRenderer
        
        self.voxels = AdaptiveSparseVoxels(config)
        self.volume_renderer = VolumeRenderer(config)  # 用于训练
        
        # 注册体素参数为模型参数
        self._register_voxel_parameters()
        
        # 懒加载真正的光栅化器（用于推理）
        self._true_rasterizer = None

        # Initialize AMP scaler
        self.scaler = GradScaler()

        # Move model to device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
    
    def _register_voxel_parameters(self):
        """注册体素参数为模型参数"""
        # 获取体素参数并注册为 nn.Parameter
        voxel_params = self.voxels.parameters()
        for i, param in enumerate(voxel_params):
            self.register_parameter(f'voxel_param_{i}', nn.Parameter(param))
    
    
    @property
    def true_rasterizer(self):
        """懒加载真正的光栅化器"""
        if self._true_rasterizer is None:
            from .true_rasterizer import TrueVoxelRasterizer
            self._true_rasterizer = TrueVoxelRasterizer(self.config)
        return self._true_rasterizer

    def forward(
        self,
        ray_origins: torch.Tensor,
        ray_directions: torch.Tensor,
        camera_params: Optional[Dict[str, torch.Tensor]] = None,
        mode: str = "training",
    ) -> Dict[str, torch.Tensor]:
        """Forward pass with modern optimizations.

        Args:
            ray_origins: Ray origins [N, 3]
            ray_directions: Ray directions [N, 3]
            camera_params: Optional camera parameters
            mode: Rendering mode - "training" for volume rendering, "inference" for rasterization

        Returns:
            Dictionary containing rendered outputs
        """
        # Move inputs to device efficiently
        ray_origins = ray_origins.to(self.device, non_blocking=True)
        ray_directions = ray_directions.to(self.device, non_blocking=True)
        if camera_params is not None:
            camera_params = {
                k: v.to(self.device, non_blocking=True) for k, v in camera_params.items()
            }

        # Get current voxel representation
        device_type = "cuda" if torch.cuda.is_available() else "cpu"
        with autocast(device_type=device_type):
            voxels = self.voxels.get_all_voxels()

            if mode == "training":
                # 使用体积渲染（用于训练）
                outputs = self.volume_renderer(
                    voxels,
                    ray_origins,
                    ray_directions,
                    camera_params,
                )
            elif mode == "inference":
                # 使用光栅化渲染（用于推理）
                from .true_rasterizer import rays_to_camera_matrix
                camera_matrix, intrinsics = rays_to_camera_matrix(ray_origins, ray_directions)
                
                # 推断视口尺寸
                viewport_size = (self.config.image_width, self.config.image_height)
                
                outputs = self.true_rasterizer(
                    voxels,
                    camera_matrix,
                    intrinsics,
                    viewport_size,
                )
            else:
                raise ValueError(f"Unknown rendering mode: {mode}")

        return outputs

    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
        optimizer: torch.optim.Optimizer,
    ) -> Dict[str, torch.Tensor]:
        """Perform a single training step with modern optimizations.

        Args:
            batch: Dictionary containing training data
            optimizer: PyTorch optimizer

        Returns:
            Dictionary containing loss values
        """
        # Move batch to device efficiently
        batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}

        # Forward pass with automatic mixed precision
        with autocast(device_type="cuda" if torch.cuda.is_available() else "cpu"):
            # Handle camera_params type correctly
            camera_params = batch.get("camera_params")
            if camera_params is not None and not isinstance(camera_params, dict):
                camera_params = None
            
            outputs = self.forward(
                batch["rays_o"],
                batch["rays_d"],
                camera_params,
            )

            # Compute losses
            loss_dict = {}

            # RGB loss
            if self.config.use_pointwise_rgb_loss:
                rgb_loss = F.mse_loss(outputs["rgb"], batch["rgb"])
                loss_dict["rgb_loss"] = rgb_loss * self.config.pointwise_rgb_loss_weight

            # SSIM loss
            if self.config.use_ssim_loss:
                ssim_loss = 1.0 - compute_ssim(outputs["rgb"], batch["rgb"])
                loss_dict["ssim_loss"] = ssim_loss * self.config.ssim_loss_weight

            # Distortion loss
            if self.config.use_distortion_loss and "weights" in outputs and "depth" in outputs:
                # Create temporary loss calculator
                loss_calc = SVRasterLoss(self.config)
                distortion_loss = loss_calc._compute_distortion_loss(outputs["weights"], outputs["depth"])
                loss_dict["distortion_loss"] = distortion_loss * self.config.distortion_loss_weight

            # Opacity regularization
            if self.config.use_opacity_regularization:
                opacity_reg = torch.mean(torch.abs(outputs.get("opacity", torch.tensor(0.0))))
                loss_dict["opacity_reg"] = opacity_reg * self.config.opacity_reg_weight

            # Total loss
            total_loss = sum(loss_dict.values())
            if not isinstance(total_loss, torch.Tensor):
                total_loss = torch.tensor(total_loss, device=self.device, requires_grad=True)
            loss_dict["total_loss"] = total_loss

        # Optimization step with gradient scaling
        optimizer.zero_grad(set_to_none=True)
        self.scaler.scale(total_loss).backward()
        self.scaler.step(optimizer)
        self.scaler.update()

        return loss_dict

    @torch.inference_mode()
    def evaluate(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Evaluate model with modern optimizations.

        Args:
            batch: Dictionary containing evaluation data

        Returns:
            Dictionary containing evaluation metrics
        """
        # Move batch to device efficiently
        batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}

        # Forward pass with automatic mixed precision
        with autocast(device_type="cuda" if torch.cuda.is_available() else "cpu"):
            # Handle camera_params type correctly
            camera_params = batch.get("camera_params")
            if camera_params is not None and not isinstance(camera_params, dict):
                camera_params = None
                
            outputs = self.forward(
                batch["rays_o"],
                batch["rays_d"],
                camera_params,
            )

            # Compute metrics
            metrics = {}

            # MSE and PSNR
            mse = F.mse_loss(outputs["rgb"], batch["rgb"])
            psnr = -10.0 * torch.log10(mse)

            # SSIM
            if self.config.use_ssim_loss:
                ssim = compute_ssim(outputs["rgb"], batch["rgb"])
                metrics["ssim"] = ssim.item()

            metrics.update(
                {
                    "mse": mse.item(),
                    "psnr": psnr.item(),
                }
            )

        return metrics

    def render_image(
        self,
        camera_pose: torch.Tensor,
        camera_intrinsics: torch.Tensor,
        image_size: Tuple[int, int],
        device: Optional[torch.device] = None,
    ) -> Dict[str, torch.Tensor]:
        """Render a full image efficiently.

        Args:
            camera_pose: Camera-to-world transform [4, 4]
            camera_intrinsics: Camera intrinsics [3, 3]
            image_size: Output image size (H, W)
            device: Optional device to render on

        Returns:
            Dictionary containing rendered image and auxiliary outputs
        """
        device = device or self.device

        # Move inputs to device efficiently
        camera_pose = camera_pose.to(device, non_blocking=True)
        camera_intrinsics = camera_intrinsics.to(device, non_blocking=True)

        # Generate rays
        H, W = image_size
        i, j = torch.meshgrid(
            torch.arange(W, device=device),
            torch.arange(H, device=device),
            indexing="xy",
        )

        # Camera coordinates
        dirs = torch.stack(
            [
                (i - camera_intrinsics[0, 2]) / camera_intrinsics[0, 0],
                -(j - camera_intrinsics[1, 2]) / camera_intrinsics[1, 1],
                -torch.ones_like(i, device=device),
            ],
            dim=-1,
        )

        # Transform to world coordinates
        rays_d = torch.sum(
            dirs.unsqueeze(-2) * camera_pose[:3, :3],
            dim=-1,
        )
        rays_o = camera_pose[:3, 3].expand_as(rays_d)

        # Render in chunks for memory efficiency
        chunk_size = 4096
        outputs_list = []

        for i in range(0, rays_o.shape[0], chunk_size):
            chunk_rays_o = rays_o[i : i + chunk_size]
            chunk_rays_d = rays_d[i : i + chunk_size]

            with autocast(device_type="cuda" if torch.cuda.is_available() else "cpu"):
                chunk_outputs = self.forward(chunk_rays_o, chunk_rays_d)
                outputs_list.append(chunk_outputs)

        # Combine chunk outputs efficiently
        outputs = {
            k: torch.cat([out[k] for out in outputs_list], dim=0) for k in outputs_list[0].keys()
        }

        # Reshape outputs to image dimensions
        for k, v in outputs.items():
            if v.dim() == 2:
                outputs[k] = v.reshape(H, W, v.shape[-1])
            else:
                outputs[k] = v.reshape(H, W)

        return outputs


class SVRasterLoss:
    """Modern loss functions for SVRaster.

    Features:
    - Multiple loss terms with configurable weights
    - SSIM loss for perceptual quality
    - Distortion loss for geometry regularization
    - Opacity regularization
    - Pointwise RGB loss
    """

    def __init__(self, config: SVRasterConfig):
        """Initialize loss functions.

        Args:
            config: Model configuration
        """
        self.config = config

    def __call__(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        model: Optional[SVRasterModel] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute all loss terms.

        Args:
            outputs: Model outputs
            targets: Ground truth targets
            model: Optional model for regularization

        Returns:
            Dictionary containing all loss terms and total loss
        """
        losses = {}

        # RGB loss (always used)
        if self.config.use_pointwise_rgb_loss:
            rgb_loss = F.mse_loss(outputs["rgb"], targets["rgb"])
            losses["rgb"] = rgb_loss * self.config.pointwise_rgb_loss_weight

        # SSIM loss
        if self.config.use_ssim_loss and "rgb" in outputs:
            ssim_loss = 1.0 - compute_ssim(outputs["rgb"], targets["rgb"])
            losses["ssim"] = ssim_loss * self.config.ssim_loss_weight

        # Distortion loss
        if self.config.use_distortion_loss and "weights" in outputs and "depth" in outputs:
            distortion_loss = self._compute_distortion_loss(
                outputs["weights"],
                outputs["depth"],
            )
            losses["distortion"] = distortion_loss * self.config.distortion_loss_weight

        # Opacity regularization
        if self.config.use_opacity_regularization and "opacity" in outputs:
            opacity_reg = outputs["opacity"].mean()
            losses["opacity_reg"] = opacity_reg * self.config.opacity_reg_weight

        # Compute total loss
        total_loss = sum(losses.values())
        losses["total_loss"] = total_loss

        return losses

    def _compute_distortion_loss(
        self,
        weights: torch.Tensor,
        depths: torch.Tensor,
    ) -> torch.Tensor:
        """Compute distortion loss for geometry regularization.

        Args:
            weights: Ray weights [..., N_samples]
            depths: Sample depths [..., N_samples]

        Returns:
            Distortion loss
        """
        with torch.inference_mode():
            # Sort depths
            depths_mid = 0.5 * (depths[..., 1:] + depths[..., :-1])
            weights_mid = 0.5 * (weights[..., 1:] + weights[..., :-1])

            # Compute loss
            loss = (
                weights_mid * torch.abs(torch.log(depths[..., 1:]) - torch.log(depths[..., :-1]))
            ).mean()

        return loss


def compute_ssim(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    """Compute SSIM between two images"""
    # Ensure inputs are 4D: [B, C, H, W]
    orig_shape = img1.shape

    # For small images, reduce window size
    window_size = min(window_size, min(orig_shape[-2:]) // 2)
    if window_size % 2 == 0:
        window_size -= 1

    if len(img1.shape) == 2:
        img1 = img1.unsqueeze(0).unsqueeze(0)
        img2 = img2.unsqueeze(0).unsqueeze(0)
    elif len(img1.shape) == 3:
        # If last dimension is 3 (RGB), transpose it
        if img1.shape[-1] == 3:
            img1 = img1.permute(2, 0, 1).unsqueeze(0)
            img2 = img2.permute(2, 0, 1).unsqueeze(0)
        else:
            img1 = img1.unsqueeze(0)
            img2 = img2.unsqueeze(0)
    elif len(img1.shape) == 4 and img1.shape[1] != 1 and img1.shape[1] != 3:
        # If channels not in correct position, transpose
        if img1.shape[-1] in [1, 3]:
            img1 = img1.permute(0, 3, 1, 2)
            img2 = img2.permute(0, 3, 1, 2)

    # Create a Gaussian kernel
    sigma = 1.5
    gauss = torch.exp(
        torch.arange(-(window_size // 2), window_size // 2 + 1).float() ** 2 / (-2 * sigma**2)
    )
    kernel = gauss.unsqueeze(0) * gauss.unsqueeze(1)
    kernel = kernel / kernel.sum()
    kernel = kernel.unsqueeze(0).unsqueeze(0).to(img1.device)

    # Pad images
    pad = window_size // 2
    img1 = F.pad(img1, (pad, pad, pad, pad), mode="constant", value=0)
    img2 = F.pad(img2, (pad, pad, pad, pad), mode="constant", value=0)

    # Compute means
    mu1 = F.conv2d(img1, kernel)
    mu2 = F.conv2d(img2, kernel)
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2

    # Compute variances and covariance
    sigma1_sq = F.conv2d(img1**2, kernel) - mu1_sq
    sigma2_sq = F.conv2d(img2**2, kernel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, kernel) - mu1_mu2

    # SSIM formula
    C1 = 0.01**2
    C2 = 0.03**2
    ssim = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )

    return ssim.mean()
