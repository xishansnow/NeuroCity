"""
Mega-NeRF Core Module

This module contains the core Mega-NeRF components including:
- Main model architecture
- Configuration classes
- Positional encoding
- Submodule management
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass

# 使用 Python 3.10+ 内置类型，无需导入 typing 模块
import logging

logger = logging.getLogger(__name__)


@dataclass
class MegaNeRFConfig:
    """Mega-NeRF 主配置类"""

    # 场景分解参数
    num_submodules: int = 8
    grid_size: tuple[int, int] = (4, 2)  # 2D网格分解
    overlap_factor: float = 0.15

    # 网络参数
    hidden_dim: int = 256
    num_layers: int = 8
    skip_connections: list[int] | None = None
    use_viewdirs: bool = True

    # 训练参数
    batch_size: int = 1024
    learning_rate: float = 5e-4
    lr_decay: float = 0.1
    max_iterations: int = 500000

    # 采样参数
    num_coarse: int = 256
    num_fine: int = 512
    near: float = 0.1
    far: float = 1000.0

    # 外观嵌入
    use_appearance_embedding: bool = True
    appearance_dim: int = 48

    # 边界参数
    scene_bounds: tuple[float, float, float, float, float, float] = (-100, -100, -10, 100, 100, 50)
    foreground_ratio: float = 0.8

    def __post_init__(self):
        """后初始化验证"""
        if self.skip_connections is None:
            self.skip_connections = [4]

        if self.num_submodules <= 0:
            raise ValueError("num_submodules must be positive")

        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")

        if self.overlap_factor < 0 or self.overlap_factor > 1:
            raise ValueError("overlap_factor must be between 0 and 1")


class PositionalEncoding(nn.Module):
    """位置编码模块"""

    def __init__(self, input_dim: int, max_freq_log2: int = 10, num_freqs: int = 10):
        super().__init__()
        self.input_dim = input_dim
        self.max_freq_log2 = max_freq_log2
        self.num_freqs = num_freqs

        self.freq_bands = 2.0 ** torch.linspace(0.0, max_freq_log2, num_freqs)
        self.output_dim = input_dim * (1 + 2 * num_freqs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        对输入进行位置编码

        Args:
            x: [..., input_dim] 输入张量

        Returns:
            encoded: [..., output_dim] 编码后的张量
        """
        encoded = [x]
        for freq in self.freq_bands:
            encoded.append(torch.sin(freq * x))
            encoded.append(torch.cos(freq * x))
        return torch.cat(encoded, dim=-1)


class MegaNeRFSubmodule(nn.Module):
    """Mega-NeRF 子模块，处理场景的一个空间区域"""

    def __init__(self, config: MegaNeRFConfig, centroid: np.ndarray):
        super().__init__()
        self.config = config
        self.centroid = torch.tensor(centroid, dtype=torch.float32)

        # 位置编码
        self.pos_encoder = PositionalEncoding(3, max_freq_log2=10, num_freqs=10)
        self.dir_encoder = PositionalEncoding(3, max_freq_log2=4, num_freqs=4)

        # 计算输入维度
        pos_dim = self.pos_encoder.output_dim
        input_dim = pos_dim

        if config.use_viewdirs:
            dir_dim = self.dir_encoder.output_dim
        else:
            dir_dim = 0

        # 外观嵌入
        if config.use_appearance_embedding:
            self.appearance_embedding = nn.Embedding(1000, config.appearance_dim)
            input_dim += config.appearance_dim

        # 主网络 - 预测密度和特征
        layers = []
        current_dim = input_dim

        for i in range(config.num_layers):
            if config.skip_connections is not None and i in config.skip_connections:
                layers.append(nn.Linear(current_dim + input_dim, config.hidden_dim))
            else:
                layers.append(nn.Linear(current_dim, config.hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            current_dim = config.hidden_dim

        self.main_layers = nn.ModuleList(layers)

        # 密度预测头
        self.density_head = nn.Linear(config.hidden_dim, 1)

        # 颜色预测网络
        if config.use_viewdirs:
            self.feature_head = nn.Linear(config.hidden_dim, config.hidden_dim)
            color_input_dim = config.hidden_dim + dir_dim
            self.color_layers = nn.Sequential(
                nn.Linear(color_input_dim, config.hidden_dim // 2),
                nn.ReLU(inplace=True),
                nn.Linear(config.hidden_dim // 2, 3),
                nn.Sigmoid(),
            )
        else:
            self.color_head = nn.Sequential(nn.Linear(config.hidden_dim, 3), nn.Sigmoid())

    def forward(
        self,
        points: torch.Tensor,
        viewdirs: torch.Tensor | None = None,
        appearance_idx: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播

        Args:
            points: [N, 3] 3D坐标
            viewdirs: [N, 3] 视角方向 (可选)
            appearance_idx: [N] 外观嵌入索引 (可选)

        Returns:
            density: [N, 1] 密度值
            color: [N, 3] RGB颜色
        """
        # 位置编码
        pos_encoded = self.pos_encoder(points)

        # 外观嵌入
        features = [pos_encoded]
        if self.config.use_appearance_embedding and appearance_idx is not None:
            app_embed = self.appearance_embedding(appearance_idx)
            features.append(app_embed)

        x = torch.cat(features, dim=-1)
        input_x = x

        # 主网络前向传播
        for i, layer in enumerate(self.main_layers):
            if (
                self.config.skip_connections is not None
                and i // 2 in self.config.skip_connections
                and i % 2 == 0
            ):
                x = torch.cat([x, input_x], dim=-1)
            x = layer(x)

        # 预测密度
        density = self.density_head(x)
        density = F.relu(density)

        # 预测颜色
        if self.config.use_viewdirs and viewdirs is not None:
            features = self.feature_head(x)
            dir_encoded = self.dir_encoder(viewdirs)
            color_input = torch.cat([features, dir_encoded], dim=-1)
            color = self.color_layers(color_input)
        else:
            color = self.color_head(x)

        return density, color

    def get_centroid(self) -> torch.Tensor:
        """获取子模块的中心点"""
        return self.centroid.clone()

    def set_centroid(self, centroid: torch.Tensor) -> None:
        """设置子模块的中心点"""
        self.centroid = centroid.clone()


class MegaNeRF(nn.Module):
    """Mega-NeRF 主模型"""

    def __init__(self, config: MegaNeRFConfig):
        super().__init__()
        self.config = config

        # 创建空间网格
        self._create_spatial_grid()

        # 创建子模块
        self.submodules = nn.ModuleList(
            [MegaNeRFSubmodule(config, centroid) for centroid in self.centroids]
        )

        # 计算前景边界
        self._compute_foreground_bounds()

        logger.info(f"MegaNeRF initialized with {len(self.submodules)} submodules")

    def _create_spatial_grid(self) -> None:
        """创建空间网格"""
        grid_rows, grid_cols = self.config.grid_size
        total_modules = grid_rows * grid_cols

        if total_modules != self.config.num_submodules:
            raise ValueError(
                f"Grid size {self.config.grid_size} must produce {self.config.num_submodules} modules"
            )

        # 计算网格边界
        x_min, y_min, z_min, x_max, y_max, z_max = self.config.scene_bounds

        # 创建网格点
        x_coords = torch.linspace(x_min, x_max, grid_cols + 1)
        y_coords = torch.linspace(y_min, y_max, grid_rows + 1)

        # 计算中心点
        self.centroids = []
        for i in range(grid_rows):
            for j in range(grid_cols):
                centroid = np.array(
                    [
                        (x_coords[j] + x_coords[j + 1]) / 2,
                        (y_coords[i] + y_coords[i + 1]) / 2,
                        (z_min + z_max) / 2,
                    ]
                )
                self.centroids.append(centroid)

    def _compute_foreground_bounds(self) -> None:
        """计算前景边界"""
        x_min, y_min, z_min, x_max, y_max, z_max = self.config.scene_bounds

        # 根据前景比例调整边界
        x_range = x_max - x_min
        y_range = y_max - y_min
        z_range = z_max - z_min

        margin = (1 - self.config.foreground_ratio) / 2

        self.foreground_bounds = [
            x_min + x_range * margin,
            y_min + y_range * margin,
            z_min + z_range * margin,
            x_max - x_range * margin,
            y_max - y_range * margin,
            z_max - z_range * margin,
        ]

    def _assign_points_to_submodules(self, points: torch.Tensor) -> list[list[int]]:
        """将点分配给相应的子模块"""
        assignments = [[] for _ in range(len(self.submodules))]

        for i, point in enumerate(points):
            # 计算到每个子模块中心的距离
            distances = []
            for centroid in self.centroids:
                dist = torch.norm(point - torch.tensor(centroid, device=point.device))
                distances.append(dist)

            # 选择最近的子模块
            closest_module = torch.argmin(torch.tensor(distances))
            assignments[closest_module].append(i)

        return assignments

    def _is_in_foreground(self, points: torch.Tensor) -> torch.Tensor:
        """检查点是否在前景区域内"""
        x_min, y_min, z_min, x_max, y_max, z_max = self.foreground_bounds

        in_bounds = (
            (points[:, 0] >= x_min)
            & (points[:, 0] <= x_max)
            & (points[:, 1] >= y_min)
            & (points[:, 1] <= y_max)
            & (points[:, 2] >= z_min)
            & (points[:, 2] <= z_max)
        )

        return in_bounds

    def forward(
        self,
        points: torch.Tensor,
        viewdirs: torch.Tensor | None = None,
        appearance_idx: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播

        Args:
            points: [N, 3] 3D坐标
            viewdirs: [N, 3] 视角方向 (可选)
            appearance_idx: [N] 外观嵌入索引 (可选)

        Returns:
            density: [N, 1] 密度值
            color: [N, 3] RGB颜色
        """
        # 检查前景区域
        foreground_mask = self._is_in_foreground(points)

        # 初始化输出
        batch_size = points.shape[0]
        device = points.device

        density = torch.zeros(batch_size, 1, device=device)
        color = torch.zeros(batch_size, 3, device=device)

        # 只处理前景点
        if not foreground_mask.any():
            return density, color

        foreground_points = points[foreground_mask]
        foreground_viewdirs = viewdirs[foreground_mask] if viewdirs is not None else None
        foreground_appearance = (
            appearance_idx[foreground_mask] if appearance_idx is not None else None
        )

        # 分配点到子模块
        assignments = self._assign_points_to_submodules(foreground_points)

        # 处理每个子模块
        for module_idx, point_indices in enumerate(assignments):
            if not point_indices:
                continue

            module_points = foreground_points[point_indices]
            module_viewdirs = (
                foreground_viewdirs[point_indices] if foreground_viewdirs is not None else None
            )
            module_appearance = (
                foreground_appearance[point_indices] if foreground_appearance is not None else None
            )

            # 子模块前向传播
            module_density, module_color = self.submodules[module_idx](
                module_points, module_viewdirs, module_appearance
            )

            # 将结果放回原位置
            original_indices = torch.where(foreground_mask)[0][point_indices]
            density[original_indices] = module_density
            color[original_indices] = module_color

        return density, color

    def get_submodule_centroids(self) -> list[np.ndarray]:
        """获取所有子模块的中心点"""
        return [centroid.copy() for centroid in self.centroids]

    def get_submodule_bounds(
        self, submodule_idx: int
    ) -> tuple[float, float, float, float, float, float]:
        """获取指定子模块的边界"""
        if submodule_idx >= len(self.submodules):
            raise ValueError(f"Invalid submodule index: {submodule_idx}")

        grid_rows, grid_cols = self.config.grid_size
        row = submodule_idx // grid_cols
        col = submodule_idx % grid_cols

        x_min, y_min, z_min, x_max, y_max, z_max = self.config.scene_bounds

        x_step = (x_max - x_min) / grid_cols
        y_step = (y_max - y_min) / grid_rows

        bounds = (
            x_min + col * x_step,
            y_min + row * y_step,
            z_min,
            x_min + (col + 1) * x_step,
            y_min + (row + 1) * y_step,
            z_max,
        )

        return bounds

    def get_relevant_submodules(
        self, camera_position: torch.Tensor, max_distance: float | None = None
    ) -> list[int]:
        """获取与相机位置相关的子模块"""
        if max_distance is None:
            max_distance = 100.0  # 默认最大距离

        relevant_modules = []

        for i, centroid in enumerate(self.centroids):
            centroid_tensor = torch.tensor(centroid, device=camera_position.device)
            distance = torch.norm(camera_position - centroid_tensor)

            if distance <= max_distance:
                relevant_modules.append(i)

        return relevant_modules

    def save_submodule(self, submodule_idx: int, path: str) -> None:
        """保存指定的子模块"""
        if submodule_idx >= len(self.submodules):
            raise ValueError(f"Invalid submodule index: {submodule_idx}")

        torch.save(
            {
                "state_dict": self.submodules[submodule_idx].state_dict(),
                "centroid": self.centroids[submodule_idx],
                "config": self.config,
            },
            path,
        )

        logger.info(f"Submodule {submodule_idx} saved to {path}")

    def load_submodule(self, submodule_idx: int, path: str) -> None:
        """加载指定的子模块"""
        if submodule_idx >= len(self.submodules):
            raise ValueError(f"Invalid submodule index: {submodule_idx}")

        checkpoint = torch.load(path, map_location="cpu")
        self.submodules[submodule_idx].load_state_dict(checkpoint["state_dict"])

        logger.info(f"Submodule {submodule_idx} loaded from {path}")

    def get_model_info(self) -> dict[str, object]:
        """获取模型信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            "num_submodules": len(self.submodules),
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "grid_size": self.config.grid_size,
            "scene_bounds": self.config.scene_bounds,
            "foreground_bounds": self.foreground_bounds,
        }
