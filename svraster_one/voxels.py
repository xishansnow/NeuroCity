"""
稀疏体素网格

实现可微分的稀疏体素表示，支持动态细分和剪枝。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class MortonCode:
    """Morton 编码工具类"""

    @staticmethod
    def encode_3d(
        x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, bits: int = 21
    ) -> torch.Tensor:
        """3D Morton 编码"""
        # 限制坐标范围
        mask = (1 << bits) - 1
        x = torch.clamp(x, 0, mask)
        y = torch.clamp(y, 0, mask)
        z = torch.clamp(z, 0, mask)

        # 位交错
        morton = torch.zeros_like(x, dtype=torch.int64)
        for i in range(bits):
            bit_pos = 1 << i
            morton |= (x & bit_pos) << (2 * i)
            morton |= (y & bit_pos) << (2 * i + 1)
            morton |= (z & bit_pos) << (2 * i + 2)

        return morton

    @staticmethod
    def decode_3d(
        morton: torch.Tensor, bits: int = 21
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """3D Morton 解码"""
        x = torch.zeros_like(morton, dtype=torch.int64)
        y = torch.zeros_like(morton, dtype=torch.int64)
        z = torch.zeros_like(morton, dtype=torch.int64)

        for i in range(bits):
            bit_pos = 1 << i
            x |= (morton >> (2 * i)) & bit_pos
            y |= (morton >> (2 * i + 1)) & bit_pos
            z |= (morton >> (2 * i + 2)) & bit_pos

        return x, y, z


class SparseVoxelGrid(nn.Module):
    """
    稀疏体素网格

    实现可微分的稀疏体素表示，支持：
    - 动态体素细分
    - 自适应剪枝
    - Morton 排序
    - 梯度传播
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # 体素参数
        self.max_voxels = config.voxel.max_voxels
        self.grid_resolution = config.voxel.grid_resolution
        self.voxel_size = config.voxel.voxel_size
        self.sparsity_threshold = config.voxel.sparsity_threshold

        # 初始化体素数据
        self._init_voxel_data()

        # 训练状态
        self.training_stats = {
            "total_voxels": 0,
            "active_voxels": 0,
            "subdivision_count": 0,
            "pruning_count": 0,
        }

    def _init_voxel_data(self):
        """初始化体素数据"""
        # 创建初始体素网格
        grid_coords = torch.meshgrid(
            torch.arange(self.grid_resolution),
            torch.arange(self.grid_resolution),
            torch.arange(self.grid_resolution),
            indexing="ij",
        )

        # 扁平化坐标
        coords = torch.stack(
            [grid_coords[0].flatten(), grid_coords[1].flatten(), grid_coords[2].flatten()], dim=1
        )

        # 随机选择初始体素
        num_initial_voxels = min(self.max_voxels // 4, coords.shape[0])
        indices = torch.randperm(coords.shape[0])[:num_initial_voxels]

        initial_coords = coords[indices]

        # 初始化体素参数
        self.register_parameter(
            "voxel_coords", nn.Parameter(initial_coords.float(), requires_grad=False)
        )

        # 体素特征（密度 + 颜色）
        feature_dim = 4  # [density, r, g, b]
        self.register_parameter(
            "voxel_features",
            nn.Parameter(torch.randn(num_initial_voxels, feature_dim) * 0.01, requires_grad=True),
        )

        # Morton 编码
        if self.config.voxel.use_morton_ordering:
            morton_codes = MortonCode.encode_3d(
                initial_coords[:, 0],
                initial_coords[:, 1],
                initial_coords[:, 2],
                self.config.voxel.morton_bits,
            )
            self.register_buffer("morton_codes", morton_codes)

        # 体素大小（支持各向异性）
        self.register_parameter(
            "voxel_sizes",
            nn.Parameter(torch.ones(num_initial_voxels) * self.voxel_size, requires_grad=True),
        )

        # 活跃状态掩码
        self.register_buffer("active_mask", torch.ones(num_initial_voxels, dtype=torch.bool))

        logger.info(f"Initialized sparse voxel grid with {num_initial_voxels} voxels")

    def forward(self, query_points: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        前向传播：查询体素特征

        Args:
            query_points: 查询点 [N, 3]

        Returns:
            体素特征字典
        """
        # 获取活跃体素
        active_coords = self.voxel_coords[self.active_mask]
        active_features = self.voxel_features[self.active_mask]
        active_sizes = self.voxel_sizes[self.active_mask]

        if active_coords.shape[0] == 0:
            # 没有活跃体素，返回零特征
            return {
                "densities": torch.zeros(query_points.shape[0], device=query_points.device),
                "colors": torch.zeros(query_points.shape[0], 3, device=query_points.device),
                "sizes": torch.zeros(query_points.shape[0], device=query_points.device),
            }

        # 计算查询点到体素中心的距离
        distances = torch.cdist(query_points, active_coords)

        # 软分配权重（可微分）
        if self.config.rendering.soft_rasterization:
            # 使用高斯核进行软分配
            sigma = self.config.rendering.sigma
            weights = torch.exp(-distances / (2 * sigma**2))
            weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)
        else:
            # 硬分配（最近邻）
            nearest_indices = torch.argmin(distances, dim=1)
            weights = torch.zeros_like(distances)
            weights[torch.arange(weights.shape[0]), nearest_indices] = 1.0

        # 插值体素特征
        interpolated_features = torch.matmul(weights, active_features)
        interpolated_sizes = torch.matmul(weights, active_sizes.unsqueeze(1)).squeeze(1)

        # 分离密度和颜色
        densities = interpolated_features[:, 0]
        colors = interpolated_features[:, 1:4]

        # 应用激活函数
        densities = F.softplus(densities)  # 确保密度为正
        colors = torch.sigmoid(colors)  # 确保颜色在 [0, 1]

        return {
            "densities": densities,
            "colors": colors,
            "sizes": interpolated_sizes,
            "weights": weights,  # 用于梯度传播
        }

    def adaptive_subdivision(self, gradient_magnitudes: torch.Tensor):
        """自适应体素细分"""
        if not self.config.voxel.adaptive_subdivision:
            return

        # 基于梯度幅度决定是否细分
        threshold = self.config.voxel.subdivision_threshold
        high_gradient_mask = gradient_magnitudes > threshold

        if not torch.any(high_gradient_mask):
            return

        # 获取需要细分的体素
        high_gradient_indices = torch.where(high_gradient_mask)[0]

        # 细分体素
        new_coords = []
        new_features = []
        new_sizes = []

        for idx in high_gradient_indices:
            if not self.active_mask[idx]:
                continue

            # 获取原体素信息
            old_coord = self.voxel_coords[idx]
            old_feature = self.voxel_features[idx]
            old_size = self.voxel_sizes[idx]

            # 创建8个子体素
            for i in range(8):
                # 计算子体素坐标
                offset = torch.tensor(
                    [(i >> 0) & 1, (i >> 1) & 1, (i >> 2) & 1],
                    dtype=torch.float32,
                    device=old_coord.device,
                )

                new_coord = old_coord + offset * old_size * 0.5
                new_coords.append(new_coord)

                # 继承特征（添加噪声）
                noise = torch.randn_like(old_feature) * 0.01
                new_features.append(old_feature + noise)

                # 子体素大小减半
                new_sizes.append(old_size * 0.5)

        if new_coords:
            # 添加新体素
            new_coords = torch.stack(new_coords)
            new_features = torch.stack(new_features)
            new_sizes = torch.stack(new_sizes)

            # 更新体素数据
            self._add_voxels(new_coords, new_features, new_sizes)

            # 停用原体素
            self.active_mask[high_gradient_indices] = False

            self.training_stats["subdivision_count"] += len(high_gradient_indices)
            logger.info(
                f"Subdivided {len(high_gradient_indices)} voxels into {len(new_coords)} sub-voxels"
            )

    def adaptive_pruning(self):
        """自适应体素剪枝"""
        if not self.active_mask.any():
            return

        # 基于密度阈值剪枝
        active_features = self.voxel_features[self.active_mask]
        densities = F.softplus(active_features[:, 0])

        # 低密度体素剪枝
        low_density_mask = densities < self.sparsity_threshold
        low_density_indices = torch.where(low_density_mask)[0]

        if len(low_density_indices) > 0:
            # 将低密度体素标记为非活跃
            active_indices = torch.where(self.active_mask)[0]
            prune_indices = active_indices[low_density_indices]
            self.active_mask[prune_indices] = False

            self.training_stats["pruning_count"] += len(prune_indices)
            logger.info(f"Pruned {len(prune_indices)} low-density voxels")

    def _add_voxels(self, coords: torch.Tensor, features: torch.Tensor, sizes: torch.Tensor):
        """添加新体素"""
        # 检查是否超过最大体素数量
        current_count = self.voxel_coords.shape[0]
        new_count = coords.shape[0]

        if current_count + new_count > self.max_voxels:
            # 如果超过限制，随机移除一些体素
            remove_count = current_count + new_count - self.max_voxels
            remove_indices = torch.randperm(current_count)[:remove_count]

            # 移除体素
            self.voxel_coords = torch.cat(
                [
                    self.voxel_coords[: remove_indices[0]],
                    self.voxel_coords[remove_indices[-1] + 1 :],
                ]
            )
            self.voxel_features = torch.cat(
                [
                    self.voxel_features[: remove_indices[0]],
                    self.voxel_features[remove_indices[-1] + 1 :],
                ]
            )
            self.voxel_sizes = torch.cat(
                [self.voxel_sizes[: remove_indices[0]], self.voxel_sizes[remove_indices[-1] + 1 :]]
            )
            self.active_mask = torch.cat(
                [self.active_mask[: remove_indices[0]], self.active_mask[remove_indices[-1] + 1 :]]
            )

            current_count = self.voxel_coords.shape[0]

        # 添加新体素
        self.voxel_coords = torch.cat([self.voxel_coords, coords])
        self.voxel_features = torch.cat([self.voxel_features, features])
        self.voxel_sizes = torch.cat([self.voxel_sizes, sizes])

        # 扩展活跃掩码
        new_active_mask = torch.ones(new_count, dtype=torch.bool, device=self.active_mask.device)
        self.active_mask = torch.cat([self.active_mask, new_active_mask])

        # 更新 Morton 编码
        if self.config.voxel.use_morton_ordering:
            new_morton_codes = MortonCode.encode_3d(
                coords[:, 0], coords[:, 1], coords[:, 2], self.config.voxel.morton_bits
            )
            self.morton_codes = torch.cat([self.morton_codes, new_morton_codes])

    def sort_by_morton(self):
        """按 Morton 编码排序"""
        if not self.config.voxel.use_morton_ordering:
            return

        # 获取活跃体素的 Morton 编码
        active_morton = self.morton_codes[self.active_mask]
        active_indices = torch.where(self.active_mask)[0]

        # 排序
        sort_indices = torch.argsort(active_morton)
        sorted_active_indices = active_indices[sort_indices]

        # 重新排列体素数据
        self.voxel_coords = self.voxel_coords[sorted_active_indices]
        self.voxel_features = self.voxel_features[sorted_active_indices]
        self.voxel_sizes = self.voxel_sizes[sorted_active_indices]
        self.morton_codes = self.morton_codes[sorted_active_indices]

        # 更新活跃掩码
        self.active_mask = torch.ones_like(self.active_mask)
        self.active_mask[len(sorted_active_indices) :] = False

    def get_active_voxels(self) -> Dict[str, torch.Tensor]:
        """获取活跃体素数据"""
        active_coords = self.voxel_coords[self.active_mask]
        active_features = self.voxel_features[self.active_mask]
        active_sizes = self.voxel_sizes[self.active_mask]

        # 分离密度和颜色
        densities = F.softplus(active_features[:, 0])
        colors = torch.sigmoid(active_features[:, 1:4])

        result = {
            "positions": active_coords,
            "densities": densities,
            "colors": colors,
            "sizes": active_sizes,
        }

        if self.config.voxel.use_morton_ordering:
            result["morton_codes"] = self.morton_codes[self.active_mask]

        return result

    def get_stats(self) -> Dict[str, float]:
        """获取统计信息"""
        active_count = self.active_mask.sum().item()
        total_count = self.voxel_coords.shape[0]

        return {
            "total_voxels": float(total_count),
            "active_voxels": float(active_count),
            "sparsity_ratio": 1.0 - active_count / total_count if total_count > 0 else 0.0,
            "subdivision_count": float(self.training_stats["subdivision_count"]),
            "pruning_count": float(self.training_stats["pruning_count"]),
        }
