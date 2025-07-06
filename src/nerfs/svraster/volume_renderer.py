"""
Volume Renderer for SVRaster Training

This module implements the volume rendering algorithm used during training phase.
It performs ray casting with volumetric integration, supporting adaptive sampling
and spherical harmonics for view-dependent appearance.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from .utils.rendering_utils import ray_direction_dependent_ordering
from .spherical_harmonics import eval_sh_basis
import math


class VolumeRenderer:
    """
    体积渲染器（原 VoxelRasterizer）

    实现了基于光线投射的体积渲染算法，用于训练阶段。
    支持自适应采样和深度剥离，使用体积渲染积分。
    """

    def __init__(self, config):
        self.config = config
        self.step_size = config.ray_samples_per_voxel
        self.depth_layers = config.depth_peeling_layers
        self.use_morton = config.morton_ordering
        self.background_color = torch.tensor(config.background_color)

    def __call__(
        self,
        voxels: Dict[str, torch.Tensor],
        ray_origins: torch.Tensor,
        ray_directions: torch.Tensor,
        camera_params: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        执行光栅化过程

        Args:
            voxels: 包含体素属性的字典
            ray_origins: 光线起点 [N, 3]
            ray_directions: 光线方向 [N, 3]
            camera_params: 可选的相机参数

        Returns:
            包含渲染结果的字典
        """
        # Sort voxels by view direction if needed
        if self.use_morton:
            mean_ray_direction = ray_directions.mean(dim=0)
            voxels = self._sort_voxels_by_ray_direction(voxels, mean_ray_direction)

        # Batch process rays
        batch_size = ray_origins.shape[0]
        all_colors = []
        all_depths = []
        all_weights = []

        for i in range(0, batch_size, 1024):
            batch_o = ray_origins[i : i + 1024]
            batch_d = ray_directions[i : i + 1024]

            # Get ray-voxel intersections
            intersections = self._ray_voxel_intersections(batch_o, batch_d, voxels)

            # Render each ray
            colors, depths, weights = self._render_ray(batch_o, batch_d, intersections, voxels)

            all_colors.append(colors)
            all_depths.append(depths)
            all_weights.append(weights)

        # Combine results
        colors = torch.cat(all_colors, dim=0)
        depths = torch.cat(all_depths, dim=0)
        weights = torch.cat(all_weights, dim=0)

        return {"rgb": colors, "depth": depths, "weights": weights}

    def _sort_voxels_by_ray_direction(
        self,
        voxels: Dict[str, torch.Tensor],
        mean_ray_direction: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        根据平均光线方向对体素进行排序

        将体素中心投影到平均光线方向上，按投影距离排序，
        确保正确的深度顺序。

        Args:
            voxels: 体素数据
            mean_ray_direction: 平均光线方向

        Returns:
            排序后的体素数据
        """
        if not self.config.morton_ordering:
            return voxels

        # 使用真正的视角相关 Morton 排序
        sort_indices = ray_direction_dependent_ordering(
            voxels["positions"], voxels["morton_codes"], mean_ray_direction
        )

        # Sort all voxel attributes using the returned indices
        sorted_voxels = {}
        for key, value in voxels.items():
            sorted_voxels[key] = value[sort_indices]

        return sorted_voxels

    def _ray_voxel_intersections(
        self,
        ray_o: torch.Tensor,
        ray_d: torch.Tensor,
        voxels: Dict[str, torch.Tensor],
    ) -> List[Tuple[int, float, float]]:
        """
        计算光线与体素的相交

        使用向量化的 AABB（轴对齐包围盒）进行快速光线-体素相交测试，
        返回相交的体素索引和相交参数。

        Args:
            ray_o: 光线起点 [N, 3]
            ray_d: 光线方向 [N, 3]
            voxels: 体素数据

        Returns:
            相交列表，每个元素为（体素索引，t_near, t_far）
        """
        positions = voxels["positions"]
        sizes = voxels["sizes"]
        device = positions.device

        # 计算所有体素的 AABB
        # sizes is [N_voxels, 3], so half_sizes should also be [N_voxels, 3]
        half_sizes = sizes / 2
        box_mins = positions - half_sizes
        box_maxs = positions + half_sizes

        # 简化处理：只处理第一条光线
        # TODO: 支持批量光线处理
        if ray_o.dim() > 1:
            ray_o = ray_o[0]  # Take first ray
            ray_d = ray_d[0]
        
        # Ensure ray tensors are on the same device
        ray_o = ray_o.to(device)
        ray_d = ray_d.to(device)
        
        # 向量化的光线-AABB 相交检测
        # Expand ray to match voxel count
        ray_o_expanded = ray_o.unsqueeze(0).expand(positions.shape[0], -1)
        ray_d_expanded = ray_d.unsqueeze(0).expand(positions.shape[0], -1)
        
        t_mins = (box_mins - ray_o_expanded) / ray_d_expanded
        t_maxs = (box_maxs - ray_o_expanded) / ray_d_expanded

        t1 = torch.min(t_mins, t_maxs)
        t2 = torch.max(t_mins, t_maxs)

        t_near = torch.max(t1, dim=-1)[0]
        t_far = torch.min(t2, dim=-1)[0]

        # 找到有效的相交
        valid_mask = (t_far > t_near) & (t_far > 0)
        valid_indices = torch.where(valid_mask)[0]

        # 构建相交列表
        intersections = []
        for idx in valid_indices:
            intersections.append((idx.item(), t_near[idx].item(), t_far[idx].item()))

        # Sort by t_near
        intersections.sort(key=lambda x: x[1])
        return intersections

    def _render_ray(
        self,
        ray_o: torch.Tensor,
        ray_d: torch.Tensor,
        intersections: List[Tuple[int, float, float]],
        voxels: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        渲染单条光线（支持体积积分、球谐函数、视角相关颜色）
        """
        device = ray_o.device
        sh_degree = getattr(self.config, "sh_degree", 2)
        num_sh_coeffs = (sh_degree + 1) ** 2

        # Handle batched rays - only process first ray for now
        if ray_o.dim() > 1:
            ray_o = ray_o[0]
            ray_d = ray_d[0]

        if not intersections:
            # 确保返回的张量有梯度
            bg_color = torch.tensor(self.config.background_color, device=device, requires_grad=True)
            far_depth = torch.tensor([self.config.far_plane], device=device, requires_grad=True)
            zero_weight = torch.tensor([0.0], device=device, requires_grad=True)
            return bg_color, far_depth, zero_weight

        rgb_acc = torch.zeros(3, device=device, requires_grad=True)
        depth_acc = torch.zeros(1, device=device, requires_grad=True)
        weight_acc = torch.zeros(1, device=device, requires_grad=True)
        transmittance = 1.0

        for voxel_idx, t_near, t_far in intersections:
            # 多点采样
            n_samples = self.config.ray_samples_per_voxel
            t_samples = torch.linspace(t_near, t_far, n_samples, device=device)
            sample_points = ray_o + ray_d * t_samples.unsqueeze(-1)  # [n_samples, 3]
            # 所有采样点方向都用 ray_d
            dirs = ray_d.expand(n_samples, 3)

            # 获取体素 SH 系数
            color_coeffs = voxels["colors"][voxel_idx]  # [3 * num_sh_coeffs]
            color_coeffs = color_coeffs.to(device)  # Ensure correct device
            color_coeffs = color_coeffs.view(3, num_sh_coeffs)  # [3, num_sh_coeffs]
            sh_basis = eval_sh_basis(sh_degree, dirs)  # [n_samples, num_sh_coeffs]
            rgb_samples = torch.matmul(sh_basis, color_coeffs.t())  # [n_samples, 3]

            # 激活函数
            if self.config.color_activation == "sigmoid":
                rgb_samples = torch.sigmoid(rgb_samples)
            elif self.config.color_activation == "tanh":
                rgb_samples = (torch.tanh(rgb_samples) + 1) / 2
            elif self.config.color_activation == "clamp":
                rgb_samples = torch.clamp(rgb_samples, 0, 1)

            # 密度
            density = voxels["densities"][voxel_idx]
            density = density.to(device)  # Ensure correct device
            if self.config.density_activation == "exp":
                sigma = torch.exp(density)
            else:
                sigma = F.relu(density)
            # 假设体素内密度均匀
            sigmas = sigma.expand(n_samples)

            # 体积渲染积分
            delta_t = (t_far - t_near) / n_samples
            alphas = 1.0 - torch.exp(-sigmas * delta_t)  # [n_samples]
            trans = torch.cumprod(
                torch.cat([torch.ones(1, device=device), 1 - alphas + 1e-8]), dim=0
            )[:-1]
            weights = alphas * trans  # [n_samples]

            rgb = torch.sum(weights.unsqueeze(-1) * rgb_samples, dim=0)  # [3]
            depth = torch.sum(weights * t_samples)
            weight_sum = torch.sum(weights)

            rgb_acc = rgb_acc + transmittance * rgb
            depth_acc = depth_acc + transmittance * depth
            weight_acc = weight_acc + transmittance * weight_sum

            transmittance = transmittance * (1 - weight_sum)
            if transmittance < 0.01:
                break

        # 背景
        if transmittance > 0:
            bg_color = torch.tensor(self.config.background_color, device=device, requires_grad=True)
            rgb_acc = rgb_acc + transmittance * bg_color

        # 输出 shape 兼容
        return rgb_acc, depth_acc, weight_acc
