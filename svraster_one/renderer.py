"""
可微分光栅化渲染器

实现基于 SVRaster 论文的可微分光栅化渲染，
支持端到端的梯度传播和训练。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
import math

logger = logging.getLogger(__name__)


class DifferentiableVoxelRasterizer(nn.Module):
    """
    可微分体素光栅化渲染器

    实现基于投影的光栅化渲染，支持：
    - 软光栅化（可微分）
    - 软深度排序（可微分）
    - Alpha 混合（可微分）
    - 梯度传播
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # 渲染参数
        self.image_width = config.rendering.image_width
        self.image_height = config.rendering.image_height
        self.background_color = torch.tensor(config.rendering.background_color, dtype=torch.float32)

        # 可微分参数
        self.temperature = config.rendering.temperature
        self.sigma = config.rendering.sigma
        self.soft_rasterization = config.rendering.soft_rasterization
        self.use_soft_sorting = config.rendering.use_soft_sorting
        self.alpha_blending = config.rendering.alpha_blending

        # 深度排序配置
        self.depth_sorting = config.rendering.depth_sorting

        logger.info(
            f"Initialized differentiable voxel rasterizer: {self.image_width}x{self.image_height}"
        )

    def forward(
        self,
        voxels: Dict[str, torch.Tensor],
        camera_matrix: torch.Tensor,
        intrinsics: torch.Tensor,
        viewport_size: Optional[Tuple[int, int]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        可微分光栅化渲染

        Args:
            voxels: 体素数据字典
            camera_matrix: 相机变换矩阵 [4, 4]
            intrinsics: 相机内参矩阵 [3, 3]
            viewport_size: 视口尺寸 (width, height)

        Returns:
            渲染结果字典
        """
        if viewport_size is None:
            viewport_size = (self.image_width, self.image_height)

        # 1. 投影变换（可微分）
        screen_voxels = self._project_voxels_to_screen(
            voxels, camera_matrix, intrinsics, viewport_size
        )

        # 2. 视锥剔除
        visible_voxels = self._frustum_culling(screen_voxels, viewport_size)

        if len(visible_voxels) == 0:
            # 没有可见体素，返回背景
            return self._create_background_image(viewport_size, voxels["positions"].device)

        # 3. 软深度排序（可微分）
        if self.use_soft_sorting:
            sorted_voxels = self._soft_depth_sort(visible_voxels)
        else:
            sorted_voxels = self._hard_depth_sort(visible_voxels)

        # 4. 软光栅化（可微分）
        if self.soft_rasterization:
            framebuffer = self._soft_rasterize_voxels(sorted_voxels, viewport_size)
        else:
            framebuffer = self._hard_rasterize_voxels(sorted_voxels, viewport_size)

        return framebuffer

    def _project_voxels_to_screen(
        self,
        voxels: Dict[str, torch.Tensor],
        camera_matrix: torch.Tensor,
        intrinsics: torch.Tensor,
        viewport_size: Tuple[int, int],
    ) -> List[Dict[str, torch.Tensor]]:
        """
        将体素投影到屏幕空间（可微分）
        """
        positions = voxels["positions"]  # [N, 3]
        sizes = voxels["sizes"]  # [N]
        densities = voxels["densities"]  # [N]
        colors = voxels["colors"]  # [N, 3]
        device = positions.device

        # 确保矩阵在正确设备上
        camera_matrix = camera_matrix.to(device, dtype=torch.float32)
        intrinsics = intrinsics.to(device, dtype=torch.float32)

        # 转换到相机坐标系
        positions_hom = torch.cat(
            [positions, torch.ones(positions.shape[0], 1, device=device)], dim=1
        )  # [N, 4]

        camera_positions = torch.matmul(positions_hom, camera_matrix.T)  # [N, 4]
        camera_positions = camera_positions[:, :3] / camera_positions[:, 3:4]  # [N, 3]

        # 投影到屏幕空间
        screen_positions = torch.matmul(camera_positions, intrinsics.T)  # [N, 3]
        screen_positions = screen_positions[:, :2] / screen_positions[:, 2:3]  # [N, 2]

        # 计算屏幕上的体素尺寸（透视投影）
        depths = camera_positions[:, 2]
        screen_sizes = sizes * intrinsics[0, 0] / torch.clamp(depths, min=0.1)

        # 组装投影后的体素数据
        screen_voxels = []
        for i in range(positions.shape[0]):
            screen_voxels.append(
                {
                    "screen_pos": screen_positions[i],
                    "depth": depths[i],
                    "screen_size": screen_sizes[i],
                    "density": densities[i],
                    "color": colors[i],
                    "world_pos": positions[i],
                    "world_size": sizes[i],
                    "voxel_idx": i,
                }
            )

        return screen_voxels

    def _frustum_culling(
        self,
        screen_voxels: List[Dict[str, torch.Tensor]],
        viewport_size: Tuple[int, int],
    ) -> List[Dict[str, torch.Tensor]]:
        """
        视锥剔除
        """
        width, height = viewport_size
        visible_voxels = []

        for voxel in screen_voxels:
            x, y = voxel["screen_pos"]
            size = voxel["screen_size"]
            depth = voxel["depth"]

            # 深度剔除
            if depth <= 0.1 or depth >= 100.0:
                continue

            # 屏幕边界剔除（考虑体素尺寸）
            if x + size >= 0 and x - size < width and y + size >= 0 and y - size < height:
                visible_voxels.append(voxel)

        return visible_voxels

    def _soft_depth_sort(
        self, visible_voxels: List[Dict[str, torch.Tensor]]
    ) -> List[Dict[str, torch.Tensor]]:
        """
        软深度排序（可微分）
        """
        if len(visible_voxels) <= 1:
            return visible_voxels

        # 提取深度值
        depths = torch.stack([v["depth"] for v in visible_voxels])

        # 软排序：使用 softmax 计算排序权重
        if self.depth_sorting == "back_to_front":
            # 后向前排序（深度大的在前）
            sort_weights = F.softmax(depths / self.temperature, dim=0)
        else:
            # 前向后排序（深度小的在前）
            sort_weights = F.softmax(-depths / self.temperature, dim=0)

        # 按权重排序
        sorted_indices = torch.argsort(sort_weights, descending=True)

        return [visible_voxels[i] for i in sorted_indices]

    def _hard_depth_sort(
        self, visible_voxels: List[Dict[str, torch.Tensor]]
    ) -> List[Dict[str, torch.Tensor]]:
        """
        硬深度排序（不可微分）
        """
        if self.depth_sorting == "back_to_front":
            return sorted(visible_voxels, key=lambda v: v["depth"].item(), reverse=True)
        elif self.depth_sorting == "front_to_back":
            return sorted(visible_voxels, key=lambda v: v["depth"].item())
        else:
            return visible_voxels

    def _soft_rasterize_voxels(
        self,
        sorted_voxels: List[Dict[str, torch.Tensor]],
        viewport_size: Tuple[int, int],
    ) -> Dict[str, torch.Tensor]:
        """
        软光栅化（可微分）
        """
        width, height = viewport_size
        device = sorted_voxels[0]["screen_pos"].device if sorted_voxels else torch.device("cpu")

        # 初始化帧缓冲
        color_buffer = self.background_color.expand(height, width, -1).to(device).clone()
        depth_buffer = torch.full((height, width), 100.0, device=device)
        alpha_buffer = torch.zeros(height, width, device=device)

        # 软光栅化每个体素
        for voxel in sorted_voxels:
            self._soft_rasterize_single_voxel(
                voxel, color_buffer, depth_buffer, alpha_buffer, viewport_size
            )

        return {"rgb": color_buffer, "depth": depth_buffer, "alpha": alpha_buffer}

    def _soft_rasterize_single_voxel(
        self,
        voxel: Dict[str, torch.Tensor],
        color_buffer: torch.Tensor,
        depth_buffer: torch.Tensor,
        alpha_buffer: torch.Tensor,
        viewport_size: Tuple[int, int],
    ) -> None:
        """
        软光栅化单个体素（可微分）
        """
        width, height = viewport_size

        # 体素屏幕位置和尺寸
        screen_x, screen_y = voxel["screen_pos"]
        screen_size = voxel["screen_size"]
        depth = voxel["depth"]

        # 计算体素在屏幕上的像素范围
        half_size = screen_size * 0.5
        min_x = max(0, int(screen_x - half_size))
        max_x = min(width, int(screen_x + half_size) + 1)
        min_y = max(0, int(screen_y - half_size))
        max_y = min(height, int(screen_y + half_size) + 1)

        if min_x >= max_x or min_y >= max_y:
            return

        # 计算体素颜色和 alpha
        voxel_color = voxel["color"]
        density = voxel["density"]

        # 软 alpha 计算（可微分）
        sigma = F.softplus(density)
        voxel_alpha = 1.0 - torch.exp(-sigma * voxel["world_size"])
        voxel_alpha = torch.clamp(voxel_alpha, 0.0, 1.0)

        # 对覆盖的像素进行软着色
        for y in range(min_y, max_y):
            for x in range(min_x, max_x):
                # 计算像素到体素中心的距离
                dx = x - screen_x
                dy = y - screen_y
                distance = torch.sqrt(dx * dx + dy * dy)

                # 软距离权重（可微分）
                if distance <= half_size:
                    # 使用高斯核计算软权重
                    pixel_weight = torch.exp(-distance / (2 * self.sigma**2))
                    pixel_weight = torch.clamp(pixel_weight, 0.0, 1.0)

                    # 软 alpha 混合（可微分）
                    pixel_alpha = voxel_alpha * pixel_weight
                    current_alpha = alpha_buffer[y, x]

                    # 软混合因子
                    blend_factor = pixel_alpha * (1.0 - current_alpha)
                    blend_factor = torch.clamp(blend_factor, 0.0, 1.0)

                    # 软颜色混合
                    color_buffer[y, x] = (
                        color_buffer[y, x] * (1.0 - blend_factor) + voxel_color * blend_factor
                    )

                    # 软 alpha 累积
                    alpha_buffer[y, x] = current_alpha + blend_factor

                    # 软深度混合
                    if blend_factor > 0:
                        depth_buffer[y, x] = (
                            depth_buffer[y, x] * (1.0 - blend_factor) + depth * blend_factor
                        )

    def _hard_rasterize_voxels(
        self,
        sorted_voxels: List[Dict[str, torch.Tensor]],
        viewport_size: Tuple[int, int],
    ) -> Dict[str, torch.Tensor]:
        """
        硬光栅化（不可微分，用于推理）
        """
        width, height = viewport_size
        device = sorted_voxels[0]["screen_pos"].device if sorted_voxels else torch.device("cpu")

        # 初始化帧缓冲
        color_buffer = self.background_color.expand(height, width, -1).to(device).clone()
        depth_buffer = torch.full((height, width), 100.0, device=device)

        # 硬光栅化每个体素
        for voxel in sorted_voxels:
            self._hard_rasterize_single_voxel(voxel, color_buffer, depth_buffer, viewport_size)

        return {"rgb": color_buffer, "depth": depth_buffer}

    def _hard_rasterize_single_voxel(
        self,
        voxel: Dict[str, torch.Tensor],
        color_buffer: torch.Tensor,
        depth_buffer: torch.Tensor,
        viewport_size: Tuple[int, int],
    ) -> None:
        """
        硬光栅化单个体素（不可微分）
        """
        width, height = viewport_size

        # 体素屏幕位置和尺寸
        screen_x, screen_y = voxel["screen_pos"]
        screen_size = voxel["screen_size"]
        depth = voxel["depth"]

        # 计算体素在屏幕上的像素范围
        half_size = screen_size * 0.5
        min_x = max(0, int(screen_x - half_size))
        max_x = min(width, int(screen_x + half_size) + 1)
        min_y = max(0, int(screen_y - half_size))
        max_y = min(height, int(screen_y + half_size) + 1)

        if min_x >= max_x or min_y >= max_y:
            return

        # 计算体素颜色和 alpha
        voxel_color = voxel["color"]
        density = voxel["density"]

        # 硬 alpha 计算
        sigma = F.softplus(density)
        voxel_alpha = 1.0 - torch.exp(-sigma * voxel["world_size"])
        voxel_alpha = torch.clamp(voxel_alpha, 0.0, 1.0)

        # 对覆盖的像素进行硬着色
        for y in range(min_y, max_y):
            for x in range(min_x, max_x):
                # 计算像素到体素中心的距离
                dx = x - screen_x
                dy = y - screen_y
                distance = torch.sqrt(dx * dx + dy * dy)

                if distance <= half_size:
                    # 硬距离权重
                    pixel_weight = 1.0 - distance / half_size
                    pixel_weight = torch.clamp(pixel_weight, 0.0, 1.0)

                    # 硬 alpha 混合
                    pixel_alpha = voxel_alpha * pixel_weight

                    # 深度测试
                    if depth < depth_buffer[y, x]:
                        # 硬颜色混合
                        color_buffer[y, x] = (
                            color_buffer[y, x] * (1.0 - pixel_alpha) + voxel_color * pixel_alpha
                        )
                        depth_buffer[y, x] = depth

    def _create_background_image(
        self, viewport_size: Tuple[int, int], device: torch.device
    ) -> Dict[str, torch.Tensor]:
        """创建背景图像"""
        width, height = viewport_size
        background = self.background_color.expand(height, width, -1).to(device)
        depth = torch.full((height, width), 100.0, device=device)

        return {"rgb": background, "depth": depth}

    def compute_gradients(
        self,
        rendered_image: torch.Tensor,
        target_image: torch.Tensor,
        voxels: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        计算梯度（用于反向传播）

        Args:
            rendered_image: 渲染图像 [H, W, 3]
            target_image: 目标图像 [H, W, 3]
            voxels: 体素数据

        Returns:
            梯度字典
        """
        # 计算图像损失
        image_loss = F.mse_loss(rendered_image, target_image)

        # 计算梯度
        gradients = {}

        # 对体素特征的梯度
        if "densities" in voxels and voxels["densities"].requires_grad:
            density_grad = torch.autograd.grad(image_loss, voxels["densities"], retain_graph=True)[
                0
            ]
            gradients["density_gradients"] = density_grad

        # 对体素颜色的梯度
        if "colors" in voxels and voxels["colors"].requires_grad:
            color_grad = torch.autograd.grad(image_loss, voxels["colors"], retain_graph=True)[0]
            gradients["color_gradients"] = color_grad

        # 对体素位置的梯度
        if "positions" in voxels and voxels["positions"].requires_grad:
            pos_grad = torch.autograd.grad(image_loss, voxels["positions"], retain_graph=True)[0]
            gradients["position_gradients"] = pos_grad

        return gradients

    def render_with_gradients(
        self,
        voxels: Dict[str, torch.Tensor],
        camera_matrix: torch.Tensor,
        intrinsics: torch.Tensor,
        target_image: torch.Tensor,
        viewport_size: Optional[Tuple[int, int]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        渲染并计算梯度

        Args:
            voxels: 体素数据
            camera_matrix: 相机矩阵
            intrinsics: 相机内参
            target_image: 目标图像
            viewport_size: 视口尺寸

        Returns:
            包含梯度的结果字典
        """
        # 启用梯度计算
        for key in voxels:
            if isinstance(voxels[key], torch.Tensor):
                voxels[key].requires_grad_(True)

        # 渲染
        rendered_result = self.forward(voxels, camera_matrix, intrinsics, viewport_size)
        rendered_image = rendered_result["rgb"]

        # 计算梯度
        gradients = self.compute_gradients(rendered_image, target_image, voxels)

        # 合并结果
        result = rendered_result.copy()
        result.update(gradients)
        result["loss"] = F.mse_loss(rendered_image, target_image)

        return result
