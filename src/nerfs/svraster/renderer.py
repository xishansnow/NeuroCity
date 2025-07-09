"""
SVRaster Renderer - 与 VoxelRasterizer 紧密耦合

这个渲染器专门用于推理阶段，与 VoxelRasterizer 紧密耦合，
使用光栅化进行快速渲染，符合 SVRaster 论文的设计理念。

渲染器负责：
1. 加载训练好的模型进行推理
2. 与 VoxelRasterizer 配合进行光栅化渲染
3. 生成新视点的高质量图像
4. 支持批量渲染和实时渲染
5. 实现基于投影的光栅化设计（图像分块、视锥剔除、深度排序）
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import numpy as np

from dataclasses import dataclass
from pathlib import Path
import logging
import json
from tqdm import tqdm
import os
import math
from typing import Optional, Tuple, List

logger = logging.getLogger(__name__)

# 尝试导入 imageio
try:
    import imageio

    IMAGEIO_AVAILABLE = True
except ImportError:
    IMAGEIO_AVAILABLE = False
    logger.warning("imageio not available for video rendering")

from .core import SVRasterModel, SVRasterConfig
from .voxel_rasterizer import VoxelRasterizer

# 创建 VoxelRasterizer 的简单配置类
from dataclasses import dataclass


@dataclass
class VoxelRasterizerConfig:
    """VoxelRasterizer 配置"""

    background_color: tuple[float, float, float] = (1.0, 1.0, 1.0)
    near_plane: float = 0.1
    far_plane: float = 100.0
    density_activation: str = "exp"
    color_activation: str = "sigmoid"
    sh_degree: int = 2


@dataclass
class TileConfig:
    """图像分块配置 - 基于 SVRaster 论文设计"""

    tile_size: int = 64  # 分块大小
    overlap: int = 8  # 分块重叠像素
    use_adaptive_tiling: bool = True  # 自适应分块
    min_tile_size: int = 32  # 最小分块大小
    max_tile_size: int = 128  # 最大分块大小


@dataclass
class FrustumCullingConfig:
    """视锥剔除配置"""

    enable_frustum_culling: bool = True
    culling_margin: float = 0.1  # 剔除边界余量
    use_octree_culling: bool = True  # 使用八叉树加速剔除
    max_culling_depth: int = 8  # 最大剔除深度


@dataclass
class DepthSortingConfig:
    """深度排序配置"""

    enable_depth_sorting: bool = True
    sort_method: str = "back_to_front"  # back_to_front, front_to_back, or none
    use_bucket_sort: bool = True  # 使用桶排序加速
    bucket_count: int = 100  # 桶数量


@dataclass
class SVRasterRendererConfig:
    """SVRaster 渲染器配置 - 专门为光栅化推理设计"""

    # 渲染质量设置
    image_width: int = 800
    image_height: int = 600
    render_batch_size: int = 4096
    render_chunk_size: int = 1024

    # 光栅化参数（与 VoxelRasterizer 紧密相关）
    background_color: tuple[float, float, float] = (1.0, 1.0, 1.0)
    use_alpha_blending: bool = True
    depth_threshold: float = 1e-6

    # 基于投影的光栅化设计参数
    tile_config: TileConfig = TileConfig()
    frustum_config: FrustumCullingConfig = FrustumCullingConfig()
    depth_config: DepthSortingConfig = DepthSortingConfig()

    # 渲染质量控制
    max_rays_per_batch: int = 8192
    use_hierarchical_sampling: bool = True

    # 渲染模式和输出设置
    render_mode: str = "rasterization"
    output_dir: str = "outputs/rendered"
    output_format: str = "png"
    save_format: str = "png"  # 为了向后兼容
    save_depth: bool = False
    save_alpha: bool = False

    # 优化设置
    use_cached_features: bool = True
    enable_gradient_checkpointing: bool = False

    # 性能监控
    enable_performance_monitoring: bool = False
    log_render_stats: bool = True

    def __post_init__(self):
        """Post-initialization validation."""
        if self.image_width <= 0:
            raise ValueError("image_width must be positive")
        if self.image_height <= 0:
            raise ValueError("image_height must be positive")
        if self.render_batch_size <= 0:
            raise ValueError("render_batch_size must be positive")


class ImageTile:
    """图像分块类 - 用于并行处理"""

    def __init__(self, x: int, y: int, width: int, height: int, tile_id: int):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.tile_id = tile_id
        self.voxels_in_tile: List[dict] = []

    def contains_point(self, x: float, y: float) -> bool:
        """检查点是否在分块内"""
        return self.x <= x < self.x + self.width and self.y <= y < self.y + self.height

    def get_bounds(self) -> Tuple[int, int, int, int]:
        """获取分块边界"""
        return (self.x, self.y, self.x + self.width, self.y + self.height)


class SVRasterRenderer:
    """
    SVRaster 渲染器 - 与 VoxelRasterizer 紧密耦合

    专门负责：
    - 加载训练好的模型进行推理
    - 使用光栅化进行快速渲染
    - 生成新视点的图像
    - 支持批量渲染和实时渲染
    - 实现基于投影的光栅化设计
    """

    def __init__(
        self, model: SVRasterModel, rasterizer: VoxelRasterizer, config: SVRasterRendererConfig
    ):
        self.model = model
        self.rasterizer = rasterizer  # 紧密耦合的体素光栅化器
        self.config = config

        # 确保模型处于评估模式
        self.model.eval()

        # 设置设备
        self.device = next(self.model.parameters()).device

        # 性能监控
        self.render_stats = {}
        self._reset_render_stats()

        logger.info(f"SVRasterRenderer initialized with VoxelRasterizer coupling")
        logger.info(f"Model device: {self.device}")
        logger.info(f"Render config: {self.config}")

    def _reset_render_stats(self):
        """重置渲染统计信息"""
        self.render_stats = {
            "total_voxels": 0,
            "visible_voxels": 0,
            "culled_voxels": 0,
            "render_time_ms": 0.0,
            "projection_time_ms": 0.0,
            "culling_time_ms": 0.0,
            "sorting_time_ms": 0.0,
            "rasterization_time_ms": 0.0,
            "tile_count": 0,
            "voxels_per_tile": [],
        }

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        rasterizer_config: VoxelRasterizerConfig | None = None,
        renderer_config: SVRasterRendererConfig | None = None,
    ) -> SVRasterRenderer:
        """
        从检查点加载渲染器
        """
        # 加载检查点
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        # 创建模型配置
        if "model_config" in checkpoint:
            model_config = checkpoint["model_config"]
        else:
            model_config = SVRasterConfig()
            logger.warning("No model config found in checkpoint, using default")

        # 创建模型
        model = SVRasterModel(model_config)
        model.load_state_dict(checkpoint["model_state_dict"])

        # 移到正确的设备
        if torch.cuda.is_available():
            model = model.cuda()

        # 创建光栅化器
        if rasterizer_config is None:
            rasterizer_config = VoxelRasterizerConfig()

        rasterizer = VoxelRasterizer(rasterizer_config)

        # 创建渲染器配置
        if renderer_config is None:
            renderer_config = SVRasterRendererConfig()

        logger.info(f"Loading renderer from checkpoint: {checkpoint_path}")

        return cls(model, rasterizer, renderer_config)

    def render(
        self,
        camera_pose: torch.Tensor,
        intrinsics: torch.Tensor,
        width: int | None = None,
        height: int | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        渲染方法 - render_image 的别名，用于测试兼容性
        """
        return self.render_image(camera_pose, intrinsics, width, height)

    def render_image(
        self,
        camera_pose: torch.Tensor,
        intrinsics: torch.Tensor,
        width: int | None = None,
        height: int | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        渲染单张图像 - 使用增强的光栅化设计
        """
        import time

        start_time = time.time()

        width = width or self.config.image_width
        height = height or self.config.image_height

        # 重置统计信息
        self._reset_render_stats()

        with torch.no_grad():
            # 1. 从模型提取体素数据
            voxels = self._extract_voxels_from_model()
            self.render_stats["total_voxels"] = voxels["positions"].shape[0]

            # 2. 投影体素到屏幕空间
            projection_start = time.time()
            screen_voxels = self._project_voxels_to_screen_enhanced(
                voxels, camera_pose, intrinsics, (width, height)
            )
            self.render_stats["projection_time_ms"] = (time.time() - projection_start) * 1000

            # 3. 视锥剔除
            culling_start = time.time()
            visible_voxels = self._frustum_culling_enhanced(screen_voxels, (width, height))
            self.render_stats["culled_voxels"] = self.render_stats["total_voxels"] - len(
                visible_voxels
            )
            self.render_stats["visible_voxels"] = len(visible_voxels)
            self.render_stats["culling_time_ms"] = (time.time() - culling_start) * 1000

            if len(visible_voxels) == 0:
                # 没有可见体素，返回背景
                rgb = torch.tensor(self.config.background_color, device=self.device).expand(
                    height, width, -1
                )
                depth = torch.full(
                    (height, width), self.rasterizer.config.far_plane, device=self.device
                )
                return {"rgb": rgb, "depth": depth}

            # 4. 深度排序
            sorting_start = time.time()
            sorted_voxels = self._depth_sort_enhanced(visible_voxels)
            self.render_stats["sorting_time_ms"] = (time.time() - sorting_start) * 1000

            # 5. 图像分块处理
            rasterization_start = time.time()
            result = self._rasterize_with_tiles(sorted_voxels, (width, height))
            self.render_stats["rasterization_time_ms"] = (time.time() - rasterization_start) * 1000

            # 6. 记录总渲染时间
            self.render_stats["render_time_ms"] = (time.time() - start_time) * 1000

            # 7. 记录渲染统计信息
            if self.config.log_render_stats:
                self._log_render_stats()

            return result

    def _project_voxels_to_screen_enhanced(
        self,
        voxels: dict[str, torch.Tensor],
        camera_pose: torch.Tensor,
        intrinsics: torch.Tensor,
        viewport_size: tuple[int, int],
    ) -> list[dict[str, torch.Tensor]]:
        """
        增强的体素投影到屏幕空间
        """
        positions = voxels["positions"]  # [N, 3]
        sizes = voxels["sizes"]  # [N]
        densities = voxels["densities"]  # [N]
        colors = voxels["colors"]  # [N, color_dim]
        device = positions.device

        # 确保相机矩阵在正确的设备和数据类型上
        camera_pose = camera_pose.to(device, dtype=torch.float32)
        intrinsics = intrinsics.to(device, dtype=torch.float32)

        # 转换到相机坐标系
        positions_hom = torch.cat(
            [positions, torch.ones(positions.shape[0], 1, device=device)], dim=1
        )  # [N, 4]

        camera_positions = torch.matmul(positions_hom, camera_pose.T)  # [N, 4]
        camera_positions = camera_positions[:, :3] / camera_positions[:, 3:4]  # [N, 3]

        # 投影到屏幕空间
        screen_positions = torch.matmul(camera_positions, intrinsics.T)  # [N, 3]
        screen_positions = screen_positions[:, :2] / screen_positions[:, 2:3]  # [N, 2]

        # 计算屏幕上的体素尺寸（考虑透视投影）
        depths = camera_positions[:, 2]
        if sizes.dim() == 2:
            sizes_scalar = sizes[:, 0]
        else:
            sizes_scalar = sizes
        screen_sizes = sizes_scalar * intrinsics[0, 0] / torch.clamp(depths, min=0.1)

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
                    "projected_bounds": self._compute_projected_bounds(
                        screen_positions[i], screen_sizes[i]
                    ),
                }
            )

        return screen_voxels

    def _compute_projected_bounds(
        self, screen_pos: torch.Tensor, screen_size: torch.Tensor
    ) -> tuple[float, float, float, float]:
        """计算投影后的边界框"""
        x, y = screen_pos
        size = screen_size
        half_size = size * 0.5

        return (
            float(x - half_size),  # min_x
            float(y - half_size),  # min_y
            float(x + half_size),  # max_x
            float(y + half_size),  # max_y
        )

    def _frustum_culling_enhanced(
        self, screen_voxels: list[dict[str, torch.Tensor]], viewport_size: tuple[int, int]
    ) -> list[dict[str, torch.Tensor]]:
        """
        增强的视锥剔除
        """
        if not self.config.frustum_config.enable_frustum_culling:
            return screen_voxels

        width, height = viewport_size
        margin = self.config.frustum_config.culling_margin
        visible_voxels = []

        for voxel in screen_voxels:
            x, y = voxel["screen_pos"]
            size = voxel["screen_size"]
            depth = voxel["depth"]
            min_x, min_y, max_x, max_y = voxel["projected_bounds"]

            # 深度剔除
            if (
                depth <= self.rasterizer.config.near_plane
                or depth >= self.rasterizer.config.far_plane
            ):
                continue

            # 屏幕边界剔除（考虑边界余量）
            margin_pixels = margin * size
            if (
                min_x - margin_pixels < width
                and max_x + margin_pixels >= 0
                and min_y - margin_pixels < height
                and max_y + margin_pixels >= 0
            ):
                visible_voxels.append(voxel)

        return visible_voxels

    def _depth_sort_enhanced(
        self, visible_voxels: list[dict[str, torch.Tensor]]
    ) -> list[dict[str, torch.Tensor]]:
        """
        增强的深度排序
        """
        if not self.config.depth_config.enable_depth_sorting:
            return visible_voxels

        method = self.config.depth_config.sort_method

        if method == "back_to_front":
            return sorted(visible_voxels, key=lambda v: v["depth"].item(), reverse=True)
        elif method == "front_to_back":
            return sorted(visible_voxels, key=lambda v: v["depth"].item(), reverse=False)
        else:
            return visible_voxels

    def _create_image_tiles(self, width: int, height: int) -> list[ImageTile]:
        """
        创建图像分块
        """
        tile_config = self.config.tile_config
        tile_size = tile_config.tile_size
        overlap = tile_config.overlap

        tiles = []
        tile_id = 0

        for y in range(0, height, tile_size - overlap):
            for x in range(0, width, tile_size - overlap):
                # 计算实际分块大小
                actual_width = min(tile_size, width - x)
                actual_height = min(tile_size, height - y)

                if actual_width > 0 and actual_height > 0:
                    tile = ImageTile(x, y, actual_width, actual_height, tile_id)
                    tiles.append(tile)
                    tile_id += 1

        self.render_stats["tile_count"] = len(tiles)
        return tiles

    def _assign_voxels_to_tiles(
        self, sorted_voxels: list[dict[str, torch.Tensor]], tiles: list[ImageTile]
    ) -> None:
        """
        将体素分配到对应的图像分块
        """
        for voxel in sorted_voxels:
            min_x, min_y, max_x, max_y = voxel["projected_bounds"]

            # 找到与体素重叠的所有分块
            for tile in tiles:
                if (
                    min_x < tile.x + tile.width
                    and max_x > tile.x
                    and min_y < tile.y + tile.height
                    and max_y > tile.y
                ):
                    tile.voxels_in_tile.append(voxel)

    def _rasterize_with_tiles(
        self, sorted_voxels: list[dict[str, torch.Tensor]], viewport_size: tuple[int, int]
    ) -> dict[str, torch.Tensor]:
        """
        使用分块进行光栅化渲染
        """
        width, height = viewport_size
        device = sorted_voxels[0]["screen_pos"].device if sorted_voxels else torch.device("cpu")

        # 创建图像分块
        tiles = self._create_image_tiles(width, height)

        # 将体素分配到分块
        self._assign_voxels_to_tiles(sorted_voxels, tiles)

        # 初始化帧缓冲
        color_buffer = (
            torch.tensor(self.config.background_color, device=device, dtype=torch.float32)
            .expand(height, width, -1)
            .clone()
        )
        depth_buffer = torch.full((height, width), self.rasterizer.config.far_plane, device=device)
        alpha_buffer = torch.zeros(height, width, device=device)

        # 记录每个分块的体素数量
        for tile in tiles:
            self.render_stats["voxels_per_tile"].append(len(tile.voxels_in_tile))

        # 并行处理每个分块（这里简化为串行，实际可以并行）
        for tile in tiles:
            self._rasterize_tile(tile, color_buffer, depth_buffer, alpha_buffer)

        return {"rgb": color_buffer, "depth": depth_buffer}

    def _rasterize_tile(
        self,
        tile: ImageTile,
        color_buffer: torch.Tensor,
        depth_buffer: torch.Tensor,
        alpha_buffer: torch.Tensor,
    ) -> None:
        """
        光栅化单个分块
        """
        for voxel in tile.voxels_in_tile:
            self._rasterize_single_voxel_enhanced(voxel, color_buffer, depth_buffer, alpha_buffer)

    def _rasterize_single_voxel_enhanced(
        self,
        voxel: dict[str, torch.Tensor],
        color_buffer: torch.Tensor,
        depth_buffer: torch.Tensor,
        alpha_buffer: torch.Tensor,
    ) -> None:
        """
        增强的单体素光栅化
        """
        height, width = color_buffer.shape[:2]

        # 体素屏幕位置和尺寸
        screen_x, screen_y = voxel["screen_pos"]
        screen_size = voxel["screen_size"]
        depth = voxel["depth"]
        min_x, min_y, max_x, max_y = voxel["projected_bounds"]

        # 计算体素在屏幕上的像素范围
        min_x = max(0, int(min_x))
        max_x = min(width, int(max_x) + 1)
        min_y = max(0, int(min_y))
        max_y = min(height, int(max_y) + 1)

        if min_x >= max_x or min_y >= max_y:
            return

        # 计算体素颜色
        voxel_color = self._compute_voxel_color_enhanced(voxel)

        # 计算体素 alpha
        density = voxel["density"]
        if self.rasterizer.config.density_activation == "exp":
            sigma = torch.exp(density)
        else:
            sigma = F.relu(density)

        voxel_alpha = 1.0 - torch.exp(-sigma * voxel["world_size"])
        voxel_alpha = torch.clamp(voxel_alpha, 0.0, 1.0)

        # 对覆盖的像素进行着色
        for y in range(min_y, max_y):
            for x in range(min_x, max_x):
                # 距离衰减
                dx = x - screen_x
                dy = y - screen_y
                distance = torch.sqrt(dx * dx + dy * dy)
                half_size = screen_size * 0.5

                if distance <= half_size:
                    # 深度测试和 alpha blending
                    if depth < depth_buffer[y, x]:
                        pixel_alpha = voxel_alpha * (1.0 - distance / half_size)
                        pixel_alpha = torch.clamp(pixel_alpha, 0.0, 1.0)

                        current_alpha = alpha_buffer[y, x]
                        blend_factor = pixel_alpha * (1.0 - current_alpha)

                        color_buffer[y, x] = (
                            color_buffer[y, x] * (1.0 - blend_factor) + voxel_color * blend_factor
                        )

                        alpha_buffer[y, x] = current_alpha + blend_factor

                        if blend_factor > 0:
                            depth_buffer[y, x] = (
                                depth_buffer[y, x] * (1.0 - blend_factor) + depth * blend_factor
                            )

    def _compute_voxel_color_enhanced(self, voxel: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        增强的体素颜色计算
        """
        # 简化实现：假设观察方向为 z 轴负方向
        view_dir = torch.tensor([0.0, 0.0, -1.0], device=voxel["color"].device)

        # 获取球谐系数
        sh_degree = getattr(self.rasterizer.config, "sh_degree", 2)
        num_sh_coeffs = (sh_degree + 1) ** 2

        color_coeffs = voxel["color"]

        if color_coeffs.numel() >= 3 * num_sh_coeffs:
            color_coeffs = color_coeffs[: 3 * num_sh_coeffs].view(3, num_sh_coeffs)

            # 计算球谐基函数
            from .spherical_harmonics import eval_sh_basis

            sh_basis = eval_sh_basis(sh_degree, view_dir.unsqueeze(0))

            # 计算颜色
            rgb = torch.matmul(sh_basis, color_coeffs.t()).squeeze(0)
        else:
            # 退化为简单颜色
            rgb = (
                color_coeffs[:3]
                if color_coeffs.numel() >= 3
                else torch.ones(3, device=color_coeffs.device)
            )

        # 应用激活函数
        if self.rasterizer.config.color_activation == "sigmoid":
            rgb = torch.sigmoid(rgb)
        elif self.rasterizer.config.color_activation == "tanh":
            rgb = (torch.tanh(rgb) + 1) / 2
        elif self.rasterizer.config.color_activation == "clamp":
            rgb = torch.clamp(rgb, 0, 1)

        rgb = rgb[:3]  # 保证 shape 始终为 [3]
        return rgb

    def _log_render_stats(self):
        """记录渲染统计信息"""
        stats = self.render_stats
        logger.info(f"Render Stats:")
        logger.info(f"  Total voxels: {stats['total_voxels']}")
        logger.info(f"  Visible voxels: {stats['visible_voxels']}")
        logger.info(f"  Culled voxels: {stats['culled_voxels']}")
        logger.info(f"  Tile count: {stats['tile_count']}")
        logger.info(f"  Avg voxels per tile: {np.mean(stats['voxels_per_tile']):.1f}")
        logger.info(f"  Render time: {stats['render_time_ms']:.2f}ms")
        logger.info(f"  Projection time: {stats['projection_time_ms']:.2f}ms")
        logger.info(f"  Culling time: {stats['culling_time_ms']:.2f}ms")
        logger.info(f"  Sorting time: {stats['sorting_time_ms']:.2f}ms")
        logger.info(f"  Rasterization time: {stats['rasterization_time_ms']:.2f}ms")

    def render_batch(
        self,
        camera_poses: torch.Tensor,
        intrinsics: torch.Tensor,
        width: int | None = None,
        height: int | None = None,
    ) -> list[dict[str, torch.Tensor]]:
        """
        批量渲染多张图像
        """
        results = []

        logger.info(f"Rendering batch of {camera_poses.shape[0]} images")

        for i in tqdm(range(camera_poses.shape[0]), desc="Rendering images"):
            # 选择相机参数
            pose = camera_poses[i]
            intrinsic = intrinsics[i] if intrinsics.ndim > 2 else intrinsics

            # 渲染单张图像
            result = self.render_image(pose, intrinsic, width, height)
            results.append(result)

        return results

    def render_video(
        self,
        camera_trajectory: torch.Tensor,
        intrinsics: torch.Tensor,
        output_path: str,
        fps: int = 30,
        width: int | None = None,
        height: int | None = None,
    ) -> None:
        """
        渲染视频序列
        """
        if not IMAGEIO_AVAILABLE:
            raise ImportError(
                "imageio is required for video rendering. Install with: pip install imageio"
            )

        width = width or self.config.image_width
        height = height or self.config.image_height

        frames = []

        logger.info(f"Rendering {len(camera_trajectory)} frames for video...")

        for i, pose in enumerate(tqdm(camera_trajectory, desc="Rendering video frames")):
            result = self.render_image(pose, intrinsics, width, height)

            # 转换为 numpy 格式 (0-255)
            rgb_np = (result["rgb"].cpu().numpy() * 255).astype(np.uint8)
            frames.append(rgb_np)

        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # 保存视频
        imageio.mimsave(output_path, frames, fps=fps)
        logger.info(f"Video saved to {output_path}")

    def render_spiral_video(
        self,
        center: torch.Tensor,
        radius: float,
        num_frames: int,
        intrinsics: torch.Tensor,
        output_path: str,
        fps: int = 30,
        height_offset: float = 0.0,
        width: int | None = None,
        height: int | None = None,
    ) -> None:
        """
        渲染螺旋轨迹视频
        """
        # 生成螺旋轨迹
        trajectory = self._generate_spiral_trajectory(center, radius, num_frames, height_offset)

        # 渲染视频
        self.render_video(trajectory, intrinsics, output_path, fps, width, height)

    def save_renders(
        self, renders: list[dict[str, torch.Tensor]], output_dir: str, prefix: str = "render"
    ) -> None:
        """
        保存渲染结果到文件
        """
        os.makedirs(output_dir, exist_ok=True)

        for i, render in enumerate(renders):
            # 保存 RGB 图像
            rgb_path = os.path.join(output_dir, f"{prefix}_{i:04d}.png")
            self._save_image(render["rgb"], rgb_path)

            # 保存深度图（如果启用）
            if self.config.save_depth and "depth" in render:
                depth_path = os.path.join(output_dir, f"{prefix}_{i:04d}_depth.png")
                self._save_depth_image(render["depth"], depth_path)

            # 保存 alpha 通道（如果启用）
            if self.config.save_alpha and "alpha" in render:
                alpha_path = os.path.join(output_dir, f"{prefix}_{i:04d}_alpha.png")
                self._save_image(render["alpha"], alpha_path)

        logger.info(f"Saved {len(renders)} renders to {output_dir}")

    def _extract_voxels_from_model(self) -> dict[str, torch.Tensor]:
        """
        从模型中提取体素数据

        Returns:
            体素数据字典
        """
        # 这里需要从 SVRasterModel 中提取体素
        # 实际实现需要根据模型的具体结构调整
        with torch.no_grad():
            # 从 SVRasterModel 中提取体素
            if hasattr(self.model, "voxels"):
                voxels_obj = self.model.voxels

                # 提取所有层级的体素数据
                all_positions = []
                all_sizes = []
                all_densities = []
                all_colors = []
                all_morton_codes = []

                for level_idx in range(len(voxels_obj.voxel_positions)):
                    all_positions.append(voxels_obj.voxel_positions[level_idx])
                    all_sizes.append(voxels_obj.voxel_sizes[level_idx])
                    all_densities.append(voxels_obj.voxel_densities[level_idx])
                    all_colors.append(voxels_obj.voxel_colors[level_idx])
                    all_morton_codes.append(voxels_obj.voxel_morton_codes[level_idx])

                # 合并所有层级
                if all_positions:
                    positions = torch.cat(all_positions, dim=0)
                    sizes = torch.cat(all_sizes, dim=0)
                    densities = torch.cat(all_densities, dim=0)
                    colors = torch.cat(all_colors, dim=0)
                    morton_codes = torch.cat(all_morton_codes, dim=0)

                    return {
                        "positions": positions.float(),
                        "sizes": sizes.float(),
                        "densities": densities.float(),
                        "colors": colors.float(),
                        "morton_codes": morton_codes.long(),
                    }
                else:
                    return self._create_dummy_voxels()
            else:
                # 如果没有找到体素数据，创建一个简单的测试网格
                return self._create_dummy_voxels()

    def _create_dummy_voxels(self) -> dict[str, torch.Tensor]:
        """
        创建测试用的体素数据
        """
        # 创建一个简单的体素网格用于测试
        n_voxels = 100

        positions = torch.randn(n_voxels, 3, device=self.device) * 2.0
        sizes = torch.ones(n_voxels, device=self.device) * 0.1
        densities = torch.randn(n_voxels, device=self.device)
        colors = torch.randn(n_voxels, 3, device=self.device)
        morton_codes = torch.randint(0, 1000000, (n_voxels,), device=self.device)

        return {
            "positions": positions,
            "sizes": sizes,
            "densities": densities,
            "colors": colors,
            "morton_codes": morton_codes,
        }

    def _pose_to_camera_matrix(self, pose: torch.Tensor) -> torch.Tensor:
        """
        将相机位姿转换为相机矩阵

        Args:
            pose: 相机位姿矩阵 [4, 4]

        Returns:
            相机变换矩阵 [4, 4]
        """
        # 假设输入的 pose 是 world-to-camera 变换
        return pose

    def _generate_rays(
        self, camera_pose: torch.Tensor, intrinsics: torch.Tensor, width: int, height: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        根据相机参数生成光线
        """
        device = camera_pose.device

        # 生成像素坐标
        i, j = torch.meshgrid(
            torch.linspace(0, width - 1, width, device=device),
            torch.linspace(0, height - 1, height, device=device),
            indexing="ij",
        )
        i = i.t()
        j = j.t()

        # 提取内参
        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]

        # 计算归一化设备坐标
        dirs = torch.stack([(i - cx) / fx, -(j - cy) / fy, -torch.ones_like(i)], -1)

        # 转换到世界坐标系
        rotation = camera_pose[:3, :3]
        translation = camera_pose[:3, 3]

        rays_d = torch.sum(dirs[..., None, :] * rotation, -1)
        rays_o = translation.expand(rays_d.shape)

        return rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)

    def _generate_spiral_trajectory(
        self, center: torch.Tensor, radius: float, num_frames: int, height_offset: float = 0.0
    ) -> torch.Tensor:
        """
        生成螺旋相机轨迹
        """
        device = center.device

        # 生成角度
        angles = torch.linspace(0, 2 * np.pi, num_frames, device=device)

        # 生成位置
        positions = []
        for angle in angles:
            x = center[0] + radius * torch.cos(angle)
            y = center[1] + radius * torch.sin(angle)
            z = center[2] + height_offset
            positions.append(torch.tensor([x, y, z], device=device))

        # 生成相机姿态
        poses = []
        for pos in positions:
            # 计算朝向中心的旋转矩阵
            forward = F.normalize(center - pos, dim=0)
            up = torch.tensor([0, 0, 1], device=device, dtype=torch.float32)
            right = F.normalize(torch.cross(forward, up), dim=0)
            up = torch.cross(right, forward)

            # 构建变换矩阵
            pose = torch.eye(4, device=device, dtype=torch.float32)
            pose[:3, 0] = right
            pose[:3, 1] = up
            pose[:3, 2] = -forward
            pose[:3, 3] = pos

            poses.append(pose)

        return torch.stack(poses)

    def _save_image(self, image: torch.Tensor, path: str) -> None:
        """
        保存图像到文件
        """
        if not IMAGEIO_AVAILABLE:
            logger.warning("imageio not available, cannot save image")
            return

        # 转换为 numpy 格式
        if image.dim() == 3 and image.shape[2] == 1:
            image = image.squeeze(2)

        image_np = (image.cpu().numpy() * 255).astype(np.uint8)
        imageio.imwrite(path, image_np)

    def _save_depth_image(self, depth: torch.Tensor, path: str) -> None:
        """
        保存深度图到文件
        """
        if not IMAGEIO_AVAILABLE:
            logger.warning("imageio not available, cannot save depth image")
            return

        # 归一化深度值
        depth_np = depth.cpu().numpy().squeeze()
        depth_np = (depth_np - depth_np.min()) / (depth_np.max() - depth_np.min())
        depth_np = (depth_np * 255).astype(np.uint8)

        imageio.imwrite(path, depth_np)

    def get_memory_usage(self) -> dict[str, float]:
        """
        获取内存使用情况
        """
        if not torch.cuda.is_available():
            return {}

        return {
            "allocated_gb": torch.cuda.memory_allocated() / 1024**3,
            "reserved_gb": torch.cuda.memory_reserved() / 1024**3,
            "max_allocated_gb": torch.cuda.max_memory_allocated() / 1024**3,
        }

    def clear_cache(self) -> None:
        """
        清理缓存
        """
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 注意：VoxelRasterizer 当前没有 clear_cache 方法
        # 如果需要，可以在 VoxelRasterizer 中添加该方法


# 使用示例函数
def create_svraster_renderer(
    checkpoint_path: str,
    rasterizer_config: VoxelRasterizerConfig | None = None,
    renderer_config: SVRasterRendererConfig | None = None,
) -> SVRasterRenderer:
    """
    创建 SVRaster 渲染器的便捷函数
    """
    return SVRasterRenderer.from_checkpoint(
        checkpoint_path=checkpoint_path,
        rasterizer_config=rasterizer_config,
        renderer_config=renderer_config,
    )


def render_demo_images(
    renderer: SVRasterRenderer, num_views: int = 8, output_dir: str = "demo_renders"
) -> None:
    """
    渲染演示图像
    """
    # 创建演示相机轨迹
    center = torch.tensor([0.0, 0.0, 0.0], device=renderer.device)
    radius = 3.0

    # 生成相机姿态
    poses = renderer._generate_spiral_trajectory(center, radius, num_views)

    # 创建内参矩阵
    intrinsics = torch.tensor(
        [[800, 0, 400], [0, 800, 300], [0, 0, 1]], dtype=torch.float32, device=renderer.device
    )

    # 批量渲染
    renders = renderer.render_batch(poses, intrinsics)

    # 保存结果
    renderer.save_renders(renders, output_dir, "demo")

    logger.info(f"Demo rendering completed. Results saved to {output_dir}")


if __name__ == "__main__":
    # 使用示例
    print("SVRaster Renderer - 与 VoxelRasterizer 紧密耦合")
    print("主要功能：")
    print("1. 从检查点加载模型")
    print("2. 使用光栅化进行快速渲染")
    print("3. 支持图像、批量和视频渲染")
    print("4. 与 VoxelRasterizer 紧密耦合优化")
