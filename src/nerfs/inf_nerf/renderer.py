"""
InfNeRF Renderer - 与训练器解耦的推理渲染模块

这个渲染器专门用于推理阶段，与训练器完全解耦，
专注于高质量的图像渲染和视频生成。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Union, Any
from dataclasses import dataclass
from pathlib import Path
import logging
import json
from tqdm import tqdm
import os

logger = logging.getLogger(__name__)

# 尝试导入 imageio
try:
    import imageio

    IMAGEIO_AVAILABLE = True
except ImportError:
    IMAGEIO_AVAILABLE = False
    logger.warning("imageio not available for video rendering")

from .core import InfNeRF, InfNeRFConfig


@dataclass
class InfNeRFRendererConfig:
    """InfNeRF 渲染器配置 - 专门为推理设计"""

    # 渲染质量设置
    image_width: int = 800
    image_height: int = 600
    render_batch_size: int = 4096
    render_chunk_size: int = 1024

    # 渲染参数
    background_color: tuple[float, float, float] = (1.0, 1.0, 1.0)
    use_alpha_blending: bool = True
    depth_threshold: float = 1e-6

    # 渲染质量控制
    max_rays_per_batch: int = 8192
    use_hierarchical_sampling: bool = True

    # 渲染模式和输出设置
    render_mode: str = "volume_rendering"
    output_dir: str = "outputs/rendered"
    output_format: str = "png"
    save_depth: bool = False
    save_alpha: bool = False

    # 优化设置
    use_cached_features: bool = True
    enable_gradient_checkpointing: bool = False

    def __post_init__(self):
        """Post-initialization validation."""
        if self.image_width <= 0:
            raise ValueError("image_width must be positive")
        if self.image_height <= 0:
            raise ValueError("image_height must be positive")
        if self.render_batch_size <= 0:
            raise ValueError("render_batch_size must be positive")


class InfNeRFRenderer:
    """
    InfNeRF 渲染器 - 与训练器解耦

    专门负责：
    - 加载训练好的模型进行推理
    - 生成高质量的渲染图像
    - 支持批量渲染和视频生成
    - 提供多种渲染模式和输出格式
    """

    def __init__(self, model: InfNeRF, config: InfNeRFRendererConfig):
        self.model = model
        self.config = config

        # 确保模型处于评估模式
        self.model.eval()

        # 设置设备
        self.device = next(self.model.parameters()).device

        # 创建输出目录
        self.output_path = Path(self.config.output_dir)
        self.output_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"InfNeRFRenderer initialized on device: {self.device}")
        logger.info(f"Render config: {self.config}")

    @classmethod
    def from_checkpoint(
        cls, checkpoint_path: str, renderer_config: Optional[InfNeRFRendererConfig] = None
    ) -> InfNeRFRenderer:
        """
        从检查点加载渲染器
        """
        # 加载检查点
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        # 创建模型配置
        if "model_config" in checkpoint:
            model_config = checkpoint["model_config"]
        else:
            model_config = InfNeRFConfig()
            logger.warning("No model config found in checkpoint, using default")

        # 创建模型
        model = InfNeRF(model_config)
        model.load_state_dict(checkpoint["model_state_dict"])

        # 移到正确的设备
        if torch.cuda.is_available():
            model = model.cuda()

        # 创建渲染器配置
        if renderer_config is None:
            renderer_config = InfNeRFRendererConfig()

        logger.info(f"Loading renderer from checkpoint: {checkpoint_path}")

        return cls(model, renderer_config)

    def render_image(
        self,
        camera_pose: torch.Tensor,
        intrinsics: torch.Tensor,
        width: Optional[int] = None,
        height: Optional[int] = None,
    ) -> dict[str, torch.Tensor]:
        """
        渲染单张图像
        """
        width = width or self.config.image_width
        height = height or self.config.image_height

        with torch.no_grad():
            # 生成光线
            rays_o, rays_d = self._generate_rays(camera_pose, intrinsics, width, height)

            # 分块渲染以节省内存
            results = []
            for i in range(0, rays_o.shape[0], self.config.render_chunk_size):
                chunk_rays_o = rays_o[i : i + self.config.render_chunk_size]
                chunk_rays_d = rays_d[i : i + self.config.render_chunk_size]

                chunk_result = self.model(
                    rays_o=chunk_rays_o,
                    rays_d=chunk_rays_d,
                    near=0.1,
                    far=100.0,
                    focal_length=intrinsics[0, 0].item(),
                    pixel_width=1.0 / width,
                )
                results.append(chunk_result)

            # 合并结果
            rendered = {}
            for key in results[0].keys():
                rendered[key] = torch.cat([r[key] for r in results], dim=0)

            # 重塑为图像格式
            for key in rendered.keys():
                if key in ["rgb", "depth", "acc"]:
                    rendered[key] = rendered[key].view(height, width, -1)

            return rendered

    def render_batch(
        self,
        camera_poses: torch.Tensor,
        intrinsics: torch.Tensor,
        width: Optional[int] = None,
        height: Optional[int] = None,
    ) -> list[dict[str, torch.Tensor]]:
        """
        批量渲染多张图像
        """
        results = []
        for i in tqdm(range(len(camera_poses)), desc="Rendering batch"):
            result = self.render_image(camera_poses[i], intrinsics[i], width, height)
            results.append(result)
        return results

    def render_video(
        self,
        camera_trajectory: torch.Tensor,
        intrinsics: torch.Tensor,
        output_path: str,
        fps: int = 30,
        width: Optional[int] = None,
        height: Optional[int] = None,
    ) -> None:
        """
        渲染视频
        """
        if not IMAGEIO_AVAILABLE:
            raise ImportError("imageio required for video rendering")

        # 渲染所有帧
        frames = []
        for i in tqdm(range(len(camera_trajectory)), desc="Rendering video"):
            result = self.render_image(camera_trajectory[i], intrinsics, width, height)

            # 转换为图像格式
            rgb = result["rgb"].cpu().numpy()
            rgb = (rgb * 255).astype(np.uint8)
            frames.append(rgb)

        # 保存视频
        imageio.mimsave(output_path, frames, fps=fps)
        logger.info(f"Video saved to: {output_path}")

    def render_spiral_video(
        self,
        center: torch.Tensor,
        radius: float,
        num_frames: int,
        intrinsics: torch.Tensor,
        output_path: str,
        fps: int = 30,
        height_offset: float = 0.0,
        width: Optional[int] = None,
        height: Optional[int] = None,
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
        保存渲染结果
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        for i, render in enumerate(renders):
            # 保存RGB图像
            rgb = render["rgb"].cpu().numpy()
            rgb = (rgb * 255).astype(np.uint8)

            rgb_path = output_path / f"{prefix}_{i:04d}_rgb.png"
            self._save_image(rgb, str(rgb_path))

            # 保存深度图像
            if self.config.save_depth and "depth" in render:
                depth = render["depth"].cpu().numpy()
                depth_path = output_path / f"{prefix}_{i:04d}_depth.png"
                self._save_depth_image(depth, str(depth_path))

            # 保存alpha图像
            if self.config.save_alpha and "acc" in render:
                alpha = render["acc"].cpu().numpy()
                alpha_path = output_path / f"{prefix}_{i:04d}_alpha.png"
                self._save_image((alpha * 255).astype(np.uint8), str(alpha_path))

    def _generate_rays(
        self, camera_pose: torch.Tensor, intrinsics: torch.Tensor, width: int, height: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        生成光线
        """
        # 创建像素坐标网格
        i, j = torch.meshgrid(
            torch.arange(height, device=self.device),
            torch.arange(width, device=self.device),
            indexing="ij",
        )

        # 转换为相机坐标
        directions = torch.stack(
            [
                (j - intrinsics[0, 2]) / intrinsics[0, 0],
                (i - intrinsics[1, 2]) / intrinsics[1, 1],
                torch.ones_like(i),
            ],
            dim=-1,
        )

        # 转换到世界坐标
        directions = directions @ camera_pose[:3, :3].T
        origins = camera_pose[:3, -1].expand_as(directions)

        # 扁平化
        rays_o = origins.reshape(-1, 3)
        rays_d = F.normalize(directions.reshape(-1, 3), dim=-1)

        return rays_o, rays_d

    def _generate_spiral_trajectory(
        self, center: torch.Tensor, radius: float, num_frames: int, height_offset: float = 0.0
    ) -> torch.Tensor:
        """
        生成螺旋轨迹
        """
        t = torch.linspace(0, 2 * np.pi, num_frames, device=self.device)

        # 螺旋参数
        x = center[0] + radius * torch.cos(t)
        y = center[1] + radius * torch.sin(t)
        z = center[2] + height_offset

        # 创建相机姿态
        poses = []
        for i in range(num_frames):
            # 简单的相机姿态，始终看向中心
            pos = torch.tensor([x[i], y[i], z], device=self.device)
            forward = F.normalize(center - pos, dim=0)
            right = F.normalize(
                torch.cross(forward, torch.tensor([0, 0, 1], device=self.device)), dim=0
            )
            up = F.normalize(torch.cross(right, forward), dim=0)

            pose = torch.eye(4, device=self.device)
            pose[:3, 0] = right
            pose[:3, 1] = up
            pose[:3, 2] = forward
            pose[:3, 3] = pos

            poses.append(pose)

        return torch.stack(poses)

    def _save_image(self, image: np.ndarray, path: str) -> None:
        """保存图像"""
        if not IMAGEIO_AVAILABLE:
            logger.warning("imageio not available, skipping image save")
            return

        imageio.imwrite(path, image)

    def _save_depth_image(self, depth: np.ndarray, path: str) -> None:
        """保存深度图像"""
        if not IMAGEIO_AVAILABLE:
            logger.warning("imageio not available, skipping depth save")
            return

        # 归一化深度值
        depth_normalized = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        depth_image = (depth_normalized * 255).astype(np.uint8)

        imageio.imwrite(path, depth_image)

    def get_memory_usage(self) -> dict[str, float]:
        """获取内存使用情况"""
        if torch.cuda.is_available():
            return {
                "gpu_memory_allocated_gb": torch.cuda.memory_allocated() / 1024**3,
                "gpu_memory_reserved_gb": torch.cuda.memory_reserved() / 1024**3,
            }
        else:
            return {"cpu_memory_gb": 0.0}

    def clear_cache(self) -> None:
        """清理缓存"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def create_inf_nerf_renderer(
    checkpoint_path: str, renderer_config: Optional[InfNeRFRendererConfig] = None
) -> InfNeRFRenderer:
    """
    创建 InfNeRF 渲染器的便捷函数
    """
    return InfNeRFRenderer.from_checkpoint(checkpoint_path, renderer_config)


def render_demo_images(
    renderer: InfNeRFRenderer, num_views: int = 8, output_dir: str = "demo_renders"
) -> None:
    """
    渲染演示图像
    """
    # 创建简单的相机轨迹
    center = torch.tensor([0.0, 0.0, 0.0], device=renderer.device)
    radius = 2.0

    t = torch.linspace(0, 2 * np.pi, num_views, device=renderer.device)
    x = center[0] + radius * torch.cos(t)
    y = center[1] + radius * torch.sin(t)
    z = center[2] + torch.zeros_like(t)

    # 创建相机姿态
    poses = []
    for i in range(num_views):
        pos = torch.tensor([x[i], y[i], z[i]], device=renderer.device)
        forward = F.normalize(center - pos, dim=0)
        right = F.normalize(
            torch.cross(forward, torch.tensor([0, 0, 1], device=renderer.device)), dim=0
        )
        up = F.normalize(torch.cross(right, forward), dim=0)

        pose = torch.eye(4, device=renderer.device)
        pose[:3, 0] = right
        pose[:3, 1] = up
        pose[:3, 2] = forward
        pose[:3, 3] = pos

        poses.append(pose)

    poses = torch.stack(poses)

    # 简单的相机内参
    intrinsics = torch.tensor([[800, 0, 400], [0, 800, 300], [0, 0, 1]], device=renderer.device)

    # 渲染并保存
    renders = renderer.render_batch(poses, intrinsics)
    renderer.save_renders(renders, output_dir, "demo")
