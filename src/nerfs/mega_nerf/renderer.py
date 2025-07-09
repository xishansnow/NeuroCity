"""
Mega-NeRF Renderer Module

This module contains the rendering pipeline for Mega-NeRF models including:
- Renderer configuration
- Main renderer class
- Volume rendering for inference
- Image and video generation
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
from typing import Optional
from pathlib import Path
import logging
import json
import os

logger = logging.getLogger(__name__)

# 尝试导入可选依赖
try:
    from tqdm import tqdm

    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    logger.warning("tqdm not available, progress bars will be disabled")

try:
    import imageio

    IMAGEIO_AVAILABLE = True
except ImportError:
    IMAGEIO_AVAILABLE = False
    logger.warning("imageio not available for image/video saving")

try:
    import cv2

    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    logger.warning("cv2 not available for image processing")


@dataclass
class MegaNeRFRendererConfig:
    """Mega-NeRF 渲染器配置"""

    # 渲染质量设置
    image_width: int = 800
    image_height: int = 600
    render_batch_size: int = 4096
    render_chunk_size: int = 1024

    # 渲染参数
    background_color: tuple[float, float, float] = (1.0, 1.0, 1.0)
    use_alpha_blending: bool = True
    depth_threshold: float = 1e-6

    # 采样参数
    num_coarse_samples: int = 256
    num_fine_samples: int = 512
    near: float = 0.1
    far: float = 1000.0
    use_hierarchical_sampling: bool = True

    # 渲染模式和输出设置
    render_mode: str = "volume_rendering"  # "volume_rendering", "rasterization"
    output_dir: str = "outputs/rendered"
    output_format: str = "png"
    save_depth: bool = False
    save_alpha: bool = False

    # 优化设置
    use_cached_features: bool = True
    enable_gradient_checkpointing: bool = False

    # 视频设置
    video_fps: int = 30
    video_quality: int = 95

    def __post_init__(self):
        """后初始化验证"""
        if self.image_width <= 0:
            raise ValueError("image_width must be positive")
        if self.image_height <= 0:
            raise ValueError("image_height must be positive")
        if self.render_batch_size <= 0:
            raise ValueError("render_batch_size must be positive")
        if self.num_coarse_samples <= 0:
            raise ValueError("num_coarse_samples must be positive")
        if self.num_fine_samples <= 0:
            raise ValueError("num_fine_samples must be positive")


class MegaNeRFRenderer:
    """
    Mega-NeRF 渲染器 - 与训练器解耦

    专门负责：
    - 加载训练好的模型进行推理
    - 生成高质量的渲染图像
    - 支持批量渲染和视频生成
    - 提供多种渲染模式和输出格式
    """

    def __init__(self, model: "MegaNeRF", config: MegaNeRFRendererConfig):
        self.model = model
        self.config = config

        # 确保模型处于评估模式
        self.model.eval()

        # 设置设备
        self.device = next(self.model.parameters()).device

        # 创建输出目录
        self.output_path = Path(self.config.output_dir)
        self.output_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"MegaNeRFRenderer initialized on device: {self.device}")
        logger.info(f"Render config: {self.config}")

    @classmethod
    def from_checkpoint(
        cls, checkpoint_path: str, renderer_config: MegaNeRFRendererConfig | None = None
    ) -> MegaNeRFRenderer:
        """
        从检查点加载渲染器
        """
        # 加载检查点
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        # 创建模型配置
        if "model_config" in checkpoint:
            model_config = checkpoint["model_config"]
        else:
            from .core import MegaNeRFConfig

            model_config = MegaNeRFConfig()
            logger.warning("No model config found in checkpoint, using default")

        # 创建模型
        from .core import MegaNeRF

        model = MegaNeRF(model_config)
        model.load_state_dict(checkpoint["model_state_dict"])

        # 移到正确的设备
        if torch.cuda.is_available():
            model = model.cuda()

        # 创建渲染器配置
        if renderer_config is None:
            renderer_config = MegaNeRFRendererConfig()

        logger.info(f"Loading renderer from checkpoint: {checkpoint_path}")

        return cls(model, renderer_config)

    def render_image(
        self,
        camera_pose: torch.Tensor,
        intrinsics: torch.Tensor,
        width: int | None = None,
        height: int | None = None,
        appearance_idx: int | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        渲染单张图像

        Args:
            camera_pose: [4, 4] 相机姿态矩阵
            intrinsics: [3, 3] 相机内参矩阵
            width: 图像宽度
            height: 图像高度
            appearance_idx: 外观嵌入索引

        Returns:
            渲染结果字典
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

                chunk_result = self._render_rays(chunk_rays_o, chunk_rays_d, appearance_idx)
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
        width: int | None = None,
        height: int | None = None,
        appearance_idx: int | None = None,
    ) -> list[dict[str, torch.Tensor]]:
        """
        批量渲染多张图像

        Args:
            camera_poses: [N, 4, 4] 相机姿态矩阵
            intrinsics: [3, 3] 相机内参矩阵
            width: 图像宽度
            height: 图像高度
            appearance_idx: 外观嵌入索引

        Returns:
            渲染结果列表
        """
        results = []

        # 创建进度条
        if TQDM_AVAILABLE:
            pbar = tqdm(range(len(camera_poses)), desc="Rendering batch")
        else:
            pbar = range(len(camera_poses))

        for i in pbar:
            result = self.render_image(camera_poses[i], intrinsics, width, height, appearance_idx)
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
        appearance_idx: int | None = None,
    ) -> None:
        """
        渲染视频

        Args:
            camera_trajectory: [N, 4, 4] 相机轨迹
            intrinsics: [3, 3] 相机内参矩阵
            output_path: 输出视频路径
            fps: 帧率
            width: 图像宽度
            height: 图像高度
            appearance_idx: 外观嵌入索引
        """
        if not IMAGEIO_AVAILABLE:
            raise ImportError("imageio required for video rendering")

        # 渲染所有帧
        frames = []

        if TQDM_AVAILABLE:
            pbar = tqdm(range(len(camera_trajectory)), desc="Rendering video")
        else:
            pbar = range(len(camera_trajectory))

        for i in pbar:
            result = self.render_image(
                camera_trajectory[i], intrinsics, width, height, appearance_idx
            )

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
        width: int | None = None,
        height: int | None = None,
        appearance_idx: int | None = None,
    ) -> None:
        """
        渲染螺旋轨迹视频

        Args:
            center: 螺旋中心点
            radius: 螺旋半径
            num_frames: 帧数
            intrinsics: 相机内参矩阵
            output_path: 输出视频路径
            fps: 帧率
            height_offset: 高度偏移
            width: 图像宽度
            height: 图像高度
            appearance_idx: 外观嵌入索引
        """
        # 生成螺旋轨迹
        trajectory = self._generate_spiral_trajectory(center, radius, num_frames, height_offset)

        # 渲染视频
        self.render_video(trajectory, intrinsics, output_path, fps, width, height, appearance_idx)

    def save_renders(
        self, renders: list[dict[str, torch.Tensor]], output_dir: str, prefix: str = "render"
    ) -> None:
        """
        保存渲染结果

        Args:
            renders: 渲染结果列表
            output_dir: 输出目录
            prefix: 文件前缀
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

    def _render_rays(
        self,
        rays_o: torch.Tensor,
        rays_d: torch.Tensor,
        appearance_idx: int | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        渲染光线

        Args:
            rays_o: [N, 3] 光线原点
            rays_d: [N, 3] 光线方向
            appearance_idx: 外观嵌入索引

        Returns:
            渲染结果
        """
        # 分层采样
        if self.config.use_hierarchical_sampling:
            # 粗采样
            t_coarse = torch.linspace(
                self.config.near,
                self.config.far,
                self.config.num_coarse_samples,
                device=self.device,
            )
            t_coarse = t_coarse.expand(rays_o.shape[0], -1)

            # 添加噪声
            t_coarse = (
                t_coarse
                + torch.rand_like(t_coarse)
                * (self.config.far - self.config.near)
                / self.config.num_coarse_samples
            )

            # 采样点
            points_coarse = rays_o.unsqueeze(1) + t_coarse.unsqueeze(-1) * rays_d.unsqueeze(1)

            # 查询模型
            density_coarse, color_coarse = self.model(
                points_coarse.reshape(-1, 3),
                rays_d.unsqueeze(1).expand(-1, self.config.num_coarse_samples, -1).reshape(-1, 3),
                appearance_idx,
            )

            density_coarse = density_coarse.view(rays_o.shape[0], self.config.num_coarse_samples)
            color_coarse = color_coarse.view(rays_o.shape[0], self.config.num_coarse_samples, 3)

            # 体积渲染
            weights_coarse = self._compute_weights(density_coarse, t_coarse)
            rgb_coarse = torch.sum(weights_coarse.unsqueeze(-1) * color_coarse, dim=1)
            depth_coarse = torch.sum(weights_coarse * t_coarse, dim=1)
            acc_coarse = torch.sum(weights_coarse, dim=1)

            # 细采样
            t_fine = self._importance_sampling(weights_coarse, t_coarse)
            points_fine = rays_o.unsqueeze(1) + t_fine.unsqueeze(-1) * rays_d.unsqueeze(1)

            # 查询模型
            density_fine, color_fine = self.model(
                points_fine.reshape(-1, 3),
                rays_d.unsqueeze(1).expand(-1, self.config.num_fine_samples, -1).reshape(-1, 3),
                appearance_idx,
            )

            density_fine = density_fine.view(rays_o.shape[0], self.config.num_fine_samples)
            color_fine = color_fine.view(rays_o.shape[0], self.config.num_fine_samples, 3)

            # 体积渲染
            weights_fine = self._compute_weights(density_fine, t_fine)
            rgb_fine = torch.sum(weights_fine.unsqueeze(-1) * color_fine, dim=1)
            depth_fine = torch.sum(weights_fine * t_fine, dim=1)
            acc_fine = torch.sum(weights_fine, dim=1)

            return {"rgb": rgb_fine, "depth": depth_fine, "acc": acc_fine}
        else:
            # 简单均匀采样
            t = torch.linspace(
                self.config.near,
                self.config.far,
                self.config.num_coarse_samples,
                device=self.device,
            )
            t = t.expand(rays_o.shape[0], -1)

            points = rays_o.unsqueeze(1) + t.unsqueeze(-1) * rays_d.unsqueeze(1)

            density, color = self.model(
                points.reshape(-1, 3),
                rays_d.unsqueeze(1).expand(-1, self.config.num_coarse_samples, -1).reshape(-1, 3),
                appearance_idx,
            )

            density = density.view(rays_o.shape[0], self.config.num_coarse_samples)
            color = color.view(rays_o.shape[0], self.config.num_coarse_samples, 3)

            weights = self._compute_weights(density, t)
            rgb = torch.sum(weights.unsqueeze(-1) * color, dim=1)
            depth = torch.sum(weights * t, dim=1)
            acc = torch.sum(weights, dim=1)

            return {"rgb": rgb, "depth": depth, "acc": acc}

    def _compute_weights(self, density: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """计算体积渲染权重"""
        # 计算距离
        dists = t[..., 1:] - t[..., :-1]
        dists = torch.cat(
            [dists, torch.tensor([1e10], device=self.device).expand(dists[..., :1].shape)], dim=-1
        )

        # 计算透明度
        alpha = 1.0 - torch.exp(-density * dists)

        # 计算权重
        weights = (
            alpha
            * torch.cumprod(
                torch.cat(
                    [torch.ones((alpha.shape[0], 1), device=self.device), 1.0 - alpha + 1e-10],
                    dim=-1,
                ),
                dim=-1,
            )[:, :-1]
        )

        return weights

    def _importance_sampling(self, weights: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """重要性采样"""
        # 计算累积分布函数
        cdf = torch.cumsum(weights, dim=-1)
        cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], dim=-1)

        # 采样
        u = torch.rand(weights.shape[0], self.config.num_fine_samples, device=self.device)
        u = u.contiguous()

        # 二分查找
        inds = torch.searchsorted(cdf, u, right=True)
        below = torch.max(torch.zeros_like(inds - 1), inds - 1)
        above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
        inds_g = torch.stack([below, above], dim=-1)

        cdf_g = torch.gather(cdf, 1, inds_g)
        bins_g = torch.gather(t, 1, inds_g)

        denom = cdf_g[..., 1] - cdf_g[..., 0]
        denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
        t_n = (u - cdf_g[..., 0]) / denom
        samples = bins_g[..., 0] + t_n * (bins_g[..., 1] - bins_g[..., 0])

        return samples

    def _generate_rays(
        self, camera_pose: torch.Tensor, intrinsics: torch.Tensor, width: int, height: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        生成光线

        Args:
            camera_pose: [4, 4] 相机姿态矩阵
            intrinsics: [3, 3] 相机内参矩阵
            width: 图像宽度
            height: 图像高度

        Returns:
            rays_o: [N, 3] 光线原点
            rays_d: [N, 3] 光线方向
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

        Args:
            center: 螺旋中心点
            radius: 螺旋半径
            num_frames: 帧数
            height_offset: 高度偏移

        Returns:
            trajectory: [num_frames, 4, 4] 相机轨迹
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


def create_mega_nerf_renderer(
    checkpoint_path: str, renderer_config: MegaNeRFRendererConfig | None = None
) -> MegaNeRFRenderer:
    """
    创建 Mega-NeRF 渲染器的便捷函数
    """
    return MegaNeRFRenderer.from_checkpoint(checkpoint_path, renderer_config)


def render_demo_images(
    renderer: MegaNeRFRenderer, num_views: int = 8, output_dir: str = "demo_renders"
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
