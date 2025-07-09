"""
Instant NGP Renderer - 专门用于推理阶段

这个渲染器专门用于 Instant NGP 的推理阶段，使用优化的渲染策略
实现快速推理，与训练阶段的体积渲染分离。

渲染器负责：
1. 加载训练好的模型进行推理
2. 使用优化的采样策略进行快速渲染
3. 生成新视点的高质量图像
4. 支持批量渲染和实时渲染
5. 提供多种渲染模式和质量级别
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

logger = logging.getLogger(__name__)

# 尝试导入 imageio 和 OpenCV
try:
    import imageio

    IMAGEIO_AVAILABLE = True
except ImportError:
    IMAGEIO_AVAILABLE = False
    logger.warning("imageio not available for video rendering")

try:
    import cv2

    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False
    logger.warning("OpenCV not available for some rendering features")

from .core import InstantNGPModel, InstantNGPConfig


@dataclass
class InstantNGPRendererConfig:
    """Instant NGP 渲染器配置 - 专门为推理阶段设计"""

    # 渲染质量设置
    num_samples: int = 64  # 基础采样数量 (训练时通常128)
    num_samples_fine: int = 32  # 精细采样数量 (可选)
    density_threshold: float = 0.01  # 密度阈值

    # 渲染优化
    use_early_termination: bool = True  # 早期终止优化
    early_termination_threshold: float = 0.99  # 累积透明度阈值
    use_hierarchical_sampling: bool = True  # 层次采样

    # 输出设置
    background_color: tuple[float, float, float] = (1.0, 1.0, 1.0)  # 背景色
    output_alpha: bool = False  # 是否输出Alpha通道

    # 批处理设置
    chunk_size: int = 8192  # 批处理大小
    max_batch_size: int = 32768  # 最大批处理大小

    # 设备设置
    device: str = "auto"  # "auto", "cuda", "cpu"

    # 输出路径
    output_dir: str = "renders/instant_ngp"

    # 高级设置
    use_mixed_precision: bool = True  # 混合精度推理
    deterministic: bool = False  # 确定性渲染（关闭随机采样）


class InstantNGPRenderer:
    """
    Instant NGP 渲染器 - 专门用于推理阶段

    这个渲染器专注于推理阶段的优化，使用高效的渲染策略
    确保快速生成高质量图像，与训练阶段的体积渲染分离。
    """

    def __init__(
        self,
        model: InstantNGPModel,
        config: InstantNGPRendererConfig,
        device: torch.device | None = None,
    ):
        self.config = config
        self.model = model

        # 设备配置
        if device is not None:
            self.device = device
        elif config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(config.device)

        self.model = self.model.to(self.device)
        self.model.eval()  # 设置为评估模式

        # 创建输出目录
        os.makedirs(config.output_dir, exist_ok=True)

        logger.info(f"Instant NGP Renderer initialized on {self.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

    @torch.no_grad()
    def render_rays(
        self,
        rays_o: torch.Tensor,
        rays_d: torch.Tensor,
        near: torch.Tensor | None = None,
        far: torch.Tensor | None = None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """
        渲染光线 - 推理优化版本

        Args:
            rays_o: 光线起点 [N, 3]
            rays_d: 光线方向 [N, 3]
            near: 近距离 [N, 1]
            far: 远距离 [N, 1]

        Returns:
            Dict包含 rgb, depth, acc 等信息
        """
        N = rays_o.shape[0]

        # 默认边界
        if near is None:
            near = torch.full((N, 1), 0.01, device=self.device)
        if far is None:
            far = torch.full((N, 1), self.model.config.bound * 1.5, device=self.device)

        # 分批处理大量光线
        if N > self.config.chunk_size:
            return self._render_rays_chunked(rays_o, rays_d, near, far, **kwargs)
        else:
            return self._render_rays_core(rays_o, rays_d, near, far, **kwargs)

    def _render_rays_chunked(
        self,
        rays_o: torch.Tensor,
        rays_d: torch.Tensor,
        near: torch.Tensor,
        far: torch.Tensor,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """分批渲染光线"""
        N = rays_o.shape[0]
        results = {}

        for i in range(0, N, self.config.chunk_size):
            end_i = min(i + self.config.chunk_size, N)

            chunk_results = self._render_rays_core(
                rays_o[i:end_i], rays_d[i:end_i], near[i:end_i], far[i:end_i], **kwargs
            )

            # 合并结果
            for key, value in chunk_results.items():
                if key not in results:
                    results[key] = []
                results[key].append(value)

        # 连接所有结果
        return {key: torch.cat(values, dim=0) for key, values in results.items()}

    def _render_rays_core(
        self,
        rays_o: torch.Tensor,
        rays_d: torch.Tensor,
        near: torch.Tensor,
        far: torch.Tensor,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """核心光线渲染逻辑"""
        N = rays_o.shape[0]

        # 生成采样点 - 推理时使用较少采样点
        t_vals = torch.linspace(0.0, 1.0, self.config.num_samples, device=self.device)
        z_vals = near * (1.0 - t_vals) + far * t_vals  # [N, S]

        # 推理时不使用随机扰动（除非显式要求）
        if not self.config.deterministic and kwargs.get("perturb", False):
            mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
            upper = torch.cat([mids, z_vals[..., -1:]], dim=-1)
            lower = torch.cat([z_vals[..., :1], mids], dim=-1)
            t_rand = torch.rand_like(z_vals)
            z_vals = lower + (upper - lower) * t_rand

        # 计算3D点位置
        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]  # [N, S, 3]
        dirs = rays_d[..., None, :].expand_as(pts)  # [N, S, 3]

        # 模型前向传播
        pts_flat = pts.view(-1, 3)
        dirs_flat = dirs.view(-1, 3)

        # 使用混合精度加速推理
        if self.config.use_mixed_precision and self.device.type == "cuda":
            with torch.cuda.amp.autocast():
                raw_outputs = self.model(pts_flat, dirs_flat)
        else:
            raw_outputs = self.model(pts_flat, dirs_flat)

        raw_outputs = raw_outputs.view(N, self.config.num_samples, -1)

        # 分离密度和颜色
        density = raw_outputs[..., 0]  # [N, S]
        rgb = raw_outputs[..., 1:4]  # [N, S, 3]

        # 体积渲染积分
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([dists, torch.full_like(dists[..., :1], 1e10)], dim=-1)

        # Alpha合成
        alpha = 1.0 - torch.exp(-torch.relu(density) * dists)

        # 早期终止优化
        if self.config.use_early_termination:
            alpha = self._apply_early_termination(alpha)

        transmittance = torch.cumprod(
            torch.cat([torch.ones_like(alpha[..., :1]), 1.0 - alpha[..., :-1]], dim=-1), dim=-1
        )
        weights = alpha * transmittance  # [N, S]

        # 最终渲染结果
        rgb_final = torch.sum(weights[..., None] * rgb, dim=-2)  # [N, 3]
        depth = torch.sum(weights * z_vals, dim=-1)  # [N]
        acc = torch.sum(weights, dim=-1)  # [N]

        # 添加背景色
        bg_color = torch.tensor(self.config.background_color, device=self.device)
        rgb_final = rgb_final + (1.0 - acc[..., None]) * bg_color

        results = {
            "rgb": torch.clamp(rgb_final, 0.0, 1.0),
            "depth": depth,
            "acc": acc,
            "weights": weights,
            "z_vals": z_vals,
        }

        # 可选的层次采样
        if self.config.use_hierarchical_sampling and self.config.num_samples_fine > 0:
            fine_results = self._hierarchical_sampling(rays_o, rays_d, z_vals, weights, **kwargs)
            # 合并粗糙和精细采样的结果
            results.update(fine_results)

        return results

    def _apply_early_termination(self, alpha: torch.Tensor) -> torch.Tensor:
        """应用早期终止优化"""
        # 计算累积透明度
        cumulative_alpha = torch.cumsum(alpha, dim=-1)

        # 当累积透明度超过阈值时，后续采样点的alpha设为0
        mask = cumulative_alpha > self.config.early_termination_threshold
        alpha = alpha * (~mask).float()

        return alpha

    def _hierarchical_sampling(
        self,
        rays_o: torch.Tensor,
        rays_d: torch.Tensor,
        z_vals: torch.Tensor,
        weights: torch.Tensor,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """层次采样 - 在重要区域进行精细采样"""
        N = rays_o.shape[0]

        # 基于权重进行重要性采样
        weights_pad = torch.cat([weights, torch.full_like(weights[..., :1], 1e-5)], dim=-1)

        # 创建累积分布函数
        pdf = weights_pad / torch.sum(weights_pad, dim=-1, keepdim=True)
        cdf = torch.cumsum(pdf, dim=-1)
        cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], dim=-1)

        # 均匀采样
        u = torch.linspace(0.0, 1.0, self.config.num_samples_fine, device=self.device)
        u = u.expand(N, self.config.num_samples_fine)

        if not self.config.deterministic:
            u = u + torch.rand_like(u) / self.config.num_samples_fine

        # 逆变换采样
        inds = torch.searchsorted(cdf, u, right=True)
        below = torch.max(torch.zeros_like(inds - 1), inds - 1)
        above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)

        # 线性插值获取精细采样点
        cdf_g0 = torch.gather(cdf, -1, below)
        cdf_g1 = torch.gather(cdf, -1, above)

        bins_g0 = torch.gather(z_vals, -1, below)
        bins_g1 = torch.gather(z_vals, -1, above)

        denom = cdf_g1 - cdf_g0
        denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
        t = (u - cdf_g0) / denom

        z_vals_fine = bins_g0 + t * (bins_g1 - bins_g0)

        # 合并粗糙和精细采样点
        z_vals_combined, _ = torch.sort(torch.cat([z_vals, z_vals_fine], dim=-1), dim=-1)

        # 在合并的采样点上进行渲染
        pts_fine = rays_o[..., None, :] + rays_d[..., None, :] * z_vals_combined[..., :, None]
        dirs_fine = rays_d[..., None, :].expand_as(pts_fine)

        pts_fine_flat = pts_fine.view(-1, 3)
        dirs_fine_flat = dirs_fine.view(-1, 3)

        # 前向传播
        if self.config.use_mixed_precision and self.device.type == "cuda":
            with torch.cuda.amp.autocast():
                raw_outputs_fine = self.model(pts_fine_flat, dirs_fine_flat)
        else:
            raw_outputs_fine = self.model(pts_fine_flat, dirs_fine_flat)

        raw_outputs_fine = raw_outputs_fine.view(N, -1, raw_outputs_fine.shape[-1])

        # 体积渲染
        density_fine = raw_outputs_fine[..., 0]
        rgb_fine = raw_outputs_fine[..., 1:4]

        dists_fine = z_vals_combined[..., 1:] - z_vals_combined[..., :-1]
        dists_fine = torch.cat([dists_fine, torch.full_like(dists_fine[..., :1], 1e10)], dim=-1)

        alpha_fine = 1.0 - torch.exp(-torch.relu(density_fine) * dists_fine)

        if self.config.use_early_termination:
            alpha_fine = self._apply_early_termination(alpha_fine)

        transmittance_fine = torch.cumprod(
            torch.cat([torch.ones_like(alpha_fine[..., :1]), 1.0 - alpha_fine[..., :-1]], dim=-1),
            dim=-1,
        )
        weights_fine = alpha_fine * transmittance_fine

        # 最终精细渲染结果
        rgb_fine_final = torch.sum(weights_fine[..., None] * rgb_fine, dim=-2)
        depth_fine = torch.sum(weights_fine * z_vals_combined, dim=-1)
        acc_fine = torch.sum(weights_fine, dim=-1)

        # 添加背景色
        bg_color = torch.tensor(self.config.background_color, device=self.device)
        rgb_fine_final = rgb_fine_final + (1.0 - acc_fine[..., None]) * bg_color

        return {
            "rgb_fine": torch.clamp(rgb_fine_final, 0.0, 1.0),
            "depth_fine": depth_fine,
            "acc_fine": acc_fine,
            "weights_fine": weights_fine,
            "z_vals_fine": z_vals_combined,
        }

    def render_image(
        self, camera_pose: torch.Tensor, intrinsics: torch.Tensor, width: int, height: int, **kwargs
    ) -> dict[str, torch.Tensor]:
        """
        渲染单张图像

        Args:
            camera_pose: 相机姿态 [4, 4]
            intrinsics: 相机内参 [3, 3]
            width: 图像宽度
            height: 图像高度

        Returns:
            Dict包含渲染结果
        """
        self.model.eval()

        # 生成光线
        rays_o, rays_d = self._generate_rays(camera_pose, intrinsics, width, height)

        # 渲染光线
        results = self.render_rays(rays_o, rays_d, **kwargs)

        # 重新整理为图像格式
        for key in results:
            if results[key].dim() == 2:
                results[key] = results[key].view(height, width, -1)
            elif results[key].dim() == 1:
                results[key] = results[key].view(height, width)

        return results

    def _generate_rays(
        self, camera_pose: torch.Tensor, intrinsics: torch.Tensor, width: int, height: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """生成相机光线"""
        # 生成像素坐标
        i, j = torch.meshgrid(
            torch.arange(width, device=self.device),
            torch.arange(height, device=self.device),
            indexing="xy",
        )

        # 转换为相机坐标
        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]

        dirs = torch.stack(
            [(i - cx) / fx, -(j - cy) / fy, -torch.ones_like(i)], dim=-1  # 负号因为图像y轴向下
        ).float()

        # 转换到世界坐标
        rays_d = torch.sum(dirs[..., None, :] * camera_pose[:3, :3], dim=-1)
        rays_o = camera_pose[:3, 3].expand(rays_d.shape)

        # 归一化方向
        rays_d = F.normalize(rays_d, dim=-1)

        # 展平为光线批次
        rays_o = rays_o.view(-1, 3)
        rays_d = rays_d.view(-1, 3)

        return rays_o, rays_d

    def render_video(
        self,
        camera_poses: list[torch.Tensor],
        intrinsics: torch.Tensor,
        width: int,
        height: int,
        output_path: str,
        fps: int = 30,
        **kwargs,
    ):
        """
        渲染视频序列

        Args:
            camera_poses: 相机姿态列表
            intrinsics: 相机内参
            width: 视频宽度
            height: 视频高度
            output_path: 输出路径
            fps: 帧率
        """
        if not IMAGEIO_AVAILABLE:
            raise ImportError("imageio is required for video rendering")

        self.model.eval()

        frames = []
        for i, pose in enumerate(tqdm(camera_poses, desc="Rendering video")):
            result = self.render_image(pose, intrinsics, width, height, **kwargs)

            # 转换为图像
            rgb_image = result["rgb"].cpu().numpy()
            rgb_image = (rgb_image * 255).astype(np.uint8)

            frames.append(rgb_image)

        # 保存视频
        imageio.mimsave(output_path, frames, fps=fps)
        logger.info(f"Video saved to {output_path}")

    def render_depth_map(
        self, camera_pose: torch.Tensor, intrinsics: torch.Tensor, width: int, height: int, **kwargs
    ) -> np.ndarray:
        """渲染深度图"""
        result = self.render_image(camera_pose, intrinsics, width, height, **kwargs)

        depth = result["depth"].cpu().numpy()

        # 归一化深度值
        depth_normalized = (depth - depth.min()) / (depth.max() - depth.min())

        return depth_normalized

    def save_checkpoint(self, filepath: str):
        """保存渲染器状态"""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "config": self.config,
            "model_config": self.model.config,
        }

        torch.save(checkpoint, filepath)
        logger.info(f"Renderer checkpoint saved: {filepath}")

    @classmethod
    def load_checkpoint(
        cls,
        filepath: str,
        model_class: type = InstantNGPModel,
        device: torch.device | None = None,
    ) -> "InstantNGPRenderer":
        """加载渲染器检查点"""
        checkpoint = torch.load(filepath, map_location=device)

        # 重建模型
        model = model_class(checkpoint["model_config"])
        model.load_state_dict(checkpoint["model_state_dict"])

        # 重建渲染器
        renderer = cls(model, checkpoint["config"], device=device)

        logger.info(f"Renderer checkpoint loaded: {filepath}")
        return renderer
