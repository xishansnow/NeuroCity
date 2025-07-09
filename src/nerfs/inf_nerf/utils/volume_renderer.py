"""
InfNeRF Volume Renderer - 体积渲染工具模块

这个模块提供体积渲染的核心算法实现，与模型和训练器解耦。
专门用于训练阶段的体积渲染计算。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class VolumeRendererConfig:
    """体积渲染器配置"""

    # 采样参数
    num_samples: int = 64
    num_importance_samples: int = 128
    perturb: bool = True

    # 渲染参数
    white_background: bool = False
    background_color: tuple[float, float, float] = (1.0, 1.0, 1.0)

    # 损失权重
    lambda_rgb: float = 1.0
    lambda_depth: float = 0.1
    lambda_distortion: float = 0.01
    lambda_transparency: float = 1e-3

    def __post_init__(self):
        """Post-initialization validation."""
        if self.num_samples <= 0:
            raise ValueError("num_samples must be positive")
        if self.num_importance_samples <= 0:
            raise ValueError("num_importance_samples must be positive")


class VolumeRenderer:
    """
    体积渲染器 - 专门用于训练阶段的体积渲染
    """

    def __init__(self, config: VolumeRendererConfig):
        self.config = config

    def sample_rays(
        self,
        rays_o: torch.Tensor,
        rays_d: torch.Tensor,
        near: float,
        far: float,
        num_samples: Optional[int] = None,
        perturb: Optional[bool] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        沿光线采样点

        Args:
            rays_o: [N, 3] 光线起点
            rays_d: [N, 3] 光线方向
            near: 近平面距离
            far: 远平面距离
            num_samples: 采样点数量
            perturb: 是否添加扰动

        Returns:
            z_vals: [N, num_samples] 采样距离
            pts: [N, num_samples, 3] 采样点坐标
        """
        num_samples = num_samples or self.config.num_samples
        perturb = perturb if perturb is not None else self.config.perturb

        batch_size = rays_o.shape[0]
        device = rays_o.device

        # 分层采样
        t_vals = torch.linspace(0.0, 1.0, num_samples, device=device)
        z_vals = near + (far - near) * t_vals
        z_vals = z_vals.expand(batch_size, num_samples)

        # 添加扰动
        if perturb:
            mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
            upper = torch.cat([mids, z_vals[..., -1:]], dim=-1)
            lower = torch.cat([z_vals[..., :1], mids], dim=-1)
            t_rand = torch.rand_like(z_vals)
            z_vals = lower + (upper - lower) * t_rand

        # 计算采样点
        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]

        return z_vals, pts

    def volume_render(
        self,
        colors: torch.Tensor,
        densities: torch.Tensor,
        z_vals: torch.Tensor,
        rays_d: torch.Tensor,
        white_background: Optional[bool] = None,
    ) -> dict[str, torch.Tensor]:
        """
        体积渲染

        Args:
            colors: [N, num_samples, 3] RGB颜色
            densities: [N, num_samples, 1] 体积密度
            z_vals: [N, num_samples] 采样距离
            rays_d: [N, 3] 光线方向
            white_background: 是否使用白色背景

        Returns:
            渲染结果字典
        """
        white_background = (
            white_background if white_background is not None else self.config.white_background
        )

        # 计算采样点之间的距离
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([dists, torch.full_like(dists[..., :1], 1e10)], dim=-1)

        # 考虑光线方向的大小
        dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

        # 计算alpha值
        alpha = 1.0 - torch.exp(-densities[..., 0] * dists)

        # 计算透射率
        T = torch.cumprod(1.0 - alpha + 1e-10, dim=-1)
        T = torch.cat([torch.ones_like(T[..., :1]), T[..., :-1]], dim=-1)

        # 计算权重
        weights = alpha * T

        # 渲染RGB
        rgb = torch.sum(weights[..., None] * colors, dim=-2)

        # 渲染深度
        depth = torch.sum(weights * z_vals, dim=-1)

        # 渲染累积权重
        acc = torch.sum(weights, dim=-1)

        # 添加背景
        if white_background:
            rgb = rgb + (1.0 - acc[..., None])

        return {
            "rgb": rgb,
            "depth": depth,
            "acc": acc,
            "weights": weights,
            "z_vals": z_vals,
            "alpha": alpha,
        }

    def compute_losses(
        self,
        outputs: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
        model: Optional[Any] = None,
    ) -> dict[str, torch.Tensor]:
        """
        计算训练损失

        Args:
            outputs: 模型输出
            targets: 目标值
            model: 模型（用于正则化）

        Returns:
            损失字典
        """
        losses = {}

        # RGB重建损失
        if "rgb" in outputs and "target_rgb" in targets:
            rgb_loss = F.mse_loss(outputs["rgb"], targets["target_rgb"])
            losses["rgb_loss"] = rgb_loss

        # 深度损失
        if "depth" in outputs and "target_depth" in targets:
            depth_loss = F.l1_loss(outputs["depth"], targets["target_depth"])
            losses["depth_loss"] = depth_loss

        # 失真损失
        if "weights" in outputs and "z_vals" in outputs:
            distortion_loss = self._compute_distortion_loss(outputs["weights"], outputs["z_vals"])
            losses["distortion_loss"] = distortion_loss

        # 透明度损失
        if "alpha" in outputs:
            transparency_loss = self._compute_transparency_loss(outputs["alpha"])
            losses["transparency_loss"] = transparency_loss

        # 总损失
        device = outputs["rgb"].device
        total_loss = (
            self.config.lambda_rgb * losses.get("rgb_loss", torch.tensor(0.0, device=device))
            + self.config.lambda_depth * losses.get("depth_loss", torch.tensor(0.0, device=device))
            + self.config.lambda_distortion
            * losses.get("distortion_loss", torch.tensor(0.0, device=device))
            + self.config.lambda_transparency
            * losses.get("transparency_loss", torch.tensor(0.0, device=device))
        )

        losses["total_loss"] = total_loss

        # 计算PSNR
        if "rgb" in outputs and "target_rgb" in targets:
            with torch.no_grad():
                mse = F.mse_loss(outputs["rgb"], targets["target_rgb"])
                psnr = -10.0 * torch.log10(mse + 1e-8)
                losses["psnr"] = psnr

        return losses

    def _compute_distortion_loss(self, weights: torch.Tensor, z_vals: torch.Tensor) -> torch.Tensor:
        """
        计算失真损失（来自Mip-NeRF 360）
        """
        # 计算区间
        intervals = z_vals[..., 1:] - z_vals[..., :-1]
        mid_points = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])

        # 权重归一化
        w_normalized = weights / (weights.sum(-1, keepdim=True) + 1e-8)

        # 单峰性损失
        loss_uni = (weights[..., :-1] * intervals).sum(-1).mean()

        # 双峰性损失
        loss_bi_terms = []
        for i in range(weights.shape[-1]):
            for j in range(i + 1, weights.shape[-1]):
                term = (
                    w_normalized[..., i]
                    * w_normalized[..., j]
                    * torch.abs(mid_points[..., i] - mid_points[..., j]).sum(dim=-1).mean()
                )
                loss_bi_terms.append(term)

        loss_bi = (
            torch.stack(loss_bi_terms).mean()
            if loss_bi_terms
            else torch.tensor(0.0, device=weights.device)
        )

        return loss_uni + loss_bi

    def _compute_transparency_loss(self, alpha: torch.Tensor) -> torch.Tensor:
        """
        计算透明度损失，鼓励空空间的低密度
        """
        # 简单的透明度正则化
        transparency_loss = torch.mean(alpha**2)
        return transparency_loss


def create_volume_renderer(config: Optional[VolumeRendererConfig] = None) -> VolumeRenderer:
    """
    创建体积渲染器的便捷函数
    """
    if config is None:
        config = VolumeRendererConfig()
    return VolumeRenderer(config)
