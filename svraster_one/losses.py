"""
SVRaster One 损失函数

实现可微分光栅化渲染的损失函数，支持端到端训练。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class SVRasterOneLoss(nn.Module):
    """
    SVRaster One 损失函数

    包含多个损失组件：
    - RGB 重建损失
    - 深度损失
    - 密度正则化损失
    - 稀疏性损失
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # 损失权重
        self.rgb_weight = config.training.rgb_loss_weight
        self.depth_weight = config.training.depth_loss_weight
        self.density_reg_weight = config.training.density_reg_weight
        self.sparsity_weight = config.training.sparsity_weight

        # 损失函数
        self.rgb_loss_fn = nn.MSELoss()
        self.depth_loss_fn = nn.L1Loss()

        logger.info(
            f"Initialized SVRaster One loss with weights: "
            f"RGB={self.rgb_weight}, Depth={self.depth_weight}, "
            f"Density={self.density_reg_weight}, Sparsity={self.sparsity_weight}"
        )

    def forward(
        self,
        rendered_output: Dict[str, torch.Tensor],
        target_data: Dict[str, torch.Tensor],
        voxel_data: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        计算总损失

        Args:
            rendered_output: 渲染输出
            target_data: 目标数据
            voxel_data: 体素数据（用于正则化）

        Returns:
            损失字典
        """
        losses = {}

        # 1. RGB 重建损失
        if "rgb" in rendered_output and "rgb" in target_data:
            rgb_loss = self.rgb_loss_fn(rendered_output["rgb"], target_data["rgb"])
            losses["rgb_loss"] = rgb_loss

        # 2. 深度损失
        if "depth" in rendered_output and "depth" in target_data:
            depth_loss = self.depth_loss_fn(rendered_output["depth"], target_data["depth"])
            losses["depth_loss"] = depth_loss

        # 3. 密度正则化损失
        if voxel_data is not None and "densities" in voxel_data:
            density_reg_loss = self._compute_density_regularization(voxel_data["densities"])
            losses["density_reg_loss"] = density_reg_loss

        # 4. 稀疏性损失
        if voxel_data is not None and "densities" in voxel_data:
            sparsity_loss = self._compute_sparsity_loss(voxel_data["densities"])
            losses["sparsity_loss"] = sparsity_loss

        # 5. 计算总损失
        total_loss = self._compute_total_loss(losses)
        losses["total_loss"] = total_loss

        return losses

    def _compute_density_regularization(self, densities: torch.Tensor) -> torch.Tensor:
        """
        计算密度正则化损失

        鼓励密度值在合理范围内，避免过度稀疏或密集
        """
        # L2 正则化
        l2_loss = torch.mean(densities**2)

        # 稀疏性正则化（鼓励大部分体素密度接近0）
        sparsity_loss = torch.mean(torch.abs(densities))

        return l2_loss + 0.1 * sparsity_loss

    def _compute_sparsity_loss(self, densities: torch.Tensor) -> torch.Tensor:
        """
        计算稀疏性损失

        鼓励体素网格保持稀疏性，减少内存使用
        """
        # 使用 L1 正则化鼓励稀疏性
        sparsity_loss = torch.mean(torch.abs(densities))

        # 可选：使用 KL 散度鼓励稀疏性
        # target_density = torch.zeros_like(densities)
        # kl_loss = F.kl_div(
        #     F.log_softmax(densities, dim=0),
        #     F.softmax(target_density, dim=0),
        #     reduction='batchmean'
        # )

        return sparsity_loss

    def _compute_total_loss(self, losses: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        计算总损失
        """
        total_loss = 0.0

        # RGB 损失
        if "rgb_loss" in losses:
            total_loss += self.rgb_weight * losses["rgb_loss"]

        # 深度损失
        if "depth_loss" in losses:
            total_loss += self.depth_weight * losses["depth_loss"]

        # 密度正则化损失
        if "density_reg_loss" in losses:
            total_loss += self.density_reg_weight * losses["density_reg_loss"]

        # 稀疏性损失
        if "sparsity_loss" in losses:
            total_loss += self.sparsity_weight * losses["sparsity_loss"]

        return total_loss


class PerceptualLoss(nn.Module):
    """
    感知损失

    使用预训练网络提取特征，计算感知相似性
    """

    def __init__(self, feature_extractor: str = "vgg16"):
        super().__init__()
        self.feature_extractor = self._load_feature_extractor(feature_extractor)

        # 冻结特征提取器
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

    def _load_feature_extractor(self, name: str) -> nn.Module:
        """加载特征提取器"""
        if name == "vgg16":
            import torchvision.models as models

            vgg = models.vgg16(pretrained=True)
            # 移除分类层，只保留特征提取部分
            features = nn.Sequential(*list(vgg.features.children())[:23])  # 到 relu4_2
            return features
        else:
            raise ValueError(f"Unsupported feature extractor: {name}")

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        计算感知损失

        Args:
            pred: 预测图像 [B, 3, H, W]
            target: 目标图像 [B, 3, H, W]

        Returns:
            感知损失
        """
        # 确保输入格式正确
        if pred.dim() == 3:
            pred = pred.unsqueeze(0)  # [H, W, 3] -> [1, 3, H, W]
        if target.dim() == 3:
            target = target.unsqueeze(0)

        # 调整通道顺序
        if pred.shape[-1] == 3:  # [B, H, W, 3]
            pred = pred.permute(0, 3, 1, 2)  # [B, 3, H, W]
        if target.shape[-1] == 3:
            target = target.permute(0, 3, 1, 2)

        # 提取特征
        pred_features = self.feature_extractor(pred)
        target_features = self.feature_extractor(target)

        # 计算特征损失
        loss = F.mse_loss(pred_features, target_features)

        return loss


class SSIMLoss(nn.Module):
    """
    SSIM 损失

    结构相似性损失，更好地保持图像结构
    """

    def __init__(self, window_size: int = 11, sigma: float = 1.5):
        super().__init__()
        self.window_size = window_size
        self.sigma = sigma

        # 创建高斯窗口
        self.register_buffer("window", self._create_window(window_size, sigma))

    def _create_window(self, window_size: int, sigma: float) -> torch.Tensor:
        """创建高斯窗口"""

        def gaussian(window_size, sigma):
            gauss = torch.Tensor(
                [
                    torch.exp(torch.tensor(-((x - window_size // 2) ** 2) / float(2 * sigma**2)))
                    for x in range(window_size)
                ]
            )
            return gauss / gauss.sum()

        _1D_window = gaussian(window_size, sigma).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(3, 1, window_size, window_size).contiguous()
        return window

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        计算 SSIM 损失

        Args:
            pred: 预测图像 [B, 3, H, W] 或 [H, W, 3]
            target: 目标图像 [B, 3, H, W] 或 [H, W, 3]

        Returns:
            SSIM 损失 (1 - SSIM)
        """
        # 确保输入格式正确
        if pred.dim() == 3:
            pred = pred.unsqueeze(0)  # [H, W, 3] -> [1, 3, H, W]
        if target.dim() == 3:
            target = target.unsqueeze(0)

        # 调整通道顺序
        if pred.shape[-1] == 3:  # [B, H, W, 3]
            pred = pred.permute(0, 3, 1, 2)  # [B, 3, H, W]
        if target.shape[-1] == 3:
            target = target.permute(0, 3, 1, 2)

        # 计算 SSIM
        ssim_value = self._ssim(pred, target)

        # 返回损失 (1 - SSIM)
        return 1.0 - ssim_value

    def _ssim(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """计算 SSIM"""
        C1 = 0.01**2
        C2 = 0.03**2

        mu1 = F.conv2d(pred, self.window, padding=self.window_size // 2, groups=3)
        mu2 = F.conv2d(target, self.window, padding=self.window_size // 2, groups=3)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = (
            F.conv2d(pred * pred, self.window, padding=self.window_size // 2, groups=3) - mu1_sq
        )
        sigma2_sq = (
            F.conv2d(target * target, self.window, padding=self.window_size // 2, groups=3) - mu2_sq
        )
        sigma12 = (
            F.conv2d(pred * target, self.window, padding=self.window_size // 2, groups=3) - mu1_mu2
        )

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
            (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
        )

        return ssim_map.mean()


class CombinedLoss(nn.Module):
    """
    组合损失函数

    结合多种损失函数，提供更全面的训练目标
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # 基础损失
        self.base_loss = SVRasterOneLoss(config)

        # 感知损失（可选）
        self.use_perceptual = getattr(config.training, "use_perceptual_loss", False)
        if self.use_perceptual:
            self.perceptual_loss = PerceptualLoss()
            self.perceptual_weight = getattr(config.training, "perceptual_weight", 0.1)

        # SSIM 损失（可选）
        self.use_ssim = getattr(config.training, "use_ssim_loss", False)
        if self.use_ssim:
            self.ssim_loss = SSIMLoss()
            self.ssim_weight = getattr(config.training, "ssim_weight", 0.1)

    def forward(
        self,
        rendered_output: Dict[str, torch.Tensor],
        target_data: Dict[str, torch.Tensor],
        voxel_data: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        计算组合损失
        """
        # 基础损失
        losses = self.base_loss(rendered_output, target_data, voxel_data)

        # 感知损失
        if self.use_perceptual and "rgb" in rendered_output and "rgb" in target_data:
            perceptual_loss = self.perceptual_loss(rendered_output["rgb"], target_data["rgb"])
            losses["perceptual_loss"] = perceptual_loss
            losses["total_loss"] += self.perceptual_weight * perceptual_loss

        # SSIM 损失
        if self.use_ssim and "rgb" in rendered_output and "rgb" in target_data:
            ssim_loss = self.ssim_loss(rendered_output["rgb"], target_data["rgb"])
            losses["ssim_loss"] = ssim_loss
            losses["total_loss"] += self.ssim_weight * ssim_loss

        return losses
