"""
Instant NGP Trainer - 专门用于训练阶段

这个训练器专门用于 Instant NGP 的训练阶段，使用体积渲染进行训练，
与推理阶段的优化渲染器分离，确保训练和推理的专业化。

训练器负责：
1. 管理训练循环和优化过程
2. 使用体积渲染进行高质量训练
3. 计算损失和梯度更新
4. 监控训练进度和保存检查点
5. 管理多分辨率哈希编码的学习率调度
"""

from __future__ import annotations

import os
import time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp.grad_scaler import GradScaler
from torch.amp.autocast_mode import autocast
from contextlib import nullcontext
import numpy as np
from dataclasses import dataclass
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Optional TensorBoard import
try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    logger.warning("TensorBoard not available. Install with: pip install tensorboard")

from .core import InstantNGPModel, InstantNGPConfig, InstantNGPLoss


@dataclass
class InstantNGPTrainerConfig:
    """Instant NGP 训练器配置 - 专门为训练阶段设计"""

    # 训练参数
    num_epochs: int = 20
    batch_size: int = 8192  # Instant NGP uses large ray batches
    learning_rate: float = 1e-2
    learning_rate_hash: float = 1e-1  # Higher LR for hash encoding
    weight_decay: float = 1e-6

    # 学习率调度
    decay_step: int = 1000
    learning_rate_decay: float = 0.33
    min_learning_rate: float = 1e-4

    # 采样参数
    num_samples: int = 128  # Samples per ray for training
    num_samples_importance: int = 0  # Additional importance samples
    perturb: bool = True  # Perturb sample positions

    # 训练策略
    use_mixed_precision: bool = True
    gradient_clip_norm: float = 1.0

    # 哈希编码特定参数
    hash_grid_update_freq: int = 100  # Hash grid pruning frequency
    density_threshold: float = 0.01  # Density threshold for pruning

    # 监控和保存
    log_freq: int = 100
    eval_freq: int = 1000
    save_freq: int = 5000
    render_freq: int = 2000

    # 输出配置
    log_dir: str = "logs/instant_ngp"
    checkpoint_dir: str = "checkpoints/instant_ngp"
    render_dir: str = "renders/instant_ngp"

    # 设备配置
    device: str = "auto"  # "auto", "cuda", "cpu"


class InstantNGPTrainer:
    """
    Instant NGP 训练器 - 专门用于训练阶段

    这个训练器专注于训练阶段的优化，使用体积渲染确保训练质量，
    而推理阶段则使用专门的快速渲染器。
    """

    def __init__(
        self,
        model: InstantNGPModel,
        config: InstantNGPTrainerConfig,
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

        # 损失函数
        self.criterion = InstantNGPLoss(self.model.config)

        # 优化器设置 - 对哈希编码使用更高的学习率
        self._setup_optimizer()

        # 学习率调度器
        self._setup_scheduler()

        # 混合精度训练
        if config.use_mixed_precision and self.device.type == "cuda":
            self.scaler = GradScaler()
            self.autocast_context = autocast
        else:
            self.scaler = None
            self.autocast_context = nullcontext

        # 训练状态
        self.global_step = 0
        self.epoch = 0
        self.best_psnr = 0.0

        # 日志记录
        self.writer = None
        if TENSORBOARD_AVAILABLE:
            os.makedirs(config.log_dir, exist_ok=True)
            self.writer = SummaryWriter(config.log_dir)

    def _setup_optimizer(self):
        """设置优化器 - 对哈希编码使用更高学习率"""
        # 分离哈希编码参数和MLP参数
        hash_param_ids = {id(p) for p in self.model.encoding.parameters()}

        hash_params = []
        mlp_params = []

        for name, param in self.model.named_parameters():
            if id(param) in hash_param_ids:
                hash_params.append(param)
            else:
                mlp_params.append(param)

        # 创建参数组
        param_groups = [
            {"params": hash_params, "lr": self.config.learning_rate_hash, "name": "hash_encoding"},
            {"params": mlp_params, "lr": self.config.learning_rate, "name": "mlp_networks"},
        ]

        self.optimizer = optim.Adam(param_groups, weight_decay=self.config.weight_decay)

        logger.info(f"Hash encoding params: {len(hash_params)}")
        logger.info(f"MLP params: {len(mlp_params)}")

    def _setup_scheduler(self):
        """设置学习率调度器"""
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=self.config.decay_step, gamma=self.config.learning_rate_decay
        )

    def train_step(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        """单步训练"""
        self.model.train()

        # 移动数据到设备
        rays_o = batch["rays_o"].to(self.device)  # [N, 3]
        rays_d = batch["rays_d"].to(self.device)  # [N, 3]
        targets = batch["rgb"].to(self.device)  # [N, 3]

        # 边界设置
        near = torch.full_like(rays_o[..., :1], 0.01)
        far = torch.full_like(rays_o[..., :1], self.model.config.bound * 1.5)

        # 前向传播
        with self.autocast_context():
            # 体积渲染
            outputs = self._volume_render(rays_o, rays_d, near, far)

            # 计算损失
            loss_dict = self.criterion(outputs, targets)
            total_loss = loss_dict["total_loss"]

        # 反向传播
        self.optimizer.zero_grad()

        if self.scaler is not None:
            self.scaler.scale(total_loss).backward()
            if self.config.gradient_clip_norm > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.gradient_clip_norm
                )
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            total_loss.backward()
            if self.config.gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.gradient_clip_norm
                )
            self.optimizer.step()

        # 学习率调度
        self.scheduler.step()

        self.global_step += 1

        # 返回损失信息
        return {k: v.item() if torch.is_tensor(v) else v for k, v in loss_dict.items()}

    def _volume_render(
        self, rays_o: torch.Tensor, rays_d: torch.Tensor, near: torch.Tensor, far: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """体积渲染 - 训练阶段使用高质量渲染"""
        N = rays_o.shape[0]

        # 生成采样点
        t_vals = torch.linspace(0.0, 1.0, self.config.num_samples, device=self.device)
        z_vals = near * (1.0 - t_vals) + far * t_vals  # [N, S]

        if self.config.perturb and self.model.training:
            # 添加随机扰动
            mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
            upper = torch.cat([mids, z_vals[..., -1:]], dim=-1)
            lower = torch.cat([z_vals[..., :1], mids], dim=-1)
            t_rand = torch.rand_like(z_vals)
            z_vals = lower + (upper - lower) * t_rand

        # 计算3D点位置
        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]  # [N, S, 3]

        # 获取方向编码
        dirs = rays_d[..., None, :].expand_as(pts)  # [N, S, 3]

        # 模型前向传播
        pts_flat = pts.view(-1, 3)
        dirs_flat = dirs.view(-1, 3)

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
        transmittance = torch.cumprod(
            torch.cat([torch.ones_like(alpha[..., :1]), 1.0 - alpha[..., :-1]], dim=-1), dim=-1
        )
        weights = alpha * transmittance  # [N, S]

        # 最终渲染结果
        rgb_final = torch.sum(weights[..., None] * rgb, dim=-2)  # [N, 3]
        depth = torch.sum(weights * z_vals, dim=-1)  # [N]
        acc = torch.sum(weights, dim=-1)  # [N]

        return {"rgb": rgb_final, "depth": depth, "acc": acc, "weights": weights, "z_vals": z_vals}

    def train_epoch(self, dataloader: DataLoader) -> dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        epoch_losses = {}

        pbar = tqdm(dataloader, desc=f"Epoch {self.epoch}")
        for batch_idx, batch in enumerate(pbar):
            # 训练步骤
            losses = self.train_step(batch)

            # 累积损失
            for key, value in losses.items():
                if key not in epoch_losses:
                    epoch_losses[key] = []
                epoch_losses[key].append(value)

            # 更新进度条
            pbar.set_postfix(
                {
                    "loss": f"{losses.get('total_loss', 0):.4f}",
                    "psnr": f"{losses.get('psnr', 0):.2f}",
                    "lr": f"{self.optimizer.param_groups[0]['lr']:.2e}",
                }
            )

            # 日志记录
            if self.global_step % self.config.log_freq == 0 and self.writer:
                for key, value in losses.items():
                    self.writer.add_scalar(f"train/{key}", value, self.global_step)

                # 记录学习率
                for i, group in enumerate(self.optimizer.param_groups):
                    self.writer.add_scalar(
                        f"lr/{group.get('name', f'group_{i}')}", group["lr"], self.global_step
                    )

        # 计算epoch平均损失
        avg_losses = {key: np.mean(values) for key, values in epoch_losses.items()}

        self.epoch += 1
        return avg_losses

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        num_epochs: int | None = None,
    ):
        """完整训练流程"""
        if num_epochs is None:
            num_epochs = self.config.num_epochs

        logger.info(f"Starting Instant NGP training for {num_epochs} epochs")
        logger.info(f"Device: {self.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        best_psnr = 0.0

        for epoch in range(num_epochs):
            # 训练一个epoch
            train_losses = self.train_epoch(train_loader)

            logger.info(
                f"Epoch {epoch}: "
                f"Loss={train_losses.get('total_loss', 0):.4f}, "
                f"PSNR={train_losses.get('psnr', 0):.2f}"
            )

            # 验证
            if val_loader is not None and epoch % self.config.eval_freq == 0:
                val_metrics = self.validate(val_loader)

                # 保存最佳模型
                if val_metrics.get("psnr", 0) > best_psnr:
                    best_psnr = val_metrics["psnr"]
                    self.save_checkpoint("best_model.pth")

            # 定期保存检查点
            if epoch % self.config.save_freq == 0:
                self.save_checkpoint(f"epoch_{epoch:04d}.pth")

        logger.info("Training completed!")
        if self.writer:
            self.writer.close()

    def validate(self, val_loader: DataLoader) -> dict[str, float]:
        """验证模型"""
        self.model.eval()
        val_losses = {}

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                rays_o = batch["rays_o"].to(self.device)
                rays_d = batch["rays_d"].to(self.device)
                targets = batch["rgb"].to(self.device)

                near = torch.full_like(rays_o[..., :1], 0.01)
                far = torch.full_like(rays_o[..., :1], self.model.config.bound * 1.5)

                outputs = self._volume_render(rays_o, rays_d, near, far)
                losses = self.criterion(outputs, targets)

                for key, value in losses.items():
                    if key not in val_losses:
                        val_losses[key] = []
                    val_losses[key].append(value.item())

        # 平均验证损失
        avg_val_losses = {key: np.mean(values) for key, values in val_losses.items()}

        # 记录到tensorboard
        if self.writer:
            for key, value in avg_val_losses.items():
                self.writer.add_scalar(f"val/{key}", value, self.global_step)

        self.model.train()
        return avg_val_losses

    def save_checkpoint(self, filename: str):
        """保存检查点"""
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        filepath = os.path.join(self.config.checkpoint_dir, filename)

        checkpoint = {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "config": self.config,
            "model_config": self.model.config,
        }

        if self.scaler is not None:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()

        torch.save(checkpoint, filepath)
        logger.info(f"Checkpoint saved: {filepath}")

    def load_checkpoint(self, filepath: str, load_optimizer: bool = True):
        """加载检查点"""
        checkpoint = torch.load(filepath, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])

        if load_optimizer:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

            if self.scaler is not None and "scaler_state_dict" in checkpoint:
                self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        self.epoch = checkpoint["epoch"]
        self.global_step = checkpoint["global_step"]

        logger.info(f"Checkpoint loaded: {filepath}")
