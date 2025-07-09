"""
InfNeRF Trainer - 与渲染器解耦的训练模块

这个训练器专门用于训练阶段，与渲染器完全解耦，
专注于训练循环、损失计算和模型优化。
"""

from __future__ import annotations

from typing import Optional, Any
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp.grad_scaler import GradScaler
from dataclasses import dataclass
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)

from .core import InfNeRF, InfNeRFConfig
from .utils.volume_renderer import VolumeRenderer, VolumeRendererConfig


@dataclass
class InfNeRFTrainerConfig:
    """InfNeRF 训练器配置"""

    # 训练参数
    num_epochs: int = 100
    lr_init: float = 1e-2
    lr_final: float = 1e-4
    lr_decay_start: int = 50000
    lr_decay_steps: int = 250000
    weight_decay: float = 1e-6

    # 批次处理
    rays_batch_size: int = 4096
    max_batch_rays: int = 16384
    accumulate_grad_batches: int = 1
    gradient_clip_val: float = 1.0

    # 渲染参数
    num_samples_coarse: int = 64
    num_samples_fine: int = 128
    use_white_background: bool = False

    # 日志和检查点
    log_dir: str = "logs"
    ckpt_dir: str = "checkpoints"
    save_freq: int = 5000
    eval_freq: int = 1000
    log_freq: int = 100

    # 内存管理
    mixed_precision: bool = True
    memory_threshold_gb: float = 16.0
    chunk_size: int = 8192


class InfNeRFTrainer:
    """
    InfNeRF 训练器 - 与渲染器解耦

    专门负责：
    - 训练循环和优化过程
    - 损失计算和梯度更新
    - 模型保存和加载
    - 训练监控和日志记录
    """

    def __init__(
        self,
        model: InfNeRF,
        train_dataset: Any,
        config: InfNeRFTrainerConfig,
        val_dataset: Optional[Any] = None,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 移动模型到设备
        self.model.to(self.device)

        # 设置优化器
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=config.lr_init, weight_decay=config.weight_decay
        )

        # 设置学习率调度器
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.95)

        # 混合精度训练
        self.scaler = GradScaler() if config.mixed_precision else None

        # 体积渲染器
        volume_config = VolumeRendererConfig(
            num_samples=config.num_samples_coarse,
            num_importance_samples=config.num_samples_fine,
            white_background=config.use_white_background,
        )
        self.volume_renderer = VolumeRenderer(volume_config)

        # 训练状态
        self.global_step = 0
        self.epoch = 0
        self.best_loss = float("inf")

        logger.info(f"InfNeRFTrainer initialized on device: {self.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

    def train(self):
        """主训练循环"""
        dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.config.rays_batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )

        logger.info(f"Starting training for {self.config.num_epochs} epochs")

        for epoch in range(self.config.num_epochs):
            self.model.train()
            epoch_losses = []

            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{self.config.num_epochs}")
            for batch in pbar:
                # 移动数据到设备
                for key in batch:
                    if torch.is_tensor(batch[key]):
                        batch[key] = batch[key].to(self.device)

                # 训练步骤
                loss = self.train_step(batch)
                epoch_losses.append(loss)

                # 更新进度条
                pbar.set_postfix({"loss": f"{loss:.4f}"})

                # 日志记录
                if self.global_step % self.config.log_freq == 0:
                    logger.info(f"Step {self.global_step}: loss = {loss:.4f}")

                # 验证
                if self.global_step % self.config.eval_freq == 0:
                    val_loss = self.validate()
                    if val_loss is not None:
                        logger.info(f"Validation loss: {val_loss:.4f}")
                        if val_loss < self.best_loss:
                            self.best_loss = val_loss
                            self.save_checkpoint("best.pth")

                # 保存检查点
                if self.global_step % self.config.save_freq == 0:
                    self.save_checkpoint(f"checkpoint_step_{self.global_step}.pth")

                self.global_step += 1

            self.epoch += 1
            self.scheduler.step()

            # 计算平均损失
            avg_loss = sum(epoch_losses) / len(epoch_losses)
            logger.info(f"Epoch {epoch+1} average loss: {avg_loss:.4f}")

        # 最终保存
        self.save_checkpoint("final.pth")
        logger.info("Training completed!")

    def train_step(self, batch: dict[str, torch.Tensor]) -> float:
        """单步训练"""
        self.optimizer.zero_grad()

        # 前向传播
        if self.scaler:
            with torch.cuda.amp.autocast():
                outputs = self.model(
                    rays_o=batch["rays_o"],
                    rays_d=batch["rays_d"],
                    near=batch.get("near", 0.1),
                    far=batch.get("far", 100.0),
                    focal_length=batch.get("focal_length", 800.0),
                    pixel_width=batch.get("pixel_width", 1.0 / 800),
                )
                loss = self.compute_loss(outputs, batch)
        else:
            outputs = self.model(
                rays_o=batch["rays_o"],
                rays_d=batch["rays_d"],
                near=batch.get("near", 0.1),
                far=batch.get("far", 100.0),
                focal_length=batch.get("focal_length", 800.0),
                pixel_width=batch.get("pixel_width", 1.0 / 800),
            )
            loss = self.compute_loss(outputs, batch)

        # 反向传播
        if self.scaler:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_val)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_val)
            self.optimizer.step()

        return loss.item()

    def compute_loss(
        self, outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """计算损失"""
        targets = {"target_rgb": batch["target_rgb"]}

        if "target_depth" in batch:
            targets["target_depth"] = batch["target_depth"]

        losses = self.volume_renderer.compute_losses(outputs, targets, self.model)
        return losses["total_loss"]

    def validate(self) -> Optional[float]:
        """验证"""
        if self.val_dataset is None:
            return None

        self.model.eval()
        val_losses = []

        val_loader = DataLoader(
            self.val_dataset, batch_size=self.config.rays_batch_size, shuffle=False, num_workers=2
        )

        with torch.no_grad():
            for batch in val_loader:
                # 移动数据到设备
                for key in batch:
                    if torch.is_tensor(batch[key]):
                        batch[key] = batch[key].to(self.device)

                # 前向传播
                outputs = self.model(
                    rays_o=batch["rays_o"],
                    rays_d=batch["rays_d"],
                    near=batch.get("near", 0.1),
                    far=batch.get("far", 100.0),
                    focal_length=batch.get("focal_length", 800.0),
                    pixel_width=batch.get("pixel_width", 1.0 / 800),
                )

                loss = self.compute_loss(outputs, batch)
                val_losses.append(loss.item())

        self.model.train()
        return sum(val_losses) / len(val_losses) if val_losses else None

    def save_checkpoint(self, filename: str):
        """保存检查点"""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "epoch": self.epoch,
            "global_step": self.global_step,
            "best_loss": self.best_loss,
            "config": self.config,
        }

        import os

        os.makedirs(self.config.ckpt_dir, exist_ok=True)
        checkpoint_path = os.path.join(self.config.ckpt_dir, filename)
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """加载检查点"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.epoch = checkpoint["epoch"]
        self.global_step = checkpoint["global_step"]
        self.best_loss = checkpoint.get("best_loss", float("inf"))

        logger.info(f"Checkpoint loaded: {checkpoint_path}")


def create_inf_nerf_trainer(
    model: InfNeRF,
    train_dataset: Any,
    config: Optional[InfNeRFTrainerConfig] = None,
    val_dataset: Optional[Any] = None,
    device: Optional[torch.device] = None,
) -> InfNeRFTrainer:
    """
    创建 InfNeRF 训练器的便捷函数
    """
    if config is None:
        config = InfNeRFTrainerConfig()

    return InfNeRFTrainer(model, train_dataset, config, val_dataset, device)
