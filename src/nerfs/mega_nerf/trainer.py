"""
Mega-NeRF Trainer Module

This module contains the training pipeline for Mega-NeRF models including:
- Training configuration
- Main trainer class
- Parallel training support
- Loss computation and optimization
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from dataclasses import dataclass
from typing import Optional, Union, Any, dict
import logging
import os
import time
from pathlib import Path
import json

logger = logging.getLogger(__name__)

# 尝试导入可选依赖
try:
    from tqdm import tqdm

    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    logger.warning("tqdm not available, progress bars will be disabled")

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    logger.warning("wandb not available, logging will be disabled")


@dataclass
class MegaNeRFTrainerConfig:
    """Mega-NeRF 训练器配置"""

    # 训练参数
    num_epochs: int = 100
    batch_size: int = 1024
    learning_rate: float = 5e-4
    weight_decay: float = 1e-6
    lr_decay: float = 0.1
    max_iterations: int = 500000

    # 优化器设置
    optimizer_type: str = "adam"  # "adam", "adamw", "sgd"
    scheduler_type: str = "exponential"  # "exponential", "cosine", "step"
    warmup_steps: int = 1000

    # 训练策略
    training_strategy: str = "sequential"  # "sequential", "parallel"
    num_parallel_workers: int = 4

    # 日志和检查点
    log_interval: int = 100
    val_interval: int = 1000
    save_interval: int = 5000
    checkpoint_dir: str = "checkpoints"

    # 损失函数权重
    rgb_loss_weight: float = 1.0
    depth_loss_weight: float = 0.1
    density_loss_weight: float = 0.01

    # 混合精度训练
    use_mixed_precision: bool = True
    gradient_clip_val: Optional[float] = 1.0

    # 早停设置
    patience: int = 10
    min_delta: float = 1e-4

    def __post_init__(self):
        """后初始化验证"""
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")

        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")

        if self.num_epochs <= 0:
            raise ValueError("num_epochs must be positive")

        if self.training_strategy not in ["sequential", "parallel"]:
            raise ValueError("training_strategy must be 'sequential' or 'parallel'")


class MegaNeRFTrainer:
    """Mega-NeRF 主训练器"""

    def __init__(
        self,
        model: "MegaNeRF",
        dataset: "MegaNeRFDataset",
        config: MegaNeRFTrainerConfig,
        device: str = "cuda",
    ):
        """
        初始化训练器

        Args:
            model: Mega-NeRF 模型
            dataset: 训练数据集
            config: 训练器配置
            device: 训练设备
        """
        self.model = model.to(device)
        self.dataset = dataset
        self.config = config
        self.device = device

        # 创建输出目录
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # 训练状态
        self.global_step = 0
        self.current_epoch = 0
        self.best_psnr = 0.0
        self.patience_counter = 0

        # 优化器和调度器
        self.optimizers = {}
        self.schedulers = {}

        # 损失历史
        self.loss_history = []
        self.psnr_history = []

        # 混合精度训练
        self.scaler = torch.cuda.amp.GradScaler() if config.use_mixed_precision else None

        logger.info(f"MegaNeRFTrainer initialized with {len(self.model.submodules)} submodules")

    def setup_optimizers(self, submodule_indices: list[int] | None = None) -> None:
        """设置指定子模块的优化器"""
        if submodule_indices is None:
            submodule_indices = list(range(len(self.model.submodules)))

        for idx in submodule_indices:
            if idx >= len(self.model.submodules):
                continue

            submodule = self.model.submodules[idx]

            # 选择优化器
            if self.config.optimizer_type.lower() == "adam":
                optimizer = optim.Adam(
                    submodule.parameters(),
                    lr=self.config.learning_rate,
                    weight_decay=self.config.weight_decay,
                )
            elif self.config.optimizer_type.lower() == "adamw":
                optimizer = optim.AdamW(
                    submodule.parameters(),
                    lr=self.config.learning_rate,
                    weight_decay=self.config.weight_decay,
                )
            elif self.config.optimizer_type.lower() == "sgd":
                optimizer = optim.SGD(
                    submodule.parameters(),
                    lr=self.config.learning_rate,
                    momentum=0.9,
                    weight_decay=self.config.weight_decay,
                )
            else:
                raise ValueError(f"Unsupported optimizer: {self.config.optimizer_type}")

            self.optimizers[idx] = optimizer

            # 选择调度器
            if self.config.scheduler_type.lower() == "exponential":
                scheduler = optim.lr_scheduler.ExponentialLR(
                    optimizer, gamma=self.config.lr_decay ** (1 / 10000)
                )
            elif self.config.scheduler_type.lower() == "cosine":
                scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=self.config.max_iterations
                )
            elif self.config.scheduler_type.lower() == "step":
                scheduler = optim.lr_scheduler.StepLR(
                    optimizer, step_size=10000, gamma=self.config.lr_decay
                )
            else:
                raise ValueError(f"Unsupported scheduler: {self.config.scheduler_type}")

            self.schedulers[idx] = scheduler

        logger.info(f"Setup optimizers for {len(submodule_indices)} submodules")

    def train_submodule(
        self,
        submodule_idx: int,
        num_iterations: int = 10000,
        log_interval: int = 100,
        val_interval: int = 1000,
    ) -> dict[str, float]:
        """
        训练单个子模块

        Args:
            submodule_idx: 子模块索引
            num_iterations: 训练迭代次数
            log_interval: 日志间隔
            val_interval: 验证间隔

        Returns:
            训练统计信息
        """
        if submodule_idx >= len(self.model.submodules):
            raise ValueError(f"Submodule {submodule_idx} does not exist")

        submodule = self.model.submodules[submodule_idx]

        # 设置优化器
        if submodule_idx not in self.optimizers:
            self.setup_optimizers([submodule_idx])

        optimizer = self.optimizers[submodule_idx]
        scheduler = self.schedulers[submodule_idx]

        # 获取分区数据
        partition_data = self.dataset.get_partition_data(submodule_idx)
        rays_data = partition_data.get("rays", {})

        if not rays_data:
            logger.warning(f"No data for submodule {submodule_idx}")
            return {}

        # 训练循环
        submodule.train()
        losses = []
        psnrs = []

        # 创建进度条
        if TQDM_AVAILABLE:
            pbar = tqdm(range(num_iterations), desc=f"Training submodule {submodule_idx}")
        else:
            pbar = range(num_iterations)

        for iteration in pbar:
            # 采样光线
            ray_batch = self._sample_rays_from_partition(rays_data, self.config.batch_size)

            # 前向传播
            optimizer.zero_grad()

            if self.config.use_mixed_precision and self.scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = self._forward_pass(submodule, ray_batch)
                    loss = self._compute_loss(outputs, ray_batch["colors"])

                # 反向传播
                self.scaler.scale(loss).backward()

                # 梯度裁剪
                if self.config.gradient_clip_val is not None:
                    self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        submodule.parameters(), self.config.gradient_clip_val
                    )

                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                outputs = self._forward_pass(submodule, ray_batch)
                loss = self._compute_loss(outputs, ray_batch["colors"])

                # 反向传播
                loss.backward()

                # 梯度裁剪
                if self.config.gradient_clip_val is not None:
                    torch.nn.utils.clip_grad_norm_(
                        submodule.parameters(), self.config.gradient_clip_val
                    )

                optimizer.step()

            scheduler.step()

            # 计算指标
            with torch.no_grad():
                mse = torch.mean((outputs["rgb"] - ray_batch["colors"]) ** 2)
                psnr = -10 * torch.log10(mse + 1e-8)

            losses.append(loss.item())
            psnrs.append(psnr.item())

            # 更新进度条
            if TQDM_AVAILABLE:
                pbar.set_postfix({"loss": f"{loss.item():.4f}", "psnr": f"{psnr.item():.2f}"})

            # 日志记录
            if iteration % log_interval == 0:
                avg_loss = np.mean(losses[-log_interval:])
                avg_psnr = np.mean(psnrs[-log_interval:])

                logger.info(
                    f"Submodule {submodule_idx} - Iter {iteration}: "
                    f"Loss={avg_loss:.4f}, PSNR={avg_psnr:.2f}"
                )

                if WANDB_AVAILABLE and wandb.run is not None:
                    wandb.log(
                        {
                            f"submodule_{submodule_idx}_loss": avg_loss,
                            f"submodule_{submodule_idx}_psnr": avg_psnr,
                        },
                        step=self.global_step,
                    )

            # 验证
            if iteration % val_interval == 0 and iteration > 0:
                val_metrics = self._validate_submodule(submodule_idx)
                logger.info(f"Validation - Submodule {submodule_idx}: {val_metrics}")

            self.global_step += 1

        # 返回训练统计
        return {
            "final_loss": np.mean(losses[-100:]),
            "final_psnr": np.mean(psnrs[-100:]),
            "best_psnr": np.max(psnrs),
            "iterations": num_iterations,
        }

    def train_sequential(
        self,
        num_iterations_per_submodule: int = 10000,
        log_interval: int = 100,
        val_interval: int = 1000,
    ) -> dict[str, object]:
        """
        顺序训练所有子模块

        Args:
            num_iterations_per_submodule: 每个子模块的训练迭代次数
            log_interval: 日志间隔
            val_interval: 验证间隔

        Returns:
            训练统计信息
        """
        logger.info("Starting sequential training of all submodules")

        # 设置所有优化器
        self.setup_optimizers()

        # 训练统计
        training_stats = {}

        for submodule_idx in range(len(self.model.submodules)):
            logger.info(f"Training submodule {submodule_idx}")

            stats = self.train_submodule(
                submodule_idx, num_iterations_per_submodule, log_interval, val_interval
            )

            training_stats[f"submodule_{submodule_idx}"] = stats

            # 保存检查点
            if (submodule_idx + 1) % self.config.save_interval == 0:
                self.save_checkpoint(f"submodule_{submodule_idx}.pth", submodule_idx)

        # 保存完整模型
        self.save_checkpoint("final_model.pth")

        logger.info("Sequential training completed")
        return training_stats

    def _forward_pass(
        self, submodule: nn.Module, ray_batch: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """执行前向传播"""
        # 这里需要根据实际的数据格式和渲染器实现
        # 简化版本，实际使用时需要根据具体实现调整
        points = ray_batch["points"]
        viewdirs = ray_batch.get("viewdirs")
        appearance_idx = ray_batch.get("appearance_idx")

        density, color = submodule(points, viewdirs, appearance_idx)

        return {"rgb": color, "density": density}

    def _compute_loss(
        self, outputs: dict[str, torch.Tensor], target_rgb: torch.Tensor
    ) -> torch.Tensor:
        """计算损失"""
        # RGB 损失
        rgb_loss = torch.mean((outputs["rgb"] - target_rgb) ** 2)

        # 密度正则化损失
        density = outputs["density"]
        density_loss = torch.mean(torch.relu(density - 0.1))  # 稀疏性正则化

        # 总损失
        total_loss = (
            self.config.rgb_loss_weight * rgb_loss + self.config.density_loss_weight * density_loss
        )

        return total_loss

    def _sample_rays_from_partition(
        self, rays_data: dict[str, np.ndarray], batch_size: int
    ) -> dict[str, torch.Tensor]:
        """从分区数据中采样光线"""
        # 随机采样索引
        num_rays = len(rays_data.get("ray_origins", []))
        if num_rays == 0:
            raise ValueError("No rays available in partition data")

        indices = np.random.choice(num_rays, min(batch_size, num_rays), replace=False)

        # 构建批次数据
        batch = {}
        for key, data in rays_data.items():
            if isinstance(data, np.ndarray):
                batch[key] = torch.tensor(data[indices], device=self.device, dtype=torch.float32)
            else:
                batch[key] = torch.tensor(data, device=self.device, dtype=torch.float32)

        return batch

    def _validate_submodule(self, submodule_idx: int) -> dict[str, float]:
        """验证子模块"""
        submodule = self.model.submodules[submodule_idx]
        submodule.eval()

        # 获取验证数据
        val_data = self.dataset.get_validation_data(submodule_idx)
        if not val_data:
            return {"psnr": 0.0, "loss": float("inf")}

        with torch.no_grad():
            total_loss = 0.0
            total_psnr = 0.0
            num_batches = 0

            for batch in val_data:
                outputs = self._forward_pass(submodule, batch)
                loss = self._compute_loss(outputs, batch["colors"])

                mse = torch.mean((outputs["rgb"] - batch["colors"]) ** 2)
                psnr = -10 * torch.log10(mse + 1e-8)

                total_loss += loss.item()
                total_psnr += psnr.item()
                num_batches += 1

            avg_loss = total_loss / num_batches if num_batches > 0 else float("inf")
            avg_psnr = total_psnr / num_batches if num_batches > 0 else 0.0

        submodule.train()
        return {"psnr": avg_psnr, "loss": avg_loss}

    def save_checkpoint(self, path: str, submodule_idx: int | None = None) -> None:
        """保存检查点"""
        checkpoint_path = self.checkpoint_dir / path

        if submodule_idx is not None:
            # 保存单个子模块
            if submodule_idx >= len(self.model.submodules):
                raise ValueError(f"Invalid submodule index: {submodule_idx}")

            checkpoint = {
                "submodule_state_dict": self.model.submodules[submodule_idx].state_dict(),
                "optimizer_state_dict": self.optimizers[submodule_idx].state_dict(),
                "scheduler_state_dict": self.schedulers[submodule_idx].state_dict(),
                "global_step": self.global_step,
                "config": self.config,
            }
        else:
            # 保存完整模型
            checkpoint = {
                "model_state_dict": self.model.state_dict(),
                "optimizer_states": {idx: opt.state_dict() for idx, opt in self.optimizers.items()},
                "scheduler_states": {
                    idx: sched.state_dict() for idx, sched in self.schedulers.items()
                },
                "global_step": self.global_step,
                "best_psnr": self.best_psnr,
                "config": self.config,
            }

        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved to {checkpoint_path}")

    def load_checkpoint(self, path: str, submodule_idx: int | None = None) -> None:
        """加载检查点"""
        checkpoint_path = self.checkpoint_dir / path

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        if submodule_idx is not None:
            # 加载单个子模块
            if submodule_idx >= len(self.model.submodules):
                raise ValueError(f"Invalid submodule index: {submodule_idx}")

            self.model.submodules[submodule_idx].load_state_dict(checkpoint["submodule_state_dict"])

            if submodule_idx in self.optimizers:
                self.optimizers[submodule_idx].load_state_dict(checkpoint["optimizer_state_dict"])
                self.schedulers[submodule_idx].load_state_dict(checkpoint["scheduler_state_dict"])
        else:
            # 加载完整模型
            self.model.load_state_dict(checkpoint["model_state_dict"])

            # 加载优化器状态
            for idx, opt_state in checkpoint.get("optimizer_states", {}).items():
                if int(idx) in self.optimizers:
                    self.optimizers[int(idx)].load_state_dict(opt_state)

            # 加载调度器状态
            for idx, sched_state in checkpoint.get("scheduler_states", {}).items():
                if int(idx) in self.schedulers:
                    self.schedulers[int(idx)].load_state_dict(sched_state)

            self.best_psnr = checkpoint.get("best_psnr", 0.0)

        self.global_step = checkpoint.get("global_step", 0)
        logger.info(f"Checkpoint loaded from {checkpoint_path}")

    def get_training_stats(self) -> dict[str, object]:
        """获取训练统计信息"""
        return {
            "global_step": self.global_step,
            "current_epoch": self.current_epoch,
            "best_psnr": self.best_psnr,
            "loss_history": self.loss_history,
            "psnr_history": self.psnr_history,
            "num_submodules": len(self.model.submodules),
        }
