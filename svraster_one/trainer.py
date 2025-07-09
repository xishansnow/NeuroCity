"""
SVRaster One 训练器

实现可微分光栅化渲染的端到端训练。
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Callable
import logging
import time
import os
from pathlib import Path
import json
from tqdm import tqdm

from .core import SVRasterOne
from .config import SVRasterOneConfig

logger = logging.getLogger(__name__)


class SVRasterOneTrainer:
    """
    SVRaster One 训练器

    实现端到端的可微分光栅化渲染训练，
    支持自适应优化、混合精度训练等。
    """

    def __init__(self, model: SVRasterOne, config: SVRasterOneConfig):
        self.model = model
        self.config = config
        self.device = model.device

        # 优化器
        self.optimizer = optim.Adam(
            model.get_trainable_parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )

        # 学习率调度器
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=config.training.num_epochs // 4,
            gamma=0.5,
        )

        # 混合精度训练
        self.use_amp = config.training.use_amp
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()

        # 训练状态
        self.current_epoch = 0
        self.global_step = 0
        self.best_loss = float("inf")

        # 日志记录
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []

        logger.info(f"Initialized SVRaster One trainer on device: {self.device}")

    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int,
        callbacks: Optional[List[Callable]] = None,
    ) -> Dict[str, float]:
        """
        训练一个 epoch

        Args:
            train_loader: 训练数据加载器
            epoch: 当前 epoch
            callbacks: 回调函数列表

        Returns:
            训练统计信息
        """
        self.model.train()
        epoch_losses = []
        epoch_start_time = time.time()

        # 进度条
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")

        for batch_idx, batch in enumerate(pbar):
            # 准备数据
            camera_matrix = batch["camera_matrix"].to(self.device)
            intrinsics = batch["intrinsics"].to(self.device)
            target_data = {k: v.to(self.device) for k, v in batch["target"].items()}

            # 前向传播
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    result = self.model.training_step_forward(
                        camera_matrix, intrinsics, target_data
                    )
                    loss = result["total_loss"]
            else:
                result = self.model.training_step_forward(camera_matrix, intrinsics, target_data)
                loss = result["total_loss"]

            # 反向传播
            self.optimizer.zero_grad()

            if self.use_amp:
                self.scaler.scale(loss).backward()

                # 梯度裁剪
                if self.config.training.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.training.grad_clip
                    )

                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()

                # 梯度裁剪
                if self.config.training.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.training.grad_clip
                    )

                self.optimizer.step()

            # 记录损失
            epoch_losses.append(loss.item())

            # 更新进度条
            pbar.set_postfix(
                {
                    "Loss": f"{loss.item():.6f}",
                    "LR": f"{self.optimizer.param_groups[0]['lr']:.6f}",
                }
            )

            # 自适应优化（每隔一定步数）
            if self.global_step % 100 == 0:
                self._adaptive_optimization_step(result)

            # 执行回调
            if callbacks:
                for callback in callbacks:
                    callback(self, batch_idx, result)

            self.global_step += 1

        # 计算 epoch 统计信息
        epoch_time = time.time() - epoch_start_time
        avg_loss = sum(epoch_losses) / len(epoch_losses)

        # 更新学习率
        self.scheduler.step()

        # 记录统计信息
        self.train_losses.append(avg_loss)
        self.learning_rates.append(self.optimizer.param_groups[0]["lr"])

        logger.info(
            f"Epoch {epoch} completed in {epoch_time:.2f}s, "
            f"Avg Loss: {avg_loss:.6f}, LR: {self.optimizer.param_groups[0]['lr']:.6f}"
        )

        return {
            "epoch": epoch,
            "avg_loss": avg_loss,
            "epoch_time": epoch_time,
            "learning_rate": self.optimizer.param_groups[0]["lr"],
        }

    def validate(
        self,
        val_loader: DataLoader,
        epoch: int,
    ) -> Dict[str, float]:
        """
        验证

        Args:
            val_loader: 验证数据加载器
            epoch: 当前 epoch

        Returns:
            验证统计信息
        """
        self.model.eval()
        val_losses = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Validation Epoch {epoch}"):
                # 准备数据
                camera_matrix = batch["camera_matrix"].to(self.device)
                intrinsics = batch["intrinsics"].to(self.device)
                target_data = {k: v.to(self.device) for k, v in batch["target"].items()}

                # 前向传播
                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        result = self.model.training_step_forward(
                            camera_matrix, intrinsics, target_data
                        )
                        loss = result["total_loss"]
                else:
                    result = self.model.training_step_forward(
                        camera_matrix, intrinsics, target_data
                    )
                    loss = result["total_loss"]

                val_losses.append(loss.item())

        avg_val_loss = sum(val_losses) / len(val_losses)
        self.val_losses.append(avg_val_loss)

        # 更新最佳损失
        if avg_val_loss < self.best_loss:
            self.best_loss = avg_val_loss
            self.model.best_loss = avg_val_loss

        logger.info(f"Validation Epoch {epoch}, Avg Loss: {avg_val_loss:.6f}")

        return {
            "epoch": epoch,
            "avg_val_loss": avg_val_loss,
            "best_loss": self.best_loss,
        }

    def _adaptive_optimization_step(self, result: Dict[str, torch.Tensor]):
        """
        自适应优化步骤
        """
        # 计算梯度幅度
        if "density_gradients" in result:
            gradient_magnitudes = torch.abs(result["density_gradients"])
            self.model.adaptive_optimization(gradient_magnitudes)

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        num_epochs: Optional[int] = None,
        save_dir: str = "checkpoints",
        callbacks: Optional[List[Callable]] = None,
    ):
        """
        完整训练流程

        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            num_epochs: 训练轮数
            save_dir: 保存目录
            callbacks: 回调函数列表
        """
        if num_epochs is None:
            num_epochs = self.config.training.num_epochs

        # 创建保存目录
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        # 训练循环
        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch

            # 训练
            train_stats = self.train_epoch(train_loader, epoch, callbacks)

            # 验证
            if val_loader is not None:
                val_stats = self.validate(val_loader, epoch)

                # 保存最佳模型
                if val_stats["avg_val_loss"] < self.best_loss:
                    self.save_checkpoint(save_path / "best_model.pth")

            # 定期保存检查点
            if (epoch + 1) % self.config.training.save_interval == 0:
                self.save_checkpoint(save_path / f"checkpoint_epoch_{epoch+1}.pth")

            # 保存训练日志
            self.save_training_log(save_path / "training_log.json")

    def save_checkpoint(self, filepath: str):
        """保存检查点"""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "config": self.config.to_dict(),
            "current_epoch": self.current_epoch,
            "global_step": self.global_step,
            "best_loss": self.best_loss,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "learning_rates": self.learning_rates,
        }

        if self.use_amp:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()

        torch.save(checkpoint, filepath)
        logger.info(f"Saved checkpoint to {filepath}")

    def load_checkpoint(self, filepath: str):
        """加载检查点"""
        checkpoint = torch.load(filepath, map_location=self.device)

        # 加载模型状态
        self.model.load_state_dict(checkpoint["model_state_dict"])

        # 加载优化器状态
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        # 加载训练状态
        self.current_epoch = checkpoint.get("current_epoch", 0)
        self.global_step = checkpoint.get("global_step", 0)
        self.best_loss = checkpoint.get("best_loss", float("inf"))
        self.train_losses = checkpoint.get("train_losses", [])
        self.val_losses = checkpoint.get("val_losses", [])
        self.learning_rates = checkpoint.get("learning_rates", [])

        # 加载混合精度训练状态
        if self.use_amp and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        logger.info(f"Loaded checkpoint from {filepath}")
        logger.info(f"Resuming from epoch {self.current_epoch}, step {self.global_step}")

    def save_training_log(self, filepath: str):
        """保存训练日志"""
        log_data = {
            "config": self.config.to_dict(),
            "current_epoch": self.current_epoch,
            "global_step": self.global_step,
            "best_loss": self.best_loss,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "learning_rates": self.learning_rates,
            "model_info": self.model.get_model_info(),
        }

        with open(filepath, "w") as f:
            json.dump(log_data, f, indent=2)

        logger.info(f"Saved training log to {filepath}")

    def get_training_stats(self) -> Dict[str, any]:
        """获取训练统计信息"""
        return {
            "current_epoch": self.current_epoch,
            "global_step": self.global_step,
            "best_loss": self.best_loss,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "learning_rates": self.learning_rates,
            "model_info": self.model.get_model_info(),
        }

    def plot_training_curves(self, save_path: str = "training_curves.png"):
        """绘制训练曲线"""
        try:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(2, 2, figsize=(12, 8))

            # 训练损失
            axes[0, 0].plot(self.train_losses, label="Train Loss")
            if self.val_losses:
                axes[0, 0].plot(self.val_losses, label="Val Loss")
            axes[0, 0].set_title("Training Loss")
            axes[0, 0].set_xlabel("Epoch")
            axes[0, 0].set_ylabel("Loss")
            axes[0, 0].legend()
            axes[0, 0].grid(True)

            # 学习率
            axes[0, 1].plot(self.learning_rates)
            axes[0, 1].set_title("Learning Rate")
            axes[0, 1].set_xlabel("Epoch")
            axes[0, 1].set_ylabel("Learning Rate")
            axes[0, 1].grid(True)

            # 体素统计
            model_info = self.model.get_model_info()
            voxel_stats = model_info["voxel_stats"]

            axes[1, 0].bar(
                ["Total", "Active"], [voxel_stats["total_voxels"], voxel_stats["active_voxels"]]
            )
            axes[1, 0].set_title("Voxel Count")
            axes[1, 0].set_ylabel("Count")

            # 内存使用
            memory_usage = model_info["memory_usage"]
            axes[1, 1].pie(
                [
                    memory_usage["active_memory_mb"],
                    memory_usage["total_memory_mb"] - memory_usage["active_memory_mb"],
                ],
                labels=["Active", "Inactive"],
                autopct="%1.1f%%",
            )
            axes[1, 1].set_title("Memory Usage")

            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close()

            logger.info(f"Saved training curves to {save_path}")

        except ImportError:
            logger.warning("matplotlib not available, skipping training curves plot")
