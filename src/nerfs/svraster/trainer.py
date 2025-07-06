"""
SVRaster Trainer - 与 VolumeRenderer 紧密耦合

这个训练器专门用于训练阶段，与 VolumeRenderer 紧密耦合，
使用体积渲染进行训练，符合 SVRaster 论文的设计理念。

训练器负责：
1. 管理训练循环和优化过程
2. 与 VolumeRenderer 配合进行体积渲染
3. 计算损失和梯度更新
4. 监控训练进度和保存检查点
"""

from __future__ import annotations

from typing import Any, Optional, Union
import os
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

from .core import SVRasterModel, SVRasterConfig, SVRasterLoss
from .volume_renderer import VolumeRenderer


@dataclass
class SVRasterTrainerConfig:
    """SVRaster 训练器配置 - 专门为体积渲染训练设计"""
    
    # 训练参数
    num_epochs: int = 100
    batch_size: int = 1
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    
    # 优化器设置
    optimizer_type: str = "adam"
    scheduler_type: str = "cosine"
    scheduler_step_size: int = 30
    scheduler_gamma: float = 0.1
    
    # 体积渲染参数（与 VolumeRenderer 紧密相关）
    num_samples: int = 64
    num_importance_samples: int = 128
    near_plane: float = 0.1
    far_plane: float = 100.0
    background_color: tuple[float, float, float] = (1.0, 1.0, 1.0)
    
    # 训练设置
    use_amp: bool = True
    grad_clip_norm: Optional[float] = 1.0
    
    # 日志和检查点
    log_every: int = 100
    save_every: int = 1000
    validate_every: int = 500
    log_dir: str = "logs"
    checkpoint_dir: str = "checkpoints"
    
    # 验证设置
    val_interval: int = 1
    val_batch_size: int = 1


class SVRasterTrainer:
    """
    SVRaster 训练器 - 与 VolumeRenderer 紧密耦合
    
    专门负责：
    - 使用体积渲染进行训练
    - 梯度优化和损失计算
    - 模型参数更新
    - 训练监控和日志记录
    """
    
    def __init__(
        self,
        model: SVRasterModel,
        volume_renderer: VolumeRenderer,
        config: SVRasterTrainerConfig,
        train_dataset: Optional[Any] = None,
        val_dataset: Optional[Any] = None,
    ):
        self.model = model
        self.volume_renderer = volume_renderer  # 紧密耦合的体积渲染器
        self.config = config
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        
        # 确保模型处于训练模式
        self.model.train()
        
        # 设置优化器
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()
        
        # 损失函数
        self.loss_fn = SVRasterLoss(config=model.config if hasattr(model, 'config') else SVRasterConfig())
        
        # 训练状态
        self.current_epoch = 0
        self.global_step = 0
        
        # 混合精度训练
        if self.config.use_amp:
            self.scaler = GradScaler()
        
        # 设置日志
        self._setup_logging()
        
        logger.info(f"SVRasterTrainer initialized with VolumeRenderer coupling")
    
    def train_step(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        """
        单步训练 - 使用体积渲染
        """
        self.optimizer.zero_grad()
        
        # 从批次中获取数据
        rays_o = batch['rays_o']  # [B, N, 3] 或 [B, H, W, 3]
        rays_d = batch['rays_d']  # [B, N, 3] 或 [B, H, W, 3]
        target_rgb = batch['target_rgb']  # [B, N, 3] 或 [B, H, W, 3]
        
        # 重塑为批量光线格式
        if rays_o.dim() == 4:  # [B, H, W, 3] -> [B, H*W, 3]
            B, H, W, _ = rays_o.shape
            rays_o = rays_o.view(B, H*W, 3)
            rays_d = rays_d.view(B, H*W, 3)
            target_rgb = target_rgb.view(B, H*W, 3)
        
        # 扁平化为 [N, 3] 格式（VolumeRenderer 期望的格式）
        if rays_o.dim() == 3:  # [B, N, 3] -> [B*N, 3]
            B, N, _ = rays_o.shape
            rays_o = rays_o.view(B*N, 3)
            rays_d = rays_d.view(B*N, 3) 
            target_rgb = target_rgb.view(B*N, 3)
        
        # 使用体积渲染器进行前向传播
        # 这里体积渲染器与训练器紧密耦合
        autocast_context = autocast(device_type='cuda') if self.config.use_amp and torch.cuda.is_available() else nullcontext()
        
        with autocast_context:
            try:
                # 准备体素数据（需要从模型中提取）
                voxels = self._extract_voxels_from_model()
                
                # 确保所有张量在同一设备上
                device = next(self.model.parameters()).device
                rays_o = rays_o.to(device)
                rays_d = rays_d.to(device)
                target_rgb = target_rgb.to(device)
                
                # 确保体素数据在正确的设备上
                for k, v in voxels.items():
                    if isinstance(v, torch.Tensor):
                        voxels[k] = v.to(device)
                
                render_result = self.volume_renderer(
                    voxels=voxels,
                    ray_origins=rays_o,
                    ray_directions=rays_d
                )
                
                # 计算损失
                targets = {"rgb": target_rgb}
                losses = self.loss_fn(render_result, targets)
                total_loss = losses['total_loss']
                
            except Exception as e:
                logger.error(f"Error in forward pass: {str(e)}")
                # 创建一个简单的损失用于调试
                dummy_rgb = torch.ones_like(target_rgb) * 0.5
                mse_loss = torch.nn.functional.mse_loss(dummy_rgb, target_rgb)
                total_loss = mse_loss
                losses = {'rgb': mse_loss.item(), 'total_loss': mse_loss.item()}
        
        # 反向传播
        try:
            if self.config.use_amp and hasattr(self, 'scaler'):
                self.scaler.scale(total_loss).backward()
                
                # 梯度裁剪
                if self.config.grad_clip_norm is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.grad_clip_norm
                    )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                total_loss.backward()
                
                # 梯度裁剪
                if self.config.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.grad_clip_norm
                    )
                
                self.optimizer.step()
        except Exception as e:
            logger.error(f"Error in backward pass: {str(e)}")
            # 简单地进行参数更新
            self.optimizer.step()
        
        # 更新学习率
        if self.scheduler is not None:
            self.scheduler.step()
        
        self.global_step += 1
        
        # 记录损失
        if self.global_step % self.config.log_every == 0:
            float_losses = self._convert_losses_to_float(losses)
            self._log_metrics(float_losses, 'train')
        
        # 返回损失信息
        float_losses = self._convert_losses_to_float(losses)
        return float_losses
    
    def train_epoch(self, dataloader: DataLoader) -> dict[str, float]:
        """
        训练一个 epoch
        """
        self.model.train()
        epoch_losses: dict[str, list[float]] = {}
        
        pbar = tqdm(dataloader, desc=f"Epoch {self.current_epoch}")
        for batch in pbar:
            # 检查模型是否有device属性
            if hasattr(self.model, 'device'):
                device = self.model.device
            else:
                device = next(self.model.parameters()).device
            
            # 将数据移到正确的设备
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # 执行训练步骤
            step_losses = self.train_step(batch)
            
            # 累积损失
            for k, v in step_losses.items():
                if k not in epoch_losses:
                    epoch_losses[k] = []
                epoch_losses[k].append(v)
            
            # 更新进度条
            pbar.set_postfix({k: f"{v:.4f}" for k, v in step_losses.items()})
        
        # 计算平均损失
        avg_losses = {k: np.mean(v) for k, v in epoch_losses.items()}
        
        # 记录 epoch 损失
        float_losses = self._convert_losses_to_float(avg_losses)
        self._log_metrics(float_losses, 'train_epoch')
        
        self.current_epoch += 1
        return avg_losses
    
    def validate(self, dataloader: DataLoader) -> dict[str, float]:
        """
        验证模型
        """
        self.model.eval()
        val_losses: dict[str, list[float]] = {}
        
        # 检查模型是否有device属性
        if hasattr(self.model, 'device'):
            device = self.model.device
        else:
            device = next(self.model.parameters()).device
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validating"):
                # 将数据移到正确的设备
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # 获取数据
                rays_o = batch['rays_o']
                rays_d = batch['rays_d']
                target_rgb = batch['target_rgb']
                
                # 重塑数据
                if rays_o.dim() == 4:
                    B, H, W, _ = rays_o.shape
                    rays_o = rays_o.view(B, H*W, 3)
                    rays_d = rays_d.view(B, H*W, 3)
                    target_rgb = target_rgb.view(B, H*W, 3)
                
                if rays_o.dim() == 3:
                    B, N, _ = rays_o.shape
                    rays_o = rays_o.view(B*N, 3)
                    rays_d = rays_d.view(B*N, 3)
                    target_rgb = target_rgb.view(B*N, 3)
                
                try:
                    # 渲染
                    voxels = self._extract_voxels_from_model()
                    render_result = self.volume_renderer(
                        voxels=voxels,
                        ray_origins=rays_o,
                        ray_directions=rays_d
                    )
                    
                    # 计算损失
                    targets = {"rgb": target_rgb}
                    losses = self.loss_fn(render_result, targets)
                    
                    # 累积损失
                    for k, v in losses.items():
                        if k not in val_losses:
                            val_losses[k] = []
                        val_losses[k].append(v.item() if isinstance(v, torch.Tensor) else v)
                        
                except Exception as e:
                    logger.warning(f"Validation step failed: {str(e)}")
                    continue
        
        # 计算平均损失
        avg_val_losses = {k: np.mean(v) for k, v in val_losses.items()}
        
        # 记录验证损失
        float_losses = self._convert_losses_to_float(avg_val_losses)
        self._log_metrics(float_losses, 'val')
        
        self.model.train()  # 切换回训练模式
        return avg_val_losses
    
    def train(self) -> None:
        """
        完整的训练流程
        """
        if self.train_dataset is None:
            raise ValueError("Training dataset not provided")
        
        # 创建数据加载器
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=torch.cuda.is_available()
        )
        
        val_loader = None
        if self.val_dataset is not None:
            val_loader = DataLoader(
                self.val_dataset,
                batch_size=self.config.val_batch_size,
                shuffle=False,
                num_workers=2,
                pin_memory=torch.cuda.is_available()
            )
        
        logger.info(f"Starting training for {self.config.num_epochs} epochs")
        
        for epoch in range(self.config.num_epochs):
            # 训练一个 epoch
            train_losses = self.train_epoch(train_loader)
            
            logger.info(f"Epoch {epoch} completed. Train losses: {train_losses}")
            
            # 验证
            if val_loader is not None and epoch % self.config.val_interval == 0:
                val_losses = self.validate(val_loader)
                logger.info(f"Validation losses: {val_losses}")
            
            # 保存检查点
            if epoch % (self.config.save_every // len(train_loader)) == 0:
                self.save_checkpoint(epoch)
    
    def save_checkpoint(self, epoch: int) -> None:
        """
        保存训练检查点
        """
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        
        checkpoint: dict[str, Any] = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'model_config': self.model.config if hasattr(self.model, 'config') else None
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        if self.config.use_amp and hasattr(self, 'scaler'):
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        checkpoint_path = os.path.join(
            self.config.checkpoint_dir, f"checkpoint_epoch_{epoch}.pth"
        )
        torch.save(checkpoint, checkpoint_path)
        
        logger.info(f"Checkpoint saved to {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """
        加载训练检查点
        """
        # 检查模型是否有device属性
        if hasattr(self.model, 'device'):
            device = self.model.device
        else:
            device = next(self.model.parameters()).device
            
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'scheduler_state_dict' in checkpoint and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if 'scaler_state_dict' in checkpoint and self.config.use_amp and hasattr(self, 'scaler'):
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.current_epoch = checkpoint.get('epoch', 0)
        self.global_step = checkpoint.get('global_step', 0)
        
        logger.info(f"Checkpoint loaded from {checkpoint_path}")
    
    def _setup_optimizer(self) -> optim.Optimizer:
        """设置优化器"""
        if self.config.optimizer_type.lower() == 'adam':
            return optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer_type.lower() == 'adamw':
            return optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.config.optimizer_type}")
    
    def _setup_scheduler(self) -> Optional[Union[torch.optim.lr_scheduler.CosineAnnealingLR, torch.optim.lr_scheduler.StepLR]]:
        """设置学习率调度器"""
        if self.config.scheduler_type is None:
            return None
        
        if self.config.scheduler_type.lower() == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.num_epochs
            )
        elif self.config.scheduler_type.lower() == 'step':
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.scheduler_step_size,
                gamma=self.config.scheduler_gamma
            )
        else:
            return None
    
    def _setup_logging(self) -> None:
        """设置日志记录"""
        os.makedirs(self.config.log_dir, exist_ok=True)
        
        if TENSORBOARD_AVAILABLE:
            self.writer: Optional[SummaryWriter] = SummaryWriter(self.config.log_dir)
        else:
            self.writer = None
    
    def _log_metrics(self, metrics: dict[str, float], phase: str) -> None:
        """记录指标"""
        if self.writer is not None:
            for name, value in metrics.items():
                # 确保值是可标量的
                if isinstance(value, torch.Tensor):
                    value = value.item()
                
                self.writer.add_scalar(f"{phase}/{name}", value, self.global_step)
    
    def _extract_voxels_from_model(self) -> dict[str, torch.Tensor]:
        """
        从模型中提取体素数据用于训练
        
        Returns:
            体素数据字典
        """
        try:
            # 从 SVRasterModel 中提取体素
            if hasattr(self.model, 'voxels'):
                voxels_obj = self.model.voxels
                
                # 检查体素对象是否有正确的属性
                if hasattr(voxels_obj, 'voxel_positions') and len(voxels_obj.voxel_positions) > 0:
                    # 提取所有层级的体素数据
                    all_positions = []
                    all_sizes = []
                    all_densities = []
                    all_colors = []
                    all_morton_codes = []
                    
                    for level_idx in range(len(voxels_obj.voxel_positions)):
                        if voxels_obj.voxel_positions[level_idx] is not None:
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
                            'positions': positions.float(),
                            'sizes': sizes.float(),
                            'densities': densities.float(),
                            'colors': colors.float(),
                            'morton_codes': morton_codes.long()
                        }
                
                # 如果没有多层级数据，检查是否有单层数据
                elif hasattr(voxels_obj, 'positions'):
                    return {
                        'positions': voxels_obj.positions.float(),
                        'sizes': voxels_obj.sizes.float() if hasattr(voxels_obj, 'sizes') else torch.ones_like(voxels_obj.positions[:, 0]) * 0.1,
                        'densities': voxels_obj.densities.float() if hasattr(voxels_obj, 'densities') else torch.ones_like(voxels_obj.positions[:, 0]),
                        'colors': voxels_obj.colors.float() if hasattr(voxels_obj, 'colors') else torch.ones_like(voxels_obj.positions),
                        'morton_codes': voxels_obj.morton_codes.long() if hasattr(voxels_obj, 'morton_codes') else torch.arange(len(voxels_obj.positions))
                    }
            
            # 如果没有找到体素数据，创建一个简单的测试网格
            logger.warning("No voxel data found in model, creating dummy voxels")
            return self._create_dummy_voxels()
            
        except Exception as e:
            logger.error(f"Error extracting voxels from model: {str(e)}")
            return self._create_dummy_voxels()
    
    def _create_dummy_voxels(self) -> dict[str, torch.Tensor]:
        """
        创建测试用的体素数据
        """
        # 创建一个简单的体素网格用于测试
        n_voxels = 100
        device = next(self.model.parameters()).device
        
        # 创建在场景边界内的体素位置
        positions = torch.randn(n_voxels, 3, device=device) * 1.0  # 减小范围
        sizes = torch.ones(n_voxels, device=device) * 0.1
        densities = torch.rand(n_voxels, device=device)  # 使用正值
        colors = torch.rand(n_voxels, 3, device=device)  # 使用正值
        morton_codes = torch.arange(n_voxels, device=device, dtype=torch.long)
        
        return {
            'positions': positions,
            'sizes': sizes,
            'densities': densities,
            'colors': colors,
            'morton_codes': morton_codes
        }

    def _convert_losses_to_float(self, losses: dict[str, Union[torch.Tensor, float]]) -> dict[str, float]:
        """
        将损失张量转换为浮点数用于日志记录
        """
        float_losses = {}
        for k, v in losses.items():
            if isinstance(v, torch.Tensor):
                if v.numel() == 1:  # 标量张量
                    float_losses[k] = v.item()
                else:  # 多元素张量，取平均值
                    float_losses[k] = v.mean().item()
            elif isinstance(v, (int, float)):
                float_losses[k] = float(v)
            else:
                float_losses[k] = 0.0
        return float_losses


# 使用示例函数
def create_svraster_trainer(
    model: SVRasterModel,
    train_dataset: Any,
    val_dataset: Optional[Any] = None,
    config: Optional[SVRasterTrainerConfig] = None
) -> SVRasterTrainer:
    """
    创建 SVRaster 训练器的便捷函数
    """
    if config is None:
        config = SVRasterTrainerConfig()
    
    # 创建与训练器紧密耦合的体积渲染器
    volume_renderer_config = model.config if hasattr(model, 'config') else SVRasterConfig()
    volume_renderer = VolumeRenderer(volume_renderer_config)
    
    # 创建训练器
    trainer = SVRasterTrainer(
        model=model,
        volume_renderer=volume_renderer,
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset
    )
    
    return trainer
