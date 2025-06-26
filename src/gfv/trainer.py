"""
GFV Trainer Module - 训练器组件

This module contains training components for GFV library including:
- GFVTrainer: Traditional PyTorch trainer
- GFVLightningModule: PyTorch Lightning module
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from tqdm import tqdm
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from torch.utils.data import DataLoader

try:
    import pytorch_lightning as pl
    from pytorch_lightning.utilities.types import STEP_OUTPUT
    LIGHTNING_AVAILABLE = True
except ImportError:
    LIGHTNING_AVAILABLE = False
    pl = None

from .core import GlobalHashConfig, GlobalFeatureLibrary
from .dataset import GlobalFeatureDataset, MultiScaleDataset

logger = logging.getLogger(__name__)


class GFVTrainer:
    """GFV传统训练器"""
    
    def __init__(self, model: GlobalFeatureLibrary, config: Optional[dict[str, Any]] = None):
        """
        初始化训练器
        
        Args:
            model: GFV模型
            config: 训练配置
        """
        self.model = model
        self.config = config or {}
        
        # 训练配置
        self.learning_rate = self.config.get('learning_rate', 1e-3)
        self.num_epochs = self.config.get('num_epochs', 100)
        self.batch_size = self.config.get('batch_size', 32)
        self.device = self.config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        
        # 优化器配置
        self.optimizer_type = self.config.get('optimizer', 'adam')
        self.scheduler_type = self.config.get('scheduler', 'step')
        
        # 损失函数
        self.criterion = nn.MSELoss()
        
        # 训练历史
        self.train_losses = []
        self.val_losses = []
        
        logger.info(f"GFV训练器初始化完成，设备: {self.device}")
    
    def _setup_optimizer(self):
        """设置优化器"""
        params = self.model.database.hash_encoder.parameters()
        
        if self.optimizer_type.lower() == 'adam':
            self.optimizer = Adam(params, lr=self.learning_rate)
        elif self.optimizer_type.lower() == 'adamw':
            self.optimizer = AdamW(params, lr=self.learning_rate)
        else:
            self.optimizer = Adam(params, lr=self.learning_rate)
        
        # 设置学习率调度器
        if self.scheduler_type.lower() == 'step':
            self.scheduler = StepLR(self.optimizer, step_size=30, gamma=0.1)
        elif self.scheduler_type.lower() == 'cosine':
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.num_epochs)
        else:
            self.scheduler = None
    
    def train(
        self,
        train_dataset: GlobalFeatureDataset,
        val_dataset: Optional[GlobalFeatureDataset] = None,
        save_path: Optional[str] = None,
    )
        """
        训练模型
        
        Args:
            train_dataset: 训练数据集
            val_dataset: 验证数据集
            save_path: 模型保存路径
        """
        logger.info("开始训练GFV模型...")
        
        # 设置优化器
        self._setup_optimizer()
        
        # 数据加载器
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4
        )
        
        val_loader = None
        if val_dataset:
            val_loader = DataLoader(
                val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4
            )
        
        # 移动模型到设备
        self.model.database.hash_encoder.to(self.device)
        
        best_val_loss = float('inf')
        
        for epoch in range(self.num_epochs):
            # 训练阶段
            train_loss = self._train_epoch(train_loader)
            self.train_losses.append(train_loss)
            
            # 验证阶段
            val_loss = None
            if val_loader:
                val_loss = self._validate_epoch(val_loader)
                self.val_losses.append(val_loss)
                
                # 保存最佳模型
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    if save_path:
                        self.save_checkpoint(save_path)
            
            # 更新学习率
            if self.scheduler:
                self.scheduler.step()
            
            # 日志输出
            if epoch % 10 == 0:
                log_msg = f"Epoch {epoch+1}/{self.num_epochs}, Train Loss: {train_loss:.6f}"
                if val_loss is not None:
                    log_msg += f", Val Loss: {val_loss:.6f}"
                logger.info(log_msg)
        
        logger.info("训练完成")
        return {
            'train_losses': self.train_losses, 'val_losses': self.val_losses, 'best_val_loss': best_val_loss
        }
    
    def _train_epoch(self, train_loader: DataLoader) -> float:
        """训练一个epoch"""
        self.model.database.hash_encoder.train()
        total_loss = 0
        num_batches = len(train_loader)
        
        for batch in tqdm(train_loader, desc="Training"):
            coords = batch['coords'].to(self.device)
            target_features = batch['features'].to(self.device)
            
            # 前向传播
            predicted_features = self.model.database.hash_encoder(coords)
            
            # 计算损失
            loss = self.criterion(predicted_features, target_features)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / num_batches
    
    def _validate_epoch(self, val_loader: DataLoader) -> float:
        """验证一个epoch"""
        self.model.database.hash_encoder.eval()
        total_loss = 0
        num_batches = len(val_loader)
        
        with torch.no_grad():
            for batch in val_loader:
                coords = batch['coords'].to(self.device)
                target_features = batch['features'].to(self.device)
                
                # 前向传播
                predicted_features = self.model.database.hash_encoder(coords)
                
                # 计算损失
                loss = self.criterion(predicted_features, target_features)
                total_loss += loss.item()
        
        return total_loss / num_batches
    
    def save_checkpoint(self, path: str):
        """保存检查点"""
        checkpoint = {
            'model_state_dict': self.model.database.hash_encoder.state_dict(
            )
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(checkpoint, path)
        logger.info(f"检查点已保存到: {path}")
    
    def load_checkpoint(self, path: str):
        """加载检查点"""
        checkpoint = torch.load(path)
        
        self.model.database.hash_encoder.load_state_dict(checkpoint['model_state_dict'])
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        
        logger.info(f"检查点已从 {path} 加载")


if LIGHTNING_AVAILABLE:
    class GFVLightningModule(pl.LightningModule):
        """GFV PyTorch Lightning模块"""
        
        def __init__(
            self,
            config: GlobalHashConfig,
            learning_rate: float = 1e-3,
            optimizer_type: str = 'adam',
        )
            """
            初始化Lightning模块
            
            Args:
                config: GFV配置
                learning_rate: 学习率
                optimizer_type: 优化器类型
            """
            super().__init__()
            self.save_hyperparameters()
            
            self.config = config
            self.learning_rate = learning_rate
            self.optimizer_type = optimizer_type
            
            # 创建GFV库
            self.gfv_library = GlobalFeatureLibrary(config)
            
            # 损失函数
            self.criterion = nn.MSELoss()
            
            # 训练指标
            self.train_losses = []
            self.val_losses = []
        
        def forward(self, coords: torch.Tensor) -> torch.Tensor:
            """前向传播"""
            return self.gfv_library.database.hash_encoder(coords)
        
        def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
            """训练步骤"""
            coords = batch['coords']
            target_features = batch['features']
            
            # 前向传播
            predicted_features = self(coords)
            
            # 计算损失
            loss = self.criterion(predicted_features, target_features)
            
            # 记录指标
            self.log('train_loss', loss, prog_bar=True)
            
            return loss
        
        def validation_step(self, batch, batch_idx) -> STEP_OUTPUT:
            """验证步骤"""
            coords = batch['coords']
            target_features = batch['features']
            
            # 前向传播
            predicted_features = self(coords)
            
            # 计算损失
            loss = self.criterion(predicted_features, target_features)
            
            # 记录指标
            self.log('val_loss', loss, prog_bar=True)
            
            return loss
        
        def configure_optimizers(self):
            """配置优化器"""
            if self.optimizer_type.lower() == 'adam':
                optimizer = Adam(self.parameters(), lr=self.learning_rate)
            elif self.optimizer_type.lower() == 'adamw':
                optimizer = AdamW(self.parameters(), lr=self.learning_rate)
            else:
                optimizer = Adam(self.parameters(), lr=self.learning_rate)
            
            # 学习率调度器
            scheduler = CosineAnnealingLR(optimizer, T_max=100)
            
            return {
                'optimizer': optimizer, 'lr_scheduler': scheduler
            }
        
        def on_train_epoch_end(self):
            """训练epoch结束时"""
            # 可以在这里添加自定义逻辑
            pass
        
        def on_validation_epoch_end(self):
            """验证epoch结束时"""
            # 可以在这里添加自定义逻辑
            pass

else:
    class GFVLightningModule:
        """Lightning不可用时的占位类"""
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch Lightning is not installed. Please install it with: pip install pytorch-lightning")


class GFVMultiScaleTrainer(GFVTrainer):
    """多尺度GFV训练器"""
    
    def __init__(self, model: GlobalFeatureLibrary, config: Optional[dict[str, Any]] = None):
        super().__init__(model, config)
        
        # 多尺度配置
        self.scale_weights = self.config.get('scale_weights', [1.0, 1.0, 1.0, 1.0])
        self.progressive_training = self.config.get('progressive_training', False)
    
    def train_multiscale(
        self,
        train_dataset: MultiScaleDataset,
        val_dataset: Optional[MultiScaleDataset] = None,
        save_path: Optional[str] = None,
    )
        """
        多尺度训练
        
        Args:
            train_dataset: 多尺度训练数据集
            val_dataset: 多尺度验证数据集
            save_path: 模型保存路径
        """
        logger.info("开始多尺度训练...")
        
        if self.progressive_training:
            return self._progressive_training(train_dataset, val_dataset, save_path)
        else:
            return self._joint_training(train_dataset, val_dataset, save_path)
    
    def _progressive_training(
        self,
        train_dataset: MultiScaleDataset,
        val_dataset: Optional[MultiScaleDataset] = None,
        save_path: Optional[str] = None,
    )
        """渐进式训练"""
        logger.info("使用渐进式多尺度训练...")
        
        zoom_levels = sorted(train_dataset.zoom_levels)
        results = {}
        
        for zoom in zoom_levels:
            logger.info(f"训练缩放级别: {zoom}")
            
            # 过滤当前缩放级别的数据
            train_subset = self._filter_by_zoom(train_dataset, zoom)
            val_subset = self._filter_by_zoom(val_dataset, zoom) if val_dataset else None
            
            # 训练当前级别
            result = self.train(train_subset, val_subset)
            results[f'zoom_{zoom}'] = result
            
            # 保存中间结果
            if save_path:
                interim_path = save_path.replace('.pth', f'_zoom_{zoom}.pth')
                self.save_checkpoint(interim_path)
        
        return results
    
    def _joint_training(
        self,
        train_dataset: MultiScaleDataset,
        val_dataset: Optional[MultiScaleDataset] = None,
        save_path: Optional[str] = None,
    )
        """联合训练"""
        logger.info("使用联合多尺度训练...")
        
        # 直接使用多尺度数据集进行训练
        return self.train(train_dataset, val_dataset, save_path)
    
    def _filter_by_zoom(self, dataset: MultiScaleDataset, zoom: int):
        """根据缩放级别过滤数据集"""
        # 这里需要实现数据集过滤逻辑
        # 简化实现，实际应该创建Subset
        filtered_samples = [(lat, lon, z) for lat, lon, z in dataset.samples if z == zoom]
        filtered_dataset = MultiScaleDataset.__new__(MultiScaleDataset)
        filtered_dataset.samples = filtered_samples
        filtered_dataset.base_coords = dataset.base_coords
        filtered_dataset.zoom_levels = [zoom]
        return filtered_dataset 