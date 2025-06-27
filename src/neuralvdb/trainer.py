from typing import Any, Optional
"""
Training Module for NeuralVDB

This module contains training classes for NeuralVDB models, including basic and advanced trainers.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import logging
from tqdm import tqdm
import os

from .networks import AdvancedLossFunction

logger = logging.getLogger(__name__)

class NeuralVDBTrainer:
    """NeuralVDB训练器 - 基础版本"""
    
    def __init__(self, sparse_grid, config, device: str = 'auto'):
        """
        初始化训练器
        
        Args:
            sparse_grid: 稀疏体素网格
            config: NeuralVDB配置
            device: 计算设备
        """
        self.sparse_grid = sparse_grid
        self.config = config
        
        # 设置设备
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # 移动网络到设备
        self.sparse_grid.feature_network.to(self.device)
        self.sparse_grid.occupancy_network.to(self.device)
        
        # 设置优化器
        self.optimizer = optim.Adam(
            list(self.sparse_grid.feature_network.parameters()) + 
            list(
                self.sparse_grid.occupancy_network.parameters,
            )
        )
        
        # 设置损失函数
        self.criterion = nn.BCELoss()
        
        # 训练历史
        self.train_losses = []
        self.val_losses = []
        
        logger.info(f"NeuralVDB训练器初始化完成，设备: {self.device}")
    
    def train(
        self,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        num_epochs: int = 100,
        save_path: Optional[str] = None,
    )
        """
        训练模型
        
        Args:
            train_dataloader: 训练数据加载器
            val_dataloader: 验证数据加载器
            num_epochs: 训练轮数
            save_path: 模型保存路径
            
        Returns:
            训练统计信息
        """
        logger.info(f"开始训练，共 {num_epochs} 轮")
        
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            # 训练一轮
            train_loss = self._train_epoch(train_dataloader)
            self.train_losses.append(train_loss)
            
            # 验证
            val_loss = None
            if val_dataloader is not None:
                val_loss = self._validate(val_dataloader)
                self.val_losses.append(val_loss)
                
                # 保存最佳模型
                if val_loss < best_val_loss and save_path:
                    best_val_loss = val_loss
                    self.save_model(save_path)
            
            # 日志
            if epoch % 10 == 0 or epoch == num_epochs - 1:
                log_msg = f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.6f}"
                if val_loss is not None:
                    log_msg += f", Val Loss: {val_loss:.6f}"
                logger.info(log_msg)
        
        # 返回训练统计
        stats = {
            'train_losses': self.train_losses, 'val_losses': self.val_losses, 'best_val_loss': best_val_loss, 'final_train_loss': train_loss
        }
        
        if val_loss is not None:
            stats['final_val_loss'] = val_loss
        
        logger.info("训练完成")
        return stats
    
    def _train_epoch(self, dataloader: DataLoader) -> float:
        """训练一轮"""
        self.sparse_grid.feature_network.train()
        self.sparse_grid.occupancy_network.train()
        
        total_loss = 0.0
        num_batches = 0
        
        for batch_points, batch_targets in tqdm(dataloader, desc="Training"):
            batch_points = batch_points.to(self.device)
            batch_targets = batch_targets.to(self.device)
            
            # 前向传播
            features = self.sparse_grid.feature_network(batch_points)
            predictions = self.sparse_grid.occupancy_network(features)
            predictions = predictions.squeeze()
            
            # 计算损失
            loss = self.criterion(predictions, batch_targets)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches
    
    def _validate(self, dataloader: DataLoader) -> float:
        """验证"""
        self.sparse_grid.feature_network.eval()
        self.sparse_grid.occupancy_network.eval()
        
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch_points, batch_targets in dataloader:
                batch_points = batch_points.to(self.device)
                batch_targets = batch_targets.to(self.device)
                
                # 前向传播
                features = self.sparse_grid.feature_network(batch_points)
                predictions = self.sparse_grid.occupancy_network(features)
                predictions = predictions.squeeze()
                
                # 计算损失
                loss = self.criterion(predictions, batch_targets)
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def predict(self, points: np.ndarray) -> np.ndarray:
        """
        预测
        
        Args:
            points: 3D坐标点 (N, 3)
            
        Returns:
            占用概率 (N, )
        """
        self.sparse_grid.feature_network.eval()
        self.sparse_grid.occupancy_network.eval()
        
        points_tensor = torch.FloatTensor(points).to(self.device)
        
        with torch.no_grad():
            features = self.sparse_grid.feature_network(points_tensor)
            predictions = self.sparse_grid.occupancy_network(features)
            predictions = predictions.squeeze()
        
        return predictions.cpu().numpy()
    
    def save_model(self, path: str):
        """保存模型"""
        checkpoint = {
            'feature_network_state_dict': self.sparse_grid.feature_network.state_dict(
            )
        }
        
        torch.save(checkpoint, path)
        logger.info(f"模型已保存到: {path}")
    
    def load_model(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.sparse_grid.feature_network.load_state_dict(checkpoint['feature_network_state_dict'])
        self.sparse_grid.occupancy_network.load_state_dict(checkpoint['occupancy_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'train_losses' in checkpoint:
            self.train_losses = checkpoint['train_losses']
        if 'val_losses' in checkpoint:
            self.val_losses = checkpoint['val_losses']
        
        logger.info(f"模型已从 {path} 加载")

class AdvancedNeuralVDBTrainer:
    """高级NeuralVDB训练器"""
    
    def __init__(self, sparse_grid, config, device: str = 'auto'):
        """
        初始化高级训练器
        
        Args:
            sparse_grid: 高级稀疏体素网格
            config: AdvancedNeuralVDB配置
            device: 计算设备
        """
        self.sparse_grid = sparse_grid
        self.config = config
        
        # 设置设备
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # 移动网络到设备
        self.sparse_grid.feature_network.to(self.device)
        self.sparse_grid.occupancy_network.to(self.device)
        
        # 设置优化器
        self.optimizer = optim.AdamW(
            list(self.sparse_grid.feature_network.parameters()) + 
            list(
                self.sparse_grid.occupancy_network.parameters,
            )
        )
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100, eta_min=config.learning_rate * 0.01
        )
        
        # 高级损失函数
        self.criterion = AdvancedLossFunction(config)
        
        # 训练历史
        self.train_losses = []
        self.val_losses = []
        self.loss_components = []
        
        logger.info(f"高级NeuralVDB训练器初始化完成，设备: {self.device}")
    
    def train(
        self,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        num_epochs: int = 100,
        save_path: Optional[str] = None,
    )
        """
        训练模型（支持渐进式训练）
        
        Args:
            train_dataloader: 训练数据加载器
            val_dataloader: 验证数据加载器
            num_epochs: 训练轮数
            save_path: 模型保存路径
            
        Returns:
            训练统计信息
        """
        logger.info(f"开始高级训练，共 {num_epochs} 轮")
        
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            # 渐进式训练参数调整
            if self.config.progressive_training:
                self._adjust_training_parameters(epoch, num_epochs)
            
            # 训练一轮
            train_loss, loss_components = self._train_epoch(train_dataloader)
            self.train_losses.append(train_loss)
            self.loss_components.append(loss_components)
            
            # 验证
            val_loss = None
            if val_dataloader is not None:
                val_loss = self._validate(val_dataloader)
                self.val_losses.append(val_loss)
                
                # 保存最佳模型
                if val_loss < best_val_loss and save_path:
                    best_val_loss = val_loss
                    self.save_model(save_path)
            
            # 更新学习率
            self.scheduler.step()
            
            # 日志
            if epoch % 10 == 0 or epoch == num_epochs - 1:
                log_msg = f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.6f}"
                if val_loss is not None:
                    log_msg += f", Val Loss: {val_loss:.6f}"
                
                # 添加损失分解信息
                if loss_components:
                    component_str = ", ".join([f"{
                        k,
                    }
                    log_msg += f" ({component_str})"
                
                logger.info(log_msg)
        
        # 返回训练统计
        stats = {
            'train_losses': self.train_losses, 'val_losses': self.val_losses, 'loss_components': self.loss_components, 'best_val_loss': best_val_loss, 'final_train_loss': train_loss
        }
        
        if val_loss is not None:
            stats['final_val_loss'] = val_loss
        
        logger.info("高级训练完成")
        return stats
    
    def _adjust_training_parameters(self, epoch: int, total_epochs: int):
        """渐进式训练参数调整"""
        progress = epoch / total_epochs
        
        # 动态调整损失权重
        if hasattr(self.config, 'smoothness_weight'):
            # 平滑性权重随训练进度增加
            self.config.smoothness_weight = 0.05 + 0.05 * progress
        
        if hasattr(self.config, 'sparsity_weight'):
            # 稀疏性权重在后期增加
            if progress > 0.5:
                self.config.sparsity_weight = 0.01 + 0.02 * (progress - 0.5) * 2
    
    def _train_epoch(self, dataloader: DataLoader) -> tuple[float, dict[str, float]]:
        """训练一轮（返回损失分解）"""
        self.sparse_grid.feature_network.train()
        self.sparse_grid.occupancy_network.train()
        
        total_loss = 0.0
        total_components = {}
        num_batches = 0
        
        for batch_points, batch_targets in tqdm(dataloader, desc="Training"):
            batch_points = batch_points.to(self.device)
            batch_targets = batch_targets.to(self.device)
            
            # 前向传播
            features = self.sparse_grid.feature_network(batch_points)
            predictions = self.sparse_grid.occupancy_network(features)
            
            # 计算高级损失
            loss, loss_components = self.criterion(
                predictions, batch_targets, batch_points
            )
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(
                list(self.sparse_grid.feature_network.parameters()) + 
                list(self.sparse_grid.occupancy_network.parameters()), max_norm=1.0
            )
            
            self.optimizer.step()
            
            # 累计损失
            total_loss += loss.item()
            
            # 累计损失分解
            for key, value in loss_components.items():
                if key not in total_components:
                    total_components[key] = 0.0
                total_components[key] += value.item() if hasattr(value, 'item') else value
            
            num_batches += 1
        
        # 平均损失
        avg_loss = total_loss / num_batches
        avg_components = {k: v / num_batches for k, v in total_components.items()}
        
        return avg_loss, avg_components
    
    def _validate(self, dataloader: DataLoader) -> float:
        """验证"""
        self.sparse_grid.feature_network.eval()
        self.sparse_grid.occupancy_network.eval()
        
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch_points, batch_targets in dataloader:
                batch_points = batch_points.to(self.device)
                batch_targets = batch_targets.to(self.device)
                
                # 前向传播
                features = self.sparse_grid.feature_network(batch_points)
                predictions = self.sparse_grid.occupancy_network(features)
                
                # 计算损失（只用主要损失）
                if isinstance(predictions, dict):
                    pred_occupancy = predictions['occupancy'].squeeze()
                else:
                    pred_occupancy = predictions.squeeze()
                
                loss = nn.BCELoss()(pred_occupancy, batch_targets)
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def predict(self, points: np.ndarray) -> np.ndarray:
        """
        预测（支持高级输出）
        
        Args:
            points: 3D坐标点 (N, 3)
            
        Returns:
            占用概率 (N, ) 或包含额外信息的字典
        """
        self.sparse_grid.feature_network.eval()
        self.sparse_grid.occupancy_network.eval()
        
        points_tensor = torch.FloatTensor(points).to(self.device)
        
        with torch.no_grad():
            features = self.sparse_grid.feature_network(points_tensor)
            predictions = self.sparse_grid.occupancy_network(features)
            
            if isinstance(predictions, dict):
                # 高级预测，返回所有信息
                result = {}
                for key, value in predictions.items():
                    result[key] = value.cpu().numpy()
                return result
            else:
                # 基础预测，只返回占用概率
                return predictions.squeeze().cpu().numpy()
    
    def save_model(self, path: str):
        """保存模型"""
        checkpoint = {
            'feature_network_state_dict': self.sparse_grid.feature_network.state_dict(
            )
        }
        
        # 保存特征压缩器
        if hasattr(self.sparse_grid, 'feature_compressor') and self.sparse_grid.feature_compressor:
            checkpoint['feature_compressor'] = {
                'codebook': self.sparse_grid.feature_compressor.codebook, 'is_trained': self.sparse_grid.feature_compressor.is_trained, 'quantization_bits': self.sparse_grid.feature_compressor.quantization_bits
            }
        
        torch.save(checkpoint, path)
        logger.info(f"高级模型已保存到: {path}")
    
    def load_model(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.sparse_grid.feature_network.load_state_dict(checkpoint['feature_network_state_dict'])
        self.sparse_grid.occupancy_network.load_state_dict(checkpoint['occupancy_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if 'train_losses' in checkpoint:
            self.train_losses = checkpoint['train_losses']
        if 'val_losses' in checkpoint:
            self.val_losses = checkpoint['val_losses']
        if 'loss_components' in checkpoint:
            self.loss_components = checkpoint['loss_components']
        
        # 加载特征压缩器
        if 'feature_compressor' in checkpoint and hasattr(self.sparse_grid, 'feature_compressor'):
            comp_data = checkpoint['feature_compressor']
            self.sparse_grid.feature_compressor.codebook = comp_data['codebook']
            self.sparse_grid.feature_compressor.is_trained = comp_data['is_trained']
            self.sparse_grid.feature_compressor.quantization_bits = comp_data['quantization_bits']
        
        logger.info(f"高级模型已从 {path} 加载")

class NeuralSDFTrainer:
    """SDF/Occupancy神经网络训练器"""
    
    def __init__(
        self,
        model: nn.Module,
        device: str = 'auto',
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
    )
        """
        初始化训练器
        
        Args:
            model: 神经网络模型
            device: 设备
            learning_rate: 学习率
            weight_decay: 权重衰减
        """
        self.model = model
        
        # 设置设备
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.model.to(self.device)
        
        # 设置优化器
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        
        # 设置损失函数
        self.criterion = nn.MSELoss()
        
        # 训练历史
        self.train_losses = []
        self.val_losses = []
        
        logger.info(f"SDF训练器初始化完成，设备: {self.device}")
    
    def train_epoch(self, dataloader: DataLoader) -> float:
        """训练一轮"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_coords, batch_targets in tqdm(dataloader, desc="Training SDF"):
            batch_coords = batch_coords.to(self.device)
            batch_targets = batch_targets.to(self.device)
            
            # 前向传播
            predictions = self.model(batch_coords).squeeze()
            
            # 计算损失
            loss = self.criterion(predictions, batch_targets)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches
    
    def validate(self, dataloader: DataLoader) -> float:
        """验证"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch_coords, batch_targets in dataloader:
                batch_coords = batch_coords.to(self.device)
                batch_targets = batch_targets.to(self.device)
                
                # 前向传播
                predictions = self.model(batch_coords).squeeze()
                
                # 计算损失
                loss = self.criterion(predictions, batch_targets)
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def train(
        self,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        num_epochs: int = 100,
        save_path: Optional[str] = None,
        early_stopping_patience: int = 10,
    )
        """
        完整训练流程
        
        Args:
            train_dataloader: 训练数据加载器
            val_dataloader: 验证数据加载器
            num_epochs: 训练轮数
            save_path: 模型保存路径
            early_stopping_patience: 早停耐心值
            
        Returns:
            训练统计信息
        """
        logger.info(f"开始SDF训练，共 {num_epochs} 轮")
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(num_epochs):
            # 训练
            train_loss = self.train_epoch(train_dataloader)
            self.train_losses.append(train_loss)
            
            # 验证
            val_loss = None
            if val_dataloader is not None:
                val_loss = self.validate(val_dataloader)
                self.val_losses.append(val_loss)
                
                # 早停检查
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    
                    # 保存最佳模型
                    if save_path:
                        self.save_model(save_path)
                else:
                    patience_counter += 1
                    
                    if patience_counter >= early_stopping_patience:
                        logger.info(f"早停触发，在第 {epoch+1} 轮停止训练")
                        break
            
            # 日志
            if epoch % 10 == 0 or epoch == num_epochs - 1:
                log_msg = f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.6f}"
                if val_loss is not None:
                    log_msg += f", Val Loss: {val_loss:.6f}"
                logger.info(log_msg)
        
        # 返回统计信息
        stats = {
            'train_losses': self.train_losses, 'val_losses': self.val_losses, 'best_val_loss': best_val_loss, 'final_train_loss': train_loss, 'epochs_trained': epoch + 1
        }
        
        if val_loss is not None:
            stats['final_val_loss'] = val_loss
        
        logger.info("SDF训练完成")
        return stats
    
    def save_model(self, path: str):
        """保存模型"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(
            )
        }
        
        torch.save(checkpoint, path)
        logger.info(f"SDF模型已保存到: {path}")
    
    def load_model(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'train_losses' in checkpoint:
            self.train_losses = checkpoint['train_losses']
        if 'val_losses' in checkpoint:
            self.val_losses = checkpoint['val_losses']
        
        logger.info(f"SDF模型已从 {path} 加载")
    
    def predict(self, coords: np.ndarray) -> np.ndarray:
        """
        预测
        
        Args:
            coords: 坐标 (N, 3)
            
        Returns:
            预测值 (N, )
        """
        self.model.eval()
        coords_tensor = torch.FloatTensor(coords).to(self.device)
        
        with torch.no_grad():
            predictions = self.model(coords_tensor).squeeze()
        
        return predictions.cpu().numpy() 