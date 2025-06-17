#!/usr/bin/env python3
"""
Advanced NeuralVDB: Enhanced Sparse Volumetric Neural Representations
基于论文 "NeuralVDB: Efficient Sparse Volumetric Neural Representations" 的高级实现

新增特性:
- 自适应分辨率
- 多尺度特征提取
- 高级损失函数
- 渐进式训练
- 动态八叉树优化
- 特征压缩和量化
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import os
from typing import List, Tuple, Dict, Optional, Union, Any
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
from dataclasses import dataclass
from collections import defaultdict
import pickle
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class AdvancedNeuralVDBConfig:
    """高级NeuralVDB配置参数"""
    # 基础参数
    voxel_size: float = 1.0
    max_depth: int = 8
    min_depth: int = 3
    
    # 神经网络参数
    feature_dim: int = 64
    hidden_dims: List[int] = None
    activation: str = 'relu'
    dropout: float = 0.1
    
    # 训练参数
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 1024
    
    # 稀疏性参数
    sparsity_threshold: float = 0.01
    occupancy_threshold: float = 0.5
    
    # 高级参数
    adaptive_resolution: bool = True
    multi_scale_features: bool = True
    progressive_training: bool = True
    feature_compression: bool = True
    quantization_bits: int = 8
    
    # 损失函数权重
    occupancy_weight: float = 1.0
    smoothness_weight: float = 0.1
    sparsity_weight: float = 0.01
    consistency_weight: float = 0.1
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [256, 512, 512, 256, 128]

class AdaptiveOctreeNode:
    """自适应八叉树节点"""
    
    def __init__(self, center: np.ndarray, size: float, depth: int = 0):
        self.center = center
        self.size = size
        self.depth = depth
        self.children = None
        self.features = None
        self.occupancy = 0.0
        self.is_leaf = True
        self.importance = 0.0  # 节点重要性
        self.density = 0.0     # 点密度
        self.gradient = 0.0    # 梯度信息
        
    def subdivide(self):
        """细分节点"""
        if self.children is not None:
            return
            
        self.children = []
        half_size = self.size / 2
        quarter_size = half_size / 2
        
        for i in range(8):
            offset = np.array([
                (i & 1) * half_size - quarter_size,
                ((i >> 1) & 1) * half_size - quarter_size,
                ((i >> 2) & 1) * half_size - quarter_size
            ])
            child_center = self.center + offset
            
            child = AdaptiveOctreeNode(child_center, half_size, self.depth + 1)
            self.children.append(child)
        
        self.is_leaf = False
    
    def compute_importance(self, points: np.ndarray, occupancies: np.ndarray):
        """计算节点重要性"""
        # 找到属于当前节点的点
        half_size = self.size / 2
        in_node = np.all(
            (points >= self.center - half_size) & 
            (points < self.center + half_size), 
            axis=1
        )
        
        if not np.any(in_node):
            self.importance = 0.0
            return
        
        node_points = points[in_node]
        node_occupancies = occupancies[in_node]
        
        # 计算密度
        self.density = len(node_points) / (self.size ** 3)
        
        # 计算占用率变化
        self.occupancy = np.mean(node_occupancies)
        
        # 计算梯度信息（占用率的变化率）
        if len(node_points) > 1:
            # 使用KNN计算局部梯度
            tree = cKDTree(node_points)
            distances, indices = tree.query(node_points, k=min(5, len(node_points)))
            
            gradients = []
            for i, (dist, idx) in enumerate(zip(distances, indices)):
                if len(idx) > 1:
                    local_gradient = np.std(node_occupancies[idx[1:]])  # 排除自身
                    gradients.append(local_gradient)
            
            if gradients:
                self.gradient = np.mean(gradients)
            else:
                self.gradient = 0.0
        else:
            self.gradient = 0.0
        
        # 综合重要性计算
        self.importance = (
            self.density * 0.4 +
            self.occupancy * 0.3 +
            self.gradient * 0.3
        )

class MultiScaleFeatureNetwork(nn.Module):
    """多尺度特征网络"""
    
    def __init__(self, input_dim: int, feature_dim: int, hidden_dims: List[int],
                 num_scales: int = 3, activation: str = 'relu', dropout: float = 0.1):
        super(MultiScaleFeatureNetwork, self).__init__()
        
        self.num_scales = num_scales
        self.feature_dim = feature_dim
        
        # 多尺度特征提取器
        self.scale_networks = nn.ModuleList()
        for i in range(num_scales):
            scale_factor = 2 ** i
            scale_input_dim = input_dim
            
            layers = []
            prev_dim = scale_input_dim
            
            for hidden_dim in hidden_dims:
                layers.extend([
                    nn.Linear(prev_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    self._get_activation(activation),
                    nn.Dropout(dropout)
                ])
                prev_dim = hidden_dim
            
            # 输出层
            layers.append(nn.Linear(prev_dim, feature_dim // num_scales))
            
            scale_network = nn.Sequential(*layers)
            self.scale_networks.append(scale_network)
        
        # 特征融合网络
        fusion_layers = [
            nn.Linear(feature_dim, feature_dim),
            nn.BatchNorm1d(feature_dim),
            self._get_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, feature_dim)
        ]
        self.fusion_network = nn.Sequential(*fusion_layers)
        
        self.apply(self._init_weights)
    
    def _get_activation(self, activation: str) -> nn.Module:
        if activation == 'relu':
            return nn.ReLU()
        elif activation == 'leaky_relu':
            return nn.LeakyReLU()
        elif activation == 'tanh':
            return nn.Tanh()
        else:
            raise ValueError(f"未知激活函数: {activation}")
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    
    def forward(self, x):
        # 多尺度特征提取
        scale_features = []
        for i, scale_net in enumerate(self.scale_networks):
            # 应用不同尺度的变换
            scale_x = x * (2 ** i)
            scale_feat = scale_net(scale_x)
            scale_features.append(scale_feat)
        
        # 特征融合
        combined_features = torch.cat(scale_features, dim=1)
        final_features = self.fusion_network(combined_features)
        
        return final_features

class AdvancedOccupancyNetwork(nn.Module):
    """高级占用网络"""
    
    def __init__(self, feature_dim: int, hidden_dims: List[int], 
                 activation: str = 'relu', dropout: float = 0.1):
        super(AdvancedOccupancyNetwork, self).__init__()
        
        layers = []
        prev_dim = feature_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                self._get_activation(activation),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
        self.apply(self._init_weights)
    
    def _get_activation(self, activation: str) -> nn.Module:
        if activation == 'relu':
            return nn.ReLU()
        elif activation == 'leaky_relu':
            return nn.LeakyReLU()
        elif activation == 'tanh':
            return nn.Tanh()
        else:
            raise ValueError(f"未知激活函数: {activation}")
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    
    def forward(self, features):
        return self.network(features)

class AdvancedLossFunction(nn.Module):
    """高级损失函数"""
    
    def __init__(self, config: AdvancedNeuralVDBConfig):
        super(AdvancedLossFunction, self).__init__()
        self.config = config
        
        # 基础损失函数
        self.occupancy_loss = nn.BCELoss()
        self.mse_loss = nn.MSELoss()
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor, 
                features: torch.Tensor = None, points: torch.Tensor = None):
        """计算综合损失"""
        total_loss = 0.0
        
        # 1. 占用损失
        occupancy_loss = self.occupancy_loss(predictions, targets)
        total_loss += self.config.occupancy_weight * occupancy_loss
        
        # 2. 平滑性损失
        if self.config.smoothness_weight > 0 and points is not None:
            smoothness_loss = self._compute_smoothness_loss(predictions, points)
            total_loss += self.config.smoothness_weight * smoothness_loss
        
        # 3. 稀疏性损失
        if self.config.sparsity_weight > 0 and features is not None:
            sparsity_loss = self._compute_sparsity_loss(features)
            total_loss += self.config.sparsity_weight * sparsity_loss
        
        # 4. 一致性损失
        if self.config.consistency_weight > 0:
            consistency_loss = self._compute_consistency_loss(predictions, targets)
            total_loss += self.config.consistency_weight * consistency_loss
        
        return total_loss, {
            'occupancy_loss': occupancy_loss.item(),
            'smoothness_loss': smoothness_loss.item() if self.config.smoothness_weight > 0 else 0.0,
            'sparsity_loss': sparsity_loss.item() if self.config.sparsity_weight > 0 else 0.0,
            'consistency_loss': consistency_loss.item() if self.config.consistency_weight > 0 else 0.0
        }
    
    def _compute_smoothness_loss(self, predictions: torch.Tensor, points: torch.Tensor):
        """计算平滑性损失"""
        # 计算预测值在空间上的梯度
        if len(predictions) > 1:
            # 使用有限差分近似梯度
            gradients = []
            for i in range(len(predictions)):
                # 找到最近的邻居
                distances = torch.norm(points - points[i], dim=1)
                nearest_idx = torch.argsort(distances)[1:4]  # 最近的3个邻居
                
                if len(nearest_idx) > 0:
                    local_gradient = torch.mean(torch.abs(
                        predictions[i] - predictions[nearest_idx]
                    ))
                    gradients.append(local_gradient)
            
            if gradients:
                return torch.mean(torch.stack(gradients))
        
        return torch.tensor(0.0, device=predictions.device)
    
    def _compute_sparsity_loss(self, features: torch.Tensor):
        """计算稀疏性损失"""
        # L1正则化促进稀疏性
        return torch.mean(torch.abs(features))
    
    def _compute_consistency_loss(self, predictions: torch.Tensor, targets: torch.Tensor):
        """计算一致性损失"""
        # 确保预测值和目标值的一致性
        return self.mse_loss(predictions, targets)

class AdvancedSparseVoxelGrid:
    """高级稀疏体素网格"""
    
    def __init__(self, config: AdvancedNeuralVDBConfig):
        self.config = config
        self.root = None
        self.feature_network = None
        self.occupancy_network = None
        self.feature_compressor = None
        
    def build_from_points(self, points: np.ndarray, occupancies: np.ndarray):
        """从点云构建自适应稀疏体素网格"""
        logger.info("构建高级稀疏体素网格...")
        
        # 计算边界框
        min_coords = points.min(axis=0)
        max_coords = points.max(axis=0)
        center = (min_coords + max_coords) / 2
        size = max(max_coords - min_coords)
        
        # 创建根节点
        self.root = AdaptiveOctreeNode(center, size)
        
        # 递归构建自适应八叉树
        self._build_adaptive_octree_recursive(self.root, points, occupancies)
        
        # 初始化神经网络
        self._init_networks()
        
        # 初始化特征压缩器
        if self.config.feature_compression:
            self._init_feature_compressor()
        
        logger.info(f"高级稀疏体素网格构建完成，根节点大小: {size}")
    
    def _build_adaptive_octree_recursive(self, node: AdaptiveOctreeNode, 
                                       points: np.ndarray, occupancies: np.ndarray):
        """递归构建自适应八叉树"""
        # 计算节点重要性
        node.compute_importance(points, occupancies)
        
        # 找到属于当前节点的点
        node_points, node_occupancies = self._get_points_in_node(node, points, occupancies)
        
        if len(node_points) == 0:
            return
        
        # 自适应细分条件
        should_subdivide = (
            node.depth < self.config.max_depth and
            len(node_points) > 10 and
            node.importance > self.config.sparsity_threshold and
            (not self.config.adaptive_resolution or 
             node.importance > 0.1 * (1.0 / (2 ** node.depth)))  # 自适应分辨率
        )
        
        if should_subdivide:
            node.subdivide()
            
            # 递归处理子节点
            for child in node.children:
                self._build_adaptive_octree_recursive(child, points, occupancies)
    
    def _get_points_in_node(self, node: AdaptiveOctreeNode, points: np.ndarray, occupancies: np.ndarray):
        """获取属于节点的点"""
        half_size = node.size / 2
        
        in_node = np.all(
            (points >= node.center - half_size) & 
            (points < node.center + half_size), 
            axis=1
        )
        
        return points[in_node], occupancies[in_node]
    
    def _init_networks(self):
        """初始化神经网络"""
        if self.config.multi_scale_features:
            self.feature_network = MultiScaleFeatureNetwork(
                input_dim=3,
                feature_dim=self.config.feature_dim,
                hidden_dims=self.config.hidden_dims,
                num_scales=3,
                activation=self.config.activation,
                dropout=self.config.dropout
            )
        else:
            # 使用简单的特征网络
            from neural_vdb import FeatureNetwork
            self.feature_network = FeatureNetwork(
                input_dim=3,
                feature_dim=self.config.feature_dim,
                hidden_dims=self.config.hidden_dims,
                activation=self.config.activation,
                dropout=self.config.dropout
            )
        
        self.occupancy_network = AdvancedOccupancyNetwork(
            feature_dim=self.config.feature_dim,
            hidden_dims=[128, 64, 32],
            activation=self.config.activation,
            dropout=self.config.dropout
        )
    
    def _init_feature_compressor(self):
        """初始化特征压缩器"""
        # 简单的特征量化
        self.feature_compressor = FeatureCompressor(
            feature_dim=self.config.feature_dim,
            quantization_bits=self.config.quantization_bits
        )

class FeatureCompressor:
    """特征压缩器"""
    
    def __init__(self, feature_dim: int, quantization_bits: int = 8):
        self.feature_dim = feature_dim
        self.quantization_bits = quantization_bits
        self.codebook = None
        self.is_trained = False
    
    def train(self, features: np.ndarray):
        """训练压缩器"""
        # 使用K-means聚类创建码本
        n_clusters = 2 ** self.quantization_bits
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(features)
        
        self.codebook = kmeans.cluster_centers_
        self.is_trained = True
    
    def compress(self, features: np.ndarray) -> np.ndarray:
        """压缩特征"""
        if not self.is_trained:
            raise ValueError("压缩器尚未训练")
        
        # 找到最近的码本向量
        distances = np.linalg.norm(
            features[:, np.newaxis, :] - self.codebook[np.newaxis, :, :], 
            axis=2
        )
        indices = np.argmin(distances, axis=1)
        
        return self.codebook[indices]
    
    def decompress(self, indices: np.ndarray) -> np.ndarray:
        """解压缩特征"""
        if not self.is_trained:
            raise ValueError("压缩器尚未训练")
        
        return self.codebook[indices]

class AdvancedNeuralVDBTrainer:
    """高级NeuralVDB训练器"""
    
    def __init__(self, sparse_grid: AdvancedSparseVoxelGrid, 
                 config: AdvancedNeuralVDBConfig, device: str = 'auto'):
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
        self.optimizer = torch.optim.Adam([
            {'params': self.sparse_grid.feature_network.parameters()},
            {'params': self.sparse_grid.occupancy_network.parameters()}
        ], lr=config.learning_rate, weight_decay=config.weight_decay)
        
        # 设置学习率调度器
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        # 设置损失函数
        self.criterion = AdvancedLossFunction(config)
        
        # 训练历史
        self.train_losses = []
        self.val_losses = []
        self.loss_components = defaultdict(list)
    
    def train(self, train_dataloader, val_dataloader=None, num_epochs=100, save_path=None):
        """训练模型"""
        logger.info("开始训练高级NeuralVDB...")
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(num_epochs):
            # 渐进式训练
            if self.config.progressive_training:
                self._adjust_training_parameters(epoch, num_epochs)
            
            # 训练
            train_loss, train_components = self._train_epoch(train_dataloader)
            self.train_losses.append(train_loss)
            
            # 记录损失组件
            for key, value in train_components.items():
                self.loss_components[key].append(value)
            
            # 验证
            if val_dataloader is not None:
                val_loss, val_components = self._validate(val_dataloader)
                self.val_losses.append(val_loss)
                
                # 更新学习率
                self.scheduler.step(val_loss)
                
                logger.info(f"Epoch {epoch+1}/{num_epochs}: "
                          f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
                
                # 早停
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    if save_path:
                        self.save_model(save_path)
                else:
                    patience_counter += 1
                    if patience_counter >= 15:
                        logger.info("早停触发")
                        break
            else:
                logger.info(f"Epoch {epoch+1}/{num_epochs}: Train Loss: {train_loss:.6f}")
        
        logger.info("训练完成")
    
    def _adjust_training_parameters(self, epoch: int, total_epochs: int):
        """调整训练参数（渐进式训练）"""
        progress = epoch / total_epochs
        
        # 调整损失权重
        if progress < 0.3:
            # 早期阶段：专注于占用损失
            self.criterion.config.occupancy_weight = 1.0
            self.criterion.config.smoothness_weight = 0.0
            self.criterion.config.sparsity_weight = 0.0
        elif progress < 0.7:
            # 中期阶段：逐渐增加其他损失
            self.criterion.config.occupancy_weight = 1.0
            self.criterion.config.smoothness_weight = 0.1 * progress
            self.criterion.config.sparsity_weight = 0.01 * progress
        else:
            # 后期阶段：完整损失
            self.criterion.config.occupancy_weight = 1.0
            self.criterion.config.smoothness_weight = 0.1
            self.criterion.config.sparsity_weight = 0.01
    
    def _train_epoch(self, dataloader):
        """训练一个epoch"""
        self.sparse_grid.feature_network.train()
        self.sparse_grid.occupancy_network.train()
        
        total_loss = 0
        total_components = defaultdict(float)
        num_batches = 0
        
        for points, occupancies in dataloader:
            points = points.to(self.device)
            occupancies = occupancies.to(self.device)
            
            # 前向传播
            features = self.sparse_grid.feature_network(points)
            predictions = self.sparse_grid.occupancy_network(features)
            
            # 计算损失
            loss, components = self.criterion(predictions.squeeze(), occupancies, features, points)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            for key, value in components.items():
                total_components[key] += value
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        avg_components = {key: value / num_batches for key, value in total_components.items()}
        
        return avg_loss, avg_components
    
    def _validate(self, dataloader):
        """验证"""
        self.sparse_grid.feature_network.eval()
        self.sparse_grid.occupancy_network.eval()
        
        total_loss = 0
        total_components = defaultdict(float)
        num_batches = 0
        
        with torch.no_grad():
            for points, occupancies in dataloader:
                points = points.to(self.device)
                occupancies = occupancies.to(self.device)
                
                features = self.sparse_grid.feature_network(points)
                predictions = self.sparse_grid.occupancy_network(features)
                
                loss, components = self.criterion(predictions.squeeze(), occupancies, features, points)
                
                total_loss += loss.item()
                for key, value in components.items():
                    total_components[key] += value
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        avg_components = {key: value / num_batches for key, value in total_components.items()}
        
        return avg_loss, avg_components
    
    def predict(self, points: np.ndarray) -> np.ndarray:
        """预测占用值"""
        self.sparse_grid.feature_network.eval()
        self.sparse_grid.occupancy_network.eval()
        
        points_tensor = torch.FloatTensor(points).to(self.device)
        
        with torch.no_grad():
            features = self.sparse_grid.feature_network(points_tensor)
            predictions = self.sparse_grid.occupancy_network(features)
        
        return predictions.cpu().numpy().squeeze()
    
    def save_model(self, path: str):
        """保存模型"""
        torch.save({
            'feature_network_state_dict': self.sparse_grid.feature_network.state_dict(),
            'occupancy_network_state_dict': self.sparse_grid.occupancy_network.state_dict(),
            'config': self.config,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'loss_components': dict(self.loss_components)
        }, path)
        logger.info(f"模型已保存到: {path}")
    
    def load_model(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.sparse_grid.feature_network.load_state_dict(checkpoint['feature_network_state_dict'])
        self.sparse_grid.occupancy_network.load_state_dict(checkpoint['occupancy_network_state_dict'])
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.loss_components = defaultdict(list, checkpoint.get('loss_components', {}))
        logger.info(f"模型已从 {path} 加载")

class AdvancedNeuralVDB:
    """高级NeuralVDB主类"""
    
    def __init__(self, config: AdvancedNeuralVDBConfig):
        self.config = config
        self.sparse_grid = AdvancedSparseVoxelGrid(config)
        self.trainer = None
    
    def fit(self, points: np.ndarray, occupancies: np.ndarray, 
            train_ratio: float = 0.8, num_epochs: int = 100, 
            save_path: str = 'advanced_neural_vdb_model.pth'):
        """训练高级NeuralVDB模型"""
        logger.info("开始高级NeuralVDB训练流程...")
        
        # 构建稀疏体素网格
        self.sparse_grid.build_from_points(points, occupancies)
        
        # 准备数据
        from neural_vdb import NeuralVDBDataset
        dataset = NeuralVDBDataset(points, occupancies)
        
        # 分割训练集和验证集
        train_size = int(train_ratio * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.config.batch_size, shuffle=True
        )
        val_dataloader = torch.utils.data.DataLoader(
            val_dataset, batch_size=self.config.batch_size, shuffle=False
        )
        
        # 创建训练器
        self.trainer = AdvancedNeuralVDBTrainer(self.sparse_grid, self.config)
        
        # 训练模型
        self.trainer.train(train_dataloader, val_dataloader, num_epochs, save_path)
        
        logger.info("高级NeuralVDB训练完成")
    
    def predict(self, points: np.ndarray) -> np.ndarray:
        """预测占用值"""
        if self.trainer is None:
            raise ValueError("模型尚未训练，请先调用fit方法")
        
        return self.trainer.predict(points)
    
    def save(self, path: str):
        """保存整个高级NeuralVDB模型"""
        if self.trainer is None:
            raise ValueError("模型尚未训练，请先调用fit方法")
        
        self.trainer.save_model(path)
    
    def load(self, path: str):
        """加载高级NeuralVDB模型"""
        if self.trainer is None:
            self.trainer = AdvancedNeuralVDBTrainer(self.sparse_grid, self.config)
        
        self.trainer.load_model(path)
    
    def get_memory_usage(self) -> Dict[str, float]:
        """获取内存使用情况"""
        memory_info = {}
        
        # 计算网络参数内存
        if self.sparse_grid.feature_network is not None:
            feature_params = sum(p.numel() for p in self.sparse_grid.feature_network.parameters())
            memory_info['feature_network_mb'] = feature_params * 4 / (1024 * 1024)  # 假设float32
        
        if self.sparse_grid.occupancy_network is not None:
            occupancy_params = sum(p.numel() for p in self.sparse_grid.occupancy_network.parameters())
            memory_info['occupancy_network_mb'] = occupancy_params * 4 / (1024 * 1024)
        
        # 计算八叉树内存
        def count_octree_nodes(node):
            count = 1
            if node.children is not None:
                for child in node.children:
                    count += count_octree_nodes(child)
            return count
        
        if self.sparse_grid.root is not None:
            node_count = count_octree_nodes(self.sparse_grid.root)
            memory_info['octree_nodes'] = node_count
            memory_info['octree_memory_mb'] = node_count * 100 / (1024 * 1024)  # 估算
        
        return memory_info

def main():
    """主函数 - 演示高级NeuralVDB的使用"""
    logger.info("高级NeuralVDB演示开始...")
    
    # 创建高级配置
    config = AdvancedNeuralVDBConfig(
        voxel_size=1.0,
        max_depth=6,
        feature_dim=64,
        hidden_dims=[256, 512, 512, 256, 128],
        learning_rate=1e-3,
        batch_size=1024,
        adaptive_resolution=True,
        multi_scale_features=True,
        progressive_training=True,
        feature_compression=True,
        quantization_bits=8
    )
    
    # 创建示例数据
    from neural_vdb import create_sample_data
    points, occupancies = create_sample_data(10000)
    logger.info(f"创建了 {len(points)} 个训练点")
    
    # 创建高级NeuralVDB模型
    advanced_neural_vdb = AdvancedNeuralVDB(config)
    
    # 训练模型
    advanced_neural_vdb.fit(points, occupancies, num_epochs=50)
    
    # 测试预测
    test_points = np.random.rand(100, 3) * 100
    predictions = advanced_neural_vdb.predict(test_points)
    
    logger.info(f"预测完成，预测值范围: [{predictions.min():.3f}, {predictions.max():.3f}]")
    
    # 获取内存使用情况
    memory_info = advanced_neural_vdb.get_memory_usage()
    logger.info(f"内存使用情况: {memory_info}")
    
    logger.info("高级NeuralVDB演示完成")

if __name__ == "__main__":
    main() 