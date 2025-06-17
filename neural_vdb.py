#!/usr/bin/env python3
"""
NeuralVDB: Efficient Sparse Volumetric Neural Representations
基于论文 "NeuralVDB: Efficient Sparse Volumetric Neural Representations" 的实现

主要特性:
- 稀疏体素表示
- 分层数据结构
- 高效的神经网络编码
- 自适应分辨率
- 内存优化
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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class NeuralVDBConfig:
    """NeuralVDB配置参数"""
    # 体素参数
    voxel_size: float = 1.0
    max_depth: int = 8
    min_depth: int = 3
    
    # 神经网络参数
    feature_dim: int = 32
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
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [256, 512, 512, 256, 128]

class OctreeNode:
    """八叉树节点"""
    
    def __init__(self, center: np.ndarray, size: float, depth: int = 0):
        self.center = center
        self.size = size
        self.depth = depth
        self.children = None
        self.features = None
        self.occupancy = 0.0
        self.is_leaf = True
        
    def subdivide(self):
        """细分节点"""
        if self.children is not None:
            return
            
        self.children = []
        half_size = self.size / 2
        quarter_size = half_size / 2
        
        for i in range(8):
            # 计算子节点中心
            offset = np.array([
                (i & 1) * half_size - quarter_size,
                ((i >> 1) & 1) * half_size - quarter_size,
                ((i >> 2) & 1) * half_size - quarter_size
            ])
            child_center = self.center + offset
            
            child = OctreeNode(child_center, half_size, self.depth + 1)
            self.children.append(child)
        
        self.is_leaf = False

class SparseVoxelGrid:
    """稀疏体素网格"""
    
    def __init__(self, config: NeuralVDBConfig):
        self.config = config
        self.root = None
        self.feature_network = None
        self.occupancy_network = None
        
    def build_from_points(self, points: np.ndarray, occupancies: np.ndarray):
        """从点云构建稀疏体素网格"""
        logger.info("构建稀疏体素网格...")
        
        # 计算边界框
        min_coords = points.min(axis=0)
        max_coords = points.max(axis=0)
        center = (min_coords + max_coords) / 2
        size = max(max_coords - min_coords)
        
        # 创建根节点
        self.root = OctreeNode(center, size)
        
        # 递归构建八叉树
        self._build_octree_recursive(self.root, points, occupancies)
        
        # 初始化神经网络
        self._init_networks()
        
        logger.info(f"稀疏体素网格构建完成，根节点大小: {size}")
    
    def _build_octree_recursive(self, node: OctreeNode, points: np.ndarray, occupancies: np.ndarray):
        """递归构建八叉树"""
        # 找到属于当前节点的点
        node_points, node_occupancies = self._get_points_in_node(node, points, occupancies)
        
        if len(node_points) == 0:
            return
        
        # 计算节点占用率
        node.occupancy = np.mean(node_occupancies)
        
        # 判断是否需要细分
        should_subdivide = (
            node.depth < self.config.max_depth and
            len(node_points) > 10 and
            node.occupancy > self.config.sparsity_threshold
        )
        
        if should_subdivide:
            node.subdivide()
            
            # 递归处理子节点
            for child in node.children:
                self._build_octree_recursive(child, points, occupancies)
    
    def _get_points_in_node(self, node: OctreeNode, points: np.ndarray, occupancies: np.ndarray):
        """获取属于节点的点"""
        half_size = node.size / 2
        
        # 计算点是否在节点范围内
        in_node = np.all(
            (points >= node.center - half_size) & 
            (points < node.center + half_size), 
            axis=1
        )
        
        return points[in_node], occupancies[in_node]
    
    def _init_networks(self):
        """初始化神经网络"""
        # 特征网络
        self.feature_network = FeatureNetwork(
            input_dim=3,
            feature_dim=self.config.feature_dim,
            hidden_dims=self.config.hidden_dims,
            activation=self.config.activation,
            dropout=self.config.dropout
        )
        
        # 占用网络
        self.occupancy_network = OccupancyNetwork(
            feature_dim=self.config.feature_dim,
            hidden_dims=[128, 64, 32],
            activation=self.config.activation,
            dropout=self.config.dropout
        )

class FeatureNetwork(nn.Module):
    """特征网络 - 将3D坐标映射到特征向量"""
    
    def __init__(self, input_dim: int, feature_dim: int, hidden_dims: List[int], 
                 activation: str = 'relu', dropout: float = 0.1):
        super(FeatureNetwork, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                self._get_activation(activation),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(prev_dim, feature_dim))
        
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
    
    def forward(self, x):
        return self.network(x)

class OccupancyNetwork(nn.Module):
    """占用网络 - 从特征向量预测占用值"""
    
    def __init__(self, feature_dim: int, hidden_dims: List[int], 
                 activation: str = 'relu', dropout: float = 0.1):
        super(OccupancyNetwork, self).__init__()
        
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
        layers.append(nn.Sigmoid())  # 输出0-1之间的占用值
        
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

class NeuralVDBDataset(torch.utils.data.Dataset):
    """NeuralVDB数据集"""
    
    def __init__(self, points: np.ndarray, occupancies: np.ndarray):
        self.points = torch.FloatTensor(points)
        self.occupancies = torch.FloatTensor(occupancies)
    
    def __len__(self):
        return len(self.points)
    
    def __getitem__(self, idx):
        return self.points[idx], self.occupancies[idx]

class NeuralVDBTrainer:
    """NeuralVDB训练器"""
    
    def __init__(self, sparse_grid: SparseVoxelGrid, config: NeuralVDBConfig, device: str = 'auto'):
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
        
        # 设置损失函数
        self.criterion = nn.BCELoss()
        
        # 训练历史
        self.train_losses = []
        self.val_losses = []
    
    def train(self, train_dataloader, val_dataloader=None, num_epochs=100, save_path=None):
        """训练模型"""
        logger.info("开始训练NeuralVDB...")
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(num_epochs):
            # 训练
            train_loss = self._train_epoch(train_dataloader)
            self.train_losses.append(train_loss)
            
            # 验证
            if val_dataloader is not None:
                val_loss = self._validate(val_dataloader)
                self.val_losses.append(val_loss)
                
                logger.info(f"Epoch {epoch+1}/{num_epochs}: Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
                
                # 早停
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    if save_path:
                        self.save_model(save_path)
                else:
                    patience_counter += 1
                    if patience_counter >= 10:
                        logger.info("早停触发")
                        break
            else:
                logger.info(f"Epoch {epoch+1}/{num_epochs}: Train Loss: {train_loss:.6f}")
        
        logger.info("训练完成")
    
    def _train_epoch(self, dataloader):
        """训练一个epoch"""
        self.sparse_grid.feature_network.train()
        self.sparse_grid.occupancy_network.train()
        
        total_loss = 0
        num_batches = 0
        
        for points, occupancies in dataloader:
            points = points.to(self.device)
            occupancies = occupancies.to(self.device)
            
            # 前向传播
            features = self.sparse_grid.feature_network(points)
            predictions = self.sparse_grid.occupancy_network(features)
            
            # 计算损失
            loss = self.criterion(predictions.squeeze(), occupancies)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches
    
    def _validate(self, dataloader):
        """验证"""
        self.sparse_grid.feature_network.eval()
        self.sparse_grid.occupancy_network.eval()
        
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for points, occupancies in dataloader:
                points = points.to(self.device)
                occupancies = occupancies.to(self.device)
                
                features = self.sparse_grid.feature_network(points)
                predictions = self.sparse_grid.occupancy_network(features)
                
                loss = self.criterion(predictions.squeeze(), occupancies)
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
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
            'val_losses': self.val_losses
        }, path)
        logger.info(f"模型已保存到: {path}")
    
    def load_model(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.sparse_grid.feature_network.load_state_dict(checkpoint['feature_network_state_dict'])
        self.sparse_grid.occupancy_network.load_state_dict(checkpoint['occupancy_network_state_dict'])
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        logger.info(f"模型已从 {path} 加载")

class NeuralVDB:
    """NeuralVDB主类"""
    
    def __init__(self, config: NeuralVDBConfig):
        self.config = config
        self.sparse_grid = SparseVoxelGrid(config)
        self.trainer = None
    
    def fit(self, points: np.ndarray, occupancies: np.ndarray, 
            train_ratio: float = 0.8, num_epochs: int = 100, 
            save_path: str = 'neural_vdb_model.pth'):
        """训练NeuralVDB模型"""
        logger.info("开始NeuralVDB训练流程...")
        
        # 构建稀疏体素网格
        self.sparse_grid.build_from_points(points, occupancies)
        
        # 准备数据
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
        self.trainer = NeuralVDBTrainer(self.sparse_grid, self.config)
        
        # 训练模型
        self.trainer.train(train_dataloader, val_dataloader, num_epochs, save_path)
        
        logger.info("NeuralVDB训练完成")
    
    def predict(self, points: np.ndarray) -> np.ndarray:
        """预测占用值"""
        if self.trainer is None:
            raise ValueError("模型尚未训练，请先调用fit方法")
        
        return self.trainer.predict(points)
    
    def save(self, path: str):
        """保存整个NeuralVDB模型"""
        if self.trainer is None:
            raise ValueError("模型尚未训练，请先调用fit方法")
        
        self.trainer.save_model(path)
    
    def load(self, path: str):
        """加载NeuralVDB模型"""
        if self.trainer is None:
            self.trainer = NeuralVDBTrainer(self.sparse_grid, self.config)
        
        self.trainer.load_model(path)
    
    def visualize_octree(self, max_depth: int = 4, save_path: str = None):
        """可视化八叉树结构"""
        if self.sparse_grid.root is None:
            logger.warning("八叉树尚未构建")
            return
        
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        self._visualize_node_recursive(self.sparse_grid.root, ax, max_depth)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('NeuralVDB Octree Structure')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
    
    def _visualize_node_recursive(self, node: OctreeNode, ax, max_depth: int):
        """递归可视化节点"""
        if node.depth > max_depth:
            return
        
        # 绘制当前节点
        half_size = node.size / 2
        x, y, z = node.center
        
        # 根据占用率设置颜色
        color = plt.cm.viridis(node.occupancy)
        
        # 绘制立方体
        for i in range(8):
            offset = np.array([
                (i & 1) * half_size - half_size/2,
                ((i >> 1) & 1) * half_size - half_size/2,
                ((i >> 2) & 1) * half_size - half_size/2
            ])
            corner = node.center + offset
            
            ax.scatter(corner[0], corner[1], corner[2], 
                      c=[color], s=50, alpha=0.7)
        
        # 递归处理子节点
        if node.children is not None:
            for child in node.children:
                self._visualize_node_recursive(child, ax, max_depth)

def create_sample_data(n_points: int = 10000) -> Tuple[np.ndarray, np.ndarray]:
    """创建示例数据"""
    # 生成随机点
    points = np.random.rand(n_points, 3) * 100  # 0-100范围内的点
    
    # 创建简单的几何形状（球体）
    center = np.array([50, 50, 50])
    radius = 20
    
    distances = np.linalg.norm(points - center, axis=1)
    occupancies = (distances < radius).astype(np.float32)
    
    return points, occupancies

def main():
    """主函数 - 演示NeuralVDB的使用"""
    logger.info("NeuralVDB演示开始...")
    
    # 创建配置
    config = NeuralVDBConfig(
        voxel_size=1.0,
        max_depth=6,
        feature_dim=32,
        learning_rate=1e-3,
        batch_size=512
    )
    
    # 创建示例数据
    points, occupancies = create_sample_data(5000)
    logger.info(f"创建了 {len(points)} 个训练点")
    
    # 创建NeuralVDB模型
    neural_vdb = NeuralVDB(config)
    
    # 训练模型
    neural_vdb.fit(points, occupancies, num_epochs=50, save_path='neural_vdb_model.pth')
    
    # 测试预测
    test_points = np.random.rand(100, 3) * 100
    predictions = neural_vdb.predict(test_points)
    
    logger.info(f"预测完成，预测值范围: [{predictions.min():.3f}, {predictions.max():.3f}]")
    
    # 可视化八叉树
    neural_vdb.visualize_octree(max_depth=4, save_path='octree_visualization.png')
    
    logger.info("NeuralVDB演示完成")

if __name__ == "__main__":
    main() 