"""
NeuralVDB Core Module

This module contains the core NeuralVDB classes and configurations.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass

from .octree import SparseVoxelGrid, AdvancedSparseVoxelGrid
from .trainer import NeuralVDBTrainer, AdvancedNeuralVDBTrainer
from .dataset import NeuralVDBDataset

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
    hidden_dims: list[int] = None
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


@dataclass
class AdvancedNeuralVDBConfig:
    """高级NeuralVDB配置参数"""
    # 基础参数
    voxel_size: float = 1.0
    max_depth: int = 8
    min_depth: int = 3
    
    # 神经网络参数
    feature_dim: int = 64
    hidden_dims: list[int] = None
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


class NeuralVDB:
    """
    NeuralVDB主类 - 稀疏体素神经表示
    
    这是基础版本的NeuralVDB实现，提供核心功能：
    - 稀疏体素表示
    - 八叉树数据结构
    - 神经网络编码
    """
    
    def __init__(self, config: NeuralVDBConfig):
        """
        初始化NeuralVDB
        
        Args:
            config: NeuralVDB配置
        """
        self.config = config
        self.sparse_grid = None
        self.trainer = None
        
        logger.info("NeuralVDB初始化完成")
    
    def fit(
        self,
        points: np.ndarray,
        occupancies: np.ndarray,
        train_ratio: float = 0.8,
        num_epochs: int = 100,
        save_path: str = 'neural_vdb_model.pth',
        device: str = 'auto',
    ) -> dict[str, Any]:
        """
        训练NeuralVDB模型
        
        Args:
            points: 3D坐标点 (N, 3)
            occupancies: 占用值 (N, )
            train_ratio: 训练集比例
            num_epochs: 训练轮数
            save_path: 模型保存路径
            device: 计算设备
            
        Returns:
            训练统计信息
        """
        logger.info("开始训练NeuralVDB模型...")
        
        # 构建稀疏体素网格
        self.sparse_grid = SparseVoxelGrid(self.config)
        self.sparse_grid.build_from_points(points, occupancies)
        
        # 创建训练器
        self.trainer = NeuralVDBTrainer(
            self.sparse_grid, self.config, device=device
        )
        
        # 准备训练数据
        dataset = NeuralVDBDataset(points, occupancies)
        
        # 分割训练和验证集
        train_size = int(len(dataset) * train_ratio)
        val_size = len(dataset) - train_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.config.batch_size, shuffle=True
        )
        
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=self.config.batch_size, shuffle=False
        )
        
        # 开始训练
        training_stats = self.trainer.train(
            train_loader, val_loader, num_epochs=num_epochs, save_path=save_path
        )
        
        logger.info("NeuralVDB训练完成")
        return training_stats
    
    def predict(self, points: np.ndarray) -> np.ndarray:
        """
        预测点的占用概率
        
        Args:
            points: 3D坐标点 (N, 3)
            
        Returns:
            占用概率 (N, )
        """
        if self.trainer is None:
            raise ValueError("模型未训练，请先调用fit()方法")
        
        return self.trainer.predict(points)
    
    def save(self, path: str) -> None:
        """保存模型"""
        if self.trainer is None:
            raise ValueError("模型未训练，无法保存")
        
        self.trainer.save_model(path)
        logger.info(f"模型已保存到: {path}")
    
    def load(self, path: str) -> None:
        """加载模型"""
        # 创建稀疏网格和训练器
        self.sparse_grid = SparseVoxelGrid(self.config)
        self.trainer = NeuralVDBTrainer(self.sparse_grid, self.config)
        
        # 加载模型
        self.trainer.load_model(path)
        logger.info(f"模型已从 {path} 加载")
    
    def visualize_octree(self, max_depth: int = 4, save_path: str = None):
        """
        可视化八叉树结构
        
        Args:
            max_depth: 最大显示深度
            save_path: 保存路径
        """
        if self.sparse_grid is None:
            raise ValueError("稀疏网格未构建，请先调用fit()方法")
        
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # 递归绘制节点
        self._visualize_node_recursive(self.sparse_grid.root, ax, max_depth)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'NeuralVDB八叉树结构 (最大深度: {max_depth})')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"八叉树可视化已保存到: {save_path}")
        else:
            plt.show()
    
    def _visualize_node_recursive(self, node, ax, max_depth: int):
        """递归可视化八叉树节点"""
        if node is None or node.depth > max_depth:
            return
        
        # 绘制节点边界框
        center = node.center
        size = node.size
        
        # 创建立方体的8个顶点
        vertices = []
        for i in range(8):
            x = center[0] + (size/2) * (1 if i & 1 else -1)
            y = center[1] + (size/2) * (1 if i & 2 else -1)
            z = center[2] + (size/2) * (1 if i & 4 else -1)
            vertices.append([x, y, z])
        
        vertices = np.array(vertices)
        
        # 绘制立方体边框
        # 定义立方体的12条边
        edges = [
            [0, 1], [1, 3], [3, 2], [2, 0], # 底面
            [4, 5], [5, 7], [7, 6], [6, 4], # 顶面
            [0, 4], [1, 5], [2, 6], [3, 7]   # 垂直边
        ]
        
        for edge in edges:
            points = vertices[edge]
            ax.plot3D(*points.T, 'b-', alpha=0.3)
        
        # 如果有子节点，递归绘制
        if not node.is_leaf and node.children:
            for child in node.children:
                self._visualize_node_recursive(child, ax, max_depth)
    
    def get_model_info(self) -> dict[str, Any]:
        """获取模型信息"""
        info = {
            'config': self.config, 'is_trained': self.trainer is not None, 'has_sparse_grid': self.sparse_grid is not None
        }
        
        if self.sparse_grid and self.sparse_grid.root:
            info['octree_depth'] = self._get_max_depth(self.sparse_grid.root)
            info['total_nodes'] = self._count_nodes(self.sparse_grid.root)
        
        return info
    
    def _get_max_depth(self, node) -> int:
        """获取八叉树最大深度"""
        if node is None or node.is_leaf:
            return node.depth if node else 0
        
        max_child_depth = 0
        if node.children:
            for child in node.children:
                max_child_depth = max(max_child_depth, self._get_max_depth(child))
        
        return max_child_depth
    
    def _count_nodes(self, node) -> int:
        """计算八叉树节点数量"""
        if node is None:
            return 0
        
        count = 1
        if not node.is_leaf and node.children:
            for child in node.children:
                count += self._count_nodes(child)
        
        return count


class AdvancedNeuralVDB:
    """
    高级NeuralVDB类 - 增强的稀疏体素神经表示
    
    这是高级版本的NeuralVDB实现，包含以下增强功能：
    - 自适应分辨率
    - 多尺度特征提取
    - 渐进式训练
    - 特征压缩
    """
    
    def __init__(self, config: AdvancedNeuralVDBConfig):
        """
        初始化高级NeuralVDB
        
        Args:
            config: 高级NeuralVDB配置
        """
        self.config = config
        self.sparse_grid = None
        self.trainer = None
        
        logger.info("高级NeuralVDB初始化完成")
    
    def fit(
        self,
        points: np.ndarray,
        occupancies: np.ndarray,
        train_ratio: float = 0.8,
        num_epochs: int = 100,
        save_path: str = 'advanced_neural_vdb_model.pth',
        device: str = 'auto',
    ) -> dict[str, Any]:
        """
        训练高级NeuralVDB模型
        
        Args:
            points: 3D坐标点 (N, 3)
            occupancies: 占用值 (N, )
            train_ratio: 训练集比例
            num_epochs: 训练轮数
            save_path: 模型保存路径
            device: 计算设备
            
        Returns:
            训练统计信息
        """
        logger.info("开始训练高级NeuralVDB模型...")
        
        # 构建高级稀疏体素网格
        self.sparse_grid = AdvancedSparseVoxelGrid(self.config)
        self.sparse_grid.build_from_points(points, occupancies)
        
        # 创建高级训练器
        self.trainer = AdvancedNeuralVDBTrainer(
            self.sparse_grid, self.config, device=device
        )
        
        # 准备训练数据
        dataset = NeuralVDBDataset(points, occupancies)
        
        # 分割训练和验证集
        train_size = int(len(dataset) * train_ratio)
        val_size = len(dataset) - train_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.config.batch_size, shuffle=True
        )
        
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=self.config.batch_size, shuffle=False
        )
        
        # 开始训练
        training_stats = self.trainer.train(
            train_loader, val_loader, num_epochs=num_epochs, save_path=save_path
        )
        
        logger.info("高级NeuralVDB训练完成")
        return training_stats
    
    def predict(self, points: np.ndarray) -> np.ndarray:
        """
        预测点的占用概率
        
        Args:
            points: 3D坐标点 (N, 3)
            
        Returns:
            占用概率 (N, )
        """
        if self.trainer is None:
            raise ValueError("模型未训练，请先调用fit()方法")
        
        return self.trainer.predict(points)
    
    def save(self, path: str) -> None:
        """保存模型"""
        if self.trainer is None:
            raise ValueError("模型未训练，无法保存")
        
        self.trainer.save_model(path)
        logger.info(f"高级模型已保存到: {path}")
    
    def load(self, path: str) -> None:
        """加载模型"""
        # 创建稀疏网格和训练器
        self.sparse_grid = AdvancedSparseVoxelGrid(self.config)
        self.trainer = AdvancedNeuralVDBTrainer(self.sparse_grid, self.config)
        
        # 加载模型
        self.trainer.load_model(path)
        logger.info(f"高级模型已从 {path} 加载")
    
    def get_memory_usage(self) -> dict[str, float]:
        """获取内存使用情况"""
        if self.sparse_grid is None:
            return {'total_memory_mb': 0.0}
        
        # 计算八叉树节点内存
        def count_octree_nodes(node):
            if node is None:
                return 0
            count = 1
            if not node.is_leaf and node.children:
                for child in node.children:
                    count += count_octree_nodes(child)
            return count
        
        num_nodes = count_octree_nodes(self.sparse_grid.root)
        
        # 估算内存使用
        node_memory = num_nodes * 200  # 每个节点约200字节
        network_memory = 0
        
        if self.trainer:
            # 计算网络参数内存
            total_params = sum(p.numel() for p in self.sparse_grid.feature_network.parameters())
            total_params += sum(p.numel() for p in self.sparse_grid.occupancy_network.parameters())
            network_memory = total_params * 4  # float32
        
        total_memory_mb = (node_memory + network_memory) / (1024 * 1024)
        
        return {
            'octree_nodes': num_nodes, 'node_memory_mb': node_memory / (
                1024 * 1024,
            )
        }
    
    def get_model_info(self) -> dict[str, Any]:
        """获取模型信息"""
        info = {
            'config': self.config, 'is_trained': self.trainer is not None, 'has_sparse_grid': self.sparse_grid is not None, 'memory_usage': self.get_memory_usage(
            )
        }
        
        return info 