"""
Octree Data Structures for NeuralVDB

This module implements octree-based sparse voxel representations
for efficient storage and processing of volumetric data.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
import logging
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)


class OctreeNode:
    """八叉树节点 - 基础版本"""
    
    def __init__(self, center: np.ndarray, size: float, depth: int = 0):
        """
        初始化八叉树节点
        
        Args:
            center: 节点中心坐标 (3, )
            size: 节点尺寸
            depth: 节点深度
        """
        self.center = center
        self.size = size
        self.depth = depth
        self.children = None
        self.features = None
        self.occupancy = 0.0
        self.is_leaf = True
        
    def subdivide(self):
        """细分节点为8个子节点"""
        if self.children is not None:
            return
            
        self.children = []
        half_size = self.size / 2
        quarter_size = half_size / 2
        
        # 创建8个子节点
        for i in range(8):
            # 计算子节点中心偏移
            offset = np.array([
                (
                    i & 1,
                )
            ])
            child_center = self.center + offset
            
            child = OctreeNode(child_center, half_size, self.depth + 1)
            self.children.append(child)
        
        self.is_leaf = False
    
    def get_points_in_node(self, points: np.ndarray) -> np.ndarray:
        """获取属于当前节点的点的索引"""
        half_size = self.size / 2
        
        # 检查点是否在节点边界内
        in_node = np.all(
            (points >= self.center - half_size) & 
            (points < self.center + half_size), axis=1
        )
        
        return np.where(in_node)[0]
    
    def compute_occupancy(self, points: np.ndarray, occupancies: np.ndarray):
        """计算节点占用率"""
        indices = self.get_points_in_node(points)
        
        if len(indices) > 0:
            self.occupancy = np.mean(occupancies[indices])
        else:
            self.occupancy = 0.0
    
    def should_subdivide(self, points: np.ndarray, occupancies: np.ndarray, config) -> bool:
        """判断是否应该细分节点"""
        indices = self.get_points_in_node(points)
        
        # 细分条件
        conditions = [
            self.depth < config.max_depth, # 未达到最大深度
            len(indices) > 10, # 包含足够多的点
            self.occupancy > config.sparsity_threshold  # 占用率足够高
        ]
        
        return all(conditions)


class AdaptiveOctreeNode(OctreeNode):
    """自适应八叉树节点 - 高级版本"""
    
    def __init__(self, center: np.ndarray, size: float, depth: int = 0):
        """
        初始化自适应八叉树节点
        
        Args:
            center: 节点中心坐标 (3, )
            size: 节点尺寸
            depth: 节点深度
        """
        super().__init__(center, size, depth)
        
        # 额外的自适应属性
        self.importance = 0.0      # 节点重要性
        self.density = 0.0         # 点密度
        self.gradient = 0.0        # 梯度信息
        self.variance = 0.0        # 占用率方差
        
    def compute_importance(self, points: np.ndarray, occupancies: np.ndarray):
        """
        计算节点重要性，用于自适应细分决策
        
        Args:
            points: 所有点坐标
            occupancies: 所有点的占用值
        """
        indices = self.get_points_in_node(points)
        
        if len(indices) == 0:
            self.importance = 0.0
            self.density = 0.0
            self.gradient = 0.0
            self.variance = 0.0
            return
        
        node_points = points[indices]
        node_occupancies = occupancies[indices]
        
        # 1. 计算点密度
        self.density = len(node_points) / (self.size ** 3)
        
        # 2. 计算占用率
        self.occupancy = np.mean(node_occupancies)
        
        # 3. 计算占用率方差
        self.variance = np.var(node_occupancies)
        
        # 4. 计算梯度信息（使用KNN）
        if len(node_points) > 1:
            try:
                tree = cKDTree(node_points)
                k = min(5, len(node_points))
                distances, neighbors = tree.query(node_points, k=k)
                
                gradients = []
                for i in range(len(node_points)):
                    if k > 1:
                        # 计算局部梯度（占用率的标准差）
                        neighbor_occupancies = node_occupancies[neighbors[i][1:]]
                        local_gradient = np.std(neighbor_occupancies)
                        gradients.append(local_gradient)
                
                if gradients:
                    self.gradient = np.mean(gradients)
                else:
                    self.gradient = 0.0
            except:
                self.gradient = 0.0
        else:
            self.gradient = 0.0
        
        # 5. 综合重要性计算
        # 重要性 = 密度权重 * 密度 + 占用率权重 * 占用率 + 梯度权重 * 梯度 + 方差权重 * 方差
        self.importance = (
            0.3 * min(self.density / 1000, 1.0) +  # 密度归一化
            0.3 * self.occupancy +
            0.2 * min(self.gradient, 1.0) +        # 梯度归一化
            0.2 * min(self.variance, 1.0)          # 方差归一化
        )
    
    def should_subdivide_adaptive(
        self,
        points: np.ndarray,
        occupancies: np.ndarray,
        config,
    )
        """自适应细分判断"""
        # 首先计算重要性
        self.compute_importance(points, occupancies)
        
        # 基础条件
        basic_conditions = [
            self.depth < config.max_depth, self.depth >= config.min_depth, len(
                self.get_points_in_node,
            )
        ]
        
        if not all(basic_conditions):
            return False
        
        # 自适应条件
        adaptive_conditions = [
            self.importance > 0.3, # 重要性阈值
            self.variance > 0.1, # 方差阈值
            self.gradient > 0.05    # 梯度阈值
        ]
        
        # 至少满足一个自适应条件
        return any(adaptive_conditions)


class SparseVoxelGrid:
    """稀疏体素网格 - 基础版本"""
    
    def __init__(self, config):
        """
        初始化稀疏体素网格
        
        Args:
            config: NeuralVDB配置
        """
        self.config = config
        self.root = None
        self.feature_network = None
        self.occupancy_network = None
        
    def build_from_points(self, points: np.ndarray, occupancies: np.ndarray):
        """
        从点云构建稀疏体素网格
        
        Args:
            points: 3D坐标点 (N, 3)
            occupancies: 占用值 (N, )
        """
        logger.info("构建稀疏体素网格...")
        
        # 计算场景边界框
        min_coords = points.min(axis=0)
        max_coords = points.max(axis=0)
        center = (min_coords + max_coords) / 2
        size = max(max_coords - min_coords) * 1.1  # 稍微扩大边界
        
        # 创建根节点
        self.root = OctreeNode(center, size)
        
        # 递归构建八叉树
        self._build_octree_recursive(self.root, points, occupancies)
        
        # 初始化神经网络
        self._init_networks()
        
        logger.info(f"稀疏体素网格构建完成，根节点大小: {size:.2f}")
    
    def _build_octree_recursive(
        self,
        node: OctreeNode,
        points: np.ndarray,
        occupancies: np.ndarray,
    )
        """递归构建八叉树"""
        # 计算节点占用率
        node.compute_occupancy(points, occupancies)
        
        # 判断是否需要细分
        if node.should_subdivide(points, occupancies, self.config):
            node.subdivide()
            
            # 递归处理子节点
            for child in node.children:
                self._build_octree_recursive(child, points, occupancies)
    
    def _init_networks(self):
        """初始化神经网络"""
        from .networks import FeatureNetwork, OccupancyNetwork
        
        # 特征网络
        self.feature_network = FeatureNetwork(
            input_dim=3, feature_dim=self.config.feature_dim, hidden_dims=self.config.hidden_dims, activation=self.config.activation, dropout=self.config.dropout
        )
        
        # 占用网络
        self.occupancy_network = OccupancyNetwork(
            feature_dim=self.config.feature_dim, hidden_dims=[128, 64, 32], activation=self.config.activation, dropout=self.config.dropout
        )
    
    def query_occupancy(self, points: np.ndarray) -> np.ndarray:
        """查询点的占用概率"""
        if self.feature_network is None or self.occupancy_network is None:
            raise ValueError("神经网络未初始化")
        
        # 转换为tensor
        points_tensor = torch.FloatTensor(points)
        
        # 特征提取
        with torch.no_grad():
            features = self.feature_network(points_tensor)
            occupancies = self.occupancy_network(features)
        
        return occupancies.numpy().flatten()
    
    def get_leaf_nodes(self) -> list[OctreeNode]:
        """获取所有叶子节点"""
        leaf_nodes = []
        self._collect_leaf_nodes(self.root, leaf_nodes)
        return leaf_nodes
    
    def _collect_leaf_nodes(self, node: OctreeNode, leaf_nodes: list[OctreeNode]):
        """递归收集叶子节点"""
        if node is None:
            return
        
        if node.is_leaf:
            leaf_nodes.append(node)
        else:
            if node.children:
                for child in node.children:
                    self._collect_leaf_nodes(child, leaf_nodes)
    
    def get_statistics(self) -> dict[str, Any]:
        """获取网格统计信息"""
        if self.root is None:
            return {}
        
        leaf_nodes = self.get_leaf_nodes()
        total_nodes = self._count_total_nodes(self.root)
        max_depth = self._get_max_depth(self.root)
        
        return {
            'total_nodes': total_nodes, 'leaf_nodes': len(
                leaf_nodes,
            )
        }
    
    def _count_total_nodes(self, node: OctreeNode) -> int:
        """计算总节点数"""
        if node is None:
            return 0
        
        count = 1
        if not node.is_leaf and node.children:
            for child in node.children:
                count += self._count_total_nodes(child)
        
        return count
    
    def _get_max_depth(self, node: OctreeNode) -> int:
        """获取最大深度"""
        if node is None or node.is_leaf:
            return node.depth if node else 0
        
        max_child_depth = 0
        if node.children:
            for child in node.children:
                max_child_depth = max(max_child_depth, self._get_max_depth(child))
        
        return max_child_depth


class AdvancedSparseVoxelGrid(SparseVoxelGrid):
    """高级稀疏体素网格"""
    
    def __init__(self, config):
        """
        初始化高级稀疏体素网格
        
        Args:
            config: AdvancedNeuralVDB配置
        """
        super().__init__(config)
        self.feature_compressor = None
        
    def build_from_points(self, points: np.ndarray, occupancies: np.ndarray):
        """
        从点云构建自适应稀疏体素网格
        
        Args:
            points: 3D坐标点 (N, 3)
            occupancies: 占用值 (N, )
        """
        logger.info("构建高级稀疏体素网格...")
        
        # 计算场景边界框
        min_coords = points.min(axis=0)
        max_coords = points.max(axis=0)
        center = (min_coords + max_coords) / 2
        size = max(max_coords - min_coords) * 1.1
        
        # 创建自适应根节点
        self.root = AdaptiveOctreeNode(center, size)
        
        # 递归构建自适应八叉树
        self._build_adaptive_octree_recursive(self.root, points, occupancies)
        
        # 初始化高级神经网络
        self._init_advanced_networks()
        
        # 初始化特征压缩器
        if self.config.feature_compression:
            self._init_feature_compressor()
        
        logger.info(f"高级稀疏体素网格构建完成，根节点大小: {size:.2f}")
    
    def _build_adaptive_octree_recursive(
        self,
        node: AdaptiveOctreeNode,
        points: np.ndarray,
        occupancies: np.ndarray,
    )
        """递归构建自适应八叉树"""
        # 计算节点重要性和占用率
        node.compute_importance(points, occupancies)
        
        # 自适应细分判断
        if node.should_subdivide_adaptive(points, occupancies, self.config):
            node.subdivide()
            
            # 递归处理子节点
            for child in node.children:
                self._build_adaptive_octree_recursive(child, points, occupancies)
    
    def _init_advanced_networks(self):
        """初始化高级神经网络"""
        from .networks import MultiScaleFeatureNetwork, AdvancedOccupancyNetwork
        
        if self.config.multi_scale_features:
            # 多尺度特征网络
            self.feature_network = MultiScaleFeatureNetwork(
                input_dim=3, feature_dim=self.config.feature_dim, hidden_dims=self.config.hidden_dims, num_scales=3, activation=self.config.activation, dropout=self.config.dropout
            )
        else:
            # 标准特征网络
            from .networks import FeatureNetwork
            self.feature_network = FeatureNetwork(
                input_dim=3, feature_dim=self.config.feature_dim, hidden_dims=self.config.hidden_dims, activation=self.config.activation, dropout=self.config.dropout
            )
        
        # 高级占用网络
        self.occupancy_network = AdvancedOccupancyNetwork(
            feature_dim=self.config.feature_dim, hidden_dims=[256, 128, 64, 32], activation=self.config.activation, dropout=self.config.dropout
        )
    
    def _init_feature_compressor(self):
        """初始化特征压缩器"""
        from .networks import FeatureCompressor
        
        self.feature_compressor = FeatureCompressor(
            feature_dim=self.config.feature_dim, quantization_bits=self.config.quantization_bits
        )
    
    def adaptive_refinement(
        self,
        points: np.ndarray,
        occupancies: np.ndarray,
        importance_threshold: float = 0.5,
    )
        """
        自适应细化 - 根据重要性动态调整网格
        
        Args:
            points: 新的点数据
            occupancies: 新的占用数据
            importance_threshold: 重要性阈值
        """
        logger.info("执行自适应细化...")
        
        # 收集需要细化的节点
        nodes_to_refine = []
        self._collect_refinement_candidates(
            self.root, points, occupancies, importance_threshold, nodes_to_refine
        )
        
        # 执行细化
        for node in nodes_to_refine:
            if node.is_leaf:
                node.subdivide()
                # 递归构建新的子节点
                for child in node.children:
                    self._build_adaptive_octree_recursive(child, points, occupancies)
        
        logger.info(f"完成自适应细化，细化了 {len(nodes_to_refine)} 个节点")
    
    def _collect_refinement_candidates(
        self,
        node: AdaptiveOctreeNode,
        points: np.ndarray,
        occupancies: np.ndarray,
        threshold: float,
        candidates: List,
    )
        """收集需要细化的候选节点"""
        if node is None:
            return
        
        # 重新计算重要性
        node.compute_importance(points, occupancies)
        
        # 检查是否需要细化
        if (node.is_leaf and 
            node.importance > threshold and 
            node.depth < self.config.max_depth):
            candidates.append(node)
        
        # 递归检查子节点
        if not node.is_leaf and node.children:
            for child in node.children:
                self._collect_refinement_candidates(
                    child, points, occupancies, threshold, candidates
                )
    
    def compress_features(self, features: np.ndarray) -> np.ndarray:
        """压缩特征"""
        if self.feature_compressor is None:
            return features
        
        return self.feature_compressor.compress(features)
    
    def decompress_features(self, compressed_features: np.ndarray) -> np.ndarray:
        """解压缩特征"""
        if self.feature_compressor is None:
            return compressed_features
        
        return self.feature_compressor.decompress(compressed_features)
    
    def get_memory_statistics(self) -> dict[str, float]:
        """获取内存统计信息"""
        stats = super().get_statistics()
        
        # 计算内存使用
        node_memory = stats['total_nodes'] * 300  # 每个自适应节点约300字节
        
        network_memory = 0
        if self.feature_network:
            network_memory += sum(p.numel() * 4 for p in self.feature_network.parameters())
        if self.occupancy_network:
            network_memory += sum(p.numel() * 4 for p in self.occupancy_network.parameters())
        
        compression_memory = 0
        if self.feature_compressor:
            compression_memory = 1024 * 1024  # 估算1MB
        
        total_memory = node_memory + network_memory + compression_memory
        
        stats.update({
            'node_memory_bytes': node_memory, 'network_memory_bytes': network_memory, 'compression_memory_bytes': compression_memory, 'total_memory_mb': total_memory / (
                1024 * 1024,
            )
        })
        
        return stats 