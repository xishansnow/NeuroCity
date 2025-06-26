"""
Neural Networks for NeuralVDB

This module contains various neural network architectures used in NeuralVDB, including feature networks, occupancy networks, and advanced variants.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, import logging
from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)


class PositionalEncoding(nn.Module):
    """位置编码 - Fourier特征编码"""
    
    def __init__(self, input_dim: int = 3, max_freq_log2: int = 9, num_freqs: int = 10):
        """
        初始化位置编码
        
        Args:
            input_dim: 输入维度
            max_freq_log2: 最大频率的log2值
            num_freqs: 频率数量
        """
        super(PositionalEncoding, self).__init__()
        
        self.input_dim = input_dim
        self.max_freq_log2 = max_freq_log2
        self.num_freqs = num_freqs
        
        # 创建频率带
        freq_bands = 2.0 ** torch.linspace(0.0, max_freq_log2, num_freqs)
        self.register_buffer('freq_bands', freq_bands)
        
        # 输出维度：原始维度 + 2 * num_freqs * input_dim
        self.output_dim = input_dim + 2 * num_freqs * input_dim
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入坐标 (batch_size, input_dim)
            
        Returns:
            编码后的特征 (batch_size, output_dim)
        """
        # 保留原始坐标
        encoded = [x]
        
        # 对每个频率和每个维度进行编码
        for freq in self.freq_bands:
            for i in range(self.input_dim):
                encoded.append(torch.sin(freq * torch.pi * x[:, i:i+1]))
                encoded.append(torch.cos(freq * torch.pi * x[:, i:i+1]))
        
        return torch.cat(encoded, dim=-1)


class MLP(nn.Module):
    """多层感知机网络"""
    
    def __init__(
        self,
        input_dim: int = 3,
        hidden_dims: list[int] = [256,
        512,
        512,
        256,
        128],
        output_dim: int = 1,
        activation: str = 'relu',
        dropout: float = 0.1,
        use_batch_norm: bool = True,
    )
        """
        初始化MLP
        
        Args:
            input_dim: 输入维度
            hidden_dims: 隐藏层维度列表
            output_dim: 输出维度
            activation: 激活函数
            dropout: dropout比例
            use_batch_norm: 是否使用批归一化
        """
        super(MLP, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            layers.append(self._get_activation(activation))
            
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            
            prev_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
        # 初始化权重
        self.apply(self._init_weights)
    
    def _get_activation(self, activation: str) -> nn.Module:
        """获取激活函数"""
        activations = {
            'relu': nn.ReLU(
            )
        }
        
        if activation not in activations:
            raise ValueError(f"未知激活函数: {activation}")
        
        return activations[activation]
    
    def _init_weights(self, module):
        """初始化权重"""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    
    def forward(self, x):
        """前向传播"""
        return self.network(x)


class FeatureNetwork(nn.Module):
    """特征网络 - 将3D坐标映射到特征向量"""
    
    def __init__(
        self,
        input_dim: int = 3,
        feature_dim: int = 32,
        hidden_dims: list[int] = [256,
        512,
        512,
        256,
        128],
        activation: str = 'relu',
        dropout: float = 0.1,
        use_positional_encoding: bool = True,
    )
        """
        初始化特征网络
        
        Args:
            input_dim: 输入维度
            feature_dim: 特征维度
            hidden_dims: 隐藏层维度列表
            activation: 激活函数
            dropout: dropout比例
            use_positional_encoding: 是否使用位置编码
        """
        super(FeatureNetwork, self).__init__()
        
        self.use_positional_encoding = use_positional_encoding
        
        # 位置编码
        if use_positional_encoding:
            self.pos_encoding = PositionalEncoding(input_dim)
            network_input_dim = self.pos_encoding.output_dim
        else:
            self.pos_encoding = None
            network_input_dim = input_dim
        
        # MLP网络
        self.mlp = MLP(
            input_dim=network_input_dim, hidden_dims=hidden_dims, output_dim=feature_dim, activation=activation, dropout=dropout
        )
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入坐标 (batch_size, input_dim)
            
        Returns:
            特征向量 (batch_size, feature_dim)
        """
        if self.use_positional_encoding:
            x = self.pos_encoding(x)
        
        return self.mlp(x)


class OccupancyNetwork(nn.Module):
    """占用网络 - 将特征向量映射到占用概率"""
    
    def __init__(
        self,
        feature_dim: int = 32,
        hidden_dims: list[int] = [128,
        64,
        32],
        activation: str = 'relu',
        dropout: float = 0.1,
    )
        """
        初始化占用网络
        
        Args:
            feature_dim: 特征维度
            hidden_dims: 隐藏层维度列表
            activation: 激活函数
            dropout: dropout比例
        """
        super(OccupancyNetwork, self).__init__()
        
        self.mlp = MLP(
            input_dim=feature_dim, hidden_dims=hidden_dims, output_dim=1, activation=activation, dropout=dropout
        )
    
    def forward(self, features):
        """
        前向传播
        
        Args:
            features: 特征向量 (batch_size, feature_dim)
            
        Returns:
            占用概率 (batch_size, 1)
        """
        logits = self.mlp(features)
        return torch.sigmoid(logits)


class MultiScaleFeatureNetwork(nn.Module):
    """多尺度特征网络"""
    
    def __init__(
        self,
        input_dim: int = 3,
        feature_dim: int = 64,
        hidden_dims: list[int] = [256,
        512,
        512,
        256,
        128],
        num_scales: int = 3,
        activation: str = 'relu',
        dropout: float = 0.1,
    )
        """
        初始化多尺度特征网络
        
        Args:
            input_dim: 输入维度
            feature_dim: 特征维度
            hidden_dims: 隐藏层维度列表
            num_scales: 尺度数量
            activation: 激活函数
            dropout: dropout比例
        """
        super(MultiScaleFeatureNetwork, self).__init__()
        
        self.num_scales = num_scales
        self.feature_dim = feature_dim
        
        # 多尺度特征提取器
        self.scale_networks = nn.ModuleList()
        for i in range(num_scales):
            # 不同尺度使用不同的位置编码
            pos_encoding = PositionalEncoding(
                input_dim=input_dim, max_freq_log2=6 + i * 2, # 不同尺度使用不同频率
                num_freqs=8
            )
            
            mlp = MLP(
                input_dim=pos_encoding.output_dim, hidden_dims=hidden_dims, output_dim=feature_dim // num_scales, activation=activation, dropout=dropout
            )
            
            self.scale_networks.append(nn.ModuleDict({
                'pos_encoding': pos_encoding, 'mlp': mlp
            }))
        
        # 特征融合网络
        self.fusion_network = nn.Sequential(
            nn.Linear(
                feature_dim,
                feature_dim,
            )
        )
    
    def _get_activation(self, activation: str) -> nn.Module:
        """获取激活函数"""
        activations = {
            'relu': nn.ReLU(), 'leaky_relu': nn.LeakyReLU(), 'swish': nn.SiLU(), 'gelu': nn.GELU()
        }
        return activations.get(activation, nn.ReLU())
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入坐标 (batch_size, input_dim)
            
        Returns:
            多尺度特征向量 (batch_size, feature_dim)
        """
        # 多尺度特征提取
        scale_features = []
        
        for scale_net in self.scale_networks:
            # 位置编码
            encoded = scale_net['pos_encoding'](x)
            # MLP特征提取
            features = scale_net['mlp'](encoded)
            scale_features.append(features)
        
        # 拼接多尺度特征
        multi_scale_features = torch.cat(scale_features, dim=-1)
        
        # 特征融合
        fused_features = self.fusion_network(multi_scale_features)
        
        return fused_features


class AdvancedOccupancyNetwork(nn.Module):
    """高级占用网络"""
    
    def __init__(
        self,
        feature_dim: int = 64,
        hidden_dims: list[int] = [256,
        128,
        64,
        32],
        activation: str = 'relu',
        dropout: float = 0.1,
    )
        """
        初始化高级占用网络
        
        Args:
            feature_dim: 特征维度
            hidden_dims: 隐藏层维度列表
            activation: 激活函数
            dropout: dropout比例
        """
        super(AdvancedOccupancyNetwork, self).__init__()
        
        # 主要占用预测分支
        self.occupancy_branch = MLP(
            input_dim=feature_dim, hidden_dims=hidden_dims, output_dim=1, activation=activation, dropout=dropout
        )
        
        # 不确定性预测分支
        self.uncertainty_branch = MLP(
            input_dim=feature_dim, hidden_dims=hidden_dims[:2], # 更简单的网络
            output_dim=1, activation=activation, dropout=dropout
        )
        
        # 梯度预测分支（用于表面检测）
        self.gradient_branch = MLP(
            input_dim=feature_dim, hidden_dims=hidden_dims[:2], output_dim=3, # 3D梯度
            activation=activation, dropout=dropout
        )
    
    def forward(self, features):
        """
        前向传播
        
        Args:
            features: 特征向量 (batch_size, feature_dim)
            
        Returns:
            Dict包含占用概率、不确定性和梯度
        """
        # 占用概率
        occupancy_logits = self.occupancy_branch(features)
        occupancy = torch.sigmoid(occupancy_logits)
        
        # 不确定性（使用softplus确保为正）
        uncertainty_logits = self.uncertainty_branch(features)
        uncertainty = F.softplus(uncertainty_logits)
        
        # 梯度
        gradient = self.gradient_branch(features)
        
        return {
            'occupancy': occupancy, 'uncertainty': uncertainty, 'gradient': gradient, 'occupancy_logits': occupancy_logits
        }


class AdvancedLossFunction(nn.Module):
    """高级损失函数"""
    
    def __init__(self, config):
        """
        初始化高级损失函数
        
        Args:
            config: AdvancedNeuralVDBConfig
        """
        super(AdvancedLossFunction, self).__init__()
        self.config = config
    
    def forward(
        self,
        predictions: dict[str,
        torch.Tensor],
        targets: torch.Tensor,
        points: torch.Tensor = None,
    )
        """
        计算综合损失
        
        Args:
            predictions: 预测结果字典
            targets: 目标占用值
            points: 输入点坐标（用于计算平滑损失）
            
        Returns:
            总损失和损失分解
        """
        losses = {}
        
        # 1. 基础占用损失
        occupancy_pred = predictions['occupancy'].squeeze()
        occupancy_loss = F.binary_cross_entropy(occupancy_pred, targets)
        losses['occupancy'] = occupancy_loss * self.config.occupancy_weight
        
        # 2. 平滑性损失
        if points is not None and self.config.smoothness_weight > 0:
            smoothness_loss = self._compute_smoothness_loss(occupancy_pred, points)
            losses['smoothness'] = smoothness_loss * self.config.smoothness_weight
        
        # 3. 稀疏性损失
        if self.config.sparsity_weight > 0:
            sparsity_loss = self._compute_sparsity_loss(occupancy_pred)
            losses['sparsity'] = sparsity_loss * self.config.sparsity_weight
        
        # 4. 一致性损失（不确定性约束）
        if 'uncertainty' in predictions and self.config.consistency_weight > 0:
            consistency_loss = self._compute_consistency_loss(
                occupancy_pred, targets, predictions['uncertainty']
            )
            losses['consistency'] = consistency_loss * self.config.consistency_weight
        
        # 总损失
        total_loss = sum(losses.values())
        losses['total'] = total_loss
        
        return total_loss, losses
    
    def _compute_smoothness_loss(self, predictions: torch.Tensor, points: torch.Tensor):
        """计算平滑性损失"""
        # 计算相邻点的预测差异
        if len(points) < 2:
            return torch.tensor(0.0, device=predictions.device)
        
        # 随机采样点对
        num_pairs = min(1000, len(points) // 2)
        indices = torch.randperm(len(points))[:num_pairs * 2].view(-1, 2)
        
        point_pairs = points[indices]  # (num_pairs, 2, 3)
        pred_pairs = predictions[indices]  # (num_pairs, 2)
        
        # 计算点之间的距离
        distances = torch.norm(point_pairs[:, 0] - point_pairs[:, 1], dim=1)
        
        # 计算预测差异
        pred_diff = torch.abs(pred_pairs[:, 0] - pred_pairs[:, 1])
        
        # 平滑性损失：预测差异应该与距离成正比
        smoothness_loss = torch.mean(pred_diff / (distances + 1e-6))
        
        return smoothness_loss
    
    def _compute_sparsity_loss(self, predictions: torch.Tensor):
        """计算稀疏性损失"""
        # 鼓励预测值接近0或1
        sparsity_loss = torch.mean(predictions * (1 - predictions))
        return sparsity_loss
    
    def _compute_consistency_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        uncertainty: torch.Tensor,
    )
        """计算一致性损失"""
        # 不确定性应该与预测误差相关
        errors = torch.abs(predictions - targets)
        uncertainty = uncertainty.squeeze()
        
        # 高误差区域应该有高不确定性
        consistency_loss = F.mse_loss(uncertainty, errors)
        
        return consistency_loss


class FeatureCompressor:
    """特征压缩器 - 用于减少内存使用"""
    
    def __init__(self, feature_dim: int, quantization_bits: int = 8):
        """
        初始化特征压缩器
        
        Args:
            feature_dim: 特征维度
            quantization_bits: 量化位数
        """
        self.feature_dim = feature_dim
        self.quantization_bits = quantization_bits
        self.codebook = None
        self.is_trained = False
        
        # 量化级别
        self.num_levels = 2 ** quantization_bits
    
    def train(self, features: np.ndarray):
        """
        训练压缩器（学习码本）
        
        Args:
            features: 特征数据 (N, feature_dim)
        """
        logger.info(f"训练特征压缩器，特征维度: {features.shape}")
        
        # 使用K-means学习码本
        kmeans = KMeans(n_clusters=self.num_levels, random_state=42)
        kmeans.fit(features)
        
        self.codebook = kmeans.cluster_centers_
        self.is_trained = True
        
        # 计算压缩比
        original_size = features.nbytes
        compressed_size = len(features) * (self.quantization_bits / 8) + self.codebook.nbytes
        compression_ratio = original_size / compressed_size
        
        logger.info(f"特征压缩器训练完成，压缩比: {compression_ratio:.2f}")
    
    def compress(self, features: np.ndarray) -> np.ndarray:
        """
        压缩特征
        
        Args:
            features: 原始特征 (N, feature_dim)
            
        Returns:
            压缩后的索引 (N, )
        """
        if not self.is_trained:
            raise ValueError("压缩器未训练，请先调用train()方法")
        
        # 找到最近的码本条目
        distances = np.linalg.norm(
            features[:, np.newaxis] - self.codebook[np.newaxis, :], axis=2
        )
        indices = np.argmin(distances, axis=1)
        
        return indices.astype(np.uint8 if self.quantization_bits <= 8 else np.uint16)
    
    def decompress(self, indices: np.ndarray) -> np.ndarray:
        """
        解压缩特征
        
        Args:
            indices: 压缩索引 (N, )
            
        Returns:
            解压缩的特征 (N, feature_dim)
        """
        if not self.is_trained:
            raise ValueError("压缩器未训练，请先调用train()方法")
        
        return self.codebook[indices] 