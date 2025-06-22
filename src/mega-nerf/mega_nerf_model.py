"""
Mega-NeRF Core Model Implementation

This module contains the core Mega-NeRF neural network architecture
including the main model, submodules, and configuration.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict, Optional, Union
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class MegaNeRFConfig:
    """Mega-NeRF配置参数"""
    # 场景分解参数
    num_submodules: int = 8
    grid_size: Tuple[int, int] = (4, 2)  # 2D网格分解
    overlap_factor: float = 0.15
    
    # 网络参数
    hidden_dim: int = 256
    num_layers: int = 8
    skip_connections: List[int] = None
    use_viewdirs: bool = True
    
    # 训练参数
    batch_size: int = 1024
    learning_rate: float = 5e-4
    lr_decay: float = 0.1
    max_iterations: int = 500000
    
    # 采样参数
    num_coarse: int = 256
    num_fine: int = 512
    near: float = 0.1
    far: float = 1000.0
    
    # 外观嵌入
    use_appearance_embedding: bool = True
    appearance_dim: int = 48
    
    # 边界参数
    scene_bounds: Tuple[float, float, float, float, float, float] = (-100, -100, -10, 100, 100, 50)
    foreground_ratio: float = 0.8
    
    def __post_init__(self):
        if self.skip_connections is None:
            self.skip_connections = [4]


class PositionalEncoding(nn.Module):
    """位置编码模块"""
    def __init__(self, input_dim: int, max_freq_log2: int = 10, num_freqs: int = 10):
        super().__init__()
        self.input_dim = input_dim
        self.max_freq_log2 = max_freq_log2
        self.num_freqs = num_freqs
        
        self.freq_bands = 2.0 ** torch.linspace(0.0, max_freq_log2, num_freqs)
        self.output_dim = input_dim * (1 + 2 * num_freqs)
    
    def forward(self, x):
        """
        Args:
            x: [..., input_dim]
        Returns:
            encoded: [..., output_dim]
        """
        encoded = [x]
        for freq in self.freq_bands:
            encoded.append(torch.sin(freq * x))
            encoded.append(torch.cos(freq * x))
        return torch.cat(encoded, dim=-1)


class MegaNeRFSubmodule(nn.Module):
    """Mega-NeRF子模块，处理场景的一个空间区域"""
    def __init__(self, config: MegaNeRFConfig, centroid: np.ndarray):
        super().__init__()
        self.config = config
        self.centroid = torch.tensor(centroid, dtype=torch.float32)
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(3, max_freq_log2=10, num_freqs=10)
        self.dir_encoder = PositionalEncoding(3, max_freq_log2=4, num_freqs=4) if config.use_viewdirs else None
        
        # 计算输入维度
        pos_dim = self.pos_encoder.output_dim
        input_dim = pos_dim
        if config.use_viewdirs:
            dir_dim = self.dir_encoder.output_dim
        else:
            dir_dim = 0
        
        # 外观嵌入
        if config.use_appearance_embedding:
            self.appearance_embedding = nn.Embedding(1000, config.appearance_dim)  # 假设最多1000张图片
            input_dim += config.appearance_dim
        
        # 主网络 - 预测密度和特征
        layers = []
        current_dim = input_dim
        
        for i in range(config.num_layers):
            if i in config.skip_connections:
                layers.append(nn.Linear(current_dim + input_dim, config.hidden_dim))
            else:
                layers.append(nn.Linear(current_dim, config.hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            current_dim = config.hidden_dim
        
        self.main_layers = nn.ModuleList(layers)
        
        # 密度预测头
        self.density_head = nn.Linear(config.hidden_dim, 1)
        
        # 颜色预测网络
        if config.use_viewdirs:
            self.feature_head = nn.Linear(config.hidden_dim, config.hidden_dim)
            color_input_dim = config.hidden_dim + dir_dim
            self.color_layers = nn.Sequential(
                nn.Linear(color_input_dim, config.hidden_dim // 2),
                nn.ReLU(inplace=True),
                nn.Linear(config.hidden_dim // 2, 3),
                nn.Sigmoid()
            )
        else:
            self.color_head = nn.Sequential(
                nn.Linear(config.hidden_dim, 3),
                nn.Sigmoid()
            )
    
    def forward(self, points, viewdirs=None, appearance_idx=None):
        """
        Args:
            points: [N, 3] 3D坐标
            viewdirs: [N, 3] 视角方向 (可选)
            appearance_idx: [N] 外观嵌入索引 (可选)
        Returns:
            density: [N, 1] 密度值
            color: [N, 3] RGB颜色
        """
        # 位置编码
        pos_encoded = self.pos_encoder(points)
        
        # 外观嵌入
        features = [pos_encoded]
        if self.config.use_appearance_embedding and appearance_idx is not None:
            app_embed = self.appearance_embedding(appearance_idx)
            features.append(app_embed)
        
        x = torch.cat(features, dim=-1)
        input_x = x
        
        # 主网络前向传播
        for i, layer in enumerate(self.main_layers):
            if i // 2 in self.config.skip_connections and i % 2 == 0:
                x = torch.cat([x, input_x], dim=-1)
            x = layer(x)
        
        # 预测密度
        density = self.density_head(x)
        density = F.relu(density)
        
        # 预测颜色
        if self.config.use_viewdirs and viewdirs is not None:
            features = self.feature_head(x)
            dir_encoded = self.dir_encoder(viewdirs)
            color_input = torch.cat([features, dir_encoded], dim=-1)
            color = self.color_layers(color_input)
        else:
            color = self.color_head(x)
        
        return density, color
    
    def get_centroid(self):
        """获取子模块的中心点"""
        return self.centroid.clone()
    
    def set_centroid(self, centroid: torch.Tensor):
        """设置子模块的中心点"""
        self.centroid = centroid.clone()


class MegaNeRF(nn.Module):
    """Mega-NeRF主模型"""
    def __init__(self, config: MegaNeRFConfig):
        super().__init__()
        self.config = config
        
        # 创建空间网格
        self.spatial_grid = self._create_spatial_grid()
        
        # 创建子模块
        self.submodules = nn.ModuleList()
        for i, centroid in enumerate(self.spatial_grid):
            submodule = MegaNeRFSubmodule(config, centroid)
            self.submodules.append(submodule)
        
        # 前景边界
        self.foreground_bounds = self._compute_foreground_bounds()
        
        logger.info(f"创建了 {len(self.submodules)} 个子模块")
        logger.info(f"场景边界: {config.scene_bounds}")
        logger.info(f"前景边界: {self.foreground_bounds}")
    
    def _create_spatial_grid(self):
        """创建空间网格分解"""
        bounds = self.config.scene_bounds
        x_min, y_min, z_min, x_max, y_max, z_max = bounds
        
        grid_x, grid_y = self.config.grid_size
        
        # 计算网格中心点
        x_centers = np.linspace(x_min, x_max, grid_x + 1)[:-1] + (x_max - x_min) / (2 * grid_x)
        y_centers = np.linspace(y_min, y_max, grid_y + 1)[:-1] + (y_max - y_min) / (2 * grid_y)
        z_center = (z_min + z_max) / 2
        
        centroids = []
        for x in x_centers:
            for y in y_centers:
                centroids.append([x, y, z_center])
        
        return np.array(centroids)
    
    def _compute_foreground_bounds(self):
        """计算前景边界"""
        bounds = self.config.scene_bounds
        x_min, y_min, z_min, x_max, y_max, z_max = bounds
        
        # 缩小边界作为前景区域
        ratio = self.config.foreground_ratio
        center_x, center_y, center_z = (x_min + x_max) / 2, (y_min + y_max) / 2, (z_min + z_max) / 2
        width_x, width_y, width_z = (x_max - x_min) * ratio, (y_max - y_min) * ratio, (z_max - z_min) * ratio
        
        return (
            center_x - width_x / 2, center_y - width_y / 2, center_z - width_z / 2,
            center_x + width_x / 2, center_y + width_y / 2, center_z + width_z / 2
        )
    
    def _assign_points_to_submodules(self, points):
        """将点分配给最近的子模块"""
        # 计算到所有中心点的距离
        centroids = torch.tensor(self.spatial_grid, device=points.device, dtype=points.dtype)
        distances = torch.cdist(points, centroids)  # [N, num_submodules]
        
        # 分配给最近的子模块
        assignments = torch.argmin(distances, dim=1)
        return assignments
    
    def _is_in_foreground(self, points):
        """检查点是否在前景区域内"""
        x_min, y_min, z_min, x_max, y_max, z_max = self.foreground_bounds
        
        in_bounds = (
            (points[:, 0] >= x_min) & (points[:, 0] <= x_max) &
            (points[:, 1] >= y_min) & (points[:, 1] <= y_max) &
            (points[:, 2] >= z_min) & (points[:, 2] <= z_max)
        )
        
        return in_bounds
    
    def forward(self, points, viewdirs=None, appearance_idx=None):
        """
        Args:
            points: [N, 3] 3D坐标
            viewdirs: [N, 3] 视角方向 (可选)
            appearance_idx: [N] 外观嵌入索引 (可选)
        Returns:
            density: [N, 1] 密度值
            color: [N, 3] RGB颜色
        """
        device = points.device
        N = points.shape[0]
        
        # 初始化输出
        total_density = torch.zeros(N, 1, device=device)
        total_color = torch.zeros(N, 3, device=device)
        total_weights = torch.zeros(N, 1, device=device)
        
        # 检查前景/背景
        in_foreground = self._is_in_foreground(points)
        
        if in_foreground.any():
            fg_points = points[in_foreground]
            fg_viewdirs = viewdirs[in_foreground] if viewdirs is not None else None
            fg_appearance_idx = appearance_idx[in_foreground] if appearance_idx is not None else None
            
            # 分配前景点到子模块
            assignments = self._assign_points_to_submodules(fg_points)
            
            # 处理每个子模块
            for i, submodule in enumerate(self.submodules):
                mask = assignments == i
                if not mask.any():
                    continue
                
                sub_points = fg_points[mask]
                sub_viewdirs = fg_viewdirs[mask] if fg_viewdirs is not None else None
                sub_appearance_idx = fg_appearance_idx[mask] if fg_appearance_idx is not None else None
                
                # 前向传播
                density, color = submodule(sub_points, sub_viewdirs, sub_appearance_idx)
                
                # 计算权重（基于距离的软分配）
                centroid = torch.tensor(self.spatial_grid[i], device=device, dtype=sub_points.dtype)
                distances = torch.norm(sub_points - centroid[None, :], dim=1, keepdim=True)
                weights = torch.exp(-distances / 10.0)  # 可调节的衰减参数
                
                # 累积结果
                fg_indices = torch.where(in_foreground)[0][mask]
                total_density[fg_indices] += density * weights
                total_color[fg_indices] += color * weights
                total_weights[fg_indices] += weights
        
        # 背景处理（简单的恒定背景）
        if (~in_foreground).any():
            bg_density = torch.full((N, 1), 0.01, device=device)  # 低密度背景
            bg_color = torch.full((N, 3), 0.8, device=device)    # 白色背景
            
            total_density[~in_foreground] = bg_density[~in_foreground]
            total_color[~in_foreground] = bg_color[~in_foreground]
            total_weights[~in_foreground] = 1.0
        
        # 归一化颜色
        valid_weights = total_weights > 0
        total_color[valid_weights] = total_color[valid_weights] / total_weights[valid_weights]
        
        return total_density, total_color
    
    def get_submodule_centroids(self):
        """获取所有子模块的中心点"""
        return torch.tensor(self.spatial_grid, dtype=torch.float32)
    
    def get_submodule_bounds(self, submodule_idx: int):
        """获取指定子模块的边界"""
        if submodule_idx >= len(self.submodules):
            raise ValueError(f"Submodule index {submodule_idx} out of range")
        
        centroid = self.spatial_grid[submodule_idx]
        bounds = self.config.scene_bounds
        grid_x, grid_y = self.config.grid_size
        
        # 计算子模块边界
        x_size = (bounds[3] - bounds[0]) / grid_x
        y_size = (bounds[4] - bounds[1]) / grid_y
        z_size = bounds[5] - bounds[2]
        
        x_min = centroid[0] - x_size / 2
        x_max = centroid[0] + x_size / 2
        y_min = centroid[1] - y_size / 2
        y_max = centroid[1] + y_size / 2
        z_min = bounds[2]
        z_max = bounds[5]
        
        return (x_min, y_min, z_min, x_max, y_max, z_max)
    
    def get_relevant_submodules(self, camera_position: torch.Tensor, max_distance: float = None):
        """获取与相机位置相关的子模块"""
        if max_distance is None:
            # 基于场景大小自动确定距离
            bounds = self.config.scene_bounds
            scene_size = max(bounds[3] - bounds[0], bounds[4] - bounds[1])
            max_distance = scene_size * 0.5
        
        centroids = torch.tensor(self.spatial_grid, device=camera_position.device, dtype=camera_position.dtype)
        distances = torch.norm(camera_position[None, :] - centroids, dim=1)
        
        relevant_mask = distances <= max_distance
        relevant_indices = torch.where(relevant_mask)[0]
        
        return relevant_indices.tolist()
    
    def save_submodule(self, submodule_idx: int, path: str):
        """保存单个子模块"""
        if submodule_idx >= len(self.submodules):
            raise ValueError(f"Submodule index {submodule_idx} out of range")
        
        submodule = self.submodules[submodule_idx]
        torch.save({
            'state_dict': submodule.state_dict(),
            'config': self.config,
            'centroid': self.spatial_grid[submodule_idx],
            'submodule_idx': submodule_idx
        }, path)
    
    def load_submodule(self, submodule_idx: int, path: str):
        """加载单个子模块"""
        if submodule_idx >= len(self.submodules):
            raise ValueError(f"Submodule index {submodule_idx} out of range")
        
        checkpoint = torch.load(path, map_location='cpu')
        self.submodules[submodule_idx].load_state_dict(checkpoint['state_dict'])
        
        # 验证中心点匹配
        saved_centroid = checkpoint['centroid']
        current_centroid = self.spatial_grid[submodule_idx]
        if not np.allclose(saved_centroid, current_centroid, atol=1e-6):
            logger.warning(f"Centroid mismatch for submodule {submodule_idx}")
    
    def get_model_info(self):
        """获取模型信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'num_submodules': len(self.submodules),
            'grid_size': self.config.grid_size,
            'scene_bounds': self.config.scene_bounds,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'config': self.config
        } 