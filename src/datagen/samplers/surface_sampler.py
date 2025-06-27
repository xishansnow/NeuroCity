from typing import Any, Optional
"""
表面采样器模块

专门用于在几何表面附近进行精确采样，适用于SDF训练。
"""

import numpy as np
import logging
from scipy.spatial.distance import cdist
from scipy.ndimage import gaussian_filter

logger = logging.getLogger(__name__)

class SurfaceSampler:
    """表面采样器类"""
    
    def __init__(
        self,
        surface_threshold: float = 0.5,
        sampling_radius: float = 3.0,
        adaptive_sampling: bool = True,
    )
        """
        初始化表面采样器
        
        Args:
            surface_threshold: 表面检测阈值
            sampling_radius: 采样半径
            adaptive_sampling: 是否使用自适应采样
        """
        self.surface_threshold = surface_threshold
        self.sampling_radius = sampling_radius
        self.adaptive_sampling = adaptive_sampling
        
    def sample_near_surface(
        self,
        coordinates: np.ndarray,
        occupancy: np.ndarray,
        n_samples: int = 10000,
        noise_std: float = 1.0,
    )
        """
        在表面附近采样
        
        Args:
            coordinates: 坐标数组 [N, 3]
            occupancy: 占用值数组 [N]
            n_samples: 采样数量
            noise_std: 噪声标准差
            
        Returns:
            采样结果
        """
        # 检测表面点
        surface_indices = self._detect_surface_points(coordinates, occupancy)
        
        if len(surface_indices) == 0:
            logger.warning("未检测到表面点，使用随机采样")
            return self._fallback_random_sampling(coordinates, occupancy, n_samples)
        
        surface_coords = coordinates[surface_indices]
        
        # 在表面附近生成采样点
        sample_coords = []
        sample_sdf = []
        
        for _ in range(n_samples):
            # 随机选择一个表面点
            surface_idx = np.random.randint(len(surface_coords))
            surface_point = surface_coords[surface_idx]
            
            # 在该点附近生成随机偏移
            if self.adaptive_sampling:
                # 自适应噪声
                local_density = self._estimate_local_density(surface_point, surface_coords)
                adaptive_std = noise_std / max(local_density, 0.1)
            else:
                adaptive_std = noise_std
            
            offset = np.random.normal(0, adaptive_std, 3)
            sample_point = surface_point + offset
            
            # 计算SDF值
            sdf_value = self._compute_sdf(sample_point, surface_coords)
            
            sample_coords.append(sample_point)
            sample_sdf.append(sdf_value)
        
        return {
            'coordinates': np.array(
                sample_coords,
            )
        }
    
    def _detect_surface_points(self, coordinates: np.ndarray, occupancy: np.ndarray) -> np.ndarray:
        """检测表面点"""
        # 使用阈值检测
        surface_mask = np.abs(occupancy - self.surface_threshold) < 0.1
        surface_indices = np.where(surface_mask)[0]
        
        if len(surface_indices) == 0:
            # 如果没有找到，使用梯度检测
            surface_indices = self._gradient_based_detection(coordinates, occupancy)
        
        return surface_indices
    
    def _gradient_based_detection(
        self,
        coordinates: np.ndarray,
        occupancy: np.ndarray,
    )
        """基于梯度的表面检测"""
        # 计算局部梯度
        gradients = []
        
        for i in range(len(coordinates)):
            # 找到邻近点
            distances = cdist([coordinates[i]], coordinates)[0]
            neighbor_indices = np.where(distances < self.sampling_radius)[0]
            
            if len(neighbor_indices) > 3:  # 至少需要几个邻居
                neighbor_coords = coordinates[neighbor_indices]
                neighbor_occupancy = occupancy[neighbor_indices]
                
                # 计算简单梯度
                grad = self._compute_local_gradient(
                    coordinates[i],
                    neighbor_coords,
                    neighbor_occupancy,
                )
                gradients.append(np.linalg.norm(grad))
            else:
                gradients.append(0.0)
        
        gradients = np.array(gradients)
        
        # 高梯度区域可能是表面
        gradient_threshold = np.percentile(gradients, 75)
        surface_indices = np.where(gradients > gradient_threshold)[0]
        
        return surface_indices
    
    def _compute_local_gradient(
        self,
        center: np.ndarray,
        neighbors: np.ndarray,
        values: np.ndarray,
    )
        """计算局部梯度"""
        if len(neighbors) < 4:
            return np.zeros(3)
        
        # 使用最小二乘法拟合局部平面
        A = np.column_stack([neighbors - center, np.ones(len(neighbors))])
        
        try:
            coeffs = np.linalg.lstsq(A, values, rcond=None)[0]
            return coeffs[:3]  # 梯度向量
        except:
            return np.zeros(3)
    
    def _estimate_local_density(self, point: np.ndarray, surface_points: np.ndarray) -> float:
        """估计局部密度"""
        distances = cdist([point], surface_points)[0]
        nearby_count = np.sum(distances < self.sampling_radius)
        
        # 密度 = 邻居数量 / 体积
        volume = (4/3) * np.pi * (self.sampling_radius ** 3)
        density = nearby_count / volume
        
        return density
    
    def _compute_sdf(self, point: np.ndarray, surface_points: np.ndarray) -> float:
        """计算SDF值"""
        # 找到最近的表面点
        distances = cdist([point], surface_points)[0]
        min_distance = np.min(distances)
        
        # 简化SDF：距离表面的最短距离
        # 这里可以根据需要添加内外判断逻辑
        return min_distance
    
    def _fallback_random_sampling(
        self,
        coordinates: np.ndarray,
        occupancy: np.ndarray,
        n_samples: int,
    )
        """备用随机采样"""
        indices = np.random.choice(
            len,
        )
        
        sample_coords = coordinates[indices]
        sample_occupancy = occupancy[indices]
        
        # 生成简单的SDF值
        sample_sdf = np.where(sample_occupancy > self.surface_threshold, -1.0, 1.0)
        
        return {
            'coordinates': sample_coords, 'sdf': sample_sdf, 'surface_points': sample_coords[sample_occupancy > self.surface_threshold]
        }
    
    def sample_multi_resolution(
        self,
        coordinates: np.ndarray,
        occupancy: np.ndarray,
        resolution_levels: list[float] = [1.0,
        0.5,
        0.25],
        samples_per_level: int = 3000,
    )
        """
        多分辨率表面采样
        
        Args:
            coordinates: 坐标数组
            occupancy: 占用值数组
            resolution_levels: 分辨率级别列表
            samples_per_level: 每个级别的采样数量
            
        Returns:
            多分辨率采样结果
        """
        all_coords = []
        all_sdf = []
        level_labels = []
        
        for level_idx, resolution in enumerate(resolution_levels):
            # 调整采样参数
            level_noise = self.sampling_radius * resolution
            
            # 进行采样
            samples = self.sample_near_surface(
                coordinates, occupancy, n_samples=samples_per_level, noise_std=level_noise
            )
            
            all_coords.append(samples['coordinates'])
            all_sdf.append(samples['sdf'])
            level_labels.extend([level_idx] * len(samples['coordinates']))
        
        return {
            'coordinates': np.vstack(
                all_coords,
            )
        }
    
    def adaptive_surface_sampling(
        self,
        coordinates: np.ndarray,
        occupancy: np.ndarray,
        target_samples: int = 10000,
        max_iterations: int = 5,
    )
        """
        自适应表面采样
        
        根据表面复杂度自动调整采样密度
        """
        current_samples = 0
        all_coords = []
        all_sdf = []
        
        # 初始表面检测
        surface_indices = self._detect_surface_points(coordinates, occupancy)
        surface_coords = coordinates[surface_indices]
        
        for iteration in range(max_iterations):
            # 计算还需要的样本数
            remaining_samples = target_samples - current_samples
            if remaining_samples <= 0:
                break
            
            # 根据迭代次数调整采样策略
            iteration_noise = self.sampling_radius * (0.8 ** iteration)
            iteration_samples = min(remaining_samples, target_samples // max_iterations)
            
            # 采样
            samples = self.sample_near_surface(
                coordinates, occupancy, n_samples=iteration_samples, noise_std=iteration_noise
            )
            
            all_coords.append(samples['coordinates'])
            all_sdf.append(samples['sdf'])
            current_samples += len(samples['coordinates'])
            
            logger.info(f"自适应采样迭代 {iteration + 1}: {len(samples['coordinates'])} 个样本")
        
        return {
            'coordinates': np.vstack(
                all_coords,
            )
        } 