"""
点云采样器模块

专门用于从点云数据中进行采样，支持多种点云处理策略。
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging

logger = logging.getLogger(__name__)


class PointCloudSampler:
    """点云采样器类"""
    
    def __init__(self,
                 downsample_ratio: float = 0.1,
                 noise_level: float = 0.01,
                 normal_estimation: bool = True):
        """
        初始化点云采样器
        
        Args:
            downsample_ratio: 下采样比例
            noise_level: 噪声级别
            normal_estimation: 是否估计法向量
        """
        self.downsample_ratio = downsample_ratio
        self.noise_level = noise_level
        self.normal_estimation = normal_estimation
    
    def sample_from_point_cloud(self,
                               points: np.ndarray,
                               normals: Optional[np.ndarray] = None,
                               colors: Optional[np.ndarray] = None,
                               n_samples: int = 10000) -> Dict[str, np.ndarray]:
        """
        从点云中采样
        
        Args:
            points: 点云坐标 [N, 3]
            normals: 法向量 [N, 3] (可选)
            colors: 颜色 [N, 3] (可选)
            n_samples: 采样数量
            
        Returns:
            采样结果
        """
        n_points = len(points)
        n_samples = min(n_samples, n_points)
        
        # 随机采样
        indices = np.random.choice(n_points, n_samples, replace=False)
        sampled_points = points[indices]
        
        result = {'coordinates': sampled_points}
        
        # 处理法向量
        if normals is not None:
            result['normals'] = normals[indices]
        elif self.normal_estimation:
            estimated_normals = self._estimate_normals(sampled_points)
            result['normals'] = estimated_normals
        
        # 处理颜色
        if colors is not None:
            result['colors'] = colors[indices]
        
        # 添加噪声
        if self.noise_level > 0:
            noise = np.random.normal(0, self.noise_level, sampled_points.shape)
            result['coordinates_noisy'] = sampled_points + noise
        
        logger.info(f"从 {n_points} 个点中采样了 {n_samples} 个点")
        return result
    
    def _estimate_normals(self, points: np.ndarray, k: int = 10) -> np.ndarray:
        """估计点的法向量"""
        from scipy.spatial import cKDTree
        
        tree = cKDTree(points)
        normals = np.zeros_like(points)
        
        for i, point in enumerate(points):
            # 找到k个最近邻
            distances, indices = tree.query(point, k=k+1)  # +1 因为包含自己
            neighbors = points[indices[1:]]  # 排除自己
            
            if len(neighbors) >= 3:
                # 使用PCA估计法向量
                centered = neighbors - np.mean(neighbors, axis=0)
                cov_matrix = np.cov(centered.T)
                eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
                
                # 最小特征值对应的特征向量就是法向量
                normal = eigenvectors[:, 0]
                normals[i] = normal / np.linalg.norm(normal)
        
        return normals
    
    def uniform_sampling(self,
                        points: np.ndarray,
                        grid_size: float = 1.0,
                        **kwargs) -> Dict[str, np.ndarray]:
        """
        均匀网格采样
        
        Args:
            points: 点云坐标
            grid_size: 网格大小
            
        Returns:
            均匀采样结果
        """
        # 计算边界
        min_coords = np.min(points, axis=0)
        max_coords = np.max(points, axis=0)
        
        # 创建网格
        grid_points = []
        x_range = np.arange(min_coords[0], max_coords[0], grid_size)
        y_range = np.arange(min_coords[1], max_coords[1], grid_size)
        z_range = np.arange(min_coords[2], max_coords[2], grid_size)
        
        for x in x_range:
            for y in y_range:
                for z in z_range:
                    grid_points.append([x, y, z])
        
        grid_points = np.array(grid_points)
        
        # 对每个网格点，找到最近的原始点
        from scipy.spatial import cKDTree
        tree = cKDTree(points)
        distances, indices = tree.query(grid_points)
        
        # 只保留距离足够近的网格点
        valid_mask = distances < grid_size
        valid_grid_points = grid_points[valid_mask]
        valid_indices = indices[valid_mask]
        
        result = {
            'coordinates': valid_grid_points,
            'original_indices': valid_indices,
            'distances': distances[valid_mask]
        }
        
        logger.info(f"均匀采样生成了 {len(valid_grid_points)} 个点")
        return result
    
    def density_based_sampling(self,
                              points: np.ndarray,
                              radius: float = 2.0,
                              min_samples: int = 5,
                              **kwargs) -> Dict[str, np.ndarray]:
        """
        基于密度的采样
        
        Args:
            points: 点云坐标
            radius: 密度计算半径
            min_samples: 最小样本数
            
        Returns:
            密度采样结果
        """
        from scipy.spatial import cKDTree
        
        tree = cKDTree(points)
        densities = []
        
        # 计算每个点的局部密度
        for point in points:
            neighbors = tree.query_ball_point(point, radius)
            densities.append(len(neighbors))
        
        densities = np.array(densities)
        
        # 根据密度进行分层采样
        high_density_mask = densities > np.percentile(densities, 75)
        medium_density_mask = (densities > np.percentile(densities, 25)) & (densities <= np.percentile(densities, 75))
        low_density_mask = densities <= np.percentile(densities, 25)
        
        # 从不同密度区域采样
        high_density_points = points[high_density_mask]
        medium_density_points = points[medium_density_mask]
        low_density_points = points[low_density_mask]
        
        # 平衡采样
        n_high = min(len(high_density_points), min_samples * 2)
        n_medium = min(len(medium_density_points), min_samples * 3)
        n_low = min(len(low_density_points), min_samples)
        
        selected_points = []
        density_labels = []
        
        if n_high > 0:
            high_indices = np.random.choice(len(high_density_points), n_high, replace=False)
            selected_points.extend(high_density_points[high_indices])
            density_labels.extend([2] * n_high)  # 高密度
        
        if n_medium > 0:
            medium_indices = np.random.choice(len(medium_density_points), n_medium, replace=False)
            selected_points.extend(medium_density_points[medium_indices])
            density_labels.extend([1] * n_medium)  # 中密度
        
        if n_low > 0:
            low_indices = np.random.choice(len(low_density_points), n_low, replace=False)
            selected_points.extend(low_density_points[low_indices])
            density_labels.extend([0] * n_low)  # 低密度
        
        result = {
            'coordinates': np.array(selected_points),
            'density_labels': np.array(density_labels),
            'densities': densities
        }
        
        logger.info(f"密度采样生成了 {len(selected_points)} 个点")
        return result
    
    def curvature_based_sampling(self,
                                points: np.ndarray,
                                k: int = 10,
                                curvature_threshold: float = 0.1,
                                **kwargs) -> Dict[str, np.ndarray]:
        """
        基于曲率的采样
        
        Args:
            points: 点云坐标
            k: 邻居数量
            curvature_threshold: 曲率阈值
            
        Returns:
            曲率采样结果
        """
        from scipy.spatial import cKDTree
        
        tree = cKDTree(points)
        curvatures = []
        
        # 计算每个点的曲率
        for i, point in enumerate(points):
            distances, indices = tree.query(point, k=k+1)
            neighbors = points[indices[1:]]  # 排除自己
            
            if len(neighbors) >= 3:
                curvature = self._estimate_curvature(point, neighbors)
                curvatures.append(curvature)
            else:
                curvatures.append(0.0)
        
        curvatures = np.array(curvatures)
        
        # 选择高曲率点
        high_curvature_mask = curvatures > curvature_threshold
        high_curvature_points = points[high_curvature_mask]
        high_curvature_values = curvatures[high_curvature_mask]
        
        result = {
            'coordinates': high_curvature_points,
            'curvatures': high_curvature_values,
            'all_curvatures': curvatures
        }
        
        logger.info(f"曲率采样生成了 {len(high_curvature_points)} 个点")
        return result
    
    def _estimate_curvature(self, center: np.ndarray, neighbors: np.ndarray) -> float:
        """估计点的曲率"""
        if len(neighbors) < 3:
            return 0.0
        
        # 计算邻居点的协方差矩阵
        centered = neighbors - np.mean(neighbors, axis=0)
        cov_matrix = np.cov(centered.T)
        
        try:
            eigenvalues = np.linalg.eigvals(cov_matrix)
            eigenvalues = np.sort(eigenvalues)
            
            # 曲率近似为最小特征值与总特征值的比值
            if eigenvalues[-1] > 1e-6:
                curvature = eigenvalues[0] / np.sum(eigenvalues)
            else:
                curvature = 0.0
        except:
            curvature = 0.0
        
        return curvature
    
    def multi_scale_sampling(self,
                           points: np.ndarray,
                           scales: List[float] = [1.0, 0.5, 0.25],
                           samples_per_scale: int = 3000) -> Dict[str, np.ndarray]:
        """
        多尺度采样
        
        Args:
            points: 点云坐标
            scales: 尺度列表
            samples_per_scale: 每个尺度的采样数量
            
        Returns:
            多尺度采样结果
        """
        all_coords = []
        scale_labels = []
        
        for scale_idx, scale in enumerate(scales):
            # 根据尺度调整采样
            if scale < 1.0:
                # 下采样
                n_scale_points = int(len(points) * scale)
                scale_indices = np.random.choice(len(points), n_scale_points, replace=False)
                scale_points = points[scale_indices]
            else:
                scale_points = points
            
            # 从该尺度采样
            n_samples = min(samples_per_scale, len(scale_points))
            sample_indices = np.random.choice(len(scale_points), n_samples, replace=False)
            sampled_points = scale_points[sample_indices]
            
            all_coords.append(sampled_points)
            scale_labels.extend([scale_idx] * len(sampled_points))
        
        result = {
            'coordinates': np.vstack(all_coords),
            'scale_labels': np.array(scale_labels),
            'scales': scales
        }
        
        logger.info(f"多尺度采样生成了 {len(result['coordinates'])} 个点")
        return result 