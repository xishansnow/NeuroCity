from typing import Any, Optional
"""
体素采样器模块

用于从tile体素数据中采样训练数据，支持多种采样策略。
从原sampler.py迁移并重构。
"""

import numpy as np
import json
import os
import random
from scipy.spatial import cKDTree
import logging

logger = logging.getLogger(__name__)

class VoxelSampler:
    """体素采样器类"""
    
    def __init__(
        self,
        tiles_dir: str = "tiles",
        voxel_size: float = 1.0,
        sample_ratio: float = 0.1,
    )
        """
        初始化体素采样器
        
        Args:
            tiles_dir: tiles目录路径
            voxel_size: 体素大小（米）
            sample_ratio: 采样比例（0-1）
        """
        self.tiles_dir = tiles_dir
        self.voxel_size = voxel_size
        self.sample_ratio = sample_ratio
        self.tiles_data = {}
        
        if os.path.exists(tiles_dir):
            self.load_tiles()
    
    def load_tiles(self):
        """加载所有tile数据"""
        logger.info(f"加载tiles目录: {self.tiles_dir}")
        
        if not os.path.exists(self.tiles_dir):
            raise FileNotFoundError(f"Tiles目录不存在: {self.tiles_dir}")
        
        # 扫描所有tile文件
        tile_files = [f for f in os.listdir(self.tiles_dir) if f.endswith('.npy')]
        
        for tile_file in tile_files:
            # 解析tile索引
            tile_name = tile_file.replace('.npy', '')
            parts = tile_name.split('_')
            if len(parts) >= 3 and parts[0] == 'tile':
                try:
                    tile_x, tile_y = int(parts[1]), int(parts[2])
                    
                    # 加载体素数据
                    npy_path = os.path.join(self.tiles_dir, tile_file)
                    json_path = os.path.join(self.tiles_dir, f"{tile_name}.json")
                    
                    voxel_data = np.load(npy_path)
                    
                    # 加载元数据
                    metadata = {}
                    if os.path.exists(json_path):
                        with open(json_path, 'r') as f:
                            metadata = json.load(f)
                    
                    self.tiles_data[(tile_x, tile_y)] = {
                        'voxels': voxel_data, 'metadata': metadata
                    }
                    
                    logger.info(f"加载tile ({tile_x}, {tile_y}): {voxel_data.shape}")
                    
                except (ValueError, IndexError) as e:
                    logger.warning(f"跳过无效tile文件: {tile_file}, 错误: {e}")
        
        logger.info(f"成功加载 {len(self.tiles_data)} 个tiles")
    
    def get_tile_coordinates(
        self,
        tile_x: int,
        tile_y: int,
    )
        """
        获取tile的坐标网格
        
        Args:
            tile_x, tile_y: tile索引
            
        Returns:
            (x_coords, y_coords, z_coords) 全局坐标
        """
        if (tile_x, tile_y) not in self.tiles_data:
            raise ValueError(f"Tile ({tile_x}, {tile_y}) 不存在")
        
        metadata = self.tiles_data[(tile_x, tile_y)]['metadata']
        tile_origin = metadata.get('tile_origin', (tile_x * 1000, tile_y * 1000))
        grid_size = metadata.get('grid_size', (1000, 1000, 100))
        
        # 创建坐标网格
        x_coords, y_coords, z_coords = np.meshgrid(
            np.arange(
                grid_size[0],
            )
        )
        
        return x_coords, y_coords, z_coords
    
    def sample_uniform(
        self,
        tile_x: int,
        tile_y: int,
        n_samples: int = 10000,
        include_occupied: bool = True,
        include_free: bool = True,
    )
        """
        均匀采样
        
        Args:
            tile_x, tile_y: tile索引
            n_samples: 采样数量
            include_occupied: 是否包含占用体素
            include_free: 是否包含自由体素
            
        Returns:
            采样结果字典
        """
        if (tile_x, tile_y) not in self.tiles_data:
            raise ValueError(f"Tile ({tile_x}, {tile_y}) 不存在")
        
        voxel_data = self.tiles_data[(tile_x, tile_y)]['voxels']
        x_coords, y_coords, z_coords = self.get_tile_coordinates(tile_x, tile_y)
        
        # 找到所有非零体素
        occupied_indices = np.where(voxel_data > 0)
        free_indices = np.where(voxel_data == 0)
        
        samples = []
        labels = []
        
        # 采样占用体素
        if include_occupied and len(occupied_indices[0]) > 0:
            n_occupied = min(n_samples // 2, len(occupied_indices[0]))
            occupied_sample_indices = np.random.choice(
                len(occupied_indices[0]), n_occupied, replace=False
            )
            
            for idx in occupied_sample_indices:
                i, j, k = occupied_indices[0][idx], occupied_indices[1][idx], occupied_indices[2][idx]
                samples.append([x_coords[i, j, k], y_coords[i, j, k], z_coords[i, j, k]])
                labels.append(1.0)  # 占用
        
        # 采样自由体素
        if include_free and len(free_indices[0]) > 0:
            n_free = n_samples - len(samples)
            free_sample_indices = np.random.choice(
                len(free_indices[0]), n_free, replace=False
            )
            
            for idx in free_sample_indices:
                i, j, k = free_indices[0][idx], free_indices[1][idx], free_indices[2][idx]
                samples.append([x_coords[i, j, k], y_coords[i, j, k], z_coords[i, j, k]])
                labels.append(0.0)  # 自由
        
        return {
            'coordinates': np.array(
                samples,
            )
        }
    
    def sample_stratified(
        self,
        tile_x: int,
        tile_y: int,
        n_samples: int = 10000,
        occupied_ratio: float = 0.3,
    )
        """
        分层采样（确保占用和自由体素的平衡）
        
        Args:
            tile_x, tile_y: tile索引
            n_samples: 采样数量
            occupied_ratio: 占用体素比例
            
        Returns:
            采样结果字典
        """
        if (tile_x, tile_y) not in self.tiles_data:
            raise ValueError(f"Tile ({tile_x}, {tile_y}) 不存在")
        
        voxel_data = self.tiles_data[(tile_x, tile_y)]['voxels']
        x_coords, y_coords, z_coords = self.get_tile_coordinates(tile_x, tile_y)
        
        # 找到所有非零体素
        occupied_indices = np.where(voxel_data > 0)
        free_indices = np.where(voxel_data == 0)
        
        n_occupied_samples = int(n_samples * occupied_ratio)
        n_free_samples = n_samples - n_occupied_samples
        
        samples = []
        labels = []
        
        # 采样占用体素
        if len(occupied_indices[0]) > 0:
            n_occupied_actual = min(n_occupied_samples, len(occupied_indices[0]))
            occupied_sample_indices = np.random.choice(
                len(occupied_indices[0]), n_occupied_actual, replace=False
            )
            
            for idx in occupied_sample_indices:
                i, j, k = occupied_indices[0][idx], occupied_indices[1][idx], occupied_indices[2][idx]
                samples.append([x_coords[i, j, k], y_coords[i, j, k], z_coords[i, j, k]])
                labels.append(1.0)  # 占用
        
        # 采样自由体素
        if len(free_indices[0]) > 0:
            n_free_actual = min(n_free_samples, len(free_indices[0]))
            free_sample_indices = np.random.choice(
                len(free_indices[0]), n_free_actual, replace=False
            )
            
            for idx in free_sample_indices:
                i, j, k = free_indices[0][idx], free_indices[1][idx], free_indices[2][idx]
                samples.append([x_coords[i, j, k], y_coords[i, j, k], z_coords[i, j, k]])
                labels.append(0.0)  # 自由
        
        return {
            'coordinates': np.array(
                samples,
            )
        }
    
    def sample_near_surface(
        self,
        tile_x: int,
        tile_y: int,
        n_samples: int = 10000,
        surface_threshold: float = 0.5,
        noise_std: float = 2.0,
    )
        """
        表面附近采样（用于SDF训练）
        
        Args:
            tile_x, tile_y: tile索引
            n_samples: 采样数量
            surface_threshold: 表面阈值
            noise_std: 噪声标准差
            
        Returns:
            采样结果字典
        """
        if (tile_x, tile_y) not in self.tiles_data:
            raise ValueError(f"Tile ({tile_x}, {tile_y}) 不存在")
        
        voxel_data = self.tiles_data[(tile_x, tile_y)]['voxels']
        x_coords, y_coords, z_coords = self.get_tile_coordinates(tile_x, tile_y)
        
        # 找到表面附近的体素（边界检测）
        surface_mask = self._detect_surface_voxels(voxel_data, surface_threshold)
        surface_indices = np.where(surface_mask)
        
        samples = []
        sdf_values = []
        
        if len(surface_indices[0]) > 0:
            # 在表面附近采样
            n_surface_samples = min(n_samples, len(surface_indices[0]))
            surface_sample_indices = np.random.choice(
                len(surface_indices[0]), n_surface_samples, replace=False
            )
            
            for idx in surface_sample_indices:
                i, j, k = surface_indices[0][idx], surface_indices[1][idx], surface_indices[2][idx]
                
                # 基础坐标
                base_coord = np.array([x_coords[i, j, k], y_coords[i, j, k], z_coords[i, j, k]])
                
                # 添加噪声以增加样本多样性
                noise = np.random.normal(0, noise_std, 3)
                sample_coord = base_coord + noise
                
                samples.append(sample_coord)
                
                # 计算简化的SDF（距离最近表面的距离）
                sdf_value = self._compute_approximate_sdf(
                    sample_coord,
                    voxel_data,
                    x_coords,
                    y_coords,
                    z_coords,
                )
                sdf_values.append(sdf_value)
        
        return {
            'coordinates': np.array(
                samples,
            )
        }
    
    def _detect_surface_voxels(self, voxel_data: np.ndarray, threshold: float) -> np.ndarray:
        """检测表面体素"""
        # 简化的表面检测：找到占用体素附近的自由体素
        from scipy.ndimage import binary_erosion, binary_dilation
        
        occupied = voxel_data > threshold
        dilated = binary_dilation(occupied)
        eroded = binary_erosion(occupied)
        
        # 表面是膨胀和侵蚀的差异
        surface = dilated & ~eroded
        
        return surface
    
    def _compute_approximate_sdf(
        self,
        point: np.ndarray,
        voxel_data: np.ndarray,
        x_coords: np.ndarray,
        y_coords: np.ndarray,
        z_coords: np.ndarray,
    )
        """计算近似SDF值"""
        # 找到最近的体素
        distances = np.sqrt(
            (x_coords - point[0])**2 + 
            (y_coords - point[1])**2 + 
            (z_coords - point[2])**2
        )
        
        min_idx = np.unravel_index(np.argmin(distances), distances.shape)
        nearest_occupancy = voxel_data[min_idx]
        min_distance = distances[min_idx]
        
        # 如果最近体素是占用的，SDF为负；否则为正
        sdf = min_distance if nearest_occupancy == 0 else -min_distance
        
        return sdf
    
    def sample_all_tiles(
        self,
        sampling_method: str = 'uniform',
        n_samples_per_tile: int = 10000,
        **kwargs,
    )
        """
        对所有tiles进行采样
        
        Args:
            sampling_method: 采样方法
            n_samples_per_tile: 每个tile的采样数量
            **kwargs: 传递给采样方法的额外参数
            
        Returns:
            所有tile的采样结果列表
        """
        logger.info(f"使用{sampling_method}方法对{len(self.tiles_data)}个tiles进行采样")
        
        all_samples = []
        
        for (tile_x, tile_y) in self.tiles_data.keys():
            try:
                if sampling_method == 'uniform':
                    samples = self.sample_uniform(tile_x, tile_y, n_samples_per_tile, **kwargs)
                elif sampling_method == 'stratified':
                    samples = self.sample_stratified(tile_x, tile_y, n_samples_per_tile, **kwargs)
                elif sampling_method == 'surface':
                    samples = self.sample_near_surface(tile_x, tile_y, n_samples_per_tile, **kwargs)
                else:
                    raise ValueError(f"未知采样方法: {sampling_method}")
                
                all_samples.append(samples)
                logger.info(f"Tile ({tile_x}, {tile_y}) 采样完成: {len(samples['coordinates'])} 个样本")
                
            except Exception as e:
                logger.error(f"Tile ({tile_x}, {tile_y}) 采样失败: {e}")
        
        logger.info(f"总共采样完成 {len(all_samples)} 个tiles")
        return all_samples
    
    def save_samples(self, samples: list[dict[str, np.ndarray]], output_dir: str = "samples"):
        """
        保存采样结果
        
        Args:
            samples: 采样结果列表
            output_dir: 输出目录
        """
        os.makedirs(output_dir, exist_ok=True)
        
        for i, sample in enumerate(samples):
            tile_index = sample.get('tile_index', (i, 0))
            
            # 保存为npz格式
            output_file = os.path.join(
                output_dir,
                f"samples_tile_{tile_index[0]}_{tile_index[1]}.npz",
            )
            
            # 准备保存数据
            save_data = {
                'coordinates': sample['coordinates'], }
            
            if 'labels' in sample:
                save_data['labels'] = sample['labels']
            if 'sdf' in sample:
                save_data['sdf'] = sample['sdf']
            
            np.savez_compressed(output_file, **save_data)
            logger.info(f"样本已保存: {output_file}")
    
    def get_tile_info(self) -> dict[str, Any]:
        """获取tiles信息"""
        info = {
            'num_tiles': len(self.tiles_data), 'tiles': []
        }
        
        for (tile_x, tile_y), data in self.tiles_data.items():
            tile_info = {
                'index': (
                    tile_x,
                    tile_y,
                )
            }
            info['tiles'].append(tile_info)
        
        return info 