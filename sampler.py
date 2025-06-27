from typing import Optional
#!/usr/bin/env python3
"""
体素采样器模块
用于从tile体素数据中采样训练数据，支持多种采样策略
"""

import numpy as np
import json
import os
import random
from scipy.spatial import cKDTree
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VoxelSampler:
    def __init__(
        self,
        tiles_dir: str = "tiles",
        voxel_size: float = 1.0,
        sample_ratio: float = 0.1,
    ):
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
    ):
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
    ):
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
    ):
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
            n_occupied = min(n_occupied_samples, len(occupied_indices[0]))
            occupied_sample_indices = np.random.choice(
                len(occupied_indices[0]), n_occupied, replace=False
            )
            
            for idx in occupied_sample_indices:
                i, j, k = occupied_indices[0][idx], occupied_indices[1][idx], occupied_indices[2][idx]
                samples.append([x_coords[i, j, k], y_coords[i, j, k], z_coords[i, j, k]])
                labels.append(1.0)
        
        # 采样自由体素
        if len(free_indices[0]) > 0:
            n_free = min(n_free_samples, len(free_indices[0]))
            free_sample_indices = np.random.choice(
                len(free_indices[0]), n_free, replace=False
            )
            
            for idx in free_sample_indices:
                i, j, k = free_indices[0][idx], free_indices[1][idx], free_indices[2][idx]
                samples.append([x_coords[i, j, k], y_coords[i, j, k], z_coords[i, j, k]])
                labels.append(0.0)
        
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
    ):
        """
        表面附近采样（用于SDF训练）
        
        Args:
            tile_x, tile_y: tile索引
            n_samples: 采样数量
            surface_threshold: 表面阈值
            noise_std: 噪声标准差（米）
            
        Returns:
            采样结果字典
        """
        if (tile_x, tile_y) not in self.tiles_data:
            raise ValueError(f"Tile ({tile_x}, {tile_y}) 不存在")
        
        voxel_data = self.tiles_data[(tile_x, tile_y)]['voxels']
        x_coords, y_coords, z_coords = self.get_tile_coordinates(tile_x, tile_y)
        
        # 找到表面体素
        surface_indices = np.where(voxel_data > surface_threshold)
        
        if len(surface_indices[0]) == 0:
            logger.warning(f"Tile ({tile_x}, {tile_y}) 没有找到表面体素")
            return self.sample_uniform(tile_x, tile_y, n_samples)
        
        samples = []
        sdf_values = []
        
        # 从表面体素采样
        n_surface_samples = min(n_samples // 2, len(surface_indices[0]))
        surface_sample_indices = np.random.choice(
            len(surface_indices[0]), n_surface_samples, replace=False
        )
        
        for idx in surface_sample_indices:
            i, j, k = surface_indices[0][idx], surface_indices[1][idx], surface_indices[2][idx]
            
            # 表面点（SDF = 0）
            samples.append([x_coords[i, j, k], y_coords[i, j, k], z_coords[i, j, k]])
            sdf_values.append(0.0)
            
            # 表面附近的点（带噪声）
            for _ in range(3):  # 每个表面点生成3个附近点
                noise = np.random.normal(0, noise_std, 3)
                sample_point = [
                    x_coords[i, j, k] + noise[0], y_coords[i, j, k] + noise[1], z_coords[i, j, k] + noise[2]
                ]
                samples.append(sample_point)
                
                # 计算到表面的距离（简化版本）
                distance = np.linalg.norm(noise)
                sdf_values.append(distance)
        
        # 随机采样一些远离表面的点
        n_random_samples = n_samples - len(samples)
        if n_random_samples > 0:
            random_indices = np.random.choice(
                voxel_data.size, n_random_samples, replace=False
            )
            
            for idx in random_indices:
                i, j, k = np.unravel_index(idx, voxel_data.shape)
                samples.append([x_coords[i, j, k], y_coords[i, j, k], z_coords[i, j, k]])
                
                # 简化的SDF计算
                if voxel_data[i, j, k] > 0:
                    sdf_values.append(-1.0)  # 内部
                else:
                    sdf_values.append(1.0)   # 外部
        
        return {
            'coordinates': np.array(
                samples,
            )
        }
    
    def sample_all_tiles(
        self,
        sampling_method: str = 'stratified',
        n_samples_per_tile: int = 10000,
        **kwargs,
    ):
        """
        对所有tile进行采样
        
        Args:
            sampling_method: 采样方法 ('uniform', 'stratified', 'near_surface')
            n_samples_per_tile: 每个tile的采样数量
            **kwargs: 其他采样参数
            
        Returns:
            所有tile的采样结果列表
        """
        all_samples = []
        
        for tile_x, tile_y in self.tiles_data.keys():
            try:
                logger.info(f"采样tile ({tile_x}, {tile_y})")
                
                if sampling_method == 'uniform':
                    samples = self.sample_uniform(tile_x, tile_y, n_samples_per_tile, **kwargs)
                elif sampling_method == 'stratified':
                    samples = self.sample_stratified(tile_x, tile_y, n_samples_per_tile, **kwargs)
                elif sampling_method == 'near_surface':
                    samples = self.sample_near_surface(tile_x, tile_y, n_samples_per_tile, **kwargs)
                else:
                    raise ValueError(f"未知的采样方法: {sampling_method}")
                
                all_samples.append(samples)
                
            except Exception as e:
                logger.error(f"采样tile ({tile_x}, {tile_y}) 失败: {e}")
                continue
        
        logger.info(f"完成所有tile采样，共 {len(all_samples)} 个tile")
        return all_samples
    
    def save_samples(self, samples: list[dict[str, np.ndarray]], output_dir: str = "samples"):
        """
        保存采样数据
        
        Args:
            samples: 采样结果列表
            output_dir: 输出目录
        """
        os.makedirs(output_dir, exist_ok=True)
        
        for i, sample in enumerate(samples):
            tile_x, tile_y = sample['tile_index']
            
            # 保存坐标和标签
            np.save(
                os.path.join,
            )
            
            if 'labels' in sample:
                np.save(os.path.join(output_dir, f"labels_{tile_x}_{tile_y}.npy"), sample['labels'])
            
            if 'sdf_values' in sample:
                np.save(
                    os.path.join,
                )
        
        # 保存采样配置
        config = {
            'n_tiles': len(
                samples,
            )
        }
        
        with open(os.path.join(output_dir, 'sampling_config.json'), 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"采样数据已保存到: {output_dir}")

def main():
    """示例用法"""
    # 创建采样器
    sampler = VoxelSampler(tiles_dir="tiles", voxel_size=1.0)
    
    # 对单个tile进行采样
    print("测试单个tile采样...")
    samples = sampler.sample_stratified(0, 0, n_samples=5000)
    print(f"采样结果: {samples['coordinates'].shape}")
    
    # 对所有tile进行采样
    print("对所有tile进行采样...")
    all_samples = sampler.sample_all_tiles(
        sampling_method='stratified', n_samples_per_tile=5000
    )
    
    # 保存采样数据
    sampler.save_samples(all_samples, "samples")
    
    print("采样完成！")

if __name__ == "__main__":
    main() 