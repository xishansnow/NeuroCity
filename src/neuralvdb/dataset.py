"""
Dataset Module for NeuralVDB

This module contains dataset classes for loading and processing
volumetric data for NeuralVDB training.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, List, Optional, Tuple, import logging
import os
import json

logger = logging.getLogger(__name__)


class NeuralVDBDataset(Dataset):
    """NeuralVDB数据集"""
    
    def __init__(self, points: np.ndarray, occupancies: np.ndarray, transform=None):
        """
        初始化NeuralVDB数据集
        
        Args:
            points: 3D坐标点 (N, 3)
            occupancies: 占用值 (N, )
            transform: 数据变换函数
        """
        self.points = torch.FloatTensor(points)
        self.occupancies = torch.FloatTensor(occupancies)
        self.transform = transform
        
        assert len(self.points) == len(self.occupancies), \
            f"点数量 {len(self.points)} 与占用值数量 {len(self.occupancies)} 不匹配"
        
        logger.info(f"NeuralVDB数据集初始化完成，包含 {len(self.points)} 个点")
    
    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.points)
    
    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        """Get a dataset item."""
        point = self.points[idx]
        occupancy = self.occupancies[idx]
        
        if self.transform:
            point, occupancy = self.transform(point, occupancy)
        
        return point, occupancy
    
    def get_statistics(self) -> dict[str, float]:
        """Get dataset statistics."""
        points_np = self.points.numpy()
        occupancies_np = self.occupancies.numpy()
        
        return {
            'num_points': len(
                self.points,
            )
        }


class VoxelDataset(Dataset):
    """体素数据集 - 用于SDF/Occupancy训练"""
    
    def __init__(
        self,
        coords: np.ndarray,
        labels: Optional[np.ndarray] = None,
        sdf_values: Optional[np.ndarray] = None,
        task_type: str = 'occupancy',
        transform=None,
    )
        """
        初始化体素数据集
        
        Args:
            coords: 坐标数据 (N, 3)
            labels: 占用标签 (N, ) 或 None
            sdf_values: SDF值 (N, ) 或 None
            task_type: 任务类型 ('occupancy' 或 'sdf')
            transform: 数据变换函数
        """
        self.coords = torch.FloatTensor(coords)
        self.task_type = task_type
        self.transform = transform
        
        if task_type == 'occupancy':
            if labels is None:
                raise ValueError("Occupancy任务需要labels")
            self.targets = torch.FloatTensor(labels)
        elif task_type == 'sdf':
            if sdf_values is None:
                raise ValueError("SDF任务需要sdf_values")
            self.targets = torch.FloatTensor(sdf_values)
        else:
            raise ValueError(f"未知任务类型: {task_type}")
        
        assert len(self.coords) == len(self.targets), \
            f"坐标数量 {len(self.coords)} 与目标数量 {len(self.targets)} 不匹配"
        
        logger.info(f"体素数据集初始化完成，任务类型: {task_type}，包含 {len(self.coords)} 个样本")
    
    def __len__(self):
        return len(self.coords)
    
    def __getitem__(self, idx):
        coord = self.coords[idx]
        target = self.targets[idx]
        
        if self.transform:
            coord, target = self.transform(coord, target)
        
        return coord, target
    
    def get_statistics(self) -> dict[str, float]:
        """获取数据集统计信息"""
        coords_np = self.coords.numpy()
        targets_np = self.targets.numpy()
        
        stats = {
            'num_samples': len(
                self.coords,
            )
        }
        
        if self.task_type == 'occupancy':
            stats['occupied_ratio'] = float(np.mean(targets_np > 0.5))
        elif self.task_type == 'sdf':
            stats['surface_ratio'] = float(np.mean(np.abs(targets_np) < 0.1))
        
        return stats


class TileDataset(Dataset):
    """瓦片数据集 - 用于大规模城市场景"""
    
    def __init__(
        self,
        tiles_dir: str,
        tile_indices: Optional[list[tuple[int,
        int]]] = None,
        load_in_memory: bool = False,
        transform=None,
    )
        """
        初始化瓦片数据集
        
        Args:
            tiles_dir: 瓦片目录路径
            tile_indices: 要加载的瓦片索引列表，None表示加载所有
            load_in_memory: 是否将所有数据加载到内存
            transform: 数据变换函数
        """
        self.tiles_dir = tiles_dir
        self.load_in_memory = load_in_memory
        self.transform = transform
        
        # 扫描瓦片文件
        self.tile_files = self._scan_tile_files(tile_indices)
        
        # 如果需要，加载到内存
        if load_in_memory:
            self.tiles_data = self._load_all_tiles()
        else:
            self.tiles_data = None
        
        logger.info(f"瓦片数据集初始化完成，包含 {len(self.tile_files)} 个瓦片")
    
    def _scan_tile_files(self, tile_indices: Optional[list[tuple[int, int]]]) -> list[Dict]:
        """扫描瓦片文件"""
        tile_files = []
        
        if tile_indices is None:
            # 扫描所有瓦片文件
            for filename in os.listdir(self.tiles_dir):
                if filename.endswith('.npy') and filename.startswith('tile_'):
                    # 解析瓦片索引
                    parts = filename.replace('.npy', '').split('_')
                    if len(parts) >= 3:
                        try:
                            tile_x = int(parts[1])
                            tile_y = int(parts[2])
                            
                            npy_path = os.path.join(self.tiles_dir, filename)
                            json_path = os.path.join(
                                self.tiles_dir,
                                filename.replace,
                            )
                            
                            tile_files.append({
                                'tile_index': (
                                    tile_x,
                                    tile_y,
                                )
                            })
                        except ValueError:
                            continue
        else:
            # 加载指定的瓦片
            for tile_x, tile_y in tile_indices:
                npy_filename = f"tile_{tile_x}_{tile_y}.npy"
                json_filename = f"tile_{tile_x}_{tile_y}.json"
                
                npy_path = os.path.join(self.tiles_dir, npy_filename)
                json_path = os.path.join(self.tiles_dir, json_filename)
                
                if os.path.exists(npy_path):
                    tile_files.append({
                        'tile_index': (
                            tile_x,
                            tile_y,
                        )
                    })
        
        return tile_files
    
    def _load_all_tiles(self) -> list[Dict]:
        """加载所有瓦片到内存"""
        tiles_data = []
        
        for tile_file in self.tile_files:
            tile_data = self._load_single_tile(tile_file)
            tiles_data.append(tile_data)
        
        return tiles_data
    
    def _load_single_tile(self, tile_file: Dict) -> Dict:
        """Load a single tile."""
        # 加载体素数据
        voxel_grid = np.load(tile_file['npy_path'])
        
        # 加载元数据
        metadata = None
        if tile_file['json_path']:
            with open(tile_file['json_path'], 'r') as f:
                metadata = json.load(f)
        
        # 提取非零体素坐标和值
        occupied_indices = np.where(voxel_grid > 0)
        coords = np.column_stack(occupied_indices)
        values = voxel_grid[occupied_indices]
        
        return {
            'tile_index': tile_file['tile_index'], 'coords': coords, 'values': values, 'grid_shape': voxel_grid.shape, 'metadata': metadata
        }
    
    def __len__(self):
        return len(self.tile_files)
    
    def __getitem__(self, idx):
        if self.load_in_memory:
            tile_data = self.tiles_data[idx]
        else:
            tile_data = self._load_single_tile(self.tile_files[idx])
        
        coords = torch.FloatTensor(tile_data['coords'])
        values = torch.FloatTensor(tile_data['values'])
        
        if self.transform:
            coords, values = self.transform(coords, values)
        
        return {
            'coords': coords, 'values': values, 'tile_index': tile_data['tile_index'], 'grid_shape': tile_data['grid_shape'], 'metadata': tile_data['metadata']
        }
    
    def get_tile_statistics(self) -> dict[str, Any]:
        """Get tile statistics."""
        total_voxels = 0
        total_occupied = 0
        tile_info = []
        
        for i in range(len(self)):
            tile_data = self[i]
            coords = tile_data['coords']
            values = tile_data['values']
            grid_shape = tile_data['grid_shape']
            
            voxel_count = np.prod(grid_shape)
            occupied_count = len(coords)
            
            total_voxels += voxel_count
            total_occupied += occupied_count
            
            tile_info.append({
                'tile_index': tile_data['tile_index'], 'total_voxels': voxel_count, 'occupied_voxels': occupied_count, 'occupancy_ratio': occupied_count / voxel_count, 'grid_shape': grid_shape
            })
        
        return {
            'num_tiles': len(
                self,
            )
        }


class MultiScaleDataset(Dataset):
    """多尺度数据集 - 支持不同分辨率的训练"""
    
    def __init__(
        self,
        base_dataset: Dataset,
        scales: list[float] = [1.0,
        0.5,
        0.25],
        scale_weights: Optional[list[float]] = None,
    )
        """
        初始化多尺度数据集
        
        Args:
            base_dataset: 基础数据集
            scales: 尺度列表
            scale_weights: 尺度权重（用于采样）
        """
        self.base_dataset = base_dataset
        self.scales = scales
        
        if scale_weights is None:
            self.scale_weights = [1.0] * len(scales)
        else:
            self.scale_weights = scale_weights
        
        # 归一化权重
        total_weight = sum(self.scale_weights)
        self.scale_weights = [w / total_weight for w in self.scale_weights]
        
        logger.info(f"多尺度数据集初始化完成，尺度: {scales}")
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        # 获取基础数据
        if isinstance(self.base_dataset[idx], tuple):
            coords, targets = self.base_dataset[idx]
        else:
            # 处理字典格式
            data = self.base_dataset[idx]
            coords = data['coords']
            targets = data['values']
        
        # 随机选择尺度
        scale_idx = np.random.choice(len(self.scales), p=self.scale_weights)
        scale = self.scales[scale_idx]
        
        # 应用尺度变换
        scaled_coords = coords * scale
        
        return {
            'coords': scaled_coords, 'targets': targets, 'scale': scale, 'scale_idx': scale_idx, 'original_coords': coords
        }


# 数据变换函数
class DataTransforms:
    """数据变换函数集合"""
    
    @staticmethod
    def normalize_coords(
        coords: torch.Tensor,
        bbox_min: torch.Tensor,
        bbox_max: torch.Tensor,
    )
        """坐标归一化到[0, 1]"""
        return (coords - bbox_min) / (bbox_max - bbox_min)
    
    @staticmethod
    def add_noise(coords: torch.Tensor, noise_std: float = 0.01) -> torch.Tensor:
        """添加高斯噪声"""
        noise = torch.normal(0, noise_std, size=coords.shape)
        return coords + noise
    
    @staticmethod
    def random_rotation(coords: torch.Tensor, max_angle: float = 0.1) -> torch.Tensor:
        """随机旋转（仅绕Z轴）"""
        angle = torch.uniform(-max_angle, max_angle)
        cos_a, sin_a = torch.cos(angle), torch.sin(angle)
        
        rotation_matrix = torch.tensor([
            [cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]
        ], dtype=coords.dtype)
        
        return torch.matmul(coords, rotation_matrix.T)
    
    @staticmethod
    def random_scale(
        coords: torch.Tensor,
        scale_range: tuple[float,
        float] =,
    )
        """随机缩放"""
        scale = torch.uniform(scale_range[0], scale_range[1])
        return coords * scale


def create_data_loaders(
    dataset: Dataset,
    train_ratio: float = 0.8,
    batch_size: int = 1024,
    num_workers: int = 4,
    shuffle: bool = True,
)
    """
    创建训练和验证数据加载器
    
    Args:
        dataset: 数据集
        train_ratio: 训练集比例
        batch_size: 批大小
        num_workers: 工作进程数
        shuffle: 是否打乱数据
        
    Returns:
        (train_loader, val_loader)
    """
    # 分割数据集
    train_size = int(len(dataset) * train_ratio)
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=torch.cuda.is_available(
        )
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available(
        )
    )
    
    logger.info(f"数据加载器创建完成，训练集: {train_size}，验证集: {val_size}")
    
    return train_loader, val_loader 