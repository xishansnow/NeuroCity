"""
GFV Dataset Module - 数据集组件

This module contains dataset classes for GFV library including:
- SDFDataset: SDF data handling
- GlobalFeatureDataset: Global feature data handling
"""

import torch
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from torch.utils.data import Dataset
import logging

logger = logging.getLogger(__name__)


class SDFDataset(Dataset):
    """SDF数据集"""
    
    def __init__(self, npy_path: str):
        """
        初始化SDF数据集
        
        Args:
            npy_path: SDF数据文件路径，格式为(N, 4) - (x, y, z, sdf)
        """
        data = np.load(npy_path)  # shape: (N, 4)
        self.coords = data[:, :3].astype(np.float32)
        self.sdf = data[:, 3].astype(np.float32)
        
        # 归一化坐标到[0,1]
        self.coords = (self.coords - self.coords.min(0)) / (self.coords.max(0) - self.coords.min(0))
        
        logger.info(f"已加载SDF数据集: {len(self)} 个样本")
    
    def __len__(self) -> int:
        return len(self.coords)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return torch.from_numpy(self.coords[idx]), torch.from_numpy(np.array([self.sdf[idx]]))


class GlobalFeatureDataset(Dataset):
    """全球特征数据集"""
    
    def __init__(self, 
                 coords: List[Tuple[float, float]], 
                 features: List[np.ndarray],
                 zoom_levels: Optional[List[int]] = None):
        """
        初始化全球特征数据集
        
        Args:
            coords: 经纬度坐标列表
            features: 对应的特征向量列表
            zoom_levels: 对应的缩放级别列表
        """
        self.coords = coords
        self.features = features
        self.zoom_levels = zoom_levels or [10] * len(coords)
        
        assert len(self.coords) == len(self.features) == len(self.zoom_levels)
        
        logger.info(f"已创建全球特征数据集: {len(self)} 个样本")
    
    def __len__(self) -> int:
        return len(self.coords)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        lat, lon = self.coords[idx]
        features = self.features[idx]
        zoom = self.zoom_levels[idx]
        
        return {
            'coords': torch.tensor([lat, lon], dtype=torch.float32),
            'features': torch.from_numpy(features).float(),
            'zoom': zoom
        }
    
    @classmethod
    def from_database(cls, database, coords: List[Tuple[float, float]], zoom: int = 10):
        """从数据库创建数据集"""
        features = []
        valid_coords = []
        
        for lat, lon in coords:
            feature = database.query_features(lat, lon, zoom)
            if feature is not None:
                features.append(feature)
                valid_coords.append((lat, lon))
        
        return cls(valid_coords, features, [zoom] * len(valid_coords))


class GeospatialDataset(Dataset):
    """地理空间数据集"""
    
    def __init__(self, 
                 bounds: Tuple[float, float, float, float],
                 resolution: int = 256,
                 zoom: int = 10):
        """
        初始化地理空间数据集
        
        Args:
            bounds: 边界坐标 (west, south, east, north)
            resolution: 采样分辨率
            zoom: 缩放级别
        """
        self.bounds = bounds
        self.resolution = resolution
        self.zoom = zoom
        
        # 生成采样网格
        west, south, east, north = bounds
        lats = np.linspace(south, north, resolution)
        lons = np.linspace(west, east, resolution)
        
        self.lat_grid, self.lon_grid = np.meshgrid(lats, lons)
        self.coords = np.stack([
            self.lat_grid.flatten(), 
            self.lon_grid.flatten()
        ], axis=1)
        
        logger.info(f"已创建地理空间数据集: {len(self)} 个采样点")
    
    def __len__(self) -> int:
        return len(self.coords)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        lat, lon = self.coords[idx]
        
        # 计算网格索引
        grid_idx = np.unravel_index(idx, (self.resolution, self.resolution))
        
        return {
            'coords': torch.tensor([lat, lon], dtype=torch.float32),
            'grid_idx': torch.tensor(grid_idx, dtype=torch.long),
            'zoom': self.zoom
        }
    
    def get_grid_shape(self) -> Tuple[int, int]:
        """获取网格形状"""
        return (self.resolution, self.resolution)


class MultiScaleDataset(Dataset):
    """多尺度数据集"""
    
    def __init__(self, 
                 base_coords: List[Tuple[float, float]],
                 zoom_levels: List[int] = [8, 10, 12, 14]):
        """
        初始化多尺度数据集
        
        Args:
            base_coords: 基础坐标列表
            zoom_levels: 缩放级别列表
        """
        self.base_coords = base_coords
        self.zoom_levels = zoom_levels
        
        # 生成多尺度样本
        self.samples = []
        for lat, lon in base_coords:
            for zoom in zoom_levels:
                self.samples.append((lat, lon, zoom))
        
        logger.info(f"已创建多尺度数据集: {len(self)} 个样本 ({len(base_coords)} 位置 × {len(zoom_levels)} 尺度)")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        lat, lon, zoom = self.samples[idx]
        
        return {
            'coords': torch.tensor([lat, lon], dtype=torch.float32),
            'zoom': zoom,
            'scale_idx': self.zoom_levels.index(zoom)
        } 