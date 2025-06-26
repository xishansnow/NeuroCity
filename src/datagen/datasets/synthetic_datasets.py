"""
合成数据集模块

用于创建各种合成数据集，支持体素、SDF和占用网格数据。
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class SyntheticVoxelDataset(Dataset):
    """合成体素数据集"""
    
    def __init__(
        self,
        coordinates: np.ndarray,
        labels: np.ndarray,
        transform: Optional[callable] = None,
    )
        """
        初始化合成体素数据集
        
        Args:
            coordinates: 坐标数组 [N, 3]
            labels: 标签数组 [N]
            transform: 可选的数据变换
        """
        self.coordinates = torch.FloatTensor(coordinates)
        self.labels = torch.FloatTensor(labels)
        self.transform = transform
        
    def __len__(self):
        return len(self.coordinates)
    
    def __getitem__(self, idx):
        coord = self.coordinates[idx]
        label = self.labels[idx]
        
        if self.transform:
            coord = self.transform(coord)
        
        return coord, label


class SDFDataset(Dataset):
    """SDF数据集"""
    
    def __init__(
        self,
        coordinates: np.ndarray,
        sdf_values: np.ndarray,
        transform: Optional[callable] = None,
    )
        """
        初始化SDF数据集
        
        Args:
            coordinates: 坐标数组 [N, 3]
            sdf_values: SDF值数组 [N]
            transform: 可选的数据变换
        """
        self.coordinates = torch.FloatTensor(coordinates)
        self.sdf_values = torch.FloatTensor(sdf_values)
        self.transform = transform
        
    def __len__(self):
        return len(self.coordinates)
    
    def __getitem__(self, idx):
        coord = self.coordinates[idx]
        sdf = self.sdf_values[idx]
        
        if self.transform:
            coord = self.transform(coord)
        
        return coord, sdf


class OccupancyDataset(Dataset):
    """占用网格数据集"""
    
    def __init__(
        self,
        coordinates: np.ndarray,
        occupancy: np.ndarray,
        transform: Optional[callable] = None,
    )
        """
        初始化占用数据集
        
        Args:
            coordinates: 坐标数组 [N, 3]
            occupancy: 占用值数组 [N]
            transform: 可选的数据变换
        """
        self.coordinates = torch.FloatTensor(coordinates)
        self.occupancy = torch.FloatTensor(occupancy)
        self.transform = transform
        
    def __len__(self):
        return len(self.coordinates)
    
    def __getitem__(self, idx):
        coord = self.coordinates[idx]
        occ = self.occupancy[idx]
        
        if self.transform:
            coord = self.transform(coord)
        
        return coord, occ


class MultiModalDataset(Dataset):
    """多模态数据集"""
    
    def __init__(
        self,
        coordinates: np.ndarray,
        sdf_values: np.ndarray,
        occupancy: np.ndarray,
        normals: Optional[np.ndarray] = None,
    )
        """
        初始化多模态数据集
        
        Args:
            coordinates: 坐标数组 [N, 3]
            sdf_values: SDF值数组 [N]
            occupancy: 占用值数组 [N]
            normals: 法向量数组 [N, 3] (可选)
        """
        self.coordinates = torch.FloatTensor(coordinates)
        self.sdf_values = torch.FloatTensor(sdf_values)
        self.occupancy = torch.FloatTensor(occupancy)
        
        if normals is not None:
            self.normals = torch.FloatTensor(normals)
        else:
            self.normals = None
    
    def __len__(self):
        return len(self.coordinates)
    
    def __getitem__(self, idx):
        result = {
            'coordinates': self.coordinates[idx], 'sdf': self.sdf_values[idx], 'occupancy': self.occupancy[idx]
        }
        
        if self.normals is not None:
            result['normals'] = self.normals[idx]
        
        return result 