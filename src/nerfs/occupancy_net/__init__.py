"""
Occupancy Networks Package
基于论文: "Occupancy Networks: Learning 3D Reconstruction in Function Space"

实现了占用网络的完整架构，包括：
- 核心网络模型
- 数据集处理
- 训练器
- 工具函数
"""

from .core import OccupancyNetwork
from .dataset import OccupancyDataset
from .trainer import OccupancyTrainer

__all__ = [
    'OccupancyNetwork', 'OccupancyDataset', 'OccupancyTrainer'
] 