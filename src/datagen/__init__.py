"""
DataGen - NeuroCity 数据生成软件包

用于生成训练神经辐射场所需的各种仿真数据，包括：
- 体素采样
- SDF数据生成
- 占用网格生成
- 合成场景数据
"""

from .core import DataGenConfig, DataGenPipeline
from .samplers import VoxelSampler, SurfaceSampler, PointCloudSampler
from .generators import SDFGenerator, OccupancyGenerator, SyntheticSceneGenerator
from .datasets import SyntheticVoxelDataset, SDFDataset, OccupancyDataset

__version__ = "1.0.0"
__author__ = "NeuroCity Team"

__all__ = [
    # 核心组件
    'DataGenConfig',
    'DataGenPipeline',
    
    # 采样器
    'VoxelSampler',
    'SurfaceSampler', 
    'PointCloudSampler',
    
    # 生成器
    'SDFGenerator',
    'OccupancyGenerator',
    'SyntheticSceneGenerator',
    
    # 数据集
    'SyntheticVoxelDataset',
    'SDFDataset',
    'OccupancyDataset',
] 