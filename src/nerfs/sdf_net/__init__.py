"""
SDF Networks Package
基于论文: "DeepSDF: Learning Continuous Signed Distance Functions for Shape Representation"

实现了SDF网络的完整架构，包括：
- 核心网络模型
- 数据集处理
- 训练器
- 工具函数
"""

from .core import SDFNetwork
from .dataset import SDFDataset
from .trainer import SDFTrainer

__all__ = [
    'SDFNetwork', 'SDFDataset', 'SDFTrainer'
] 