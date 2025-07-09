"""
SVRaster One - 单一可微分光栅化渲染器

基于 SVRaster 原始论文实现的可微分稀疏体素光栅化渲染器。
支持端到端的梯度传播和训练。

主要特性：
- 可微分的光栅化渲染
- 稀疏体素表示
- CUDA 加速
- 端到端训练支持
- 实时推理渲染
"""

__version__ = "1.0.0"
__author__ = "SVRaster One Team"

from .core import SVRasterOne
from .config import SVRasterOneConfig
from .renderer import DifferentiableVoxelRasterizer
from .voxels import SparseVoxelGrid
from .losses import SVRasterOneLoss
from .trainer import SVRasterOneTrainer

__all__ = [
    "SVRasterOne",
    "SVRasterOneConfig",
    "DifferentiableVoxelRasterizer",
    "SparseVoxelGrid",
    "SVRasterOneLoss",
    "SVRasterOneTrainer",
]
