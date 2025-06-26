"""
GFV (Global Feature Vector) Library

A high-performance global feature vector library based on multi-resolution hash encoding
for neural graphics primitives and spatial data processing.

核心特性:
- 多分辨率哈希编码
- 全球地理坐标支持
- 分层特征表示
- 高效查询和更新
- 分布式存储支持
"""

from .core import (
    GlobalHashConfig, MultiResolutionHashEncoding, GlobalFeatureDatabase, GlobalFeatureLibrary
)

from .dataset import (
    SDFDataset, GlobalFeatureDataset
)

from .trainer import (
    GFVTrainer, GFVLightningModule
)

__version__ = "1.0.0"
__author__ = "NeuroCity Team"
__email__ = "contact@neurocity.org"

__all__ = [
    "GlobalHashConfig", "MultiResolutionHashEncoding", "GlobalFeatureDatabase", "GlobalFeatureLibrary", "SDFDataset", "GlobalFeatureDataset", "GFVTrainer", "GFVLightningModule"
] 