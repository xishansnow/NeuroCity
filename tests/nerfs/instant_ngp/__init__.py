"""
Instant NGP 测试模块

这个模块包含了对 Instant NGP 实现的全面测试，确保与 Python 3.10 的兼容性。
使用 Python 3.10 的内置容器类型，如 dict、list、tuple 而不是 typing 模块的泛型版本。

测试组件包括：
- 核心模型组件测试
- 哈希编码器测试
- 训练器测试
- 渲染器测试
- 数据集测试
- 工具函数测试
- Python 3.10 兼容性测试
"""

__version__ = "1.0.0"
__author__ = "NeuroCity Team"

# 测试模块导入
from .test_core import *
from .test_hash_encoder import *
from .test_trainer import *
from .test_renderer import *
from .test_dataset import *
from .test_utils import *
from .test_integration import *
from .test_python310_compatibility import *

__all__ = [
    "TestInstantNGPCore",
    "TestHashEncoder", 
    "TestInstantNGPTrainer",
    "TestInstantNGPRenderer",
    "TestInstantNGPDataset",
    "TestInstantNGPUtils",
    "TestInstantNGPIntegration",
    "TestPython310Compatibility",
]
