"""
SVRaster One 测试套件

提供全面的测试用例，覆盖所有核心组件：
- 配置系统
- 体素网格
- 渲染器
- 损失函数
- 训练器
- 端到端集成测试
"""

from .test_config import TestSVRasterOneConfig
from .test_voxels import TestSparseVoxelGrid, TestMortonCode
from .test_renderer import TestDifferentiableVoxelRasterizer
from .test_losses import TestSVRasterOneLoss
from .test_trainer import TestSVRasterOneTrainer
from .test_core import TestSVRasterOne
from .test_integration import TestSVRasterOneIntegration
from .test_cuda import TestSVRasterOneCUDA

__all__ = [
    "TestSVRasterOneConfig",
    "TestSparseVoxelGrid", 
    "TestMortonCode",
    "TestDifferentiableVoxelRasterizer",
    "TestSVRasterOneLoss",
    "TestSVRasterOneTrainer",
    "TestSVRasterOne",
    "TestSVRasterOneIntegration",
    "TestSVRasterOneCUDA",
] 