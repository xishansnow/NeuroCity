"""
SVRaster One 测试配置

提供测试夹具和配置。
"""

import pytest
import torch
import tempfile
from pathlib import Path

from ..config import SVRasterOneConfig
from ..core import SVRasterOne


@pytest.fixture
def basic_config():
    """基本配置夹具"""
    config = SVRasterOneConfig()
    config.voxel.grid_resolution = 64
    config.voxel.max_voxels = 1000
    config.rendering.image_width = 64
    config.rendering.image_height = 64
    config.training.batch_size = 4
    config.training.use_amp = False
    return config


@pytest.fixture
def small_config():
    """小规模配置夹具"""
    config = SVRasterOneConfig()
    config.voxel.grid_resolution = 32
    config.voxel.max_voxels = 100
    config.rendering.image_width = 32
    config.rendering.image_height = 32
    config.training.batch_size = 2
    config.training.use_amp = False
    return config


@pytest.fixture
def cuda_config():
    """CUDA 配置夹具"""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    config = SVRasterOneConfig()
    config.device = "cuda"
    config.voxel.grid_resolution = 64
    config.voxel.max_voxels = 1000
    config.rendering.image_width = 64
    config.rendering.image_height = 64
    config.training.batch_size = 4
    config.training.use_amp = True
    return config


@pytest.fixture
def basic_model(basic_config):
    """基本模型夹具"""
    return SVRasterOne(basic_config)


@pytest.fixture
def small_model(small_config):
    """小规模模型夹具"""
    return SVRasterOne(small_config)


@pytest.fixture
def cuda_model(cuda_config):
    """CUDA 模型夹具"""
    return SVRasterOne(cuda_config)


@pytest.fixture
def temp_dir():
    """临时目录夹具"""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def mock_camera_data():
    """模拟相机数据夹具"""
    camera_matrix = torch.eye(4)
    camera_matrix[2, 3] = 2.0  # 相机在 z=2 位置
    
    intrinsics = torch.tensor([
        [500, 0, 32],
        [0, 500, 32],
        [0, 0, 1]
    ], dtype=torch.float32)
    
    return {
        "camera_matrix": camera_matrix,
        "intrinsics": intrinsics,
    }


@pytest.fixture
def mock_target_data():
    """模拟目标数据夹具"""
    batch_size = 2
    height, width = 32, 32
    
    return {
        "rgb": torch.rand(batch_size, height, width, 3),
        "depth": torch.rand(batch_size, height, width),
    }


@pytest.fixture
def mock_voxel_data():
    """模拟体素数据夹具"""
    num_voxels = 50
    
    return {
        "positions": torch.randn(num_voxels, 3),
        "sizes": torch.ones(num_voxels) * 0.1,
        "densities": torch.rand(num_voxels),
        "colors": torch.rand(num_voxels, 3),
    }


def pytest_configure(config):
    """pytest 配置"""
    # 添加自定义标记
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "cuda: marks tests as requiring CUDA"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )


def pytest_collection_modifyitems(config, items):
    """修改测试收集"""
    # 为 CUDA 测试添加标记
    for item in items:
        if "cuda" in item.nodeid.lower():
            item.add_marker(pytest.mark.cuda)
        
        if "integration" in item.nodeid.lower():
            item.add_marker(pytest.mark.integration)
        
        # 标记长时间运行的测试
        if any(keyword in item.nodeid.lower() for keyword in ["benchmark", "performance", "full"]):
            item.add_marker(pytest.mark.slow) 