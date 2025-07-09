"""
SVRaster One 配置系统测试

测试配置类的初始化、验证、序列化等功能。
"""

import pytest
import torch
import tempfile
import json
from pathlib import Path

from ..config import (
    SVRasterOneConfig,
    VoxelConfig,
    RenderingConfig,
    TrainingConfig,
    CUDAConfig,
)


class TestSVRasterOneConfig:
    """测试 SVRaster One 配置类"""

    def test_config_init_default(self):
        """测试默认配置初始化"""
        config = SVRasterOneConfig()
        
        # 检查设备自动选择
        assert config.device in ["cuda", "cpu"]
        
        # 检查子配置存在
        assert isinstance(config.voxel, VoxelConfig)
        assert isinstance(config.rendering, RenderingConfig)
        assert isinstance(config.training, TrainingConfig)
        assert isinstance(config.cuda, CUDAConfig)

    def test_config_init_custom(self):
        """测试自定义配置初始化"""
        config = SVRasterOneConfig()
        config.device = "cuda"
        config.voxel.grid_resolution = 512
        config.rendering.image_width = 1024
        config.rendering.image_height = 768
        config.training.learning_rate = 1e-4
        
        assert config.device == "cuda"
        assert config.voxel.grid_resolution == 512
        assert config.rendering.image_width == 1024
        assert config.rendering.image_height == 768
        assert config.training.learning_rate == 1e-4

    def test_config_validation(self):
        """测试配置验证"""
        config = SVRasterOneConfig()
        
        # 测试无效的网格分辨率
        with pytest.raises(ValueError):
            config.voxel.grid_resolution = -1
            config.__post_init__()
        
        # 测试无效的图像尺寸
        with pytest.raises(ValueError):
            config.rendering.image_width = 0
            config.__post_init__()
        
        # 测试无效的学习率
        with pytest.raises(ValueError):
            config.training.learning_rate = 0
            config.__post_init__()

    def test_config_to_dict(self):
        """测试配置序列化为字典"""
        config = SVRasterOneConfig()
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert "voxel" in config_dict
        assert "rendering" in config_dict
        assert "training" in config_dict
        assert "cuda" in config_dict
        assert "device" in config_dict
        
        # 检查子配置
        assert config_dict["voxel"]["grid_resolution"] == config.voxel.grid_resolution
        assert config_dict["rendering"]["image_width"] == config.rendering.image_width
        assert config_dict["training"]["learning_rate"] == config.training.learning_rate

    def test_config_from_dict(self):
        """测试从字典创建配置"""
        original_config = SVRasterOneConfig()
        original_config.voxel.grid_resolution = 512
        original_config.rendering.image_width = 1024
        original_config.training.learning_rate = 1e-4
        
        config_dict = original_config.to_dict()
        new_config = SVRasterOneConfig.from_dict(config_dict)
        
        assert new_config.voxel.grid_resolution == 512
        assert new_config.rendering.image_width == 1024
        assert new_config.training.learning_rate == 1e-4

    def test_config_save_load(self):
        """测试配置保存和加载"""
        config = SVRasterOneConfig()
        config.voxel.grid_resolution = 512
        config.rendering.image_width = 1024
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config_path = f.name
        
        try:
            # 保存配置
            with open(config_path, 'w') as f:
                json.dump(config.to_dict(), f, indent=2)
            
            # 加载配置
            with open(config_path, 'r') as f:
                loaded_dict = json.load(f)
            
            loaded_config = SVRasterOneConfig.from_dict(loaded_dict)
            
            assert loaded_config.voxel.grid_resolution == 512
            assert loaded_config.rendering.image_width == 1024
            
        finally:
            Path(config_path).unlink(missing_ok=True)

    def test_voxel_config(self):
        """测试体素配置"""
        voxel_config = VoxelConfig()
        
        # 测试默认值
        assert voxel_config.grid_resolution == 256
        assert voxel_config.voxel_size == 0.01
        assert voxel_config.max_voxels == 1000000
        assert voxel_config.sparsity_threshold == 0.01
        assert voxel_config.adaptive_subdivision is True
        assert voxel_config.use_morton_ordering is True

    def test_rendering_config(self):
        """测试渲染配置"""
        rendering_config = RenderingConfig()
        
        # 测试默认值
        assert rendering_config.image_width == 800
        assert rendering_config.image_height == 600
        assert rendering_config.background_color == (0.0, 0.0, 0.0)
        assert rendering_config.soft_rasterization is True
        assert rendering_config.temperature == 0.1
        assert rendering_config.sigma == 1.0
        assert rendering_config.depth_sorting == "back_to_front"
        assert rendering_config.use_soft_sorting is True
        assert rendering_config.alpha_blending is True

    def test_training_config(self):
        """测试训练配置"""
        training_config = TrainingConfig()
        
        # 测试默认值
        assert training_config.rgb_loss_weight == 1.0
        assert training_config.depth_loss_weight == 0.1
        assert training_config.density_reg_weight == 0.01
        assert training_config.sparsity_weight == 0.001
        assert training_config.learning_rate == 1e-3
        assert training_config.weight_decay == 1e-4
        assert training_config.batch_size == 4096
        assert training_config.num_epochs == 1000
        assert training_config.use_amp is True
        assert training_config.grad_clip == 1.0

    def test_cuda_config(self):
        """测试 CUDA 配置"""
        cuda_config = CUDAConfig()
        
        # 测试默认值
        assert cuda_config.block_size == 256
        assert cuda_config.max_blocks == 65535
        assert cuda_config.memory_pool_size == 1024 * 1024 * 1024
        assert cuda_config.use_memory_pool is True
        assert cuda_config.enable_profiling is False
        assert cuda_config.sync_cuda is False

    def test_device_selection(self):
        """测试设备选择逻辑"""
        # 测试自动选择
        config = SVRasterOneConfig()
        config.device = "auto"
        config.__post_init__()
        
        if torch.cuda.is_available():
            assert config.device == "cuda"
        else:
            assert config.device == "cpu"
        
        # 测试强制选择
        config.device = "cpu"
        config.__post_init__()
        assert config.device == "cpu"
        
        if torch.cuda.is_available():
            config.device = "cuda"
            config.__post_init__()
            assert config.device == "cuda"


def test_config_edge_cases():
    """测试配置边界情况"""
    config = SVRasterOneConfig()
    
    # 测试极值
    config.voxel.grid_resolution = 1
    config.rendering.image_width = 1
    config.rendering.image_height = 1
    config.training.learning_rate = 1e-10
    
    # 应该不抛出异常
    config.__post_init__()
    
    assert config.voxel.grid_resolution == 1
    assert config.rendering.image_width == 1
    assert config.rendering.image_height == 1
    assert config.training.learning_rate == 1e-10


if __name__ == "__main__":
    pytest.main([__file__]) 