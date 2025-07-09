"""
Instant NGP 集成测试

测试 Instant NGP 各组件之间的集成和端到端功能，
确保与 Python 3.10 的兼容性，使用内置容器类型。
"""

from __future__ import annotations

import pytest
import torch
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))

from nerfs.instant_ngp.core import InstantNGPConfig, InstantNGPModel
from nerfs.instant_ngp.renderer import InstantNGPRenderer, InstantNGPRendererConfig
from nerfs.instant_ngp.trainer import InstantNGPTrainer
from nerfs.instant_ngp.dataset import InstantNGPDataset


class TestIntegration:
    def setup_method(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = InstantNGPConfig()

    def test_model_initialization(self):
        """测试模型初始化"""
        model = InstantNGPModel(self.config).to(self.device)
        assert model is not None
        assert hasattr(model, 'encoding')
        assert hasattr(model, 'sigma_net')
        assert hasattr(model, 'color_net')

    def test_renderer_initialization(self):
        """测试渲染器初始化"""
        model = InstantNGPModel(self.config).to(self.device)
        renderer_config = InstantNGPRendererConfig()
        renderer = InstantNGPRenderer(model, renderer_config)
        assert renderer is not None
        assert hasattr(renderer, 'render_rays')
        assert hasattr(renderer, 'render_image')

    def test_basic_forward_pass(self):
        """测试基本前向传播"""
        model = InstantNGPModel(self.config).to(self.device)
        
        # 创建测试数据
        batch_size = 100
        rays_o = torch.rand(batch_size, 3, device=self.device)
        rays_d = torch.rand(batch_size, 3, device=self.device)
        rays_d = torch.nn.functional.normalize(rays_d, dim=-1)
        
        # 前向传播
        with torch.no_grad():
            rgb, density = model(rays_o, rays_d)
        
        # 验证输出
        assert rgb.shape == (batch_size, 3)
        # density的形状应该是 [batch_size * num_samples]
        expected_density_size = batch_size * self.config.num_samples
        assert density.shape[0] == expected_density_size

    def test_config_validation(self):
        """测试配置验证"""
        config = InstantNGPConfig()
        
        # 检查基本配置
        assert config.num_levels > 0
        assert config.base_resolution > 0
        assert config.finest_resolution > config.base_resolution
        assert config.feature_dim > 0

    def teardown_method(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def teardown_method(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    pytest.main([__file__])
