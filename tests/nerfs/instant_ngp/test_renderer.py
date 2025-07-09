"""
Instant NGP 渲染器测试

测试 InstantNGPInferenceRenderer 和 InstantNGPRendererConfig 的功能，
确保与 Python 3.10 的兼容性，使用内置容器类型。
"""

from __future__ import annotations

import pytest
import torch
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))

from nerfs.instant_ngp.renderer import InstantNGPRenderer
from nerfs.instant_ngp.core import InstantNGPConfig, InstantNGPModel

class TestInstantNGPRenderer:
    def setup_method(self):
        """设置测试环境"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = InstantNGPConfig()
        self.model = InstantNGPModel(self.config).to(self.device)
        self.renderer = InstantNGPRenderer(self.config)

    def test_renderer_initialization(self):
        """测试渲染器初始化"""
        renderer = InstantNGPRenderer(self.config)
        
        # 检查基本属性
        assert hasattr(renderer, 'config')
        assert hasattr(renderer, 'render_rays')
        assert hasattr(renderer, 'render_image')

    def test_rays_generation(self):
        """测试光线生成"""
        renderer = InstantNGPRenderer(self.config)
        
        H, W = 32, 32
        focal = 100.0
        c2w = torch.eye(4, device=self.device)
        
        # 生成光线
        rays_o, rays_d = renderer._generate_rays(H, W, focal, c2w)
        
        # 验证输出
        assert rays_o.shape == (H * W, 3)
        assert rays_d.shape == (H * W, 3)
        
        # 检查光线方向是否归一化
        rays_d_norm = torch.norm(rays_d, dim=-1)
        assert torch.allclose(rays_d_norm, torch.ones_like(rays_d_norm), atol=1e-6)

    def test_basic_rendering(self):
        """测试基本渲染功能"""
        renderer = InstantNGPRenderer(self.config)
        
        # 创建简单的光线
        batch_size = 100
        rays_o = torch.rand(batch_size, 3, device=self.device)
        rays_d = torch.rand(batch_size, 3, device=self.device)
        rays_d = torch.nn.functional.normalize(rays_d, dim=-1)
        
        # 渲染
        with torch.no_grad():
            results = renderer.render_rays(self.model, rays_o, rays_d)
        
        # 验证结果
        assert 'rgb' in results
        assert 'depth' in results
        assert results['rgb'].shape == (batch_size, 3)
        assert results['depth'].shape == (batch_size, 1)

    def teardown_method(self):
        """清理"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    pytest.main([__file__])
