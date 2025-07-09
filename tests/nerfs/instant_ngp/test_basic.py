from __future__ import annotations

"""
Instant NGP 基础测试

只测试核心功能的基本实现，确保代码能正常运行。
"""

import pytest
import torch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))

from nerfs.instant_ngp.core import InstantNGPConfig, InstantNGPModel


class TestInstantNGPBasic:
    """基础功能测试类"""

    def setup_method(self):
        """测试设置"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = InstantNGPConfig(
            num_levels=4,  # 减少层数以提高测试速度
            base_resolution=16,
            finest_resolution=64,
            feature_dim=2,
            log2_hashmap_size=15,  # 减少哈希表大小
            num_samples=32,  # 减少采样数量
        )

    def test_config_creation(self):
        """测试配置创建"""
        config = InstantNGPConfig()
        assert config.num_levels > 0
        assert config.base_resolution > 0
        assert config.finest_resolution > config.base_resolution

    def test_model_creation(self):
        """测试模型创建"""
        model = InstantNGPModel(self.config)
        assert model is not None
        assert hasattr(model, 'encoding')
        assert hasattr(model, 'sigma_net')
        assert hasattr(model, 'color_net')

    def test_model_forward(self):
        """测试模型前向传播"""
        model = InstantNGPModel(self.config).to(self.device)
        
        # 创建测试输入
        batch_size = 10  # 小批次
        rays_o = torch.rand(batch_size, 3, device=self.device)
        rays_d = torch.rand(batch_size, 3, device=self.device)
        rays_d = torch.nn.functional.normalize(rays_d, dim=-1)
        
        # 前向传播
        with torch.no_grad():
            rgb, density = model(rays_o, rays_d)
        
        # 基本检查
        assert rgb.shape == (batch_size, 3)
        assert density.shape[0] > 0  # density应该有输出
        assert not torch.isnan(rgb).any()
        assert not torch.isnan(density).any()

    def test_encoding_forward(self):
        """测试编码器前向传播"""
        model = InstantNGPModel(self.config).to(self.device)
        
        # 测试位置编码
        positions = torch.rand(100, 3, device=self.device) * 2 - 1  # [-1, 1]
        
        with torch.no_grad():
            features = model.encoding(positions)
        
        assert features.shape[0] == 100
        assert features.shape[1] == model.encoding.output_dim
        assert not torch.isnan(features).any()

    def teardown_method(self):
        """清理"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    pytest.main([__file__])
