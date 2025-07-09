"""
Test Instant NGP Core Components

This module tests the core components of Instant NGP:
- InstantNGP model
- InstantNGPConfig
- Hash encoding
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import tempfile
import json
import os
import sys
from pathlib import Path

from dataclasses import dataclass, asdict

# Add the src directory to the path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))

try:
    from nerfs.instant_ngp import InstantNGP
    from nerfs.instant_ngp.config import InstantNGPConfig

    INSTANT_NGP_AVAILABLE = True
except ImportError as e:
    INSTANT_NGP_AVAILABLE = False
    IMPORT_ERROR = str(e)

from nerfs.instant_ngp.core import (
    InstantNGPModel,
    InstantNGPLoss,
    InstantNGPRenderer,
)


class TestInstantNGPCore:
    """Instant NGP 核心组件测试类"""

    def setup_method(self):
        """每个测试方法前的设置"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = InstantNGPConfig(
            num_levels=8,
            base_resolution=16,
            finest_resolution=256,
            hidden_dim=64,
            num_layers=2,
            batch_size=1024,
        )

    def test_config_initialization(self):
        """测试配置初始化"""
        config = InstantNGPConfig()

        # 测试默认值
        assert config.num_levels == 16
        assert config.base_resolution == 16
        assert config.finest_resolution == 512
        assert config.hidden_dim == 64
        assert config.use_amp is True

        # 测试 Python 3.10 兼容性 - 使用内置容器
        config_dict: dict[str, Any] = asdict(config)
        assert isinstance(config_dict, dict)
        assert "num_levels" in config_dict
        assert "hidden_dim" in config_dict

    def test_config_validation(self):
        """测试配置验证"""
        # 测试有效配置
        config = InstantNGPConfig(num_levels=16, base_resolution=16, finest_resolution=512)
        # 应该不抛出异常
        config.__post_init__()

        # 测试无效配置应该抛出异常
        with pytest.raises(AssertionError):
            config = InstantNGPConfig(num_levels=0)
            config.__post_init__()

        with pytest.raises(AssertionError):
            config = InstantNGPConfig(base_resolution=512, finest_resolution=256)
            config.__post_init__()

    def test_config_device_setup(self):
        """测试设备配置"""
        config = InstantNGPConfig()
        config.__post_init__()

        # 检查设备设置
        expected_device = "cuda" if torch.cuda.is_available() else "cpu"
        assert str(config.device).startswith(expected_device)

    def test_model_initialization(self):
        """测试模型初始化"""
        model = InstantNGPModel(self.config)

        # 检查模型组件
        assert hasattr(model, "encoding")
        assert hasattr(model, "direction_encoder")
        assert hasattr(model, "sigma_net")
        assert hasattr(model, "color_net")

        # 检查参数数量
        total_params = sum(p.numel() for p in model.parameters())
        assert total_params > 0

    def test_model_forward_pass(self):
        """测试模型前向传播"""
        model = InstantNGPModel(self.config).to(self.device)

        # 创建测试输入 - 使用 Python 3.10 兼容的类型注解
        batch_size = 1000
        positions: torch.Tensor = torch.randn(batch_size, 3, device=self.device)
        directions: torch.Tensor = torch.randn(batch_size, 3, device=self.device)

        # 前向传播
        with torch.no_grad():
            output: dict[str, torch.Tensor] = model(positions, directions)

        # 检查输出
        assert isinstance(output, dict)
        assert "density" in output
        assert "color" in output

        # 检查输出形状
        assert output["density"].shape == (batch_size, 1)
        assert output["color"].shape == (batch_size, 3)

        # 检查输出值范围
        assert torch.all(output["density"] >= 0)  # 密度应该非负
        assert torch.all(output["color"] >= 0) and torch.all(output["color"] <= 1)  # 颜色在[0,1]

    def test_model_encoding_dimensions(self):
        """测试编码维度"""
        model = InstantNGPModel(self.config)

        # 检查哈希编码输出维度
        expected_hash_dim = self.config.num_levels * self.config.feature_dim
        assert model.encoding.output_dim == expected_hash_dim

        # 检查方向编码输出维度
        assert model.direction_encoder.output_dim > 0

    def test_loss_function(self):
        """测试损失函数"""
        loss_fn = InstantNGPLoss()

        # 创建测试数据
        batch_size = 100
        pred_colors: torch.Tensor = torch.rand(batch_size, 3, device=self.device)
        gt_colors: torch.Tensor = torch.rand(batch_size, 3, device=self.device)
        pred_density: torch.Tensor = torch.rand(batch_size, 1, device=self.device)

        # 计算损失 - 使用 Python 3.10 兼容的字典类型
        outputs: dict[str, torch.Tensor] = {"color": pred_colors, "density": pred_density}

        targets: dict[str, torch.Tensor] = {"color": gt_colors}

        loss_info: dict[str, torch.Tensor] = loss_fn(outputs, targets)

        # 检查损失
        assert isinstance(loss_info, dict)
        assert "total_loss" in loss_info
        assert "color_loss" in loss_info

        # 检查损失值
        assert loss_info["total_loss"].item() >= 0
        assert loss_info["color_loss"].item() >= 0

    def test_legacy_renderer(self):
        """测试遗留渲染器（从 core 模块）"""
        model = InstantNGPModel(self.config).to(self.device)
        renderer = InstantNGPRenderer(model)

        # 创建测试光线
        batch_size = 500
        ray_origins: torch.Tensor = torch.randn(batch_size, 3, device=self.device)
        ray_directions: torch.Tensor = torch.randn(batch_size, 3, device=self.device)
        ray_directions = torch.nn.functional.normalize(ray_directions, dim=-1)

        # 渲染
        with torch.no_grad():
            outputs: dict[str, torch.Tensor] = renderer.render_rays(ray_origins, ray_directions)

        # 检查输出
        assert isinstance(outputs, dict)
        assert "color" in outputs
        assert "depth" in outputs
        assert "weights" in outputs

        # 检查输出形状
        assert outputs["color"].shape == (batch_size, 3)
        assert outputs["depth"].shape == (batch_size,)

    def test_model_state_dict(self):
        """测试模型状态字典序列化"""
        model = InstantNGPModel(self.config)

        # 获取状态字典
        state_dict: dict[str, torch.Tensor] = model.state_dict()
        assert isinstance(state_dict, dict)

        # 检查关键组件的参数
        hash_params: list[str] = [k for k in state_dict.keys() if "encoding" in k]
        density_params: list[str] = [k for k in state_dict.keys() if "sigma_net" in k]
        color_params: list[str] = [k for k in state_dict.keys() if "color_net" in k]

        assert len(hash_params) > 0
        assert len(density_params) > 0
        assert len(color_params) > 0

    def test_model_save_load(self):
        """测试模型保存和加载"""
        model = InstantNGPModel(self.config)

        # 创建临时文件
        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
            temp_path = f.name

        try:
            # 保存模型
            torch.save(model.state_dict(), temp_path)

            # 创建新模型并加载
            new_model = InstantNGPModel(self.config)
            new_model.load_state_dict(torch.load(temp_path, map_location="cpu"))

            # 比较参数
            for (name1, param1), (name2, param2) in zip(
                model.named_parameters(), new_model.named_parameters()
            ):
                assert name1 == name2
                assert torch.allclose(param1, param2)

        finally:
            # 清理临时文件
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_mixed_precision_compatibility(self):
        """测试混合精度训练兼容性"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available for mixed precision testing")

        model = InstantNGPModel(self.config).to(self.device)

        # 使用 autocast 测试
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            positions = torch.randn(100, 3, device=self.device)
            directions = torch.randn(100, 3, device=self.device)

            output: dict[str, torch.Tensor] = model(positions, directions)

            # 检查输出类型（可能是 float16）
            assert output["density"].dtype in [torch.float16, torch.float32]
            assert output["color"].dtype in [torch.float16, torch.float32]

    def test_batch_processing(self):
        """测试批处理能力"""
        model = InstantNGPModel(self.config).to(self.device)

        # 测试不同批次大小
        batch_sizes: list[int] = [1, 10, 100, 1000]

        for batch_size in batch_sizes:
            positions = torch.randn(batch_size, 3, device=self.device)
            directions = torch.randn(batch_size, 3, device=self.device)

            with torch.no_grad():
                output = model(positions, directions)

            assert output["density"].shape[0] == batch_size
            assert output["color"].shape[0] == batch_size

    def test_gradient_computation(self):
        """测试梯度计算"""
        model = InstantNGPModel(self.config).to(self.device)
        loss_fn = InstantNGPLoss()

        # 创建需要梯度的输入
        positions = torch.randn(100, 3, device=self.device)
        directions = torch.randn(100, 3, device=self.device)
        gt_colors = torch.rand(100, 3, device=self.device)

        # 前向传播
        outputs = model(positions, directions)

        # 计算损失
        targets = {"color": gt_colors}
        loss_info = loss_fn(outputs, targets)

        # 反向传播
        loss_info["total_loss"].backward()

        # 检查梯度
        has_gradients = False
        for param in model.parameters():
            if param.grad is not None:
                has_gradients = True
                assert not torch.isnan(param.grad).any()
                break

        assert has_gradients, "模型应该有梯度"

    def teardown_method(self):
        """每个测试方法后的清理"""
        # 清理 GPU 内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    pytest.main([__file__])
