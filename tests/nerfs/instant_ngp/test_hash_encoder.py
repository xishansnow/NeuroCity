from __future__ import annotations

"""
Test Instant NGP Hash Encoder

This module tests the hash encoding functionality of Instant NGP:
- Hash encoding computation
- Hash table operations
- Encoding dimensions
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import sys
import os

# Add the src directory to the path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))

try:
    from nerfs.instant_ngp import InstantNGP
    from nerfs.instant_ngp.hash_encoder import HashEncoder
    from nerfs.instant_ngp.core import (
        InstantNGPConfig,
        SHEncoder,
    )

    INSTANT_NGP_AVAILABLE = True
except ImportError as e:
    INSTANT_NGP_AVAILABLE = False
    IMPORT_ERROR = str(e)


class TestHashEncoder:
    """哈希编码器测试类"""

    def setup_method(self):
        """每个测试方法前的设置"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = InstantNGPConfig(
            num_levels=8,
            base_resolution=16,
            finest_resolution=256,
            feature_dim=2,
            log2_hashmap_size=19,
        )

    def test_hash_encoder_initialization(self):
        """测试哈希编码器初始化"""
        encoder = HashEncoder(self.config)

        # 检查基本属性
        assert encoder.num_levels == self.config.num_levels
        assert encoder.level_dim == self.config.feature_dim
        # filepath: tests/nerfs/instant_ngp/test_core.py
        """
        Instant NGP 核心组件测试
        测试核心模型组件、配置和损失函数
        """

        import torch.nn as nn

        sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))

        from nerfs.instant_ngp.core import (
            InstantNGPConfig,
            InstantNGPModel,
            InstantNGPLoss,
            DensityField,
            RadianceField,
        )

        class TestInstantNGPCore:
            """核心组件测试类"""

            def setup_method(self):
                """每个测试方法前的设置"""
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self.config = InstantNGPConfig(
                    num_levels=8,
                    base_resolution=16,
                    finest_resolution=256,
                    feature_dim=2,
                    log2_hashmap_size=19,
                )

            def test_model_initialization(self):
                """测试模型初始化"""
                model = InstantNGPModel(self.config).to(self.device)

                # 检查模型组件
                assert hasattr(model, "density_field")
                assert hasattr(model, "radiance_field")
                assert isinstance(model.density_field, DensityField)
                assert isinstance(model.radiance_field, RadianceField)

            def test_density_field(self):
                """测试密度场"""
                model = InstantNGPModel(self.config).to(self.device)

                # 创建测试输入
                batch_size = 1000
                positions = torch.randn(batch_size, 3, device=self.device)
                positions = torch.tanh(positions)  # 缩放到 [-1, 1]

                # 前向传播
                with torch.no_grad():
                    density = model.density_field(positions)

                # 检查输出
                assert density.shape == (batch_size, 1)
                assert torch.all(density >= 0)  # 密度应该非负

            def test_radiance_field(self):
                """测试辐射场"""
                model = InstantNGPModel(self.config).to(self.device)

                # 创建测试输入
                batch_size = 1000
                positions = torch.randn(batch_size, 3, device=self.device)
                positions = torch.tanh(positions)
                directions = torch.randn(batch_size, 3, device=self.device)
                directions = nn.functional.normalize(directions, dim=-1)

                # 前向传播
                with torch.no_grad():
                    radiance = model.radiance_field(positions, directions)

                # 检查输出
                assert radiance.shape == (batch_size, 3)  # RGB 输出
                assert torch.all(radiance >= 0) and torch.all(radiance <= 1)  # 颜色范围

            def test_model_forward_pass(self):
                """测试模型前向传播"""
                model = InstantNGPModel(self.config).to(self.device)

                # 创建测试输入
                batch_size = 1000
                rays_o = torch.randn(batch_size, 3, device=self.device)
                rays_d = torch.randn(batch_size, 3, device=self.device)
                rays_d = nn.functional.normalize(rays_d, dim=-1)

                # 前向传播
                with torch.no_grad():
                    rgb, depth, weights = model(rays_o, rays_d)

                # 检查输出
                assert rgb.shape == (batch_size, 3)
                assert depth.shape == (batch_size, 1)
                assert weights.shape[0] == batch_size

                # 检查有效范围
                assert torch.all(rgb >= 0) and torch.all(rgb <= 1)
                assert torch.all(depth >= 0)
                assert torch.all(weights >= 0) and torch.all(weights <= 1)

            def test_loss_computation(self):
                """测试损失计算"""
                criterion = InstantNGPLoss()

                # 创建模拟预测和目标
                batch_size = 100
                pred_rgb = torch.rand(batch_size, 3, device=self.device)
                target_rgb = torch.rand(batch_size, 3, device=self.device)

                # 计算损失
                loss = criterion(pred_rgb, target_rgb)

                # 检查损失
                assert isinstance(loss, torch.Tensor)
                assert loss.ndim == 0  # 标量损失
                assert loss >= 0  # 损失应该非负

            def test_model_gradient_computation(self):
                """测试模型梯度计算"""
                model = InstantNGPModel(self.config).to(self.device)
                criterion = InstantNGPLoss()

                # 创建测试数据
                rays_o = torch.randn(100, 3, device=self.device, requires_grad=True)
                rays_d = torch.randn(100, 3, device=self.device)
                rays_d = nn.functional.normalize(rays_d, dim=-1)
                target_rgb = torch.rand(100, 3, device=self.device)

                # 前向传播
                rgb, depth, weights = model(rays_o, rays_d)
                loss = criterion(rgb, target_rgb)

                # 反向传播
                loss.backward()

                # 检查梯度
                for param in model.parameters():
                    assert param.grad is not None
                    assert not torch.isnan(param.grad).any()
                    assert not torch.isinf(param.grad).any()

            def test_model_state_dict(self):
                """测试模型状态字典保存和加载"""
                model = InstantNGPModel(self.config).to(self.device)

                # 保存状态字典
                state_dict = model.state_dict()

                # 创建新模型并加载状态字典
                new_model = InstantNGPModel(self.config).to(self.device)
                new_model.load_state_dict(state_dict)

                # 比较两个模型的输出
                with torch.no_grad():
                    rays_o = torch.randn(100, 3, device=self.device)
                    rays_d = torch.randn(100, 3, device=self.device)
                    rays_d = nn.functional.normalize(rays_d, dim=-1)

                    rgb1, depth1, weights1 = model(rays_o, rays_d)
                    rgb2, depth2, weights2 = new_model(rays_o, rays_d)

                assert torch.allclose(rgb1, rgb2, atol=1e-6)
                assert torch.allclose(depth1, depth2, atol=1e-6)
                assert torch.allclose(weights1, weights2, atol=1e-6)

            def test_model_config_validation(self):
                """测试模型配置验证"""
                # 测试无效配置
                with pytest.raises(ValueError):
                    invalid_config = InstantNGPConfig(num_levels=-1)
                    InstantNGPModel(invalid_config)

                with pytest.raises(ValueError):
                    invalid_config = InstantNGPConfig(base_resolution=0)
                    InstantNGPModel(invalid_config)

            def test_model_device_transfer(self):
                """测试模型设备转移"""
                model = InstantNGPModel(self.config)

                # 转移到 GPU (如果可用)
                if torch.cuda.is_available():
                    model = model.cuda()
                    assert next(model.parameters()).is_cuda

                # 转移回 CPU
                model = model.cpu()
                assert not next(model.parameters()).is_cuda

            def teardown_method(self):
                """每个测试方法后的清理"""
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        if __name__ == "__main__":
            pytest.main([__file__])

        # 创建测试坐标
        coords: torch.Tensor = torch.tensor(
            [
                [0, 0, 0],
                [1, 1, 1],
                [15, 15, 15],  # 在分辨率范围内
            ],
            dtype=torch.long,
            device=self.device,
        )

        resolution = 16

        # 计算哈希值
        hash_values: torch.Tensor = encoder.hash_function(coords, resolution)

        # 检查输出
        assert hash_values.shape == (3,)
        assert hash_values.dtype == torch.long

        # 检查哈希值在有效范围内
        max_hash = encoder.embeddings[0].num_embeddings
        assert torch.all(hash_values >= 0)
        assert torch.all(hash_values < max_hash)

    def test_grid_coordinates(self):
        """测试网格坐标计算"""
        encoder = HashEncoder(self.config)

        # 创建测试位置
        positions: torch.Tensor = torch.tensor(
            [
                [-1.0, -1.0, -1.0],  # 最小值
                [1.0, 1.0, 1.0],  # 最大值
                [0.0, 0.0, 0.0],  # 中心值
            ],
            device=self.device,
        )

        resolution = 16

        # 获取网格坐标和权重
        grid_coords, weights = encoder.get_grid_coordinates(positions, resolution)

        # 检查形状
        assert grid_coords.shape == (3, 3)
        assert weights.shape == (3, 3)

        # 检查坐标范围
        assert torch.all(grid_coords >= 0)
        assert torch.all(grid_coords < resolution)

        # 检查权重范围
        assert torch.all(weights >= 0)
        assert torch.all(weights <= 1)

    def test_trilinear_interpolation(self):
        """测试三线性插值"""
        encoder = HashEncoder(self.config)

        # 创建测试特征（8个角点）
        batch_size = 10
        feature_dim = 4
        corner_features: torch.Tensor = torch.randn(batch_size, 8, feature_dim, device=self.device)

        # 创建测试权重
        weights: torch.Tensor = torch.rand(batch_size, 3, device=self.device)

        # 执行三线性插值
        interpolated: torch.Tensor = encoder.trilinear_interpolation(corner_features, weights)

        # 检查输出形状
        assert interpolated.shape == (batch_size, feature_dim)

        # 检查边界情况
        # 当权重为 [0, 0, 0] 时，应该返回第一个角点
        weights_zero = torch.zeros(1, 3, device=self.device)
        corner_features_test = torch.randn(1, 8, feature_dim, device=self.device)

        result_zero = encoder.trilinear_interpolation(corner_features_test, weights_zero)
        expected_zero = corner_features_test[0, 0]  # 第一个角点

        assert torch.allclose(result_zero[0], expected_zero, atol=1e-6)

    def test_encoder_levels_consistency(self):
        """测试编码器各层级的一致性"""
        encoder = HashEncoder(self.config).to(self.device)

        # 创建测试位置
        positions: torch.Tensor = torch.randn(100, 3, device=self.device)
        positions = torch.tanh(positions)  # 缩放到 [-1, 1]

        # 测试不同层级
        for level_idx in range(encoder.num_levels):
            resolution = encoder.resolutions[level_idx]

            # 获取该层级的网格坐标
            grid_coords, weights = encoder.get_grid_coordinates(positions, resolution)

            # 检查坐标有效性
            assert torch.all(grid_coords >= 0)
            assert torch.all(grid_coords < resolution)

            # 检查权重有效性
            assert torch.all(weights >= 0)
            assert torch.all(weights <= 1)

    def test_encoder_deterministic(self):
        """测试编码器的确定性"""
        encoder = HashEncoder(self.config).to(self.device)

        # 创建测试位置
        positions: torch.Tensor = torch.randn(50, 3, device=self.device)
        positions = torch.tanh(positions)

        # 多次前向传播
        with torch.no_grad():
            output1 = encoder(positions)
            output2 = encoder(positions)

        # 检查结果一致性
        assert torch.allclose(output1, output2, atol=1e-6)

    def test_encoder_different_batch_sizes(self):
        """测试编码器处理不同批次大小"""
        encoder = HashEncoder(self.config).to(self.device)

        # 测试不同批次大小
        batch_sizes: list[int] = [1, 10, 100, 1000]

        for batch_size in batch_sizes:
            positions = torch.randn(batch_size, 3, device=self.device)
            positions = torch.tanh(positions)

            with torch.no_grad():
                output = encoder(positions)

            expected_shape = (batch_size, encoder.output_dim)
            assert output.shape == expected_shape

    def test_encoder_gradient_computation(self):
        """测试编码器梯度计算"""
        encoder = HashEncoder(self.config).to(self.device)

        # 创建需要梯度的输入
        positions = torch.randn(100, 3, device=self.device, requires_grad=True)
        positions = torch.tanh(positions)

        # 前向传播
        output = encoder(positions)

        # 计算损失（简单求和）
        loss = output.sum()

        # 反向传播
        loss.backward()

        # 检查梯度
        assert positions.grad is not None
        assert not torch.isnan(positions.grad).any()
        assert not torch.isinf(positions.grad).any()

    def test_encoder_memory_efficiency(self):
        """测试编码器内存效率"""
        encoder = HashEncoder(self.config).to(self.device)

        # 获取初始内存使用
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            initial_memory = torch.cuda.memory_allocated()

        # 处理大批次
        large_batch_size = 10000
        positions = torch.randn(large_batch_size, 3, device=self.device)
        positions = torch.tanh(positions)

        with torch.no_grad():
            output = encoder(positions)

        # 检查输出
        assert output.shape[0] == large_batch_size

        # 清理内存
        del positions, output
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class TestSHEncoder:
    """球谐编码器测试类"""

    def setup_method(self):
        """每个测试方法前的设置"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_sh_encoder_initialization(self):
        """测试球谐编码器初始化"""
        for degree in [2, 3, 4]:
            encoder = SHEncoder(degree=degree)

            # 检查属性
            assert encoder.degree == degree
            assert encoder.output_dim == degree**2

    def test_sh_encoder_forward_pass(self):
        """测试球谐编码器前向传播"""
        encoder = SHEncoder(degree=4).to(self.device)

        # 创建测试方向
        batch_size = 1000
        directions: torch.Tensor = torch.randn(batch_size, 3, device=self.device)

        # 前向传播
        with torch.no_grad():
            encoded_dirs: torch.Tensor = encoder(directions)

        # 检查输出形状
        expected_shape = (batch_size, encoder.output_dim)
        assert encoded_dirs.shape == expected_shape

        # 检查输出不是 NaN 或 Inf
        assert not torch.isnan(encoded_dirs).any()
        assert not torch.isinf(encoded_dirs).any()

    def test_sh_encoder_normalization(self):
        """测试球谐编码器的方向归一化"""
        encoder = SHEncoder(degree=4).to(self.device)

        # 创建未归一化的方向
        directions = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],  # 未归一化
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            device=self.device,
        )

        with torch.no_grad():
            encoded = encoder(directions)

        # 检查第一个和第二个方向（应该相同，因为都指向 x 轴）
        assert torch.allclose(encoded[0], encoded[1], atol=1e-6)

    def test_sh_encoder_different_degrees(self):
        """测试不同阶数的球谐编码器"""
        degrees: list[int] = [1, 2, 3, 4, 5]

        for degree in degrees:
            encoder = SHEncoder(degree=degree).to(self.device)

            # 测试输出维度
            assert encoder.output_dim == degree**2

            # 测试前向传播
            directions = torch.randn(100, 3, device=self.device)
            with torch.no_grad():
                output = encoder(directions)

            assert output.shape == (100, degree**2)

    def test_sh_encoder_gradient_computation(self):
        """测试球谐编码器梯度计算"""
        encoder = SHEncoder(degree=4).to(self.device)

        # 创建需要梯度的输入
        directions = torch.randn(100, 3, device=self.device, requires_grad=True)

        # 前向传播
        output = encoder(directions)

        # 计算损失
        loss = output.sum()

        # 反向传播
        loss.backward()

        # 检查梯度
        assert directions.grad is not None
        assert not torch.isnan(directions.grad).any()
        assert not torch.isinf(directions.grad).any()

    def test_sh_encoder_canonical_directions(self):
        """测试球谐编码器对典型方向的响应"""
        encoder = SHEncoder(degree=4).to(self.device)

        # 典型方向
        canonical_directions: torch.Tensor = torch.tensor(
            [
                [1.0, 0.0, 0.0],  # +X
                [-1.0, 0.0, 0.0],  # -X
                [0.0, 1.0, 0.0],  # +Y
                [0.0, -1.0, 0.0],  # -Y
                [0.0, 0.0, 1.0],  # +Z
                [0.0, 0.0, -1.0],  # -Z
            ],
            device=self.device,
        )

        with torch.no_grad():
            encoded = encoder(canonical_directions)

        # 检查输出形状
        assert encoded.shape == (6, encoder.output_dim)

        # 检查对称性（相对方向的某些球谐系数应该有关系）
        # 这里只检查不是 NaN 或 Inf
        assert not torch.isnan(encoded).any()
        assert not torch.isinf(encoded).any()

    def teardown_method(self):
        """每个测试方法后的清理"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    pytest.main([__file__])
