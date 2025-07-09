"""
Test Instant NGP Utility Functions

This module tests the utility functions of Instant NGP:
- Coordinate transformations
- Hash encoding utilities
- Geometry utilities
"""

import pytest
import torch
import numpy as np
import tempfile
import os

# Add the src directory to the path for imports
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))

try:
    from nerfs.instant_ngp import InstantNGP
    from nerfs.instant_ngp.utils import coordinate_utils, geometry_utils

    INSTANT_NGP_AVAILABLE = True
except ImportError as e:
    INSTANT_NGP_AVAILABLE = False
    IMPORT_ERROR = str(e)

from nerfs.instant_ngp.utils import (
    contract_to_unisphere,
    uncontract_from_unisphere,
    morton_encode_3d,
    compute_tv_loss,
    adaptive_sampling,
    estimate_normals,
    compute_hash_grid_size,
)


class TestInstantNGPUtils:
    """Instant NGP 工具函数测试类"""

    def setup_method(self):
        """每个测试方法前的设置"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_contract_to_unisphere(self):
        """测试收缩到单位球体"""
        # 创建测试点
        batch_size = 1000
        points: torch.Tensor = torch.randn(batch_size, 3, device=self.device) * 10

        # 收缩到单位球体
        contracted_points = contract_to_unisphere(points)

        # 检查输出形状
        assert contracted_points.shape == points.shape

        # 检查点是否在单位球体内
        distances = torch.norm(contracted_points, dim=-1)
        assert torch.all(distances <= 1.0 + 1e-6)  # 允许数值误差

        # 检查原点附近的点不应该变化太大
        origin_points = torch.zeros(10, 3, device=self.device)
        contracted_origin = contract_to_unisphere(origin_points)
        assert torch.allclose(origin_points, contracted_origin, atol=1e-6)

    def test_uncontract_from_unisphere(self):
        """测试从单位球体还原"""
        # 创建单位球体内的点
        batch_size = 1000
        # 生成球体内的随机点
        directions = torch.randn(batch_size, 3, device=self.device)
        directions = torch.nn.functional.normalize(directions, dim=-1)
        radii = torch.rand(batch_size, 1, device=self.device) ** (1 / 3)  # 均匀分布在球体内
        contracted_points = directions * radii

        # 还原点
        uncontracted_points = uncontract_from_unisphere(contracted_points)

        # 检查输出形状
        assert uncontracted_points.shape == contracted_points.shape

        # 测试收缩和还原的逆操作
        points = torch.randn(100, 3, device=self.device) * 2
        contracted = contract_to_unisphere(points)
        uncontracted = uncontract_from_unisphere(contracted)

        # 应该接近原始点（在合理范围内）
        assert torch.allclose(points, uncontracted, atol=1e-4)

    def test_morton_encode_3d(self):
        """测试 3D Morton 编码"""
        # 创建测试坐标
        coords_list: list[tuple[int, int, int]] = [
            (0, 0, 0),
            (1, 1, 1),
            (2, 3, 4),
            (7, 7, 7),
            (10, 15, 20),
        ]

        coords = torch.tensor(coords_list, dtype=torch.long, device=self.device)

        # 计算 Morton 编码
        morton_codes = morton_encode_3d(coords)

        # 检查输出形状
        assert morton_codes.shape == (len(coords_list),)
        assert morton_codes.dtype == torch.long

        # 检查编码是否非负
        assert torch.all(morton_codes >= 0)

        # 检查相同坐标应该产生相同编码
        same_coords = torch.tensor([[1, 2, 3], [1, 2, 3]], dtype=torch.long, device=self.device)
        same_codes = morton_encode_3d(same_coords)
        assert same_codes[0] == same_codes[1]

        # 检查不同坐标应该产生不同编码
        diff_coords = torch.tensor([[1, 2, 3], [3, 2, 1]], dtype=torch.long, device=self.device)
        diff_codes = morton_encode_3d(diff_coords)
        assert diff_codes[0] != diff_codes[1]

    def test_compute_tv_loss(self):
        """测试总变差损失计算"""
        # 创建测试网格
        grid_size = 16
        grid = torch.randn(1, 1, grid_size, grid_size, grid_size, device=self.device)

        # 计算 TV 损失
        tv_loss = compute_tv_loss(grid)

        # 检查损失值
        assert isinstance(tv_loss, torch.Tensor)
        assert tv_loss.numel() == 1
        assert tv_loss.item() >= 0

        # 测试平滑网格应该有较低的 TV 损失
        smooth_grid = torch.ones(1, 1, grid_size, grid_size, grid_size, device=self.device)
        smooth_tv_loss = compute_tv_loss(smooth_grid)
        assert smooth_tv_loss.item() < 1e-6

        # 测试噪声网格应该有较高的 TV 损失
        noisy_grid = torch.randn(1, 1, grid_size, grid_size, grid_size, device=self.device) * 10
        noisy_tv_loss = compute_tv_loss(noisy_grid)
        assert noisy_tv_loss.item() > smooth_tv_loss.item()

    def test_adaptive_sampling(self):
        """测试自适应采样"""
        # 创建测试权重（采样概率）
        batch_size = 1000
        num_samples = 64
        weights = torch.rand(batch_size, num_samples, device=self.device)
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)  # 归一化

        # 执行自适应采样
        num_new_samples = 32
        new_samples, new_weights = adaptive_sampling(weights, num_new_samples)

        # 检查输出形状
        assert new_samples.shape == (batch_size, num_new_samples)
        assert new_weights.shape == (batch_size, num_new_samples)

        # 检查采样索引范围
        assert torch.all(new_samples >= 0)
        assert torch.all(new_samples < num_samples)

        # 检查权重非负
        assert torch.all(new_weights >= 0)

    def test_estimate_normals(self):
        """测试法向量估计"""
        # 创建测试密度场（球体）
        grid_size = 32
        center = grid_size // 2
        radius = grid_size // 4

        # 创建坐标网格
        x = torch.arange(grid_size, dtype=torch.float32, device=self.device)
        y = torch.arange(grid_size, dtype=torch.float32, device=self.device)
        z = torch.arange(grid_size, dtype=torch.float32, device=self.device)

        X, Y, Z = torch.meshgrid(x, y, z, indexing="ij")

        # 计算到中心的距离
        distances = torch.sqrt((X - center) ** 2 + (Y - center) ** 2 + (Z - center) ** 2)

        # 创建球体密度场
        density_field = torch.exp(-(((distances - radius) / 2) ** 2))
        density_field = density_field.unsqueeze(0).unsqueeze(0)  # 添加 batch 和 channel 维度

        # 估计法向量
        normals = estimate_normals(density_field)

        # 检查输出形状
        expected_shape = (1, 3, grid_size, grid_size, grid_size)
        assert normals.shape == expected_shape

        # 检查法向量是否已归一化
        normal_magnitudes = torch.norm(normals, dim=1)
        # 在密度较高的区域，法向量应该接近单位长度
        mask = density_field[0, 0] > 0.1
        if mask.any():
            assert torch.allclose(
                normal_magnitudes[0][mask], torch.ones_like(normal_magnitudes[0][mask]), atol=0.1
            )

    def test_compute_hash_grid_size(self):
        """测试哈希网格大小计算"""
        # 测试不同的分辨率和特征维度
        test_cases: list[tuple[int, int, int]] = [
            (16, 2, 19),  # (resolution, feature_dim, log2_hashmap_size)
            (32, 4, 20),
            (64, 2, 21),
            (128, 8, 22),
        ]

        for resolution, feature_dim, log2_hashmap_size in test_cases:
            grid_size = compute_hash_grid_size(resolution, feature_dim, log2_hashmap_size)

            # 检查网格大小
            assert isinstance(grid_size, int)
            assert grid_size > 0

            # 网格大小应该不超过哈希表大小
            max_hash_size = 2**log2_hashmap_size
            assert grid_size <= max_hash_size

            # 对于小分辨率，网格大小应该等于 resolution^3
            if resolution**3 <= max_hash_size:
                assert grid_size == resolution**3
            else:
                assert grid_size == max_hash_size

    def test_utils_device_compatibility(self):
        """测试工具函数的设备兼容性"""
        # 测试 CPU 和 GPU（如果可用）
        devices: list[torch.device] = [torch.device("cpu")]
        if torch.cuda.is_available():
            devices.append(torch.device("cuda"))

        for device in devices:
            # 测试坐标收缩
            points = torch.randn(100, 3, device=device)
            contracted = contract_to_unisphere(points)
            assert contracted.device == device

            # 测试 Morton 编码
            coords = torch.randint(0, 10, (50, 3), dtype=torch.long, device=device)
            morton_codes = morton_encode_3d(coords)
            assert morton_codes.device == device

            # 测试 TV 损失
            grid = torch.randn(1, 1, 8, 8, 8, device=device)
            tv_loss = compute_tv_loss(grid)
            assert tv_loss.device == device

    def test_utils_numerical_stability(self):
        """测试工具函数的数值稳定性"""
        # 测试极值情况

        # 1. 非常大的点坐标
        large_points = torch.tensor([[1e6, 1e6, 1e6]], dtype=torch.float32, device=self.device)
        contracted_large = contract_to_unisphere(large_points)
        assert not torch.isnan(contracted_large).any()
        assert not torch.isinf(contracted_large).any()

        # 2. 非常小的点坐标
        small_points = torch.tensor([[1e-6, 1e-6, 1e-6]], dtype=torch.float32, device=self.device)
        contracted_small = contract_to_unisphere(small_points)
        assert not torch.isnan(contracted_small).any()
        assert not torch.isinf(contracted_small).any()

        # 3. 零权重的自适应采样
        zero_weights = torch.zeros(10, 64, device=self.device)
        new_samples, new_weights = adaptive_sampling(zero_weights, 32)
        assert not torch.isnan(new_samples).any()
        assert not torch.isnan(new_weights).any()

    def test_utils_batch_processing(self):
        """测试工具函数的批处理能力"""
        # 测试不同批次大小
        batch_sizes: list[int] = [1, 10, 100, 1000]

        for batch_size in batch_sizes:
            # 测试坐标收缩
            points = torch.randn(batch_size, 3, device=self.device)
            contracted = contract_to_unisphere(points)
            assert contracted.shape == (batch_size, 3)

            # 测试 Morton 编码
            coords = torch.randint(0, 16, (batch_size, 3), dtype=torch.long, device=self.device)
            morton_codes = morton_encode_3d(coords)
            assert morton_codes.shape == (batch_size,)

    def test_utils_gradient_computation(self):
        """测试工具函数的梯度计算"""
        # 测试需要梯度的输入

        # 1. 坐标收缩的梯度
        points = torch.randn(100, 3, device=self.device, requires_grad=True)
        contracted = contract_to_unisphere(points)
        loss = contracted.sum()
        loss.backward()

        assert points.grad is not None
        assert not torch.isnan(points.grad).any()
        assert not torch.isinf(points.grad).any()

        # 2. TV 损失的梯度
        grid = torch.randn(1, 1, 16, 16, 16, device=self.device, requires_grad=True)
        tv_loss = compute_tv_loss(grid)
        tv_loss.backward()

        assert grid.grad is not None
        assert not torch.isnan(grid.grad).any()
        assert not torch.isinf(grid.grad).any()

    def test_utils_performance(self):
        """测试工具函数的性能"""
        import time

        # 测试大批次处理的性能
        large_batch_size = 10000

        # 1. 坐标收缩性能
        points = torch.randn(large_batch_size, 3, device=self.device)

        start_time = time.time()
        contracted = contract_to_unisphere(points)
        end_time = time.time()

        contraction_time = end_time - start_time
        assert contraction_time < 1.0  # 应该在1秒内完成

        # 2. Morton 编码性能
        coords = torch.randint(0, 1024, (large_batch_size, 3), dtype=torch.long, device=self.device)

        start_time = time.time()
        morton_codes = morton_encode_3d(coords)
        end_time = time.time()

        morton_time = end_time - start_time
        assert morton_time < 2.0  # 应该在2秒内完成

    def test_utils_edge_cases(self):
        """测试工具函数的边界情况"""
        # 1. 空输入
        empty_points = torch.empty(0, 3, device=self.device)
        empty_contracted = contract_to_unisphere(empty_points)
        assert empty_contracted.shape == (0, 3)

        # 2. 单个点
        single_point = torch.randn(1, 3, device=self.device)
        single_contracted = contract_to_unisphere(single_point)
        assert single_contracted.shape == (1, 3)

        # 3. 非常小的网格
        tiny_grid = torch.randn(1, 1, 2, 2, 2, device=self.device)
        tiny_tv_loss = compute_tv_loss(tiny_grid)
        assert tiny_tv_loss.item() >= 0

    def test_utils_type_consistency(self):
        """测试工具函数的类型一致性"""
        # 使用 Python 3.10 兼容的类型注解

        # 测试输入输出类型
        points: torch.Tensor = torch.randn(100, 3, device=self.device)
        contracted: torch.Tensor = contract_to_unisphere(points)

        assert isinstance(contracted, torch.Tensor)
        assert contracted.dtype == points.dtype
        assert contracted.device == points.device

        # 测试 Morton 编码类型
        coords: torch.Tensor = torch.randint(0, 10, (50, 3), dtype=torch.long, device=self.device)
        morton_codes: torch.Tensor = morton_encode_3d(coords)

        assert isinstance(morton_codes, torch.Tensor)
        assert morton_codes.dtype == torch.long
        assert morton_codes.device == coords.device

    def teardown_method(self):
        """每个测试方法后的清理"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class TestInstantNGPUtilsIntegration:
    """Instant NGP 工具函数集成测试"""

    def setup_method(self):
        """设置集成测试"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_complete_coordinate_transformation_pipeline(self):
        """测试完整的坐标变换流水线"""
        # 创建原始世界坐标
        world_coords = torch.randn(1000, 3, device=self.device) * 5

        # 1. 收缩到单位球体
        contracted_coords = contract_to_unisphere(world_coords)

        # 2. 验证收缩后的坐标在单位球体内
        distances = torch.norm(contracted_coords, dim=-1)
        assert torch.all(distances <= 1.0 + 1e-6)

        # 3. 计算 Morton 编码（用于哈希）
        # 首先需要将坐标离散化
        discretized_coords = ((contracted_coords + 1) * 127.5).long().clamp(0, 255)
        morton_codes = morton_encode_3d(discretized_coords)

        # 4. 验证 Morton 编码
        assert morton_codes.shape == (1000,)
        assert torch.all(morton_codes >= 0)

        # 5. 还原坐标
        uncontracted_coords = uncontract_from_unisphere(contracted_coords)

        # 6. 验证还原精度
        assert torch.allclose(world_coords, uncontracted_coords, atol=1e-4)

    def test_density_field_processing_pipeline(self):
        """测试密度场处理流水线"""
        # 创建密度场
        grid_size = 32
        density_field = torch.rand(1, 1, grid_size, grid_size, grid_size, device=self.device)

        # 1. 计算 TV 损失（正则化）
        tv_loss = compute_tv_loss(density_field)
        assert tv_loss.item() >= 0

        # 2. 估计法向量
        normals = estimate_normals(density_field)
        assert normals.shape == (1, 3, grid_size, grid_size, grid_size)

        # 3. 验证法向量的数值特性
        normal_magnitudes = torch.norm(normals, dim=1)
        # 在密度较高的区域检查法向量
        high_density_mask = density_field[0, 0] > 0.5
        if high_density_mask.any():
            high_density_normals = normal_magnitudes[0][high_density_mask]
            # 法向量应该接近单位长度
            assert torch.allclose(
                high_density_normals, torch.ones_like(high_density_normals), atol=0.2
            )

    def test_sampling_and_encoding_integration(self):
        """测试采样和编码的集成"""
        # 创建采样权重
        batch_size = 500
        num_samples = 128
        weights = torch.rand(batch_size, num_samples, device=self.device)
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)

        # 1. 自适应采样
        num_new_samples = 64
        new_samples, new_weights = adaptive_sampling(weights, num_new_samples)

        # 2. 验证采样结果
        assert new_samples.shape == (batch_size, num_new_samples)
        assert new_weights.shape == (batch_size, num_new_samples)
        assert torch.all(new_samples >= 0)
        assert torch.all(new_samples < num_samples)

        # 3. 基于采样结果生成坐标
        sampled_coords = torch.randn(batch_size * num_new_samples, 3, device=self.device)

        # 4. 坐标收缩和编码
        contracted_coords = contract_to_unisphere(sampled_coords)
        discretized_coords = ((contracted_coords + 1) * 127.5).long().clamp(0, 255)
        morton_codes = morton_encode_3d(discretized_coords)

        # 5. 验证整个流水线的输出
        assert morton_codes.shape == (batch_size * num_new_samples,)
        assert torch.all(morton_codes >= 0)

    def teardown_method(self):
        """清理集成测试"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    pytest.main([__file__])
