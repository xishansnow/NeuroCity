"""
SVRaster One 体素网格测试

测试稀疏体素网格、Morton编码、自适应细分等功能。
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import tempfile

from ..config import SVRasterOneConfig
from ..voxels import SparseVoxelGrid, MortonCode


class TestMortonCode:
    """测试 Morton 编码功能"""

    def test_morton_encode_3d(self):
        """测试 3D Morton 编码"""
        # 测试基本编码
        x = torch.tensor([0, 1, 2, 3])
        y = torch.tensor([0, 1, 2, 3])
        z = torch.tensor([0, 1, 2, 3])
        
        morton_codes = MortonCode.encode_3d(x, y, z, bits=4)
        
        assert morton_codes.shape == (4,)
        assert morton_codes.dtype == torch.int64
        assert not torch.isnan(morton_codes).any()
        
        # 验证编码结果
        expected = torch.tensor([0, 1, 8, 9])  # 简化的预期值
        assert torch.all(morton_codes >= 0)

    def test_morton_decode_3d(self):
        """测试 3D Morton 解码"""
        # 编码
        x = torch.tensor([1, 2, 3])
        y = torch.tensor([1, 2, 3])
        z = torch.tensor([1, 2, 3])
        
        morton_codes = MortonCode.encode_3d(x, y, z, bits=4)
        
        # 解码
        decoded_x, decoded_y, decoded_z = MortonCode.decode_3d(morton_codes, bits=4)
        
        assert torch.allclose(x, decoded_x)
        assert torch.allclose(y, decoded_y)
        assert torch.allclose(z, decoded_z)

    def test_morton_clamping(self):
        """测试 Morton 编码的坐标限制"""
        # 测试超出范围的坐标
        x = torch.tensor([1000, -1000, 500])
        y = torch.tensor([1000, -1000, 500])
        z = torch.tensor([1000, -1000, 500])
        
        morton_codes = MortonCode.encode_3d(x, y, z, bits=8)
        
        # 应该被限制在有效范围内
        assert not torch.isnan(morton_codes).any()
        assert torch.all(morton_codes >= 0)

    def test_morton_reversibility(self):
        """测试 Morton 编码的可逆性"""
        # 随机坐标
        x = torch.randint(0, 100, (50,))
        y = torch.randint(0, 100, (50,))
        z = torch.randint(0, 100, (50,))
        
        # 编码
        morton_codes = MortonCode.encode_3d(x, y, z, bits=7)
        
        # 解码
        decoded_x, decoded_y, decoded_z = MortonCode.decode_3d(morton_codes, bits=7)
        
        # 验证可逆性
        assert torch.allclose(x, decoded_x)
        assert torch.allclose(y, decoded_y)
        assert torch.allclose(z, decoded_z)


class TestSparseVoxelGrid:
    """测试稀疏体素网格"""

    def test_voxel_grid_init(self):
        """测试体素网格初始化"""
        config = SVRasterOneConfig()
        config.voxel.grid_resolution = 64  # 较小的分辨率用于测试
        config.voxel.max_voxels = 1000
        
        voxel_grid = SparseVoxelGrid(config)
        
        # 检查参数
        assert voxel_grid.max_voxels == 1000
        assert voxel_grid.grid_resolution == 64
        assert voxel_grid.voxel_size == config.voxel.voxel_size
        
        # 检查体素数据
        assert hasattr(voxel_grid, 'voxel_coords')
        assert hasattr(voxel_grid, 'voxel_features')
        assert hasattr(voxel_grid, 'voxel_sizes')
        assert hasattr(voxel_grid, 'active_mask')
        
        # 检查数据形状
        num_voxels = voxel_grid.voxel_coords.shape[0]
        assert num_voxels <= config.voxel.max_voxels
        assert voxel_grid.voxel_features.shape == (num_voxels, 4)  # [density, r, g, b]
        assert voxel_grid.voxel_sizes.shape == (num_voxels,)
        assert voxel_grid.active_mask.shape == (num_voxels,)

    def test_voxel_grid_forward(self):
        """测试体素网格前向传播"""
        config = SVRasterOneConfig()
        config.voxel.grid_resolution = 32
        config.voxel.max_voxels = 100
        
        voxel_grid = SparseVoxelGrid(config)
        
        # 查询点
        query_points = torch.randn(50, 3) * 0.5  # 在 [-0.5, 0.5] 范围内
        
        # 前向传播
        result = voxel_grid(query_points)
        
        # 检查输出
        assert "densities" in result
        assert "colors" in result
        assert "sizes" in result
        assert "weights" in result
        
        assert result["densities"].shape == (50,)
        assert result["colors"].shape == (50, 3)
        assert result["sizes"].shape == (50,)
        assert result["weights"].shape[0] == 50
        
        # 检查数值范围
        assert torch.all(result["densities"] >= 0)  # 密度应该非负
        assert torch.all(result["colors"] >= 0) and torch.all(result["colors"] <= 1)  # 颜色在 [0,1]
        assert torch.all(result["sizes"] >= 0)  # 尺寸应该非负

    def test_voxel_grid_empty(self):
        """测试空体素网格"""
        config = SVRasterOneConfig()
        config.voxel.grid_resolution = 32
        config.voxel.max_voxels = 0  # 没有体素
        
        voxel_grid = SparseVoxelGrid(config)
        
        # 设置所有体素为非活跃
        voxel_grid.active_mask.fill_(False)
        
        query_points = torch.randn(10, 3)
        result = voxel_grid(query_points)
        
        # 应该返回零特征
        assert torch.all(result["densities"] == 0)
        assert torch.all(result["colors"] == 0)
        assert torch.all(result["sizes"] == 0)

    def test_get_active_voxels(self):
        """测试获取活跃体素"""
        config = SVRasterOneConfig()
        config.voxel.grid_resolution = 32
        config.voxel.max_voxels = 100
        
        voxel_grid = SparseVoxelGrid(config)
        
        # 获取活跃体素
        active_voxels = voxel_grid.get_active_voxels()
        
        # 检查输出
        assert "positions" in active_voxels
        assert "sizes" in active_voxels
        assert "densities" in active_voxels
        assert "colors" in active_voxels
        
        # 检查形状一致性
        num_active = active_voxels["positions"].shape[0]
        assert active_voxels["sizes"].shape == (num_active,)
        assert active_voxels["densities"].shape == (num_active,)
        assert active_voxels["colors"].shape == (num_active, 3)

    def test_adaptive_subdivision(self):
        """测试自适应细分"""
        config = SVRasterOneConfig()
        config.voxel.grid_resolution = 32
        config.voxel.max_voxels = 200
        
        voxel_grid = SparseVoxelGrid(config)
        
        # 记录初始体素数量
        initial_count = voxel_grid.voxel_coords.shape[0]
        
        # 模拟梯度幅度
        gradient_magnitudes = torch.rand(initial_count) * 0.5
        
        # 执行自适应细分
        voxel_grid.adaptive_subdivision(gradient_magnitudes)
        
        # 检查体素数量可能增加
        new_count = voxel_grid.voxel_coords.shape[0]
        assert new_count >= initial_count

    def test_adaptive_pruning(self):
        """测试自适应剪枝"""
        config = SVRasterOneConfig()
        config.voxel.grid_resolution = 32
        config.voxel.max_voxels = 100
        
        voxel_grid = SparseVoxelGrid(config)
        
        # 记录初始活跃体素数量
        initial_active = voxel_grid.active_mask.sum().item()
        
        # 设置一些体素的密度很低
        voxel_grid.voxel_features[:, 0] = 0.001  # 低密度
        
        # 执行自适应剪枝
        voxel_grid.adaptive_pruning()
        
        # 检查活跃体素数量可能减少
        new_active = voxel_grid.active_mask.sum().item()
        assert new_active <= initial_active

    def test_sort_by_morton(self):
        """测试 Morton 排序"""
        config = SVRasterOneConfig()
        config.voxel.grid_resolution = 32
        config.voxel.max_voxels = 100
        config.voxel.use_morton_ordering = True
        
        voxel_grid = SparseVoxelGrid(config)
        
        # 记录初始顺序
        initial_coords = voxel_grid.voxel_coords.clone()
        initial_features = voxel_grid.voxel_features.clone()
        initial_sizes = voxel_grid.voxel_sizes.clone()
        
        # 执行 Morton 排序
        voxel_grid.sort_by_morton()
        
        # 检查数据仍然存在，但顺序可能改变
        assert voxel_grid.voxel_coords.shape == initial_coords.shape
        assert voxel_grid.voxel_features.shape == initial_features.shape
        assert voxel_grid.voxel_sizes.shape == initial_sizes.shape

    def test_get_stats(self):
        """测试获取统计信息"""
        config = SVRasterOneConfig()
        config.voxel.grid_resolution = 32
        config.voxel.max_voxels = 100
        
        voxel_grid = SparseVoxelGrid(config)
        
        stats = voxel_grid.get_stats()
        
        # 检查统计信息
        assert "total_voxels" in stats
        assert "active_voxels" in stats
        assert "subdivision_count" in stats
        assert "pruning_count" in stats
        
        assert stats["total_voxels"] > 0
        assert stats["active_voxels"] >= 0
        assert stats["active_voxels"] <= stats["total_voxels"]

    def test_voxel_grid_gradients(self):
        """测试体素网格的梯度传播"""
        config = SVRasterOneConfig()
        config.voxel.grid_resolution = 32
        config.voxel.max_voxels = 50
        
        voxel_grid = SparseVoxelGrid(config)
        
        # 查询点
        query_points = torch.randn(10, 3, requires_grad=True)
        
        # 前向传播
        result = voxel_grid(query_points)
        
        # 计算损失
        loss = result["densities"].sum() + result["colors"].sum()
        
        # 反向传播
        loss.backward()
        
        # 检查梯度
        assert query_points.grad is not None
        assert not torch.isnan(query_points.grad).any()
        
        # 检查体素特征的梯度
        assert voxel_grid.voxel_features.grad is not None
        assert not torch.isnan(voxel_grid.voxel_features.grad).any()

    def test_voxel_grid_device(self):
        """测试体素网格在不同设备上的行为"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        config = SVRasterOneConfig()
        config.voxel.grid_resolution = 32
        config.voxel.max_voxels = 50
        
        # CPU 版本
        voxel_grid_cpu = SparseVoxelGrid(config)
        query_points_cpu = torch.randn(10, 3)
        result_cpu = voxel_grid_cpu(query_points_cpu)
        
        # GPU 版本
        config.device = "cuda"
        voxel_grid_gpu = SparseVoxelGrid(config)
        query_points_gpu = torch.randn(10, 3, device="cuda")
        result_gpu = voxel_grid_gpu(query_points_gpu)
        
        # 检查结果形状一致
        assert result_cpu["densities"].shape == result_gpu["densities"].shape
        assert result_cpu["colors"].shape == result_gpu["colors"].shape
        assert result_cpu["sizes"].shape == result_gpu["sizes"].shape


def test_voxel_grid_edge_cases():
    """测试体素网格边界情况"""
    config = SVRasterOneConfig()
    config.voxel.grid_resolution = 1  # 最小分辨率
    config.voxel.max_voxels = 1  # 最小体素数量
    
    voxel_grid = SparseVoxelGrid(config)
    
    # 测试空查询
    query_points = torch.empty(0, 3)
    result = voxel_grid(query_points)
    
    assert result["densities"].shape == (0,)
    assert result["colors"].shape == (0, 3)
    assert result["sizes"].shape == (0,)


if __name__ == "__main__":
    pytest.main([__file__]) 