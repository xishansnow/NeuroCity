"""
InfNeRF 核心模型测试

测试 InfNeRF 的核心组件，包括：
- InfNeRF 主模型
- OctreeNode 八叉树节点
- LoDAwareNeRF 网络
- HashEncoder 哈希编码器
- SphericalHarmonicsEncoder 球谐编码器
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch

from src.nerfs.inf_nerf import (
    InfNeRF,
    InfNeRFConfig,
    OctreeNode,
    LoDAwareNeRF,
    HashEncoder,
    SphericalHarmonicsEncoder,
)


class TestInfNeRFConfig:
    """测试 InfNeRF 配置类"""

    def test_default_config(self):
        """测试默认配置"""
        config = InfNeRFConfig()

        assert config.max_depth == 8
        assert config.hidden_dim == 64
        assert config.num_samples == 64
        assert config.scene_bound == 1.0
        assert config.use_pruning is True

    def test_custom_config(self):
        """测试自定义配置"""
        config = InfNeRFConfig(
            max_depth=6, hidden_dim=32, num_samples=32, scene_bound=2.0, use_pruning=False
        )

        assert config.max_depth == 6
        assert config.hidden_dim == 32
        assert config.num_samples == 32
        assert config.scene_bound == 2.0
        assert config.use_pruning is False

    def test_config_validation(self):
        """测试配置验证"""
        # 测试无效参数
        with pytest.raises(ValueError):
            InfNeRFConfig(max_depth=-1)

        with pytest.raises(ValueError):
            InfNeRFConfig(hidden_dim=0)

        with pytest.raises(ValueError):
            InfNeRFConfig(scene_bound=0)


class TestOctreeNode:
    """测试八叉树节点"""

    def test_node_creation(self, small_config):
        """测试节点创建"""
        center = np.array([0.0, 0.0, 0.0])
        size = 2.0
        level = 0

        node = OctreeNode(center, size, level, small_config)

        assert node.center.shape == (3,)
        assert node.size == 2.0
        assert node.level == 0
        assert node.is_leaf is True
        assert len(node.children) == 8
        assert all(child is None for child in node.children)

    def test_node_bounds(self, small_config):
        """测试节点边界计算"""
        center = np.array([1.0, 2.0, 3.0])
        size = 2.0
        level = 0
        node = OctreeNode(center, size, level, small_config)

        expected_min = np.array([0.0, 1.0, 2.0])
        expected_max = np.array([2.0, 3.0, 4.0])

        np.testing.assert_array_almost_equal(node.bounds_min, expected_min)
        np.testing.assert_array_almost_equal(node.bounds_max, expected_max)

    def test_contains_point(self, small_config):
        """测试点包含判断"""
        center = np.array([0.0, 0.0, 0.0])
        size = 2.0
        level = 0
        node = OctreeNode(center, size, level, small_config)

        # 测试包含的点
        assert node.contains_point(np.array([0.5, 0.5, 0.5])) is True
        assert node.contains_point(np.array([0.0, 0.0, 0.0])) is True

        # 测试不包含的点
        assert node.contains_point(np.array([2.0, 2.0, 2.0])) is False
        assert node.contains_point(np.array([-1.0, -1.0, -1.0])) is False

    def test_subdivide(self, small_config):
        """测试节点细分"""
        center = np.array([0.0, 0.0, 0.0])
        size = 2.0
        level = 0
        node = OctreeNode(center, size, level, small_config)

        children = node.subdivide()

        assert len(children) == 8
        assert node.is_leaf is False
        assert all(child is not None for child in children)

        # 检查子节点属性
        for i, child in enumerate(children):
            assert child.level == 1
            assert child.size == 1.0
            assert child.parent == node

    def test_find_containing_child(self, small_config):
        """测试查找包含子节点"""
        center = np.array([0.0, 0.0, 0.0])
        size = 2.0
        node = OctreeNode(center, size, 0, small_config)
        node.subdivide()

        # 测试查找
        point = np.array([0.5, 0.5, 0.5])
        child = node.find_containing_child(point)

        assert child is not None
        assert child.contains_point(point)

    def test_memory_size(self, small_config):
        """测试内存大小计算"""
        center = np.array([0.0, 0.0, 0.0])
        size = 2.0
        node = OctreeNode(center, size, 0, small_config)

        # 创建模拟的 NeRF 网络
        node.nerf = Mock()
        node.nerf.parameters = lambda: [torch.randn(100, 100)]

        memory_size = node.get_memory_size()
        assert memory_size > 0


class TestLoDAwareNeRF:
    """测试 LoD 感知 NeRF 网络"""

    def test_network_creation(self, small_config):
        """测试网络创建"""
        network = LoDAwareNeRF(small_config)

        # 检查网络结构
        assert hasattr(network, "encoding")
        assert hasattr(network, "density_net")
        assert hasattr(network, "color_net")
        assert hasattr(network, "sh_encoder")

    def test_forward_pass(self, small_config, device):
        """测试前向传播"""
        network = LoDAwareNeRF(small_config).to(device)

        batch_size = 4
        positions = torch.randn(batch_size, 3, device=device)
        directions = torch.randn(batch_size, 3, device=device)

        output = network(positions, directions)

        assert "density" in output
        assert "color" in output
        assert output["density"].shape == (batch_size, 1)
        assert output["color"].shape == (batch_size, 3)

        # 检查输出范围
        assert torch.all(output["density"] >= 0)
        assert torch.all(output["color"] >= 0) and torch.all(output["color"] <= 1)

    def test_different_levels(self, small_config, device):
        """测试不同层级的网络"""
        network = LoDAwareNeRF(small_config).to(device)

        positions = torch.randn(4, 3, device=device)
        directions = torch.randn(4, 3, device=device)

        # 测试不同层级
        for level in range(3):
            network.set_level(level)
            output = network(positions, directions)

            assert "density" in output
            assert "color" in output


class TestHashEncoder:
    """测试哈希编码器"""

    def test_encoder_creation(self, small_config):
        """测试编码器创建"""
        encoder = HashEncoder(small_config)

        assert len(encoder.embeddings) == small_config.num_levels
        assert encoder.output_dim == small_config.num_levels * small_config.level_dim

    def test_forward_pass(self, small_config, device):
        """测试前向传播"""
        encoder = HashEncoder(small_config).to(device)

        batch_size = 8
        positions = torch.randn(batch_size, 3, device=device)

        encoded = encoder(positions)

        expected_dim = small_config.num_levels * small_config.level_dim
        assert encoded.shape == (batch_size, expected_dim)

    def test_position_scaling(self, small_config, device):
        """测试位置缩放"""
        encoder = HashEncoder(small_config).to(device)

        # 测试不同尺度的位置
        positions1 = torch.randn(4, 3, device=device)
        positions2 = positions1 * 2.0

        encoded1 = encoder(positions1)
        encoded2 = encoder(positions2)

        # 编码应该不同
        assert not torch.allclose(encoded1, encoded2)


class TestSphericalHarmonicsEncoder:
    """测试球谐编码器"""

    def test_encoder_creation(self):
        """测试编码器创建"""
        degree = 2
        encoder = SphericalHarmonicsEncoder(degree)

        expected_dim = (degree + 1) ** 2
        assert encoder.output_dim == expected_dim

    def test_forward_pass(self, device):
        """测试前向传播"""
        degree = 2
        encoder = SphericalHarmonicsEncoder(degree).to(device)

        batch_size = 8
        directions = torch.randn(batch_size, 3, device=device)
        directions = torch.nn.functional.normalize(directions, dim=-1)

        encoded = encoder(directions)

        expected_dim = (degree + 1) ** 2
        assert encoded.shape == (batch_size, expected_dim)

    def test_normalized_directions(self, device):
        """测试归一化方向"""
        degree = 1
        encoder = SphericalHarmonicsEncoder(degree).to(device)

        # 测试非归一化方向
        directions = torch.randn(4, 3, device=device)
        encoded = encoder(directions)

        # 应该自动归一化
        assert encoded.shape == (4, 4)  # degree=1 有 4 个系数


class TestInfNeRF:
    """测试 InfNeRF 主模型"""

    def test_model_creation(self, small_config):
        """测试模型创建"""
        model = InfNeRF(small_config)

        assert hasattr(model, "root_node")
        assert hasattr(model, "renderer")
        assert model.root_node is not None

    def test_octree_construction(self, small_config):
        """测试八叉树构建"""
        model = InfNeRF(small_config)

        # 创建稀疏点
        sparse_points = np.random.randn(100, 3) * 0.5

        model.build_octree(sparse_points)

        # 检查根节点
        assert model.root_node is not None
        assert len(model.root_node.sparse_points) > 0

    def test_forward_pass(self, small_config, device):
        """测试前向传播"""
        model = InfNeRF(small_config).to(device)

        batch_size = 4
        rays_o = torch.randn(batch_size, 3, device=device)
        rays_d = torch.randn(batch_size, 3, device=device)
        rays_d = torch.nn.functional.normalize(rays_d, dim=-1)

        output = model(
            rays_o=rays_o,
            rays_d=rays_d,
            near=0.1,
            far=10.0,
            focal_length=800.0,
            pixel_width=1.0 / 800,
        )

        assert "rgb" in output
        assert "depth" in output
        assert "acc" in output
        assert output["rgb"].shape == (batch_size, 3)
        assert output["depth"].shape == (batch_size,)
        assert output["acc"].shape == (batch_size,)

    def test_memory_usage(self, small_config):
        """测试内存使用计算"""
        model = InfNeRF(small_config)

        memory_info = model.get_memory_usage()

        assert "total_mb" in memory_info
        assert "by_level_mb" in memory_info
        assert memory_info["total_mb"] >= 0

    @pytest.mark.gpu
    def test_gpu_compatibility(self, small_config, device):
        """测试 GPU 兼容性"""
        if device.type == "cpu":
            pytest.skip("GPU not available")

        model = InfNeRF(small_config).to(device)

        # 检查模型是否在正确的设备上
        assert next(model.parameters()).device == device

        # 测试 GPU 前向传播
        batch_size = 4
        rays_o = torch.randn(batch_size, 3, device=device)
        rays_d = torch.randn(batch_size, 3, device=device)
        rays_d = torch.nn.functional.normalize(rays_d, dim=-1)

        output = model(
            rays_o=rays_o,
            rays_d=rays_d,
            near=0.1,
            far=10.0,
            focal_length=800.0,
            pixel_width=1.0 / 800,
        )

        assert output["rgb"].device == device
        assert output["depth"].device == device

    def test_model_parameters(self, small_config):
        """测试模型参数"""
        model = InfNeRF(small_config)

        # 检查参数数量
        total_params = sum(p.numel() for p in model.parameters())
        assert total_params > 0

        # 检查可训练参数
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert trainable_params > 0
        assert trainable_params == total_params  # 所有参数都应该可训练

    def test_model_save_load(self, small_config, temp_dir):
        """测试模型保存和加载"""
        model = InfNeRF(small_config)

        # 保存模型
        save_path = temp_dir / "test_model.pth"
        torch.save(model.state_dict(), save_path)

        # 加载模型
        new_model = InfNeRF(small_config)
        new_model.load_state_dict(torch.load(save_path))

        # 检查参数是否相同
        for p1, p2 in zip(model.parameters(), new_model.parameters()):
            assert torch.allclose(p1, p2)
