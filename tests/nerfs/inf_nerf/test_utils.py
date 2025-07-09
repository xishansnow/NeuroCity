"""
InfNeRF 工具函数测试

测试 InfNeRF 的各种工具函数，包括：
- 八叉树工具
- LoD 工具
- 渲染工具
- 体积渲染工具
"""

import pytest
import torch
import numpy as np
from pathlib import Path

from src.nerfs.inf_nerf.utils import octree_utils, lod_utils, rendering_utils
from src.nerfs.inf_nerf.utils.volume_renderer import VolumeRenderer, VolumeRendererConfig


class TestOctreeUtils:
    """测试八叉树工具函数"""

    def test_morton_code_encoding(self):
        """测试 Morton 编码"""
        # 测试简单的坐标
        x, y, z = 1, 2, 3
        morton = octree_utils.encode_morton(x, y, z)

        assert isinstance(morton, int)
        assert morton >= 0

        # 测试解码
        decoded_x, decoded_y, decoded_z = octree_utils.decode_morton(morton)
        assert decoded_x == x
        assert decoded_y == y
        assert decoded_z == z

    def test_morton_code_batch(self):
        """测试批量 Morton 编码"""
        coords = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3]])

        morton_codes = octree_utils.encode_morton_batch(coords)

        assert len(morton_codes) == len(coords)
        assert all(isinstance(code, int) for code in morton_codes)
        assert all(code >= 0 for code in morton_codes)

    def test_octree_node_creation(self):
        """测试八叉树节点创建"""
        center = np.array([0.0, 0.0, 0.0])
        size = 2.0
        level = 0

        node = octree_utils.create_octree_node(center, size, level)

        assert node.center.shape == (3,)
        assert node.size == size
        assert node.level == level
        assert node.is_leaf is True

    def test_octree_traversal(self):
        """测试八叉树遍历"""
        # 创建简单的八叉树
        root = octree_utils.create_octree_node(np.array([0.0, 0.0, 0.0]), 2.0, 0)
        root.subdivide()

        # 遍历所有节点
        nodes = octree_utils.traverse_octree(root)

        assert len(nodes) == 9  # 根节点 + 8个子节点
        assert root in nodes

    def test_spatial_query(self):
        """测试空间查询"""
        # 创建八叉树
        root = octree_utils.create_octree_node(np.array([0.0, 0.0, 0.0]), 2.0, 0)
        root.subdivide()

        # 查询点
        query_point = np.array([0.5, 0.5, 0.5])
        containing_node = octree_utils.find_containing_node(root, query_point)

        assert containing_node is not None
        assert containing_node.contains_point(query_point)

    def test_octree_statistics(self):
        """测试八叉树统计"""
        # 创建八叉树
        root = octree_utils.create_octree_node(np.array([0.0, 0.0, 0.0]), 2.0, 0)
        root.subdivide()

        stats = octree_utils.get_octree_statistics(root)

        assert "total_nodes" in stats
        assert "max_depth" in stats
        assert "leaf_nodes" in stats
        assert stats["total_nodes"] == 9
        assert stats["max_depth"] == 1
        assert stats["leaf_nodes"] == 8


class TestLoDUtils:
    """测试 LoD 工具函数"""

    def test_lod_level_calculation(self):
        """测试 LoD 层级计算"""
        # 测试不同距离的 LoD 层级
        distances = [0.1, 1.0, 10.0, 100.0]

        for distance in distances:
            level = lod_utils.calculate_lod_level(distance)
            assert isinstance(level, int)
            assert level >= 0

    def test_lod_transition(self):
        """测试 LoD 过渡"""
        # 测试平滑过渡
        level1 = 2
        level2 = 4
        alpha = 0.5

        interpolated = lod_utils.interpolate_lod_levels(level1, level2, alpha)

        assert isinstance(interpolated, float)
        assert level1 <= interpolated <= level2

    def test_lod_aware_sampling(self):
        """测试 LoD 感知采样"""
        # 创建采样点
        positions = torch.randn(10, 3)
        distances = torch.norm(positions, dim=-1)

        # 计算 LoD 感知的采样密度
        densities = lod_utils.calculate_lod_sampling_density(positions, distances)

        assert densities.shape == positions.shape[:1]
        assert torch.all(densities >= 0)

    def test_lod_consistency(self):
        """测试 LoD 一致性"""
        # 测试相邻区域的 LoD 一致性
        positions = torch.randn(100, 3)
        distances = torch.norm(positions, dim=-1)

        consistency_score = lod_utils.calculate_lod_consistency(positions, distances)

        assert isinstance(consistency_score, float)
        assert 0 <= consistency_score <= 1


class TestRenderingUtils:
    """测试渲染工具函数"""

    def test_ray_generation(self):
        """测试光线生成"""
        # 创建相机参数
        camera_pose = torch.eye(4)
        intrinsics = torch.tensor([[800, 0, 400], [0, 800, 300], [0, 0, 1]])
        width, height = 64, 64

        rays_o, rays_d = rendering_utils.generate_rays(camera_pose, intrinsics, width, height)

        assert rays_o.shape == (width * height, 3)
        assert rays_d.shape == (width * height, 3)

        # 检查光线方向归一化
        directions_norm = torch.norm(rays_d, dim=-1)
        assert torch.allclose(directions_norm, torch.ones_like(directions_norm), atol=1e-6)

    def test_ray_box_intersection(self):
        """测试光线包围盒相交"""
        # 创建光线
        rays_o = torch.tensor([[0.0, 0.0, 0.0], [5.0, 5.0, 5.0]])
        rays_d = torch.tensor([[1.0, 0.0, 0.0], [-1.0, -1.0, -1.0]])
        rays_d = torch.nn.functional.normalize(rays_d, dim=-1)

        # 创建包围盒
        box_min = torch.tensor([-1.0, -1.0, -1.0])
        box_max = torch.tensor([1.0, 1.0, 1.0])

        intersections = rendering_utils.ray_box_intersection(rays_o, rays_d, box_min, box_max)

        assert intersections.shape == (2, 2)  # 每个光线有近点和远点
        assert torch.all(intersections[:, 0] <= intersections[:, 1])

    def test_depth_to_disparity(self):
        """测试深度到视差转换"""
        depth = torch.tensor([1.0, 2.0, 5.0, 10.0])
        focal_length = 800.0

        disparity = rendering_utils.depth_to_disparity(depth, focal_length)

        assert disparity.shape == depth.shape
        assert torch.all(disparity > 0)
        # 深度越大，视差越小
        assert disparity[0] > disparity[1] > disparity[2] > disparity[3]

    def test_disparity_to_depth(self):
        """测试视差到深度转换"""
        disparity = torch.tensor([800.0, 400.0, 160.0, 80.0])
        focal_length = 800.0

        depth = rendering_utils.disparity_to_depth(disparity, focal_length)

        assert depth.shape == disparity.shape
        assert torch.all(depth > 0)
        # 视差越大，深度越小
        assert depth[0] < depth[1] < depth[2] < depth[3]

    def test_image_metrics(self):
        """测试图像质量指标"""
        # 创建模拟图像
        pred = torch.rand(64, 64, 3)
        target = torch.rand(64, 64, 3)

        # 计算 PSNR
        psnr = rendering_utils.calculate_psnr(pred, target)
        assert isinstance(psnr, float)
        assert psnr >= 0

        # 计算 SSIM
        ssim = rendering_utils.calculate_ssim(pred, target)
        assert isinstance(ssim, float)
        assert 0 <= ssim <= 1

        # 计算 LPIPS
        lpips = rendering_utils.calculate_lpips(pred, target)
        assert isinstance(lpips, float)
        assert lpips >= 0

    def test_color_conversion(self):
        """测试颜色空间转换"""
        # 测试 RGB 到 sRGB 转换
        rgb = torch.rand(10, 3)
        srgb = rendering_utils.rgb_to_srgb(rgb)

        assert srgb.shape == rgb.shape
        assert torch.all(srgb >= 0) and torch.all(srgb <= 1)

        # 测试 sRGB 到 RGB 转换
        rgb_back = rendering_utils.srgb_to_rgb(srgb)
        assert rgb_back.shape == srgb.shape

    def test_tone_mapping(self):
        """测试色调映射"""
        # 创建 HDR 图像
        hdr = torch.rand(64, 64, 3) * 10.0  # 高动态范围

        # 应用色调映射
        ldr = rendering_utils.tone_mapping(hdr, method="reinhard")

        assert ldr.shape == hdr.shape
        assert torch.all(ldr >= 0) and torch.all(ldr <= 1)


class TestVolumeRenderer:
    """测试体积渲染器"""

    def test_volume_renderer_creation(self):
        """测试体积渲染器创建"""
        config = VolumeRendererConfig()
        renderer = VolumeRenderer(config)

        assert renderer.config is config
        assert renderer.num_samples == config.num_samples
        assert renderer.num_importance_samples == config.num_importance_samples

    def test_volume_rendering(self, device):
        """测试体积渲染"""
        config = VolumeRendererConfig(num_samples=32, num_importance_samples=64)
        renderer = VolumeRenderer(config).to(device)

        # 创建模拟数据
        batch_size = 4
        num_samples = 32

        colors = torch.rand(batch_size, num_samples, 3, device=device)
        densities = torch.rand(batch_size, num_samples, 1, device=device)
        z_vals = torch.linspace(0.1, 10.0, num_samples, device=device).expand(batch_size, -1)
        rays_d = torch.randn(batch_size, 3, device=device)
        rays_d = torch.nn.functional.normalize(rays_d, dim=-1)

        # 执行体积渲染
        result = renderer.volume_render(colors, densities, z_vals, rays_d)

        assert isinstance(result, dict)
        assert "rgb" in result
        assert "depth" in result
        assert "acc" in result

        assert result["rgb"].shape == (batch_size, 3)
        assert result["depth"].shape == (batch_size,)
        assert result["acc"].shape == (batch_size,)

        # 检查值范围
        assert torch.all(result["rgb"] >= 0) and torch.all(result["rgb"] <= 1)
        assert torch.all(result["depth"] >= 0)
        assert torch.all(result["acc"] >= 0) and torch.all(result["acc"] <= 1)

    def test_hierarchical_sampling(self, device):
        """测试分层采样"""
        config = VolumeRendererConfig(num_samples=16, num_importance_samples=32, perturb=True)
        renderer = VolumeRenderer(config).to(device)

        # 创建模拟数据
        batch_size = 2
        rays_o = torch.randn(batch_size, 3, device=device)
        rays_d = torch.randn(batch_size, 3, device=device)
        rays_d = torch.nn.functional.normalize(rays_d, dim=-1)
        near = torch.tensor([0.1, 0.1], device=device)
        far = torch.tensor([10.0, 10.0], device=device)

        # 执行分层采样
        z_vals, weights = renderer.hierarchical_sampling(rays_o, rays_d, near, far)

        assert z_vals.shape == (batch_size, config.num_samples + config.num_importance_samples)
        assert weights.shape == (batch_size, config.num_samples + config.num_importance_samples)

        # 检查 z_vals 是否有序
        for i in range(batch_size):
            assert torch.all(z_vals[i, 1:] >= z_vals[i, :-1])

    def test_importance_sampling(self, device):
        """测试重要性采样"""
        config = VolumeRendererConfig(num_samples=16, num_importance_samples=32)
        renderer = VolumeRenderer(config).to(device)

        # 创建模拟权重
        weights = torch.rand(2, 16, device=device)
        z_vals = torch.linspace(0.1, 10.0, 16, device=device).expand(2, -1)

        # 执行重要性采样
        z_vals_importance = renderer.importance_sampling(weights, z_vals)

        assert z_vals_importance.shape == (2, config.num_importance_samples)

        # 检查采样点是否在合理范围内
        assert torch.all(z_vals_importance >= z_vals.min())
        assert torch.all(z_vals_importance <= z_vals.max())

    def test_volume_renderer_config(self):
        """测试体积渲染器配置"""
        config = VolumeRendererConfig(
            num_samples=64, num_importance_samples=128, perturb=True, white_background=True
        )

        assert config.num_samples == 64
        assert config.num_importance_samples == 128
        assert config.perturb is True
        assert config.white_background is True

    def test_volume_renderer_performance(self, device):
        """测试体积渲染器性能"""
        config = VolumeRendererConfig(num_samples=32, num_importance_samples=64)
        renderer = VolumeRenderer(config).to(device)

        # 创建测试数据
        batch_size = 8
        colors = torch.rand(batch_size, 32, 3, device=device)
        densities = torch.rand(batch_size, 32, 1, device=device)
        z_vals = torch.linspace(0.1, 10.0, 32, device=device).expand(batch_size, -1)
        rays_d = torch.randn(batch_size, 3, device=device)
        rays_d = torch.nn.functional.normalize(rays_d, dim=-1)

        import time

        # 测量渲染时间
        start_time = time.time()
        result = renderer.volume_render(colors, densities, z_vals, rays_d)
        end_time = time.time()

        render_time = end_time - start_time

        # 渲染应该在合理时间内完成
        assert render_time < 1.0  # 小于1秒
        assert "rgb" in result
        assert result["rgb"].shape == (batch_size, 3)


def test_utility_functions_integration():
    """测试工具函数集成"""
    # 测试八叉树和 LoD 的集成
    center = np.array([0.0, 0.0, 0.0])
    size = 2.0
    level = 0

    node = octree_utils.create_octree_node(center, size, level)

    # 计算 LoD 层级
    distance = 1.0
    lod_level = lod_utils.calculate_lod_level(distance)

    assert isinstance(lod_level, int)
    assert lod_level >= 0

    # 生成光线
    camera_pose = torch.eye(4)
    intrinsics = torch.tensor([[800, 0, 400], [0, 800, 300], [0, 0, 1]])
    width, height = 32, 32

    rays_o, rays_d = rendering_utils.generate_rays(camera_pose, intrinsics, width, height)

    assert rays_o.shape == (width * height, 3)
    assert rays_d.shape == (width * height, 3)
