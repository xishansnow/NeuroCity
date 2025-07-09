"""
SVRaster One 渲染器测试

测试可微分光栅化渲染器的各种功能。
"""

import pytest
import torch
import numpy as np
from pathlib import Path

from ..config import SVRasterOneConfig
from ..renderer import DifferentiableVoxelRasterizer


class TestDifferentiableVoxelRasterizer:
    """测试可微分体素光栅化渲染器"""

    def test_renderer_init(self):
        """测试渲染器初始化"""
        config = SVRasterOneConfig()
        config.rendering.image_width = 256
        config.rendering.image_height = 256
        
        renderer = DifferentiableVoxelRasterizer(config)
        
        # 检查参数
        assert renderer.image_width == 256
        assert renderer.image_height == 256
        assert renderer.temperature == config.rendering.temperature
        assert renderer.sigma == config.rendering.sigma
        assert renderer.soft_rasterization == config.rendering.soft_rasterization
        assert renderer.use_soft_sorting == config.rendering.use_soft_sorting
        assert renderer.alpha_blending == config.rendering.alpha_blending
        assert renderer.depth_sorting == config.rendering.depth_sorting

    def test_project_voxels_to_screen(self):
        """测试体素投影到屏幕空间"""
        config = SVRasterOneConfig()
        config.rendering.image_width = 256
        config.rendering.image_height = 256
        
        renderer = DifferentiableVoxelRasterizer(config)
        
        # 创建体素数据
        num_voxels = 10
        voxels = {
            "positions": torch.randn(num_voxels, 3) * 0.5,
            "sizes": torch.ones(num_voxels) * 0.1,
            "densities": torch.rand(num_voxels),
            "colors": torch.rand(num_voxels, 3),
        }
        
        # 相机参数
        camera_matrix = torch.eye(4)
        camera_matrix[2, 3] = 2.0  # 相机在 z=2 位置
        
        intrinsics = torch.tensor([
            [500, 0, 128],
            [0, 500, 128],
            [0, 0, 1]
        ], dtype=torch.float32)
        
        viewport_size = (256, 256)
        
        # 投影
        screen_voxels = renderer._project_voxels_to_screen(
            voxels, camera_matrix, intrinsics, viewport_size
        )
        
        # 检查输出
        assert len(screen_voxels) == num_voxels
        
        for voxel in screen_voxels:
            assert "screen_pos" in voxel
            assert "depth" in voxel
            assert "screen_size" in voxel
            assert "density" in voxel
            assert "color" in voxel
            assert "world_pos" in voxel
            assert "world_size" in voxel
            assert "voxel_idx" in voxel
            
            # 检查数据类型
            assert isinstance(voxel["screen_pos"], torch.Tensor)
            assert isinstance(voxel["depth"], torch.Tensor)
            assert isinstance(voxel["screen_size"], torch.Tensor)
            assert isinstance(voxel["density"], torch.Tensor)
            assert isinstance(voxel["color"], torch.Tensor)

    def test_frustum_culling(self):
        """测试视锥剔除"""
        config = SVRasterOneConfig()
        config.rendering.image_width = 256
        config.rendering.image_height = 256
        
        renderer = DifferentiableVoxelRasterizer(config)
        
        # 创建屏幕体素
        screen_voxels = []
        for i in range(10):
            # 一些在视锥内，一些在视锥外
            if i < 5:
                # 在视锥内
                screen_voxels.append({
                    "screen_pos": torch.tensor([128.0, 128.0]),
                    "depth": torch.tensor(1.0),
                    "screen_size": torch.tensor(10.0),
                    "density": torch.tensor(0.5),
                    "color": torch.tensor([0.5, 0.5, 0.5]),
                    "world_pos": torch.tensor([0.0, 0.0, 1.0]),
                    "world_size": torch.tensor(0.1),
                    "voxel_idx": i,
                })
            else:
                # 在视锥外
                screen_voxels.append({
                    "screen_pos": torch.tensor([1000.0, 1000.0]),
                    "depth": torch.tensor(0.05),  # 太近
                    "screen_size": torch.tensor(10.0),
                    "density": torch.tensor(0.5),
                    "color": torch.tensor([0.5, 0.5, 0.5]),
                    "world_pos": torch.tensor([0.0, 0.0, 0.05]),
                    "world_size": torch.tensor(0.1),
                    "voxel_idx": i,
                })
        
        viewport_size = (256, 256)
        visible_voxels = renderer._frustum_culling(screen_voxels, viewport_size)
        
        # 应该只有前5个体素可见
        assert len(visible_voxels) <= 5

    def test_soft_depth_sort(self):
        """测试软深度排序"""
        config = SVRasterOneConfig()
        config.rendering.temperature = 0.1
        config.rendering.depth_sorting = "back_to_front"
        
        renderer = DifferentiableVoxelRasterizer(config)
        
        # 创建屏幕体素
        screen_voxels = []
        depths = [3.0, 1.0, 2.0, 4.0]  # 不同深度
        
        for i, depth in enumerate(depths):
            screen_voxels.append({
                "screen_pos": torch.tensor([128.0, 128.0]),
                "depth": torch.tensor(depth),
                "screen_size": torch.tensor(10.0),
                "density": torch.tensor(0.5),
                "color": torch.tensor([0.5, 0.5, 0.5]),
                "world_pos": torch.tensor([0.0, 0.0, depth]),
                "world_size": torch.tensor(0.1),
                "voxel_idx": i,
            })
        
        sorted_voxels = renderer._soft_depth_sort(screen_voxels)
        
        # 检查排序结果
        assert len(sorted_voxels) == len(screen_voxels)
        
        # 后向前排序：深度大的应该在前
        sorted_depths = [v["depth"].item() for v in sorted_voxels]
        assert sorted_depths[0] >= sorted_depths[1]  # 第一个深度应该大于等于第二个

    def test_hard_depth_sort(self):
        """测试硬深度排序"""
        config = SVRasterOneConfig()
        config.rendering.depth_sorting = "front_to_back"
        
        renderer = DifferentiableVoxelRasterizer(config)
        
        # 创建屏幕体素
        screen_voxels = []
        depths = [3.0, 1.0, 2.0, 4.0]
        
        for i, depth in enumerate(depths):
            screen_voxels.append({
                "screen_pos": torch.tensor([128.0, 128.0]),
                "depth": torch.tensor(depth),
                "screen_size": torch.tensor(10.0),
                "density": torch.tensor(0.5),
                "color": torch.tensor([0.5, 0.5, 0.5]),
                "world_pos": torch.tensor([0.0, 0.0, depth]),
                "world_size": torch.tensor(0.1),
                "voxel_idx": i,
            })
        
        sorted_voxels = renderer._hard_depth_sort(screen_voxels)
        
        # 检查排序结果
        assert len(sorted_voxels) == len(screen_voxels)
        
        # 前向后排序：深度小的应该在前
        sorted_depths = [v["depth"].item() for v in sorted_voxels]
        assert sorted_depths[0] <= sorted_depths[1]  # 第一个深度应该小于等于第二个

    def test_soft_rasterize_voxels(self):
        """测试软光栅化"""
        config = SVRasterOneConfig()
        config.rendering.image_width = 64
        config.rendering.image_height = 64
        config.rendering.soft_rasterization = True
        config.rendering.sigma = 1.0
        
        renderer = DifferentiableVoxelRasterizer(config)
        
        # 创建排序后的体素
        sorted_voxels = []
        for i in range(5):
            sorted_voxels.append({
                "screen_pos": torch.tensor([32.0 + i * 5, 32.0 + i * 5]),
                "depth": torch.tensor(1.0 + i * 0.1),
                "screen_size": torch.tensor(5.0),
                "density": torch.tensor(0.5),
                "color": torch.tensor([0.5, 0.5, 0.5]),
                "world_pos": torch.tensor([0.0, 0.0, 1.0 + i * 0.1]),
                "world_size": torch.tensor(0.1),
                "voxel_idx": i,
            })
        
        viewport_size = (64, 64)
        framebuffer = renderer._soft_rasterize_voxels(sorted_voxels, viewport_size)
        
        # 检查输出
        assert "rgb" in framebuffer
        assert "depth" in framebuffer
        assert "alpha" in framebuffer
        
        assert framebuffer["rgb"].shape == (64, 64, 3)
        assert framebuffer["depth"].shape == (64, 64)
        assert framebuffer["alpha"].shape == (64, 64)
        
        # 检查数值范围
        assert torch.all(framebuffer["rgb"] >= 0) and torch.all(framebuffer["rgb"] <= 1)
        assert torch.all(framebuffer["depth"] >= 0)
        assert torch.all(framebuffer["alpha"] >= 0) and torch.all(framebuffer["alpha"] <= 1)

    def test_hard_rasterize_voxels(self):
        """测试硬光栅化"""
        config = SVRasterOneConfig()
        config.rendering.image_width = 64
        config.rendering.image_height = 64
        config.rendering.soft_rasterization = False
        
        renderer = DifferentiableVoxelRasterizer(config)
        
        # 创建排序后的体素
        sorted_voxels = []
        for i in range(5):
            sorted_voxels.append({
                "screen_pos": torch.tensor([32.0 + i * 5, 32.0 + i * 5]),
                "depth": torch.tensor(1.0 + i * 0.1),
                "screen_size": torch.tensor(5.0),
                "density": torch.tensor(0.5),
                "color": torch.tensor([0.5, 0.5, 0.5]),
                "world_pos": torch.tensor([0.0, 0.0, 1.0 + i * 0.1]),
                "world_size": torch.tensor(0.1),
                "voxel_idx": i,
            })
        
        viewport_size = (64, 64)
        framebuffer = renderer._hard_rasterize_voxels(sorted_voxels, viewport_size)
        
        # 检查输出
        assert "rgb" in framebuffer
        assert "depth" in framebuffer
        
        assert framebuffer["rgb"].shape == (64, 64, 3)
        assert framebuffer["depth"].shape == (64, 64)
        
        # 检查数值范围
        assert torch.all(framebuffer["rgb"] >= 0) and torch.all(framebuffer["rgb"] <= 1)
        assert torch.all(framebuffer["depth"] >= 0)

    def test_create_background_image(self):
        """测试背景图像创建"""
        config = SVRasterOneConfig()
        config.rendering.background_color = (0.1, 0.2, 0.3)
        
        renderer = DifferentiableVoxelRasterizer(config)
        
        viewport_size = (64, 64)
        device = torch.device("cpu")
        
        background = renderer._create_background_image(viewport_size, device)
        
        # 检查输出
        assert "rgb" in background
        assert "depth" in background
        assert "alpha" in background
        
        assert background["rgb"].shape == (64, 64, 3)
        assert background["depth"].shape == (64, 64)
        assert background["alpha"].shape == (64, 64)
        
        # 检查背景颜色
        expected_color = torch.tensor([0.1, 0.2, 0.3])
        assert torch.allclose(background["rgb"][0, 0], expected_color, atol=1e-6)

    def test_renderer_forward(self):
        """测试渲染器前向传播"""
        config = SVRasterOneConfig()
        config.rendering.image_width = 64
        config.rendering.image_height = 64
        config.rendering.soft_rasterization = True
        
        renderer = DifferentiableVoxelRasterizer(config)
        
        # 创建体素数据
        num_voxels = 10
        voxels = {
            "positions": torch.randn(num_voxels, 3) * 0.5,
            "sizes": torch.ones(num_voxels) * 0.1,
            "densities": torch.rand(num_voxels),
            "colors": torch.rand(num_voxels, 3),
        }
        
        # 相机参数
        camera_matrix = torch.eye(4)
        camera_matrix[2, 3] = 2.0
        
        intrinsics = torch.tensor([
            [500, 0, 32],
            [0, 500, 32],
            [0, 0, 1]
        ], dtype=torch.float32)
        
        viewport_size = (64, 64)
        
        # 前向传播
        result = renderer(voxels, camera_matrix, intrinsics, viewport_size)
        
        # 检查输出
        assert "rgb" in result
        assert "depth" in result
        assert "alpha" in result
        
        assert result["rgb"].shape == (64, 64, 3)
        assert result["depth"].shape == (64, 64)
        assert result["alpha"].shape == (64, 64)
        
        # 检查数值范围
        assert torch.all(result["rgb"] >= 0) and torch.all(result["rgb"] <= 1)
        assert torch.all(result["depth"] >= 0)
        assert torch.all(result["alpha"] >= 0) and torch.all(result["alpha"] <= 1)

    def test_renderer_empty_voxels(self):
        """测试空体素渲染"""
        config = SVRasterOneConfig()
        config.rendering.image_width = 64
        config.rendering.image_height = 64
        
        renderer = DifferentiableVoxelRasterizer(config)
        
        # 空体素数据
        voxels = {
            "positions": torch.empty(0, 3),
            "sizes": torch.empty(0),
            "densities": torch.empty(0),
            "colors": torch.empty(0, 3),
        }
        
        camera_matrix = torch.eye(4)
        intrinsics = torch.eye(3)
        viewport_size = (64, 64)
        
        # 应该返回背景图像
        result = renderer(voxels, camera_matrix, intrinsics, viewport_size)
        
        assert "rgb" in result
        assert "depth" in result
        assert "alpha" in result
        
        assert result["rgb"].shape == (64, 64, 3)
        assert result["depth"].shape == (64, 64)
        assert result["alpha"].shape == (64, 64)

    def test_renderer_gradients(self):
        """测试渲染器梯度传播"""
        config = SVRasterOneConfig()
        config.rendering.image_width = 32
        config.rendering.image_height = 32
        config.rendering.soft_rasterization = True
        
        renderer = DifferentiableVoxelRasterizer(config)
        
        # 创建需要梯度的体素数据
        num_voxels = 5
        voxels = {
            "positions": torch.randn(num_voxels, 3, requires_grad=True),
            "sizes": torch.ones(num_voxels, requires_grad=True),
            "densities": torch.rand(num_voxels, requires_grad=True),
            "colors": torch.rand(num_voxels, 3, requires_grad=True),
        }
        
        camera_matrix = torch.eye(4)
        intrinsics = torch.eye(3)
        viewport_size = (32, 32)
        
        # 前向传播
        result = renderer(voxels, camera_matrix, intrinsics, viewport_size)
        
        # 计算损失
        loss = result["rgb"].sum() + result["depth"].sum()
        
        # 反向传播
        loss.backward()
        
        # 检查梯度
        assert voxels["positions"].grad is not None
        assert voxels["sizes"].grad is not None
        assert voxels["densities"].grad is not None
        assert voxels["colors"].grad is not None
        
        assert not torch.isnan(voxels["positions"].grad).any()
        assert not torch.isnan(voxels["sizes"].grad).any()
        assert not torch.isnan(voxels["densities"].grad).any()
        assert not torch.isnan(voxels["colors"].grad).any()

    def test_renderer_device(self):
        """测试渲染器在不同设备上的行为"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        config = SVRasterOneConfig()
        config.rendering.image_width = 32
        config.rendering.image_height = 32
        
        # CPU 版本
        renderer_cpu = DifferentiableVoxelRasterizer(config)
        voxels_cpu = {
            "positions": torch.randn(5, 3),
            "sizes": torch.ones(5),
            "densities": torch.rand(5),
            "colors": torch.rand(5, 3),
        }
        camera_matrix_cpu = torch.eye(4)
        intrinsics_cpu = torch.eye(3)
        
        result_cpu = renderer_cpu(voxels_cpu, camera_matrix_cpu, intrinsics_cpu, (32, 32))
        
        # GPU 版本
        renderer_gpu = DifferentiableVoxelRasterizer(config)
        voxels_gpu = {k: v.cuda() for k, v in voxels_cpu.items()}
        camera_matrix_gpu = camera_matrix_cpu.cuda()
        intrinsics_gpu = intrinsics_cpu.cuda()
        
        result_gpu = renderer_gpu(voxels_gpu, camera_matrix_gpu, intrinsics_gpu, (32, 32))
        
        # 检查结果形状一致
        assert result_cpu["rgb"].shape == result_gpu["rgb"].shape
        assert result_cpu["depth"].shape == result_gpu["depth"].shape
        assert result_cpu["alpha"].shape == result_gpu["alpha"].shape


def test_renderer_edge_cases():
    """测试渲染器边界情况"""
    config = SVRasterOneConfig()
    config.rendering.image_width = 1
    config.rendering.image_height = 1
    
    renderer = DifferentiableVoxelRasterizer(config)
    
    # 最小视口
    voxels = {
        "positions": torch.tensor([[0.0, 0.0, 1.0]]),
        "sizes": torch.tensor([0.1]),
        "densities": torch.tensor([0.5]),
        "colors": torch.tensor([[0.5, 0.5, 0.5]]),
    }
    
    camera_matrix = torch.eye(4)
    intrinsics = torch.eye(3)
    viewport_size = (1, 1)
    
    result = renderer(voxels, camera_matrix, intrinsics, viewport_size)
    
    assert result["rgb"].shape == (1, 1, 3)
    assert result["depth"].shape == (1, 1)
    assert result["alpha"].shape == (1, 1)


if __name__ == "__main__":
    pytest.main([__file__]) 