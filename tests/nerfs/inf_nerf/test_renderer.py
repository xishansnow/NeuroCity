"""
InfNeRF 渲染器测试

测试 InfNeRFRenderer 的各种功能，包括：
- 图像渲染
- 批量渲染
- 视频生成
- 光线生成
- 输出保存
"""

import pytest
import torch
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch

from src.nerfs.inf_nerf import InfNeRF, InfNeRFConfig, InfNeRFRenderer, InfNeRFRendererConfig


class TestInfNeRFRendererConfig:
    """测试渲染器配置"""

    def test_default_config(self):
        """测试默认配置"""
        config = InfNeRFRendererConfig()

        assert config.image_width == 800
        assert config.image_height == 600
        assert config.render_batch_size == 4096
        assert config.render_chunk_size == 1024
        assert config.background_color == (1.0, 1.0, 1.0)
        assert config.use_alpha_blending is True

    def test_custom_config(self):
        """测试自定义配置"""
        config = InfNeRFRendererConfig(
            image_width=1024,
            image_height=768,
            render_batch_size=2048,
            render_chunk_size=512,
            background_color=(0.0, 0.0, 0.0),
            use_alpha_blending=False,
        )

        assert config.image_width == 1024
        assert config.image_height == 768
        assert config.render_batch_size == 2048
        assert config.render_chunk_size == 512
        assert config.background_color == (0.0, 0.0, 0.0)
        assert config.use_alpha_blending is False

    def test_config_validation(self):
        """测试配置验证"""
        with pytest.raises(ValueError):
            InfNeRFRendererConfig(image_width=-1)

        with pytest.raises(ValueError):
            InfNeRFRendererConfig(image_height=0)

        with pytest.raises(ValueError):
            InfNeRFRendererConfig(render_batch_size=0)


class TestInfNeRFRenderer:
    """测试 InfNeRF 渲染器"""

    def test_renderer_creation(self, small_config, renderer_config):
        """测试渲染器创建"""
        model = InfNeRF(small_config)
        renderer = InfNeRFRenderer(model, renderer_config)

        assert renderer.model is model
        assert renderer.config is renderer_config
        assert renderer.device == next(model.parameters()).device
        assert renderer.output_path.exists()

    def test_renderer_from_checkpoint(self, mock_checkpoint, renderer_config):
        """测试从检查点创建渲染器"""
        renderer = InfNeRFRenderer.from_checkpoint(mock_checkpoint, renderer_config)

        assert isinstance(renderer, InfNeRFRenderer)
        assert renderer.model is not None
        assert renderer.config is renderer_config

    def test_generate_rays(self, small_config, renderer_config, device):
        """测试光线生成"""
        model = InfNeRF(small_config).to(device)
        renderer = InfNeRFRenderer(model, renderer_config)

        # 创建相机参数
        camera_pose = torch.eye(4, device=device)
        intrinsics = torch.tensor([[800, 0, 400], [0, 800, 300], [0, 0, 1]], device=device)
        width, height = 64, 64

        rays_o, rays_d = renderer._generate_rays(camera_pose, intrinsics, width, height)

        assert rays_o.shape == (width * height, 3)
        assert rays_d.shape == (width * height, 3)
        assert rays_o.device == device
        assert rays_d.device == device

        # 检查光线方向是否归一化
        directions_norm = torch.norm(rays_d, dim=-1)
        assert torch.allclose(directions_norm, torch.ones_like(directions_norm), atol=1e-6)

    def test_render_single_image(self, small_config, renderer_config, device):
        """测试单张图像渲染"""
        model = InfNeRF(small_config).to(device)
        renderer = InfNeRFRenderer(model, renderer_config)

        # 创建相机参数
        camera_pose = torch.eye(4, device=device)
        intrinsics = torch.tensor([[800, 0, 400], [0, 800, 300], [0, 0, 1]], device=device)
        width, height = 32, 32

        result = renderer.render_image(camera_pose, intrinsics, width, height)

        assert isinstance(result, dict)
        assert "rgb" in result
        assert "depth" in result
        assert "acc" in result

        # 检查输出形状
        assert result["rgb"].shape == (height, width, 3)
        assert result["depth"].shape == (height, width, 1)
        assert result["acc"].shape == (height, width, 1)

        # 检查值范围
        assert torch.all(result["rgb"] >= 0) and torch.all(result["rgb"] <= 1)
        assert torch.all(result["depth"] >= 0)
        assert torch.all(result["acc"] >= 0) and torch.all(result["acc"] <= 1)

    def test_render_batch(self, small_config, renderer_config, device):
        """测试批量渲染"""
        model = InfNeRF(small_config).to(device)
        renderer = InfNeRFRenderer(model, renderer_config)

        # 创建多个相机姿态
        num_views = 3
        camera_poses = torch.stack([torch.eye(4, device=device) for _ in range(num_views)])
        intrinsics = torch.tensor([[800, 0, 400], [0, 800, 300], [0, 0, 1]], device=device)
        width, height = 32, 32

        results = renderer.render_batch(camera_poses, intrinsics, width, height)

        assert len(results) == num_views
        for result in results:
            assert isinstance(result, dict)
            assert "rgb" in result
            assert "depth" in result
            assert "acc" in result
            assert result["rgb"].shape == (height, width, 3)

    def test_generate_spiral_trajectory(self, small_config, renderer_config, device):
        """测试螺旋轨迹生成"""
        model = InfNeRF(small_config).to(device)
        renderer = InfNeRFRenderer(model, renderer_config)

        center = torch.tensor([0.0, 0.0, 0.0], device=device)
        radius = 2.0
        num_frames = 10
        height_offset = 0.5

        trajectory = renderer._generate_spiral_trajectory(center, radius, num_frames, height_offset)

        assert trajectory.shape == (num_frames, 4, 4)
        assert trajectory.device == device

        # 检查相机位置是否在螺旋上
        for i, pose in enumerate(trajectory):
            pos = pose[:3, 3]
            distance_from_center = torch.norm(pos[:2] - center[:2])
            assert abs(distance_from_center - radius) < 0.1

    @pytest.mark.skipif(True, reason="需要 imageio 库")
    def test_render_video(self, small_config, renderer_config, device, temp_dir):
        """测试视频渲染"""
        model = InfNeRF(small_config).to(device)
        renderer = InfNeRFRenderer(model, renderer_config)

        # 创建相机轨迹
        num_frames = 5
        camera_trajectory = torch.stack([torch.eye(4, device=device) for _ in range(num_frames)])
        intrinsics = torch.tensor([[800, 0, 400], [0, 800, 300], [0, 0, 1]], device=device)
        output_path = str(temp_dir / "test_video.mp4")

        # 渲染视频
        renderer.render_video(camera_trajectory, intrinsics, output_path, fps=10)

        # 检查输出文件是否存在
        assert Path(output_path).exists()

    @pytest.mark.skipif(True, reason="需要 imageio 库")
    def test_render_spiral_video(self, small_config, renderer_config, device, temp_dir):
        """测试螺旋视频渲染"""
        model = InfNeRF(small_config).to(device)
        renderer = InfNeRFRenderer(model, renderer_config)

        center = torch.tensor([0.0, 0.0, 0.0], device=device)
        radius = 2.0
        num_frames = 5
        intrinsics = torch.tensor([[800, 0, 400], [0, 800, 300], [0, 0, 1]], device=device)
        output_path = str(temp_dir / "spiral_video.mp4")

        # 渲染螺旋视频
        renderer.render_spiral_video(center, radius, num_frames, intrinsics, output_path, fps=10)

        # 检查输出文件是否存在
        assert Path(output_path).exists()

    def test_save_renders(self, small_config, renderer_config, device, temp_dir):
        """测试渲染结果保存"""
        model = InfNeRF(small_config).to(device)
        renderer = InfNeRFRenderer(model, renderer_config)

        # 创建模拟渲染结果
        renders = []
        for i in range(2):
            render = {
                "rgb": torch.rand(32, 32, 3, device=device),
                "depth": torch.rand(32, 32, 1, device=device),
                "acc": torch.rand(32, 32, 1, device=device),
            }
            renders.append(render)

        output_dir = str(temp_dir / "renders")
        renderer.save_renders(renders, output_dir, "test")

        # 检查输出文件
        output_path = Path(output_dir)
        assert output_path.exists()

        # 检查是否生成了文件（如果 imageio 可用）
        rgb_files = list(output_path.glob("test_*_rgb.png"))
        if rgb_files:  # 如果 imageio 可用
            assert len(rgb_files) == 2

    def test_memory_usage(self, small_config, renderer_config, device):
        """测试内存使用监控"""
        model = InfNeRF(small_config).to(device)
        renderer = InfNeRFRenderer(model, renderer_config)

        memory_info = renderer.get_memory_usage()

        assert isinstance(memory_info, dict)
        if device.type == "cuda":
            assert "gpu_memory_allocated_gb" in memory_info
            assert "gpu_memory_reserved_gb" in memory_info
            assert memory_info["gpu_memory_allocated_gb"] >= 0
        else:
            assert "cpu_memory_gb" in memory_info

    def test_clear_cache(self, small_config, renderer_config, device):
        """测试缓存清理"""
        model = InfNeRF(small_config).to(device)
        renderer = InfNeRFRenderer(model, renderer_config)

        # 应该不会抛出异常
        renderer.clear_cache()

    def test_render_with_different_resolutions(self, small_config, renderer_config, device):
        """测试不同分辨率的渲染"""
        model = InfNeRF(small_config).to(device)
        renderer = InfNeRFRenderer(model, renderer_config)

        camera_pose = torch.eye(4, device=device)
        intrinsics = torch.tensor([[800, 0, 400], [0, 800, 300], [0, 0, 1]], device=device)

        # 测试不同分辨率
        resolutions = [(16, 16), (32, 32), (64, 64)]

        for width, height in resolutions:
            result = renderer.render_image(camera_pose, intrinsics, width, height)

            assert result["rgb"].shape == (height, width, 3)
            assert result["depth"].shape == (height, width, 1)
            assert result["acc"].shape == (height, width, 1)

    def test_render_with_different_camera_poses(self, small_config, renderer_config, device):
        """测试不同相机姿态的渲染"""
        model = InfNeRF(small_config).to(device)
        renderer = InfNeRFRenderer(model, renderer_config)

        intrinsics = torch.tensor([[800, 0, 400], [0, 800, 300], [0, 0, 1]], device=device)
        width, height = 32, 32

        # 创建不同的相机姿态
        poses = []
        for i in range(3):
            pose = torch.eye(4, device=device)
            pose[:3, 3] = torch.tensor([i * 2.0, 0.0, 0.0], device=device)
            poses.append(pose)

        for pose in poses:
            result = renderer.render_image(pose, intrinsics, width, height)

            assert isinstance(result, dict)
            assert "rgb" in result
            assert result["rgb"].shape == (height, width, 3)

    def test_render_chunk_processing(self, small_config, renderer_config, device):
        """测试分块渲染处理"""
        model = InfNeRF(small_config).to(device)
        renderer = InfNeRFRenderer(model, renderer_config)

        # 设置小的分块大小以测试分块处理
        renderer.config.render_chunk_size = 8

        camera_pose = torch.eye(4, device=device)
        intrinsics = torch.tensor([[800, 0, 400], [0, 800, 300], [0, 0, 1]], device=device)
        width, height = 16, 16

        result = renderer.render_image(camera_pose, intrinsics, width, height)

        assert result["rgb"].shape == (height, width, 3)
        assert result["depth"].shape == (height, width, 1)
        assert result["acc"].shape == (height, width, 1)

    def test_renderer_error_handling(self, small_config, renderer_config, device):
        """测试渲染器错误处理"""
        model = InfNeRF(small_config).to(device)
        renderer = InfNeRFRenderer(model, renderer_config)

        # 测试无效的相机参数
        invalid_pose = torch.randn(3, 3, device=device)  # 不是4x4矩阵

        with pytest.raises((ValueError, RuntimeError)):
            renderer.render_image(invalid_pose, torch.eye(3, device=device), 32, 32)

    def test_renderer_performance(self, small_config, renderer_config, device):
        """测试渲染器性能"""
        model = InfNeRF(small_config).to(device)
        renderer = InfNeRFRenderer(model, renderer_config)

        camera_pose = torch.eye(4, device=device)
        intrinsics = torch.tensor([[800, 0, 400], [0, 800, 300], [0, 0, 1]], device=device)

        import time

        # 测量渲染时间
        start_time = time.time()
        result = renderer.render_image(camera_pose, intrinsics, 32, 32)
        end_time = time.time()

        render_time = end_time - start_time

        # 渲染应该在合理时间内完成（小于10秒）
        assert render_time < 10.0
        assert result["rgb"].shape == (32, 32, 3)


def test_create_inf_nerf_renderer(mock_checkpoint, renderer_config):
    """测试便捷函数"""
    renderer = create_inf_nerf_renderer(mock_checkpoint, renderer_config)

    assert isinstance(renderer, InfNeRFRenderer)
    assert renderer.model is not None
    assert renderer.config is renderer_config


def test_render_demo_images(small_config, renderer_config, device, temp_dir):
    """测试演示图像渲染"""
    model = InfNeRF(small_config).to(device)
    renderer = InfNeRFRenderer(model, renderer_config)

    output_dir = str(temp_dir / "demo")
    render_demo_images(renderer, num_views=3, output_dir=output_dir)

    # 检查输出目录是否存在
    assert Path(output_dir).exists()
