"""
InfNeRF 测试配置和夹具

提供测试所需的通用配置、模拟数据和测试夹具。
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import tempfile
import shutil

from src.nerfs.inf_nerf import (
    InfNeRF,
    InfNeRFConfig,
    InfNeRFTrainer,
    InfNeRFTrainerConfig,
    InfNeRFRenderer,
    InfNeRFRendererConfig,
    VolumeRenderer,
    VolumeRendererConfig,
)


@pytest.fixture(scope="session")
def device():
    """获取测试设备"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


@pytest.fixture(scope="session")
def small_config():
    """小型配置，用于快速测试"""
    return InfNeRFConfig(
        max_depth=4,  # 较小的八叉树深度
        hidden_dim=32,  # 较小的隐藏层维度
        num_samples=16,  # 较少的采样点
        scene_bound=1.0,  # 较小的场景边界
        grid_size=512,  # 较小的网格大小
        log2_hashmap_size=16,  # 较小的哈希表
    )


@pytest.fixture(scope="session")
def medium_config():
    """中等配置，用于标准测试"""
    return InfNeRFConfig(
        max_depth=6,
        hidden_dim=64,
        num_samples=32,
        scene_bound=2.0,
        grid_size=1024,
        log2_hashmap_size=18,
    )


@pytest.fixture(scope="session")
def trainer_config():
    """训练器配置"""
    return InfNeRFTrainerConfig(
        num_epochs=2,  # 少量轮数用于测试
        lr_init=1e-3,
        rays_batch_size=512,
        mixed_precision=False,  # 关闭混合精度以简化测试
        save_freq=100,
        eval_freq=50,
    )


@pytest.fixture(scope="session")
def renderer_config():
    """渲染器配置"""
    return InfNeRFRendererConfig(
        image_width=64, image_height=64, render_chunk_size=256, save_depth=True, save_alpha=True
    )


@pytest.fixture(scope="session")
def volume_renderer_config():
    """体积渲染器配置"""
    return VolumeRendererConfig(
        num_samples=32, num_importance_samples=64, perturb=True, white_background=False
    )


@pytest.fixture
def temp_dir():
    """临时目录夹具"""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def synthetic_dataset():
    """合成数据集"""
    dataset = []
    num_images = 10
    image_size = 32

    for i in range(num_images):
        # 生成随机相机参数
        angle = 2 * np.pi * i / num_images
        radius = 2.0
        height = 0.5

        # 相机位置
        camera_pos = np.array([radius * np.cos(angle), radius * np.sin(angle), height])

        # 相机朝向原点
        look_at = np.array([0.0, 0.0, 0.0])
        up = np.array([0.0, 0.0, 1.0])

        # 构建相机矩阵
        forward = (look_at - camera_pos) / np.linalg.norm(look_at - camera_pos)
        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)
        up = np.cross(right, forward)

        camera_matrix = np.eye(4)
        camera_matrix[:3, 0] = right
        camera_matrix[:3, 1] = up
        camera_matrix[:3, 2] = forward
        camera_matrix[:3, 3] = camera_pos

        # 生成光线
        rays_o, rays_d = _generate_rays(camera_matrix, image_size, image_size)

        # 生成目标图像（简单的球体）
        target_rgb = _render_simple_sphere(rays_o, rays_d)

        dataset.append(
            {
                "rays_o": torch.from_numpy(rays_o).float(),
                "rays_d": torch.from_numpy(rays_d).float(),
                "target_rgb": torch.from_numpy(target_rgb).float(),
                "near": torch.tensor(0.1),
                "far": torch.tensor(10.0),
                "focal_length": torch.tensor(800.0),
                "pixel_width": torch.tensor(1.0 / image_size),
            }
        )

    return dataset


def _generate_rays(camera_matrix, width, height):
    """生成光线"""
    # 创建像素坐标网格
    i, j = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")

    # 相机内参（简化）
    fx = fy = 800.0
    cx, cy = width / 2, height / 2

    # 转换为相机坐标
    directions = np.stack([(j - cx) / fx, (i - cy) / fy, np.ones_like(i)], axis=-1)

    # 转换到世界坐标
    directions = directions @ camera_matrix[:3, :3].T
    origins = camera_matrix[:3, -1][None, None, :].expand_as(directions)

    # 扁平化
    rays_o = origins.reshape(-1, 3)
    rays_d = directions.reshape(-1, 3)
    rays_d = rays_d / np.linalg.norm(rays_d, axis=-1, keepdims=True)

    return rays_o, rays_d


def _render_simple_sphere(rays_o, rays_d):
    """渲染简单球体作为目标"""
    # 球体参数
    sphere_center = np.array([0.0, 0.0, 0.0])
    sphere_radius = 0.5

    # 计算光线与球体的交点
    oc = rays_o - sphere_center
    a = np.sum(rays_d * rays_d, axis=-1)
    b = 2.0 * np.sum(oc * rays_d, axis=-1)
    c = np.sum(oc * oc, axis=-1) - sphere_radius * sphere_radius

    discriminant = b * b - 4 * a * c
    hit = discriminant > 0

    # 计算交点
    t = (-b - np.sqrt(discriminant)) / (2 * a)
    t = np.where(hit, t, 0)

    # 计算法向量
    hit_points = rays_o + t[:, None] * rays_d
    normals = (hit_points - sphere_center) / sphere_radius

    # 简单的着色
    light_dir = np.array([1.0, 1.0, 1.0])
    light_dir = light_dir / np.linalg.norm(light_dir)

    diffuse = np.maximum(0, np.sum(normals * light_dir, axis=-1))
    ambient = 0.3

    intensity = ambient + 0.7 * diffuse
    intensity = np.where(hit, intensity, 0.0)

    # 转换为RGB
    rgb = np.stack([intensity] * 3, axis=-1)
    return rgb


@pytest.fixture
def mock_checkpoint(temp_dir):
    """模拟检查点文件"""
    checkpoint_path = temp_dir / "test_checkpoint.pth"

    # 创建简单的模型和检查点
    config = InfNeRFConfig(max_depth=4, hidden_dim=32)
    model = InfNeRF(config)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_config": config,
        "epoch": 0,
        "global_step": 0,
    }

    torch.save(checkpoint, checkpoint_path)
    return str(checkpoint_path)


# 测试标记
pytest_plugins = []


def pytest_configure(config):
    """配置测试标记"""
    config.addinivalue_line("markers", "slow: 标记为慢速测试")
    config.addinivalue_line("markers", "gpu: 需要 GPU 的测试")
    config.addinivalue_line("markers", "integration: 集成测试")
    config.addinivalue_line("markers", "performance: 性能测试")
