"""
InfNeRF 使用示例

展示重构后的 InfNeRF 的完整使用流程，包括训练和渲染。
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path

from .core import InfNeRF, InfNeRFConfig
from .trainer import InfNeRFTrainer, InfNeRFTrainerConfig
from .renderer import InfNeRFRenderer, InfNeRFRendererConfig, render_demo_images
from .utils.volume_renderer import VolumeRenderer, VolumeRendererConfig


def create_synthetic_dataset(num_images=100, image_size=64):
    """创建合成数据集用于演示"""
    dataset = []

    for i in range(num_images):
        # 生成随机相机参数
        angle = 2 * np.pi * i / num_images
        radius = 3.0
        height = 1.0

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
        rays_o, rays_d = generate_rays(camera_matrix, image_size, image_size)

        # 生成目标图像（简单的球体）
        target_rgb = render_simple_sphere(rays_o, rays_d)

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


def generate_rays(camera_matrix, width, height):
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


def render_simple_sphere(rays_o, rays_d):
    """渲染简单球体作为目标"""
    # 球体参数
    sphere_center = np.array([0.0, 0.0, 0.0])
    sphere_radius = 1.0

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


def demo_training():
    """演示训练流程"""
    print("=== InfNeRF 训练演示 ===")

    # 1. 创建模型
    config = InfNeRFConfig(
        max_depth=6,  # 较小的八叉树深度用于演示
        hidden_dim=32,  # 较小的隐藏层维度
        num_samples=32,  # 较少的采样点
        scene_bound=2.0,  # 较小的场景边界
    )
    model = InfNeRF(config)

    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")

    # 2. 创建数据集
    print("创建合成数据集...")
    dataset = create_synthetic_dataset(num_images=50, image_size=32)

    # 3. 创建训练器
    trainer_config = InfNeRFTrainerConfig(
        num_epochs=5,  # 少量轮数用于演示
        lr_init=1e-3,
        rays_batch_size=1024,
        mixed_precision=False,  # 关闭混合精度以简化
    )
    trainer = InfNeRFTrainer(model, dataset, trainer_config)

    # 4. 开始训练
    print("开始训练...")
    trainer.train()

    return model, trainer


def demo_rendering(model):
    """演示渲染流程"""
    print("\n=== InfNeRF 渲染演示 ===")

    # 1. 创建渲染器
    renderer_config = InfNeRFRendererConfig(
        image_width=128, image_height=128, render_chunk_size=512, save_depth=True, save_alpha=True
    )
    renderer = InfNeRFRenderer(model, renderer_config)

    # 2. 渲染单张图像
    print("渲染单张图像...")

    # 创建相机参数
    camera_pose = torch.eye(4)
    camera_pose[:3, 3] = torch.tensor([2.0, 0.0, 1.0])

    intrinsics = torch.tensor([[800, 0, 64], [0, 800, 64], [0, 0, 1]])

    result = renderer.render_image(camera_pose, intrinsics)
    print(f"渲染结果形状: RGB={result['rgb'].shape}, 深度={result['depth'].shape}")

    # 3. 渲染演示图像
    print("渲染演示图像...")
    render_demo_images(renderer, num_views=8, output_dir="demo_outputs")

    return renderer


def demo_volume_renderer():
    """演示体积渲染器"""
    print("\n=== 体积渲染器演示 ===")

    # 1. 创建体积渲染器
    volume_config = VolumeRendererConfig(num_samples=64, perturb=True, white_background=False)
    volume_renderer = VolumeRenderer(volume_config)

    # 2. 生成测试数据
    batch_size = 100
    num_samples = 64

    rays_o = torch.randn(batch_size, 3)
    rays_d = torch.randn(batch_size, 3)
    rays_d = rays_d / np.linalg.norm(rays_d, axis=-1, keepdims=True)

    # 采样点
    z_vals, pts = volume_renderer.sample_rays(
        rays_o, rays_d, near=0.1, far=10.0, num_samples=num_samples
    )

    # 模拟 NeRF 输出
    colors = torch.rand(batch_size, num_samples, 3)
    densities = torch.rand(batch_size, num_samples, 1)

    # 3. 体积渲染
    result = volume_renderer.volume_render(colors, densities, z_vals, rays_d)

    print(f"体积渲染结果: RGB={result['rgb'].shape}, 深度={result['depth'].shape}")

    # 4. 计算损失
    targets = {"target_rgb": torch.rand_like(result["rgb"])}

    losses = volume_renderer.compute_losses(result, targets)
    print(f"损失: {losses['total_loss'].item():.4f}")

    return volume_renderer


def main():
    """主演示函数"""
    print("InfNeRF 重构演示")
    print("=" * 50)

    try:
        # 演示训练
        model, trainer = demo_training()

        # 演示渲染
        renderer = demo_rendering(model)

        # 演示体积渲染器
        volume_renderer = demo_volume_renderer()

        print("\n=== 演示完成 ===")
        print("输出文件保存在 'demo_outputs' 目录中")

    except Exception as e:
        print(f"演示过程中出现错误: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
