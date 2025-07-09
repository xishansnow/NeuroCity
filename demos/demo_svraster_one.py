#!/usr/bin/env python3
"""
SVRaster One 演示脚本

展示可微分光栅化渲染器的功能：
1. 稀疏体素网格初始化
2. 可微分光栅化渲染
3. 端到端训练
4. 自适应优化
"""

import sys
import os
import torch
import numpy as np
import time
from pathlib import Path

# 添加 src 目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# 添加 svraster_one 目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent / "svraster_one"))

from svraster_one import SVRasterOne, SVRasterOneConfig, SVRasterOneTrainer


def create_test_data(num_samples: int = 100, image_size: tuple = (256, 256)):
    """创建测试数据"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 创建相机参数
    camera_matrices = []
    intrinsics_list = []
    target_images = []

    for i in range(num_samples):
        # 随机相机位置
        angle = 2 * np.pi * i / num_samples
        radius = 3.0
        camera_pos = torch.tensor(
            [radius * np.cos(angle), radius * np.sin(angle), 2.0], dtype=torch.float32
        )

        # 构建相机矩阵
        look_at = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
        up = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)

        forward = (look_at - camera_pos) / torch.norm(look_at - camera_pos)
        right = torch.cross(forward, up)
        up = torch.cross(right, forward)

        rotation = torch.stack([right, up, forward], dim=1)
        translation = -torch.matmul(rotation.T, camera_pos)

        camera_matrix = torch.eye(4, dtype=torch.float32)
        camera_matrix[:3, :3] = rotation.T
        camera_matrix[:3, 3] = translation

        camera_matrices.append(camera_matrix)

        # 相机内参
        fx = fy = 500.0
        cx, cy = image_size[0] / 2, image_size[1] / 2
        intrinsics = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=torch.float32)
        intrinsics_list.append(intrinsics)

        # 创建简单的目标图像（彩色圆环）
        target_image = torch.zeros(image_size[0], image_size[1], 3, dtype=torch.float32)

        # 在图像中心绘制彩色圆环
        center_x, center_y = image_size[0] // 2, image_size[1] // 2
        for x in range(image_size[0]):
            for y in range(image_size[1]):
                dx = x - center_x
                dy = y - center_y
                distance = np.sqrt(dx**2 + dy**2)

                if 50 < distance < 100:
                    # 根据角度设置颜色
                    angle_pixel = np.arctan2(dy, dx)
                    r = (np.sin(angle_pixel) + 1) / 2
                    g = (np.cos(angle_pixel) + 1) / 2
                    b = 0.5
                    target_image[x, y] = torch.tensor([r, g, b], dtype=torch.float32)

        target_images.append(target_image)

    return {
        "camera_matrices": torch.stack(camera_matrices).to(device),
        "intrinsics": torch.stack(intrinsics_list).to(device),
        "target_images": torch.stack(target_images).to(device),
    }


def demo_basic_rendering():
    """演示基础渲染功能"""
    print("=== SVRaster One 基础渲染演示 ===")

    # 创建配置
    config = SVRasterOneConfig()
    config.rendering.image_width = 256
    config.rendering.image_height = 256
    config.voxel.grid_resolution = 64
    config.voxel.max_voxels = 10000

    # 创建模型
    model = SVRasterOne(config)
    print(f"模型设备: {model.device}")

    # 创建测试数据
    test_data = create_test_data(num_samples=1, image_size=(256, 256))

    # 渲染
    print("开始渲染...")
    start_time = time.time()

    with torch.no_grad():
        rendered_output = model.forward(
            test_data["camera_matrices"][0], test_data["intrinsics"][0], mode="inference"
        )

    render_time = time.time() - start_time
    print(f"渲染完成，耗时: {render_time:.4f}秒")

    # 显示结果
    print(f"渲染图像尺寸: {rendered_output['rgb'].shape}")
    print(f"体素统计: {rendered_output['voxel_stats']}")

    return model, test_data


def demo_differentiable_rendering():
    """演示可微分渲染功能"""
    print("\n=== SVRaster One 可微分渲染演示 ===")

    # 创建配置
    config = SVRasterOneConfig()
    config.rendering.image_width = 128
    config.rendering.image_height = 128
    config.voxel.grid_resolution = 32
    config.voxel.max_voxels = 5000
    config.rendering.soft_rasterization = True
    config.rendering.use_soft_sorting = True

    # 创建模型
    model = SVRasterOne(config)

    # 创建测试数据
    test_data = create_test_data(num_samples=1, image_size=(128, 128))

    # 启用梯度计算
    for param in model.parameters():
        param.requires_grad_(True)

    # 可微分渲染
    print("开始可微分渲染...")
    start_time = time.time()

    rendered_output = model.forward(
        test_data["camera_matrices"][0], test_data["intrinsics"][0], mode="training"
    )

    # 计算损失
    target_data = {"rgb": test_data["target_images"][0]}
    losses = model.compute_loss(rendered_output, target_data)

    # 反向传播
    losses["total_loss"].backward()

    render_time = time.time() - start_time
    print(f"可微分渲染完成，耗时: {render_time:.4f}秒")

    # 显示结果
    print(f"总损失: {losses['total_loss'].item():.6f}")
    print(f"RGB损失: {losses.get('rgb_loss', 0).item():.6f}")

    # 检查梯度
    has_gradients = any(p.grad is not None for p in model.parameters())
    print(f"梯度计算: {'成功' if has_gradients else '失败'}")

    return model, test_data


def demo_training():
    """演示训练功能"""
    print("\n=== SVRaster One 训练演示 ===")

    # 创建配置
    config = SVRasterOneConfig()
    config.rendering.image_width = 128
    config.rendering.image_height = 128
    config.voxel.grid_resolution = 32
    config.voxel.max_voxels = 5000
    config.training.num_epochs = 10
    config.training.batch_size = 4
    config.training.learning_rate = 1e-3

    # 创建模型
    model = SVRasterOne(config)

    # 创建训练器
    trainer = SVRasterOneTrainer(model, config)

    # 创建训练数据
    train_data = create_test_data(num_samples=20, image_size=(128, 128))

    # 创建数据加载器（简化版本）
    class SimpleDataset:
        def __init__(self, data, num_samples):
            self.data = data
            self.num_samples = num_samples

        def __len__(self):
            return self.num_samples

        def __getitem__(self, idx):
            return {
                "camera_matrix": self.data["camera_matrices"][
                    idx % len(self.data["camera_matrices"])
                ],
                "intrinsics": self.data["intrinsics"][idx % len(self.data["intrinsics"])],
                "target": {
                    "rgb": self.data["target_images"][idx % len(self.data["target_images"])]
                },
            }

    from torch.utils.data import DataLoader

    dataset = SimpleDataset(train_data, 20)
    train_loader = DataLoader(dataset, batch_size=4, shuffle=True)

    # 开始训练
    print("开始训练...")
    start_time = time.time()

    # 训练几个 epoch
    for epoch in range(3):
        train_stats = trainer.train_epoch(train_loader, epoch)
        print(f"Epoch {epoch}: Loss = {train_stats['avg_loss']:.6f}")

    train_time = time.time() - start_time
    print(f"训练完成，总耗时: {train_time:.2f}秒")

    # 显示训练统计
    stats = trainer.get_training_stats()
    print(f"最佳损失: {stats['best_loss']:.6f}")
    print(f"模型信息: {stats['model_info']}")

    return model, trainer


def demo_adaptive_optimization():
    """演示自适应优化功能"""
    print("\n=== SVRaster One 自适应优化演示 ===")

    # 创建配置
    config = SVRasterOneConfig()
    config.rendering.image_width = 128
    config.rendering.image_height = 128
    config.voxel.grid_resolution = 32
    config.voxel.max_voxels = 10000
    config.voxel.adaptive_subdivision = True
    config.voxel.subdivision_threshold = 0.1

    # 创建模型
    model = SVRasterOne(config)

    # 显示初始状态
    initial_stats = model.voxel_grid.get_stats()
    print(f"初始体素数量: {initial_stats['active_voxels']}")

    # 创建测试数据
    test_data = create_test_data(num_samples=1, image_size=(128, 128))

    # 模拟训练过程，触发自适应优化
    print("模拟训练过程...")
    for step in range(5):
        # 前向传播
        rendered_output = model.forward(
            test_data["camera_matrices"][0], test_data["intrinsics"][0], mode="training"
        )

        # 计算损失
        target_data = {"rgb": test_data["target_images"][0]}
        losses = model.compute_loss(rendered_output, target_data)

        # 反向传播
        losses["total_loss"].backward()

        # 获取梯度幅度
        gradient_magnitudes = torch.abs(model.voxel_grid.voxel_features.grad[:, 0])

        # 自适应优化
        model.adaptive_optimization(gradient_magnitudes)

        # 清除梯度
        model.optimizer.zero_grad()

        print(f"Step {step}: Loss = {losses['total_loss'].item():.6f}")

    # 显示最终状态
    final_stats = model.voxel_grid.get_stats()
    print(f"最终体素数量: {final_stats['active_voxels']}")
    print(f"细分次数: {final_stats['subdivision_count']}")
    print(f"剪枝次数: {final_stats['pruning_count']}")

    return model


def demo_memory_optimization():
    """演示内存优化功能"""
    print("\n=== SVRaster One 内存优化演示 ===")

    # 创建配置
    config = SVRasterOneConfig()
    config.rendering.image_width = 256
    config.rendering.image_height = 256
    config.voxel.grid_resolution = 64
    config.voxel.max_voxels = 50000

    # 创建模型
    model = SVRasterOne(config)

    # 显示内存使用情况
    memory_usage = model.get_memory_usage()
    print(f"初始内存使用: {memory_usage['total_memory_mb']:.2f} MB")
    print(f"活跃内存: {memory_usage['active_memory_mb']:.2f} MB")
    print(f"内存效率: {memory_usage['memory_efficiency']:.2%}")

    # 内存优化
    print("执行内存优化...")
    model.optimize_memory(target_memory_mb=50.0)

    # 显示优化后的内存使用情况
    memory_usage_after = model.get_memory_usage()
    print(f"优化后内存使用: {memory_usage_after['total_memory_mb']:.2f} MB")
    print(f"优化后活跃内存: {memory_usage_after['active_memory_mb']:.2f} MB")
    print(f"优化后内存效率: {memory_usage_after['memory_efficiency']:.2%}")

    return model


def main():
    """主函数"""
    print("SVRaster One 可微分光栅化渲染器演示")
    print("=" * 50)

    # 检查 CUDA 可用性
    if torch.cuda.is_available():
        print(f"CUDA 可用: {torch.cuda.get_device_name()}")
    else:
        print("使用 CPU 模式")

    try:
        # 基础渲染演示
        model1, test_data1 = demo_basic_rendering()

        # 可微分渲染演示
        model2, test_data2 = demo_differentiable_rendering()

        # 训练演示
        model3, trainer = demo_training()

        # 自适应优化演示
        model4 = demo_adaptive_optimization()

        # 内存优化演示
        model5 = demo_memory_optimization()

        print("\n" + "=" * 50)
        print("所有演示完成！")
        print("SVRaster One 成功实现了可微分光栅化渲染的端到端训练。")

    except Exception as e:
        print(f"演示过程中出现错误: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
