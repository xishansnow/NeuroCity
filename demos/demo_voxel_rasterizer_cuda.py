#!/usr/bin/env python3
"""
CUDA VoxelRasterizer 演示脚本

展示如何使用 GPU 加速的体素光栅化渲染器，
包括性能对比和功能演示。
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import time
import logging
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from nerfs.svraster.cuda.voxel_rasterizer_gpu import VoxelRasterizerGPU, benchmark_rasterizer
from nerfs.svraster.voxel_rasterizer import VoxelRasterizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleConfig:
    """简单的配置类"""

    def __init__(self):
        self.near_plane = 0.1
        self.far_plane = 100.0
        self.background_color = [0.1, 0.1, 0.1]
        self.density_activation = "exp"
        self.color_activation = "sigmoid"
        self.sh_degree = 2


def generate_test_scene(num_voxels=10000):
    """生成测试场景"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 生成体素位置（在一个球体内）
    theta = torch.rand(num_voxels) * 2 * np.pi
    phi = torch.acos(2 * torch.rand(num_voxels) - 1)
    r = torch.rand(num_voxels) ** (1 / 3) * 0.8  # 球体分布

    x = r * torch.sin(phi) * torch.cos(theta)
    y = r * torch.sin(phi) * torch.sin(theta)
    z = r * torch.cos(phi)

    positions = torch.stack([x, y, z], dim=1).to(device)

    # 生成体素大小（基于距离中心的距离）
    distances = torch.norm(positions, dim=1)
    sizes = (0.05 + 0.05 * distances).to(device)

    # 生成密度（基于距离）
    densities = torch.exp(-distances * 2).to(device)

    # 生成颜色（基于位置）
    colors = torch.zeros(num_voxels, 3, device=device)
    colors[:, 0] = (positions[:, 0] + 1) / 2  # 红色分量
    colors[:, 1] = (positions[:, 1] + 1) / 2  # 绿色分量
    colors[:, 2] = (positions[:, 2] + 1) / 2  # 蓝色分量

    return {"positions": positions, "sizes": sizes, "densities": densities, "colors": colors}


def generate_camera_params(viewport_size=(800, 600)):
    """生成相机参数"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    width, height = viewport_size

    # 相机位置
    camera_distance = 2.0
    camera_matrix = torch.eye(4, device=device)
    camera_matrix[2, 3] = camera_distance

    # 相机内参
    focal_length = min(width, height) * 0.8
    cx, cy = width / 2, height / 2

    intrinsics = torch.tensor(
        [[focal_length, 0.0, cx], [0.0, focal_length, cy], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
        device=device,
    )

    return camera_matrix, intrinsics


def compare_cpu_gpu_performance():
    """比较 CPU 和 GPU 版本的性能"""
    logger.info("开始性能对比测试...")

    # 配置
    config = SimpleConfig()
    viewport_size = (800, 600)
    num_voxels_list = [1000, 5000, 10000, 20000]

    # 生成测试数据
    camera_matrix, intrinsics = generate_camera_params(viewport_size)

    cpu_times = []
    gpu_times = []

    for num_voxels in num_voxels_list:
        logger.info(f"测试 {num_voxels} 个体素...")

        # 生成体素数据
        voxels = generate_test_scene(num_voxels)

        # CPU 测试
        try:
            cpu_rasterizer = VoxelRasterizer(config)

            start_time = time.time()
            cpu_result = cpu_rasterizer(voxels, camera_matrix, intrinsics, viewport_size)
            cpu_time = time.time() - start_time

            cpu_times.append(cpu_time)
            logger.info(f"CPU 渲染时间: {cpu_time:.4f}s")
        except Exception as e:
            logger.warning(f"CPU 渲染失败: {e}")
            cpu_times.append(float("inf"))

        # GPU 测试
        try:
            gpu_rasterizer = VoxelRasterizerGPU(config)

            start_time = time.time()
            gpu_result = gpu_rasterizer(voxels, camera_matrix, intrinsics, viewport_size)
            gpu_time = time.time() - start_time

            gpu_times.append(gpu_time)
            logger.info(f"GPU 渲染时间: {gpu_time:.4f}s")
        except Exception as e:
            logger.warning(f"GPU 渲染失败: {e}")
            gpu_times.append(float("inf"))

    # 绘制性能对比图
    plt.figure(figsize=(10, 6))

    plt.subplot(1, 2, 1)
    plt.plot(num_voxels_list, cpu_times, "b-o", label="CPU", linewidth=2, markersize=8)
    plt.plot(num_voxels_list, gpu_times, "r-s", label="GPU", linewidth=2, markersize=8)
    plt.xlabel("体素数量")
    plt.ylabel("渲染时间 (秒)")
    plt.title("CPU vs GPU 渲染时间对比")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale("log")

    # 计算加速比
    speedups = [cpu / gpu if gpu != float("inf") else 0 for cpu, gpu in zip(cpu_times, gpu_times)]

    plt.subplot(1, 2, 2)
    plt.bar(range(len(num_voxels_list)), speedups, color="green", alpha=0.7)
    plt.xlabel("体素数量")
    plt.ylabel("加速比")
    plt.title("GPU 加速比")
    plt.xticks(range(len(num_voxels_list)), [str(x) for x in num_voxels_list])
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("voxel_rasterizer_performance_comparison.png", dpi=300, bbox_inches="tight")
    plt.show()

    return cpu_times, gpu_times, speedups


def demo_rendering_quality():
    """演示渲染质量"""
    logger.info("开始渲染质量演示...")

    # 配置
    config = SimpleConfig()
    viewport_size = (800, 600)
    num_voxels = 15000

    # 生成场景
    voxels = generate_test_scene(num_voxels)
    camera_matrix, intrinsics = generate_camera_params(viewport_size)

    # GPU 渲染
    try:
        gpu_rasterizer = VoxelRasterizerGPU(config)
        gpu_result = gpu_rasterizer(voxels, camera_matrix, intrinsics, viewport_size)

        # 显示结果
        rgb_image = gpu_result["rgb"].cpu().numpy()
        depth_image = gpu_result["depth"].cpu().numpy()

        plt.figure(figsize=(15, 5))

        # RGB 图像
        plt.subplot(1, 3, 1)
        plt.imshow(rgb_image)
        plt.title("RGB 渲染结果")
        plt.axis("off")

        # 深度图
        plt.subplot(1, 3, 2)
        plt.imshow(depth_image, cmap="viridis")
        plt.title("深度图")
        plt.axis("off")
        plt.colorbar()

        # 体素分布可视化
        plt.subplot(1, 3, 3)
        positions = voxels["positions"].cpu().numpy()
        plt.scatter(
            positions[:, 0],
            positions[:, 2],
            c=voxels["colors"].cpu().numpy(),
            s=voxels["sizes"].cpu().numpy() * 100,
            alpha=0.6,
        )
        plt.title("体素分布 (X-Z 平面)")
        plt.xlabel("X")
        plt.ylabel("Z")
        plt.axis("equal")
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("voxel_rasterizer_rendering_demo.png", dpi=300, bbox_inches="tight")
        plt.show()

        logger.info("渲染质量演示完成")

    except Exception as e:
        logger.error(f"渲染质量演示失败: {e}")


def demo_interactive_camera():
    """演示交互式相机控制"""
    logger.info("开始交互式相机演示...")

    # 配置
    config = SimpleConfig()
    viewport_size = (600, 400)
    num_voxels = 8000

    # 生成场景
    voxels = generate_test_scene(num_voxels)

    # 创建多个视角
    angles = np.linspace(0, 2 * np.pi, 8)
    camera_distance = 2.0

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    for i, angle in enumerate(angles):
        # 计算相机位置
        x = camera_distance * np.cos(angle)
        z = camera_distance * np.sin(angle)

        # 构建相机矩阵
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        camera_matrix = torch.eye(4, device=device)
        camera_matrix[0, 3] = x
        camera_matrix[2, 3] = z

        # 让相机始终看向原点
        look_at = torch.tensor([0.0, 0.0, 0.0], device=device)
        camera_pos = torch.tensor([x, 0.0, z], device=device)

        forward = (look_at - camera_pos) / torch.norm(look_at - camera_pos)
        up = torch.tensor([0.0, 1.0, 0.0], device=device)
        right = torch.cross(forward, up)
        up = torch.cross(right, forward)

        camera_matrix[:3, 0] = right
        camera_matrix[:3, 1] = up
        camera_matrix[:3, 2] = forward

        # 生成内参
        width, height = viewport_size
        focal_length = min(width, height) * 0.8
        cx, cy = width / 2, height / 2

        intrinsics = torch.tensor(
            [[focal_length, 0.0, cx], [0.0, focal_length, cy], [0.0, 0.0, 1.0]],
            dtype=torch.float32,
            device=device,
        )

        try:
            # 渲染
            gpu_rasterizer = VoxelRasterizerGPU(config)
            result = gpu_rasterizer(voxels, camera_matrix, intrinsics, viewport_size)

            # 显示
            rgb_image = result["rgb"].cpu().numpy()
            axes[i].imshow(rgb_image)
            axes[i].set_title(f"角度: {angle:.1f}°")
            axes[i].axis("off")

        except Exception as e:
            logger.warning(f"视角 {i} 渲染失败: {e}")
            axes[i].text(
                0.5, 0.5, "渲染失败", ha="center", va="center", transform=axes[i].transAxes
            )
            axes[i].axis("off")

    plt.tight_layout()
    plt.savefig("voxel_rasterizer_interactive_camera.png", dpi=300, bbox_inches="tight")
    plt.show()

    logger.info("交互式相机演示完成")


def run_benchmark():
    """运行性能基准测试"""
    logger.info("开始性能基准测试...")

    try:
        results = benchmark_rasterizer(
            num_voxels=10000, viewport_size=(800, 600), num_iterations=50
        )

        logger.info("基准测试结果:")
        for key, value in results.items():
            logger.info(f"  {key}: {value:.4f}")

        return results

    except Exception as e:
        logger.error(f"基准测试失败: {e}")
        return {}


def main():
    """主函数"""
    logger.info("CUDA VoxelRasterizer 演示开始")

    # 检查 CUDA 可用性
    if torch.cuda.is_available():
        logger.info(f"CUDA 可用: {torch.cuda.get_device_name(0)}")
        logger.info(
            f"CUDA 版本: {torch.version.cuda if hasattr(torch.version, 'cuda') else 'Unknown'}"
        )
    else:
        logger.warning("CUDA 不可用，将使用 CPU 版本")

    # 运行各种演示
    try:
        # 1. 性能对比
        logger.info("=" * 50)
        logger.info("1. 性能对比测试")
        cpu_times, gpu_times, speedups = compare_cpu_gpu_performance()

        # 2. 渲染质量演示
        logger.info("=" * 50)
        logger.info("2. 渲染质量演示")
        demo_rendering_quality()

        # 3. 交互式相机演示
        logger.info("=" * 50)
        logger.info("3. 交互式相机演示")
        demo_interactive_camera()

        # 4. 性能基准测试
        logger.info("=" * 50)
        logger.info("4. 性能基准测试")
        benchmark_results = run_benchmark()

        logger.info("=" * 50)
        logger.info("所有演示完成！")

        # 保存结果摘要
        with open("voxel_rasterizer_demo_summary.txt", "w") as f:
            f.write("CUDA VoxelRasterizer 演示结果摘要\n")
            f.write("=" * 40 + "\n\n")

            f.write("性能对比结果:\n")
            f.write(f"CPU 渲染时间: {cpu_times}\n")
            f.write(f"GPU 渲染时间: {gpu_times}\n")
            f.write(f"GPU 加速比: {speedups}\n\n")

            f.write("基准测试结果:\n")
            for key, value in benchmark_results.items():
                f.write(f"{key}: {value:.4f}\n")

        logger.info("结果已保存到 voxel_rasterizer_demo_summary.txt")

    except Exception as e:
        logger.error(f"演示过程中发生错误: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
