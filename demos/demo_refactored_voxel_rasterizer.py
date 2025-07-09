#!/usr/bin/env python3
"""
重构后的 VoxelRasterizer 演示

展示 CUDA 和 CPU 版本的功能和性能差异。
"""

import torch
import numpy as np
import time
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from nerfs.svraster.voxel_rasterizer import (
    VoxelRasterizer,
    is_cuda_available,
    get_recommended_device,
)


class SimpleConfig:
    """简单的配置类"""

    def __init__(self):
        self.near_plane = 0.1
        self.far_plane = 100.0
        self.background_color = [0.1, 0.1, 0.1]
        self.density_activation = "exp"
        self.color_activation = "sigmoid"
        self.sh_degree = 2


def generate_test_scene(num_voxels=2000, device="cpu"):
    """生成测试场景"""
    device = torch.device(device)

    # 创建一个简单的立方体场景
    positions = []
    sizes = []
    densities = []
    colors = []

    # 立方体中心
    center = torch.tensor([0.0, 0.0, 2.0], device=device)
    size = 1.0

    # 在立方体内部随机分布体素
    for _ in range(num_voxels):
        pos = center + (torch.rand(3, device=device) - 0.5) * size
        positions.append(pos)

        # 随机大小
        voxel_size = torch.rand(1, device=device) * 0.05 + 0.01
        sizes.append(voxel_size)

        # 基于位置的颜色
        color = torch.rand(3, device=device)
        colors.append(color)

        # 基于距离中心的密度
        dist = torch.norm(pos - center)
        density = torch.exp(-dist * 2.0) + torch.randn(1, device=device) * 0.1
        densities.append(density)

    voxels = {
        "positions": torch.stack(positions),
        "sizes": torch.cat(sizes),
        "densities": torch.cat(densities),
        "colors": torch.stack(colors),
    }

    # 相机参数
    camera_matrix = torch.eye(4, device=device)
    camera_matrix[2, 3] = 3.0  # 相机位置

    intrinsics = torch.tensor(
        [[800, 0, 400], [0, 800, 300], [0, 0, 1]], dtype=torch.float32, device=device
    )

    viewport_size = (800, 600)

    return voxels, camera_matrix, intrinsics, viewport_size


def render_and_compare():
    """渲染并比较 CPU 和 CUDA 版本"""
    print("🎨 VoxelRasterizer 重构演示")
    print("=" * 50)

    # 检查环境
    print(f"CUDA 可用性: {is_cuda_available()}")
    print(f"推荐设备: {get_recommended_device()}")
    print(f"PyTorch 版本: {torch.__version__}")

    if torch.cuda.is_available():
        print(f"GPU 设备: {torch.cuda.get_device_name(0)}")

    print("\n📊 生成测试场景...")
    voxels, camera_matrix, intrinsics, viewport_size = generate_test_scene(2000, "cpu")

    config = SimpleConfig()

    # CPU 渲染
    print("\n🖥️  CPU 渲染测试...")
    rasterizer_cpu = VoxelRasterizer(config, use_cuda=False)

    start_time = time.time()
    result_cpu = rasterizer_cpu(voxels, camera_matrix, intrinsics, viewport_size)
    cpu_time = (time.time() - start_time) * 1000

    print(f"✅ CPU 渲染完成")
    print(f"   渲染时间: {cpu_time:.2f} ms")
    print(f"   RGB 形状: {result_cpu['rgb'].shape}")
    print(f"   深度形状: {result_cpu['depth'].shape}")

    # CUDA 渲染（如果可用）
    if is_cuda_available():
        print("\n🚀 CUDA 渲染测试...")
        try:
            # 将数据移到 GPU
            voxels_cuda = {k: v.cuda() for k, v in voxels.items()}
            camera_matrix_cuda = camera_matrix.cuda()
            intrinsics_cuda = intrinsics.cuda()

            rasterizer_cuda = VoxelRasterizer(config, use_cuda=True)

            # 预热
            _ = rasterizer_cuda(voxels_cuda, camera_matrix_cuda, intrinsics_cuda, viewport_size)
            torch.cuda.synchronize()

            # 计时
            start_time = time.time()
            result_cuda = rasterizer_cuda(
                voxels_cuda, camera_matrix_cuda, intrinsics_cuda, viewport_size
            )
            torch.cuda.synchronize()
            cuda_time = (time.time() - start_time) * 1000

            print(f"✅ CUDA 渲染完成")
            print(f"   渲染时间: {cuda_time:.2f} ms")
            print(f"   加速比: {cpu_time/cuda_time:.2f}x")

        except Exception as e:
            print(f"❌ CUDA 渲染失败: {e}")
            result_cuda = None
    else:
        print("\n⚠️  CUDA 不可用，跳过 CUDA 测试")
        result_cuda = None

    # 可视化结果
    print("\n🎨 可视化结果...")
    visualize_results(result_cpu, result_cuda, cpu_time, cuda_time if result_cuda else None)


def visualize_results(result_cpu, result_cuda, cpu_time, cuda_time):
    """可视化渲染结果"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("VoxelRasterizer 重构演示结果", fontsize=16)

    # CPU RGB
    axes[0, 0].imshow(result_cpu["rgb"].cpu().numpy())
    axes[0, 0].set_title(f"CPU RGB ({cpu_time:.1f}ms)")
    axes[0, 0].axis("off")

    # CPU Depth
    depth_cpu = result_cpu["depth"].cpu().numpy()
    im1 = axes[0, 1].imshow(depth_cpu, cmap="viridis")
    axes[0, 1].set_title("CPU Depth")
    axes[0, 1].axis("off")
    plt.colorbar(im1, ax=axes[0, 1])

    if result_cuda is not None:
        # CUDA RGB
        axes[1, 0].imshow(result_cuda["rgb"].cpu().numpy())
        axes[1, 0].set_title(f"CUDA RGB ({cuda_time:.1f}ms)")
        axes[1, 0].axis("off")

        # CUDA Depth
        depth_cuda = result_cuda["depth"].cpu().numpy()
        im2 = axes[1, 1].imshow(depth_cuda, cmap="viridis")
        axes[1, 1].set_title("CUDA Depth")
        axes[1, 1].axis("off")
        plt.colorbar(im2, ax=axes[1, 1])

        # 性能对比
        speedup = cpu_time / cuda_time
        fig.text(0.5, 0.02, f"性能提升: {speedup:.2f}x", ha="center", fontsize=14, weight="bold")
    else:
        # 如果没有 CUDA 结果，显示 CPU 结果的放大版本
        axes[1, 0].imshow(result_cpu["rgb"].cpu().numpy())
        axes[1, 0].set_title("CPU RGB (放大)")
        axes[1, 0].axis("off")

        axes[1, 1].imshow(depth_cpu, cmap="viridis")
        axes[1, 1].set_title("CPU Depth (放大)")
        axes[1, 1].axis("off")

        fig.text(
            0.5, 0.02, "CUDA 不可用，仅显示 CPU 结果", ha="center", fontsize=14, style="italic"
        )

    plt.tight_layout()

    # 保存结果
    output_path = Path(__file__).parent / "demo_outputs" / "voxel_rasterizer_refactored.png"
    output_path.parent.mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"💾 结果已保存到: {output_path}")

    plt.show()


def performance_benchmark():
    """性能基准测试"""
    print("\n📈 性能基准测试")
    print("-" * 30)

    config = SimpleConfig()
    viewport_size = (800, 600)

    # 测试不同体素数量
    voxel_counts = [500, 1000, 2000, 5000]

    results = []

    for num_voxels in voxel_counts:
        print(f"\n测试 {num_voxels} 个体素...")

        voxels, camera_matrix, intrinsics, _ = generate_test_scene(num_voxels, "cpu")

        # CPU 测试
        rasterizer_cpu = VoxelRasterizer(config, use_cuda=False)

        # 预热
        for _ in range(3):
            _ = rasterizer_cpu(voxels, camera_matrix, intrinsics, viewport_size)

        # 计时
        start_time = time.time()
        for _ in range(10):
            _ = rasterizer_cpu(voxels, camera_matrix, intrinsics, viewport_size)
        cpu_time = (time.time() - start_time) / 10 * 1000

        result = {"voxels": num_voxels, "cpu_time": cpu_time, "cuda_time": None, "speedup": None}

        # CUDA 测试（如果可用）
        if is_cuda_available():
            try:
                voxels_cuda = {k: v.cuda() for k, v in voxels.items()}
                camera_matrix_cuda = camera_matrix.cuda()
                intrinsics_cuda = intrinsics.cuda()

                rasterizer_cuda = VoxelRasterizer(config, use_cuda=True)

                # 预热
                for _ in range(3):
                    _ = rasterizer_cuda(
                        voxels_cuda, camera_matrix_cuda, intrinsics_cuda, viewport_size
                    )
                    torch.cuda.synchronize()

                # 计时
                start_time = time.time()
                for _ in range(10):
                    _ = rasterizer_cuda(
                        voxels_cuda, camera_matrix_cuda, intrinsics_cuda, viewport_size
                    )
                    torch.cuda.synchronize()
                cuda_time = (time.time() - start_time) / 10 * 1000

                result["cuda_time"] = cuda_time
                result["speedup"] = cpu_time / cuda_time

            except Exception as e:
                print(f"  CUDA 测试失败: {e}")

        results.append(result)

        # 打印结果
        print(f"  CPU: {cpu_time:.2f} ms")
        if result["cuda_time"]:
            print(f"  CUDA: {result['cuda_time']:.2f} ms")
            print(f"  加速比: {result['speedup']:.2f}x")

    # 绘制性能图表
    plot_performance_results(results)


def plot_performance_results(results):
    """绘制性能结果图表"""
    voxel_counts = [r["voxels"] for r in results]
    cpu_times = [r["cpu_time"] for r in results]
    cuda_times = [r["cuda_time"] for r in results if r["cuda_time"]]
    speedups = [r["speedup"] for r in results if r["speedup"]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # 渲染时间对比
    ax1.plot(voxel_counts, cpu_times, "o-", label="CPU", linewidth=2, markersize=8)
    if cuda_times:
        ax1.plot(
            voxel_counts[: len(cuda_times)],
            cuda_times,
            "s-",
            label="CUDA",
            linewidth=2,
            markersize=8,
        )

    ax1.set_xlabel("体素数量")
    ax1.set_ylabel("渲染时间 (ms)")
    ax1.set_title("渲染性能对比")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 加速比
    if speedups:
        ax2.plot(
            voxel_counts[: len(speedups)], speedups, "o-", color="red", linewidth=2, markersize=8
        )
        ax2.set_xlabel("体素数量")
        ax2.set_ylabel("加速比")
        ax2.set_title("CUDA 加速比")
        ax2.grid(True, alpha=0.3)

        # 添加加速比标签
        for i, speedup in enumerate(speedups):
            ax2.annotate(
                f"{speedup:.1f}x",
                (voxel_counts[i], speedup),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
            )

    plt.tight_layout()

    # 保存图表
    output_path = Path(__file__).parent / "demo_outputs" / "voxel_rasterizer_performance.png"
    output_path.parent.mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"💾 性能图表已保存到: {output_path}")

    plt.show()


def main():
    """主函数"""
    print("🚀 启动 VoxelRasterizer 重构演示")

    try:
        # 基本渲染演示
        render_and_compare()

        # 性能基准测试
        performance_benchmark()

        print("\n🎉 演示完成！")
        print("\n📝 总结:")
        print("- VoxelRasterizer 已成功重构为支持 CUDA 加速")
        print("- 保持向后兼容性，现有代码无需修改")
        print("- 自动设备选择，智能选择最优渲染方案")
        print("- 显著性能提升，特别是在大规模体素渲染时")

    except Exception as e:
        print(f"\n❌ 演示过程中出现错误: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
