#!/usr/bin/env python3
"""
SVRaster 渲染性能演示脚本

这个脚本展示了如何使用性能测试来评估 SVRaster 渲染器的性能。
可以独立运行，也可以作为性能基准测试的参考。
"""

import sys
import os
import time
import torch
import numpy as np

# 保证 src 是 sys.path[0]
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
src_path = os.path.join(project_root, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

try:
    import nerfs.svraster as svraster
    from nerfs.svraster.voxel_rasterizer import benchmark_voxel_rasterizer

    SVRASTER_AVAILABLE = True
except ImportError as e:
    print(f"SVRaster not available: {e}")
    SVRASTER_AVAILABLE = False


def demo_basic_benchmark():
    """演示基本性能基准测试"""
    if not SVRASTER_AVAILABLE:
        print("SVRaster not available, skipping benchmark demo")
        return

    print("=== 基本性能基准测试 ===")

    # 创建测试数据
    num_voxels = 1000
    voxels = {
        "positions": torch.randn(num_voxels, 3),
        "sizes": torch.randn(num_voxels),
        "densities": torch.randn(num_voxels),
        "colors": torch.randn(num_voxels, 3),
    }

    camera_matrix = torch.eye(4)
    intrinsics = torch.eye(3)
    viewport_size = (256, 256)

    # 运行基准测试
    result = benchmark_voxel_rasterizer(
        voxels, camera_matrix, intrinsics, viewport_size, num_iterations=10
    )

    print(f"渲染 {num_voxels} 个体素，分辨率 {viewport_size[0]}x{viewport_size[1]}")
    print(f"总时间: {result['total_time_ms']:.2f} ms")
    print(f"平均时间: {result['avg_time_ms']:.2f} ms")
    print(f"帧率: {result['fps']:.1f} FPS")
    print()


def demo_scalability_test():
    """演示可扩展性测试"""
    if not SVRASTER_AVAILABLE:
        print("SVRaster not available, skipping scalability demo")
        return

    print("=== 可扩展性测试 ===")

    camera_matrix = torch.eye(4)
    intrinsics = torch.eye(3)
    viewport_size = (128, 128)

    # 测试不同的体素数量
    voxel_counts = [100, 500, 1000, 2000]
    results = {}

    for num_voxels in voxel_counts:
        voxels = {
            "positions": torch.randn(num_voxels, 3),
            "sizes": torch.randn(num_voxels),
            "densities": torch.randn(num_voxels),
            "colors": torch.randn(num_voxels, 3),
        }

        result = benchmark_voxel_rasterizer(
            voxels, camera_matrix, intrinsics, viewport_size, num_iterations=5
        )

        results[num_voxels] = result
        print(f"{num_voxels:4d} 体素: {result['avg_time_ms']:6.2f} ms, {result['fps']:5.1f} FPS")

    # 分析可扩展性
    print("\n可扩展性分析:")
    voxel_counts_list = sorted(results.keys())
    for i in range(1, len(voxel_counts_list)):
        prev_count = voxel_counts_list[i - 1]
        curr_count = voxel_counts_list[i]
        prev_time = results[prev_count]["avg_time_ms"]
        curr_time = results[curr_count]["avg_time_ms"]

        voxel_ratio = curr_count / prev_count
        time_ratio = curr_time / prev_time

        print(
            f"体素数量 {prev_count} → {curr_count} (x{voxel_ratio:.1f}): "
            f"时间 {prev_time:.1f} → {curr_time:.1f} ms (x{time_ratio:.1f})"
        )
    print()


def demo_image_size_test():
    """演示图像尺寸性能测试"""
    if not SVRASTER_AVAILABLE:
        print("SVRaster not available, skipping image size demo")
        return

    print("=== 图像尺寸性能测试 ===")

    # 创建测试数据
    voxels = {
        "positions": torch.randn(500, 3),
        "sizes": torch.randn(500),
        "densities": torch.randn(500),
        "colors": torch.randn(500, 3),
    }

    camera_matrix = torch.eye(4)
    intrinsics = torch.eye(3)

    # 测试不同的图像尺寸
    image_sizes = [(64, 64), (128, 128), (256, 256), (512, 512)]
    results = {}

    for width, height in image_sizes:
        result = benchmark_voxel_rasterizer(
            voxels, camera_matrix, intrinsics, (width, height), num_iterations=5
        )

        results[(width, height)] = result
        pixels = width * height
        print(
            f"{width:3d}x{height:3d} ({pixels:6d} 像素): "
            f"{result['avg_time_ms']:6.2f} ms, {result['fps']:5.1f} FPS"
        )

    # 分析像素性能
    print("\n像素性能分析:")
    sizes_list = sorted(results.keys(), key=lambda x: x[0] * x[1])
    for i in range(1, len(sizes_list)):
        prev_size = sizes_list[i - 1]
        curr_size = sizes_list[i]
        prev_pixels = prev_size[0] * prev_size[1]
        curr_pixels = curr_size[0] * curr_size[1]
        prev_time = results[prev_size]["avg_time_ms"]
        curr_time = results[curr_size]["avg_time_ms"]

        pixel_ratio = curr_pixels / prev_pixels
        time_ratio = curr_time / prev_time

        print(
            f"像素数量 {prev_pixels} → {curr_pixels} (x{pixel_ratio:.1f}): "
            f"时间 {prev_time:.1f} → {curr_time:.1f} ms (x{time_ratio:.1f})"
        )
    print()


def demo_cuda_vs_cpu():
    """演示 CUDA vs CPU 性能对比"""
    if not SVRASTER_AVAILABLE:
        print("SVRaster not available, skipping CUDA vs CPU demo")
        return

    if not torch.cuda.is_available():
        print("CUDA not available, skipping CUDA vs CPU demo")
        return

    print("=== CUDA vs CPU 性能对比 ===")

    # 创建测试数据
    voxels = {
        "positions": torch.randn(500, 3),
        "sizes": torch.randn(500),
        "densities": torch.randn(500),
        "colors": torch.randn(500, 3),
    }

    camera_matrix = torch.eye(4)
    intrinsics = torch.eye(3)
    viewport_size = (128, 128)

    # CPU 测试
    print("运行 CPU 测试...")
    cpu_result = benchmark_voxel_rasterizer(
        voxels, camera_matrix, intrinsics, viewport_size, num_iterations=5, use_cuda=False
    )

    # CUDA 测试
    print("运行 CUDA 测试...")
    cuda_result = benchmark_voxel_rasterizer(
        voxels, camera_matrix, intrinsics, viewport_size, num_iterations=5, use_cuda=True
    )

    # 对比结果
    print(f"\nCPU 性能:  {cpu_result['avg_time_ms']:6.2f} ms, {cpu_result['fps']:5.1f} FPS")
    print(f"CUDA 性能: {cuda_result['avg_time_ms']:6.2f} ms, {cuda_result['fps']:5.1f} FPS")

    speedup = cpu_result["avg_time_ms"] / cuda_result["avg_time_ms"]
    print(f"CUDA 加速比: {speedup:.2f}x")
    print()


def demo_renderer_throughput():
    """演示渲染器吞吐量测试"""
    if not SVRASTER_AVAILABLE:
        print("SVRaster not available, skipping renderer throughput demo")
        return

    print("=== 渲染器吞吐量测试 ===")

    # 创建渲染器组件
    model_config = svraster.SVRasterConfig(max_octree_levels=3, base_resolution=16, sh_degree=1)
    model = svraster.SVRasterModel(model_config)
    raster_config = svraster.VoxelRasterizerConfig()
    rasterizer = svraster.VoxelRasterizer(raster_config)
    renderer_config = svraster.SVRasterRendererConfig()
    renderer = svraster.SVRasterRenderer(model, rasterizer, renderer_config)

    camera_pose = torch.eye(4)
    intrinsics = torch.eye(3)
    width, height = 128, 128

    # 预热
    print("预热渲染器...")
    for _ in range(3):
        _ = renderer.render(camera_pose, intrinsics, width, height)

    # 测量吞吐量
    print("测量渲染吞吐量...")
    num_renders = 20
    start_time = time.time()

    for i in range(num_renders):
        result = renderer.render(camera_pose, intrinsics, width, height)
        if i % 5 == 0:
            print(f"  完成 {i+1}/{num_renders} 次渲染")

    end_time = time.time()

    total_time = end_time - start_time
    throughput = num_renders / total_time
    avg_time = total_time / num_renders * 1000

    print(f"\n渲染 {num_renders} 张图像，分辨率 {width}x{height}")
    print(f"总时间: {total_time:.2f} 秒")
    print(f"平均时间: {avg_time:.2f} ms/图像")
    print(f"吞吐量: {throughput:.2f} 图像/秒")
    print()


def main():
    """主函数"""
    print("SVRaster 渲染性能演示")
    print("=" * 50)

    # 检查环境
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"CUDA 可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU 数量: {torch.cuda.device_count()}")
        if torch.cuda.device_count() > 0:
            print(f"当前 GPU: {torch.cuda.get_device_name(0)}")
    print()

    # 运行演示
    demo_basic_benchmark()
    demo_scalability_test()
    demo_image_size_test()
    demo_cuda_vs_cpu()
    demo_renderer_throughput()

    print("性能演示完成！")


if __name__ == "__main__":
    main()
