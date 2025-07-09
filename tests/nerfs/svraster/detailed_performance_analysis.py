#!/usr/bin/env python3
"""
SVRaster 详细性能分析脚本

深入分析 CUDA vs CPU 性能差异的原因，包括：
- 数据传输时间
- 计算时间
- 内存使用
- 不同数据规模下的性能
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


def analyze_data_transfer_overhead():
    """分析数据传输开销"""
    if not SVRASTER_AVAILABLE or not torch.cuda.is_available():
        print("CUDA not available, skipping data transfer analysis")
        return

    print("=== 数据传输开销分析 ===")

    # 测试不同大小的数据传输
    data_sizes = [100, 500, 1000, 2000, 5000]

    for num_voxels in data_sizes:
        # 创建测试数据
        voxels = {
            "positions": torch.randn(num_voxels, 3),
            "sizes": torch.randn(num_voxels),
            "densities": torch.randn(num_voxels),
            "colors": torch.randn(num_voxels, 3),
        }

        # 测量 CPU 到 GPU 传输时间
        start_time = time.time()
        voxels_gpu = {k: v.cuda() for k, v in voxels.items()}
        torch.cuda.synchronize()  # 确保传输完成
        transfer_time = (time.time() - start_time) * 1000

        # 测量 GPU 到 CPU 传输时间
        start_time = time.time()
        voxels_cpu = {k: v.cpu() for k, v in voxels_gpu.items()}
        torch.cuda.synchronize()
        transfer_back_time = (time.time() - start_time) * 1000

        total_data_size = sum(v.numel() * v.element_size() for v in voxels.values()) / 1024  # KB

        print(
            f"{num_voxels:4d} 体素 ({total_data_size:6.1f} KB): "
            f"CPU→GPU {transfer_time:6.2f} ms, "
            f"GPU→CPU {transfer_back_time:6.2f} ms, "
            f"总计 {transfer_time + transfer_back_time:6.2f} ms"
        )

    print()


def analyze_computation_scaling():
    """分析计算性能随数据规模的变化"""
    if not SVRASTER_AVAILABLE:
        print("SVRaster not available, skipping computation scaling analysis")
        return

    print("=== 计算性能随数据规模变化 ===")

    camera_matrix = torch.eye(4)
    intrinsics = torch.eye(3)
    viewport_size = (128, 128)

    # 测试更大的数据规模
    voxel_counts = [100, 500, 1000, 2000, 5000, 10000]
    cpu_results = {}
    cuda_results = {}

    for num_voxels in voxel_counts:
        print(f"测试 {num_voxels} 体素...")

        voxels = {
            "positions": torch.randn(num_voxels, 3),
            "sizes": torch.randn(num_voxels),
            "densities": torch.randn(num_voxels),
            "colors": torch.randn(num_voxels, 3),
        }

        # CPU 测试
        try:
            cpu_result = benchmark_voxel_rasterizer(
                voxels, camera_matrix, intrinsics, viewport_size, num_iterations=3, use_cuda=False
            )
            cpu_results[num_voxels] = cpu_result
        except Exception as e:
            print(f"  CPU 测试失败: {e}")
            continue

        # CUDA 测试
        if torch.cuda.is_available():
            try:
                cuda_result = benchmark_voxel_rasterizer(
                    voxels,
                    camera_matrix,
                    intrinsics,
                    viewport_size,
                    num_iterations=3,
                    use_cuda=True,
                )
                cuda_results[num_voxels] = cuda_result
            except Exception as e:
                print(f"  CUDA 测试失败: {e}")
                continue

    # 分析结果
    print("\n性能对比:")
    print("体素数量 | CPU时间(ms) | CUDA时间(ms) | 加速比 | 推荐")
    print("-" * 60)

    for num_voxels in sorted(cpu_results.keys()):
        if num_voxels in cuda_results:
            cpu_time = cpu_results[num_voxels]["avg_time_ms"]
            cuda_time = cuda_results[num_voxels]["avg_time_ms"]
            speedup = cpu_time / cuda_time
            recommendation = "CUDA" if speedup > 1.2 else "CPU"

            print(
                f"{num_voxels:8d} | {cpu_time:10.1f} | {cuda_time:11.1f} | {speedup:6.2f}x | {recommendation}"
            )

    print()


def analyze_memory_usage():
    """分析内存使用情况"""
    if not SVRASTER_AVAILABLE:
        print("SVRaster not available, skipping memory analysis")
        return

    print("=== 内存使用分析 ===")

    import psutil

    process = psutil.Process()

    # 测试不同配置的内存使用
    configs = [(100, (64, 64)), (500, (128, 128)), (1000, (256, 256)), (2000, (512, 512))]

    for num_voxels, viewport_size in configs:
        # 清理内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 测量初始内存
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # 创建数据
        voxels = {
            "positions": torch.randn(num_voxels, 3),
            "sizes": torch.randn(num_voxels),
            "densities": torch.randn(num_voxels),
            "colors": torch.randn(num_voxels, 3),
        }

        # 测量数据创建后内存
        data_memory = process.memory_info().rss / 1024 / 1024

        # 移动到 GPU（如果可用）
        if torch.cuda.is_available():
            voxels_gpu = {k: v.cuda() for k, v in voxels.items()}
            gpu_memory = process.memory_info().rss / 1024 / 1024
            gpu_vram = torch.cuda.memory_allocated() / 1024 / 1024  # MB
        else:
            gpu_memory = data_memory
            gpu_vram = 0

        print(f"{num_voxels:4d} 体素, {viewport_size[0]}x{viewport_size[1]} 分辨率:")
        print(f"  初始内存: {initial_memory:.1f} MB")
        print(f"  数据内存: {data_memory:.1f} MB (+{data_memory - initial_memory:.1f} MB)")
        if torch.cuda.is_available():
            print(f"  GPU内存: {gpu_memory:.1f} MB (+{gpu_memory - initial_memory:.1f} MB)")
            print(f"  GPU显存: {gpu_vram:.1f} MB")
        print()

    print()


def analyze_break_even_point():
    """分析 CPU vs CUDA 的盈亏平衡点"""
    if not SVRASTER_AVAILABLE:
        print("SVRaster not available, skipping break-even analysis")
        return

    print("=== CPU vs CUDA 盈亏平衡点分析 ===")

    camera_matrix = torch.eye(4)
    intrinsics = torch.eye(3)
    viewport_size = (128, 128)

    # 寻找盈亏平衡点
    voxel_counts = list(range(50, 2001, 50))  # 50 到 2000，步长 50
    break_even_found = False

    for num_voxels in voxel_counts:
        voxels = {
            "positions": torch.randn(num_voxels, 3),
            "sizes": torch.randn(num_voxels),
            "densities": torch.randn(num_voxels),
            "colors": torch.randn(num_voxels, 3),
        }

        try:
            # CPU 测试
            cpu_result = benchmark_voxel_rasterizer(
                voxels, camera_matrix, intrinsics, viewport_size, num_iterations=5, use_cuda=False
            )

            # CUDA 测试
            if torch.cuda.is_available():
                cuda_result = benchmark_voxel_rasterizer(
                    voxels,
                    camera_matrix,
                    intrinsics,
                    viewport_size,
                    num_iterations=5,
                    use_cuda=True,
                )

                speedup = cpu_result["avg_time_ms"] / cuda_result["avg_time_ms"]

                if speedup > 1.1 and not break_even_found:  # CUDA 开始明显更快
                    print(f"盈亏平衡点: {num_voxels} 体素")
                    print(f"  CPU: {cpu_result['avg_time_ms']:.1f} ms")
                    print(f"  CUDA: {cuda_result['avg_time_ms']:.1f} ms")
                    print(f"  加速比: {speedup:.2f}x")
                    break_even_found = True
                    break

                if num_voxels % 200 == 0:
                    print(
                        f"{num_voxels:4d} 体素: CPU {cpu_result['avg_time_ms']:5.1f} ms, "
                        f"CUDA {cuda_result['avg_time_ms']:5.1f} ms, 加速比 {speedup:.2f}x"
                    )

        except Exception as e:
            print(f"测试 {num_voxels} 体素时出错: {e}")
            continue

    if not break_even_found:
        print("在测试范围内未找到明显的盈亏平衡点")

    print()


def analyze_optimization_opportunities():
    """分析优化机会"""
    print("=== 优化机会分析 ===")

    if not SVRASTER_AVAILABLE:
        print("SVRaster not available, skipping optimization analysis")
        return

    print("基于性能测试结果，以下优化建议：")
    print()

    print("1. 数据传输优化:")
    print("   - 对于小规模数据（<500 体素），考虑使用 CPU 渲染")
    print("   - 批量处理多个渲染任务以减少传输开销")
    print("   - 使用 CUDA 流进行异步数据传输")
    print()

    print("2. 内存优化:")
    print("   - 使用内存池减少分配开销")
    print("   - 优化数据结构以减少内存占用")
    print("   - 考虑使用半精度浮点数（FP16）")
    print()

    print("3. 算法优化:")
    print("   - 实现更高效的体素排序算法")
    print("   - 优化光线-体素相交测试")
    print("   - 使用空间哈希或八叉树加速")
    print()

    print("4. CUDA 内核优化:")
    print("   - 优化内存访问模式（合并访问）")
    print("   - 使用共享内存减少全局内存访问")
    print("   - 调整线程块大小以获得最佳性能")
    print()

    print("5. 自适应渲染:")
    print("   - 根据数据规模自动选择 CPU 或 CUDA")
    print("   - 动态调整渲染质量以平衡性能和质量")
    print()


def main():
    """主函数"""
    print("SVRaster 详细性能分析")
    print("=" * 50)

    # 检查环境
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"CUDA 可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    # 运行分析
    analyze_data_transfer_overhead()
    analyze_computation_scaling()
    analyze_memory_usage()
    analyze_break_even_point()
    analyze_optimization_opportunities()

    print("详细性能分析完成！")


if __name__ == "__main__":
    main()
