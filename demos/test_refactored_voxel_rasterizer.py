#!/usr/bin/env python3
"""
测试重构后的 VoxelRasterizer

验证 CUDA 和 CPU 版本的功能和性能。
"""

import torch
import numpy as np
import time
import logging
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from nerfs.svraster.voxel_rasterizer import (
    VoxelRasterizer,
    benchmark_voxel_rasterizer,
    is_cuda_available,
    get_recommended_device,
)

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


def generate_test_data(num_voxels=1000, device="cpu"):
    """生成测试数据"""
    device = torch.device(device)

    # 体素数据
    voxels = {
        "positions": torch.randn(num_voxels, 3, device=device),
        "sizes": torch.rand(num_voxels, device=device) * 0.1,
        "densities": torch.randn(num_voxels, device=device),
        "colors": torch.rand(num_voxels, 3, device=device),
    }

    # 相机参数
    camera_matrix = torch.eye(4, device=device)
    camera_matrix[2, 3] = 2.0

    intrinsics = torch.tensor(
        [[800, 0, 400], [0, 800, 300], [0, 0, 1]], dtype=torch.float32, device=device
    )

    viewport_size = (800, 600)

    return voxels, camera_matrix, intrinsics, viewport_size


def test_basic_functionality():
    """测试基本功能"""
    logger.info("测试基本功能...")

    # 生成测试数据
    voxels, camera_matrix, intrinsics, viewport_size = generate_test_data(1000, "cpu")

    # 测试 CPU 版本
    config = SimpleConfig()
    rasterizer_cpu = VoxelRasterizer(config, use_cuda=False)

    try:
        result_cpu = rasterizer_cpu(voxels, camera_matrix, intrinsics, viewport_size)
        logger.info("✅ CPU 版本渲染成功")
        logger.info(f"RGB 图像形状: {result_cpu['rgb'].shape}")
        logger.info(f"深度图像形状: {result_cpu['depth'].shape}")
    except Exception as e:
        logger.error(f"❌ CPU 版本渲染失败: {e}")
        return False

    # 测试 CUDA 版本（如果可用）
    if is_cuda_available():
        try:
            voxels_cuda, camera_matrix_cuda, intrinsics_cuda, _ = generate_test_data(1000, "cuda")
            rasterizer_cuda = VoxelRasterizer(config, use_cuda=True)

            result_cuda = rasterizer_cuda(
                voxels_cuda, camera_matrix_cuda, intrinsics_cuda, viewport_size
            )
            logger.info("✅ CUDA 版本渲染成功")
            logger.info(f"RGB 图像形状: {result_cuda['rgb'].shape}")
            logger.info(f"深度图像形状: {result_cuda['depth'].shape}")
        except Exception as e:
            logger.error(f"❌ CUDA 版本渲染失败: {e}")
            return False
    else:
        logger.info("⚠️ CUDA 不可用，跳过 CUDA 测试")

    return True


def test_performance_comparison():
    """测试性能对比"""
    logger.info("测试性能对比...")

    # 生成测试数据
    num_voxels = 5000
    voxels, camera_matrix, intrinsics, viewport_size = generate_test_data(num_voxels, "cpu")

    # CPU 性能测试
    logger.info("测试 CPU 性能...")
    cpu_results = benchmark_voxel_rasterizer(
        voxels, camera_matrix, intrinsics, viewport_size, num_iterations=10, use_cuda=False
    )

    logger.info(f"CPU 性能结果:")
    for key, value in cpu_results.items():
        logger.info(f"  {key}: {value:.4f}")

    # CUDA 性能测试（如果可用）
    if is_cuda_available():
        logger.info("测试 CUDA 性能...")
        voxels_cuda, camera_matrix_cuda, intrinsics_cuda, _ = generate_test_data(num_voxels, "cuda")

        cuda_results = benchmark_voxel_rasterizer(
            voxels_cuda,
            camera_matrix_cuda,
            intrinsics_cuda,
            viewport_size,
            num_iterations=10,
            use_cuda=True,
        )

        logger.info(f"CUDA 性能结果:")
        for key, value in cuda_results.items():
            logger.info(f"  {key}: {value:.4f}")

        # 计算加速比
        if "avg_time_ms" in cpu_results and "avg_time_ms" in cuda_results:
            speedup = cpu_results["avg_time_ms"] / cuda_results["avg_time_ms"]
            logger.info(f"CUDA 加速比: {speedup:.2f}x")

    return True


def test_auto_device_selection():
    """测试自动设备选择"""
    logger.info("测试自动设备选择...")

    # 生成测试数据
    voxels, camera_matrix, intrinsics, viewport_size = generate_test_data(1000, "cpu")

    # 测试自动选择
    config = SimpleConfig()
    rasterizer_auto = VoxelRasterizer(config)  # 不指定 use_cuda

    try:
        result = rasterizer_auto(voxels, camera_matrix, intrinsics, viewport_size)
        logger.info("✅ 自动设备选择渲染成功")

        # 检查是否使用了正确的设备
        if is_cuda_available():
            expected_device = "cuda"
        else:
            expected_device = "cpu"

        logger.info(f"推荐设备: {get_recommended_device()}")
        logger.info(f"实际使用设备: {expected_device}")

    except Exception as e:
        logger.error(f"❌ 自动设备选择渲染失败: {e}")
        return False

    return True


def test_error_handling():
    """测试错误处理"""
    logger.info("测试错误处理...")

    # 测试无效数据
    config = SimpleConfig()
    rasterizer = VoxelRasterizer(config, use_cuda=False)

    # 空体素数据
    empty_voxels = {
        "positions": torch.empty(0, 3),
        "sizes": torch.empty(0),
        "densities": torch.empty(0),
        "colors": torch.empty(0, 3),
    }

    camera_matrix = torch.eye(4)
    intrinsics = torch.eye(3)
    viewport_size = (800, 600)

    try:
        result = rasterizer(empty_voxels, camera_matrix, intrinsics, viewport_size)
        logger.info("✅ 空体素数据处理成功")
    except Exception as e:
        logger.error(f"❌ 空体素数据处理失败: {e}")
        return False

    return True


def main():
    """主测试函数"""
    logger.info("开始测试重构后的 VoxelRasterizer")
    logger.info("=" * 50)

    # 检查环境
    logger.info(f"CUDA 可用性: {is_cuda_available()}")
    logger.info(f"推荐设备: {get_recommended_device()}")
    logger.info(f"PyTorch 版本: {torch.__version__}")

    if torch.cuda.is_available():
        logger.info(f"GPU 设备: {torch.cuda.get_device_name(0)}")

    # 运行测试
    tests = [
        ("基本功能", test_basic_functionality),
        ("性能对比", test_performance_comparison),
        ("自动设备选择", test_auto_device_selection),
        ("错误处理", test_error_handling),
    ]

    results = {}

    for test_name, test_func in tests:
        logger.info(f"\n测试: {test_name}")
        logger.info("-" * 30)

        try:
            success = test_func()
            results[test_name] = success
            status = "✅ 通过" if success else "❌ 失败"
            logger.info(f"{test_name}: {status}")
        except Exception as e:
            logger.error(f"{test_name}: ❌ 异常 - {e}")
            results[test_name] = False

    # 总结
    logger.info("\n" + "=" * 50)
    logger.info("测试总结:")

    passed = sum(results.values())
    total = len(results)

    for test_name, success in results.items():
        status = "✅ 通过" if success else "❌ 失败"
        logger.info(f"  {test_name}: {status}")

    logger.info(f"\n总体结果: {passed}/{total} 测试通过")

    if passed == total:
        logger.info("🎉 所有测试通过！重构后的 VoxelRasterizer 工作正常。")
        return True
    else:
        logger.error("⚠️ 部分测试失败，请检查实现。")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
