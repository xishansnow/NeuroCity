#!/usr/bin/env python3
"""
VoxelRasterizer CUDA 扩展测试脚本

用于验证 CUDA 扩展是否正确编译和运行。
"""

import torch
import numpy as np
import logging
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_cuda_availability():
    """测试 CUDA 可用性"""
    logger.info("测试 CUDA 可用性...")

    if not torch.cuda.is_available():
        logger.error("CUDA 不可用")
        return False

    logger.info(f"CUDA 可用: {torch.cuda.get_device_name(0)}")
    logger.info(f"CUDA 版本: {torch.version.cuda if hasattr(torch.version, 'cuda') else 'Unknown'}")
    return True


def test_extension_import():
    """测试 CUDA 扩展导入"""
    logger.info("测试 CUDA 扩展导入...")

    try:
        import nerfs.svraster.cuda.renderer.voxel_rasterizer_gpu as voxel_rasterizer_cuda

        logger.info("✅ CUDA 扩展导入成功")
        logger.info(f"可用函数: {list(voxel_rasterizer_cuda.__dict__.keys())}")
        return True
    except ImportError as e:
        logger.error(f"❌ CUDA 扩展导入失败: {e}")
        return False


def test_basic_functionality():
    """测试基本功能"""
    logger.info("测试基本功能...")

    try:
        import nerfs.svraster.cuda.renderer.voxel_rasterizer_gpu as voxel_rasterizer_cuda

        # 生成测试数据
        device = torch.device("cuda")
        num_voxels = 1000

        # 体素数据
        voxel_positions = torch.randn(num_voxels, 3, device=device)
        voxel_sizes = torch.rand(num_voxels, device=device) * 0.1
        voxel_densities = torch.randn(num_voxels, device=device)
        voxel_colors = torch.rand(num_voxels, 3, device=device)

        # 相机参数
        camera_matrix = torch.eye(4, device=device)
        camera_matrix[2, 3] = 2.0

        intrinsics = torch.tensor(
            [[800, 0, 400], [0, 800, 300], [0, 0, 1]], dtype=torch.float32, device=device
        )

        viewport_size = torch.tensor([800, 600], dtype=torch.int32, device=device)
        background_color = torch.tensor([0.1, 0.1, 0.1], device=device)

        # 调用 CUDA 函数
        rgb, depth = voxel_rasterizer_cuda.voxel_rasterization(
            voxel_positions,
            voxel_sizes,
            voxel_densities,
            voxel_colors,
            camera_matrix,
            intrinsics,
            viewport_size,
            0.1,  # near_plane
            100.0,  # far_plane
            background_color,
            "exp",  # density_activation
            "sigmoid",  # color_activation
            2,  # sh_degree
        )

        logger.info(f"✅ 渲染成功")
        logger.info(f"RGB 图像形状: {rgb.shape}")
        logger.info(f"深度图像形状: {depth.shape}")

        return True

    except Exception as e:
        logger.error(f"❌ 基本功能测试失败: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_performance():
    """测试性能"""
    logger.info("测试性能...")

    try:
        import voxel_rasterizer_cuda

        # 生成测试数据
        device = torch.device("cuda")
        num_voxels = 5000

        voxel_positions = torch.randn(num_voxels, 3, device=device)
        voxel_sizes = torch.rand(num_voxels, device=device) * 0.1
        voxel_densities = torch.randn(num_voxels, device=device)
        voxel_colors = torch.rand(num_voxels, 3, device=device)

        camera_matrix = torch.eye(4, device=device)
        camera_matrix[2, 3] = 2.0

        intrinsics = torch.tensor(
            [[800, 0, 400], [0, 800, 300], [0, 0, 1]], dtype=torch.float32, device=device
        )

        viewport_size = torch.tensor([800, 600], dtype=torch.int32, device=device)
        background_color = torch.tensor([0.1, 0.1, 0.1], device=device)

        # 预热
        for _ in range(3):
            _ = voxel_rasterizer_cuda.voxel_rasterization(
                voxel_positions,
                voxel_sizes,
                voxel_densities,
                voxel_colors,
                camera_matrix,
                intrinsics,
                viewport_size,
                0.1,
                100.0,
                background_color,
                "exp",
                "sigmoid",
                2,
            )

        torch.cuda.synchronize()

        # 性能测试
        num_iterations = 10
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)

        start_time.record()
        for _ in range(num_iterations):
            _ = voxel_rasterizer_cuda.voxel_rasterization(
                voxel_positions,
                voxel_sizes,
                voxel_densities,
                voxel_colors,
                camera_matrix,
                intrinsics,
                viewport_size,
                0.1,
                100.0,
                background_color,
                "exp",
                "sigmoid",
                2,
            )
        end_time.record()

        torch.cuda.synchronize()

        elapsed_time = start_time.elapsed_time(end_time) / num_iterations
        fps = 1000.0 / elapsed_time

        logger.info(f"✅ 性能测试成功")
        logger.info(f"平均渲染时间: {elapsed_time:.2f}ms")
        logger.info(f"帧率: {fps:.1f} FPS")

        return True

    except Exception as e:
        logger.error(f"❌ 性能测试失败: {e}")
        return False


def test_utility_functions():
    """测试工具函数"""
    logger.info("测试工具函数...")

    try:
        import voxel_rasterizer_cuda

        device = torch.device("cuda")

        # 测试相机矩阵创建
        camera_pose = torch.eye(4, device=device)
        camera_matrix = voxel_rasterizer_cuda.create_camera_matrix(camera_pose)
        logger.info(f"✅ 相机矩阵创建成功: {camera_matrix.shape}")

        # 测试从光线估算相机参数
        ray_origins = torch.randn(100, 3, device=device)
        ray_directions = torch.randn(100, 3, device=device)
        ray_directions = ray_directions / ray_directions.norm(dim=1, keepdim=True)

        camera_matrix, intrinsics = voxel_rasterizer_cuda.rays_to_camera_matrix(
            ray_origins, ray_directions
        )
        logger.info(f"✅ 相机参数估算成功")
        logger.info(f"相机矩阵形状: {camera_matrix.shape}")
        logger.info(f"内参矩阵形状: {intrinsics.shape}")

        return True

    except Exception as e:
        logger.error(f"❌ 工具函数测试失败: {e}")
        return False


def test_benchmark_function():
    """测试基准测试函数"""
    logger.info("测试基准测试函数...")

    try:
        import voxel_rasterizer_cuda

        device = torch.device("cuda")

        # 生成测试数据
        num_voxels = 2000
        voxel_positions = torch.randn(num_voxels, 3, device=device)
        voxel_sizes = torch.rand(num_voxels, device=device) * 0.1
        voxel_densities = torch.randn(num_voxels, device=device)
        voxel_colors = torch.rand(num_voxels, 3, device=device)

        camera_matrix = torch.eye(4, device=device)
        camera_matrix[2, 3] = 2.0

        intrinsics = torch.tensor(
            [[800, 0, 400], [0, 800, 300], [0, 0, 1]], dtype=torch.float32, device=device
        )

        viewport_size = torch.tensor([800, 600], dtype=torch.int32, device=device)

        # 运行基准测试
        results = voxel_rasterizer_cuda.benchmark(
            voxel_positions,
            voxel_sizes,
            voxel_densities,
            voxel_colors,
            camera_matrix,
            intrinsics,
            viewport_size,
            5,
        )

        logger.info(f"✅ 基准测试成功")
        for key, value in results.items():
            logger.info(f"  {key}: {value:.4f}")

        return True

    except Exception as e:
        logger.error(f"❌ 基准测试函数失败: {e}")
        return False


def main():
    """主测试函数"""
    logger.info("开始 VoxelRasterizer CUDA 扩展测试")
    logger.info("=" * 50)

    tests = [
        ("CUDA 可用性", test_cuda_availability),
        ("扩展导入", test_extension_import),
        ("基本功能", test_basic_functionality),
        ("性能测试", test_performance),
        ("工具函数", test_utility_functions),
        ("基准测试函数", test_benchmark_function),
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
        logger.info("🎉 所有测试通过！CUDA 扩展工作正常。")
        return True
    else:
        logger.error("⚠️ 部分测试失败，请检查 CUDA 扩展安装。")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
