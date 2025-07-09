#!/usr/bin/env python3
"""
测试 VolumeRenderer 的 CUDA 加速功能
"""

import torch
import time
import numpy as np
import logging

# 设置日志级别
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 添加 src 到路径
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

try:
    import nerfs.svraster as svraster

    SVRASTER_AVAILABLE = True
except ImportError as e:
    SVRASTER_AVAILABLE = False
    logger.error(f"SVRaster not available: {e}")
    sys.exit(1)


def create_test_data(device, num_rays=1024, num_voxels=1000):
    """创建测试数据"""
    # 创建光线数据
    ray_origins = torch.randn(num_rays, 3, device=device)
    ray_directions = torch.randn(num_rays, 3, device=device)
    ray_directions = ray_directions / ray_directions.norm(dim=-1, keepdim=True)

    # 创建体素数据
    voxel_positions = torch.randn(num_voxels, 3, device=device) * 2.0
    voxel_sizes = torch.ones(num_voxels, device=device) * 0.1
    voxel_densities = torch.randn(num_voxels, device=device) * 0.1
    voxel_colors = torch.rand(num_voxels, 3, device=device)
    voxel_morton_codes = torch.arange(num_voxels, device=device, dtype=torch.long)

    voxels = {
        "positions": voxel_positions,
        "sizes": voxel_sizes,
        "densities": voxel_densities,
        "colors": voxel_colors,
        "morton_codes": voxel_morton_codes,
    }

    return voxels, ray_origins, ray_directions


def benchmark_rendering(renderer, voxels, ray_origins, ray_directions, num_runs=10):
    """基准测试渲染性能"""
    # 预热
    for _ in range(3):
        _ = renderer(voxels, ray_origins, ray_directions)

    # 测试 CPU 版本
    if hasattr(renderer, "use_cuda"):
        renderer.use_cuda = False
        logger.info("Testing CPU version...")

        start_time = time.time()
        for _ in range(num_runs):
            result_cpu = renderer(voxels, ray_origins, ray_directions)
        cpu_time = (time.time() - start_time) / num_runs

        logger.info(f"CPU rendering time: {cpu_time:.4f}s per run")

    # 测试 CUDA 版本
    if hasattr(renderer, "use_cuda") and renderer.use_cuda:
        renderer.use_cuda = True
        logger.info("Testing CUDA version...")

        start_time = time.time()
        for _ in range(num_runs):
            result_cuda = renderer(voxels, ray_origins, ray_directions)
        cuda_time = (time.time() - start_time) / num_runs

        logger.info(f"CUDA rendering time: {cuda_time:.4f}s per run")

        if "cpu_time" in locals():
            speedup = cpu_time / cuda_time
            logger.info(f"Speedup: {speedup:.2f}x")

    return result_cpu if "result_cpu" in locals() else result_cuda


def test_volume_renderer_cuda():
    """测试 VolumeRenderer 的 CUDA 功能"""
    logger.info("开始测试 VolumeRenderer CUDA 加速")

    # 检查 CUDA 可用性
    if not torch.cuda.is_available():
        logger.warning("CUDA not available, skipping CUDA tests")
        return False

    device = torch.device("cuda")
    logger.info(f"Using device: {device}")

    try:
        # 1. 创建配置和渲染器
        logger.info("创建 VolumeRenderer...")
        config = svraster.SVRasterConfig(
            max_octree_levels=4,
            base_resolution=16,
            sh_degree=2,
            ray_samples_per_voxel=8,
            depth_peeling_layers=4,
            morton_ordering=True,
        )

        renderer = svraster.VolumeRenderer(config)
        logger.info(f"VolumeRenderer created, CUDA support: {getattr(renderer, 'use_cuda', False)}")

        # 2. 创建测试数据
        logger.info("创建测试数据...")
        voxels, ray_origins, ray_directions = create_test_data(
            device, num_rays=2048, num_voxels=2000
        )

        # 3. 基准测试
        logger.info("开始基准测试...")
        result = benchmark_rendering(renderer, voxels, ray_origins, ray_directions, num_runs=5)

        # 4. 验证结果
        logger.info("验证渲染结果...")
        assert "rgb" in result, "Missing 'rgb' in result"
        assert "depth" in result, "Missing 'depth' in result"
        assert "weights" in result, "Missing 'weights' in result"

        rgb = result["rgb"]
        depth = result["depth"]
        weights = result["weights"]

        logger.info(f"Rendered {rgb.shape[0]} rays")
        logger.info(f"RGB shape: {rgb.shape}, range: [{rgb.min():.3f}, {rgb.max():.3f}]")
        logger.info(f"Depth shape: {depth.shape}, range: [{depth.min():.3f}, {depth.max():.3f}]")
        logger.info(
            f"Weights shape: {weights.shape}, range: [{weights.min():.3f}, {weights.max():.3f}]"
        )

        logger.info("✅ VolumeRenderer CUDA 测试成功！")
        return True

    except Exception as e:
        logger.error(f"❌ VolumeRenderer CUDA 测试失败: {str(e)}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_volume_renderer_cuda()
    if success:
        print("✅ 所有测试通过！")
    else:
        print("❌ 测试失败！")
