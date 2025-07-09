#!/usr/bin/env python3
"""
基于投影的光栅化设计演示

展示 SVRaster 渲染器中新增的基于投影的光栅化功能：
1. 图像分块处理
2. 视锥剔除优化
3. 深度排序
4. 性能监控
"""

import sys
import os
import torch
import numpy as np
import time
from pathlib import Path

# 添加 src 目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from nerfs.svraster.renderer import (
    SVRasterRenderer,
    SVRasterRendererConfig,
    VoxelRasterizerConfig,
    TileConfig,
    FrustumCullingConfig,
    DepthSortingConfig,
)
from nerfs.svraster.core import SVRasterModel, SVRasterConfig


def create_test_model():
    """创建测试模型"""
    config = SVRasterConfig()
    model = SVRasterModel(config)

    # 创建一些测试体素
    n_voxels = 1000
    model.voxels.voxel_positions = [torch.randn(n_voxels, 3) * 2.0]
    model.voxels.voxel_sizes = [torch.ones(n_voxels) * 0.1]
    model.voxels.voxel_densities = [torch.randn(n_voxels)]
    model.voxels.voxel_colors = [torch.randn(n_voxels, 3)]
    model.voxels.voxel_morton_codes = [torch.randint(0, 1000000, (n_voxels,))]

    return model


def create_test_camera():
    """创建测试相机参数"""
    # 相机位姿 (world to camera)
    camera_pose = torch.eye(4)
    camera_pose[:3, 3] = torch.tensor([0.0, 0.0, -3.0])  # 相机位置

    # 相机内参
    intrinsics = torch.tensor([[800.0, 0.0, 400.0], [0.0, 800.0, 300.0], [0.0, 0.0, 1.0]])

    return camera_pose, intrinsics


def demo_basic_projection_rasterization():
    """演示基本的投影光栅化"""
    print("=== 基本投影光栅化演示 ===")

    # 创建测试模型
    model = create_test_model()

    # 创建配置
    rasterizer_config = VoxelRasterizerConfig(
        background_color=(0.1, 0.1, 0.1),
        near_plane=0.1,
        far_plane=100.0,
        density_activation="exp",
        color_activation="sigmoid",
        sh_degree=2,
    )

    renderer_config = SVRasterRendererConfig(
        image_width=800, image_height=600, background_color=(0.1, 0.1, 0.1), log_render_stats=True
    )

    # 创建渲染器
    from nerfs.svraster.voxel_rasterizer import VoxelRasterizer

    rasterizer = VoxelRasterizer(rasterizer_config)
    renderer = SVRasterRenderer(model, rasterizer, renderer_config)

    # 创建测试相机
    camera_pose, intrinsics = create_test_camera()

    # 渲染
    print("开始渲染...")
    start_time = time.time()
    result = renderer.render_image(camera_pose, intrinsics)
    render_time = time.time() - start_time

    print(f"渲染完成，耗时: {render_time:.3f}秒")
    print(f"输出图像尺寸: {result['rgb'].shape}")
    print(f"深度图尺寸: {result['depth'].shape}")

    return result


def demo_tile_configuration():
    """演示不同的分块配置"""
    print("\n=== 分块配置演示 ===")

    model = create_test_model()
    camera_pose, intrinsics = create_test_camera()

    # 测试不同的分块配置
    tile_configs = [
        ("小分块", TileConfig(tile_size=32, overlap=4)),
        ("中等分块", TileConfig(tile_size=64, overlap=8)),
        ("大分块", TileConfig(tile_size=128, overlap=16)),
    ]

    results = {}

    for name, tile_config in tile_configs:
        print(f"\n测试 {name} 配置...")

        rasterizer_config = VoxelRasterizerConfig()
        renderer_config = SVRasterRendererConfig(
            image_width=800, image_height=600, tile_config=tile_config, log_render_stats=True
        )

        from nerfs.svraster.voxel_rasterizer import VoxelRasterizer

        rasterizer = VoxelRasterizer(rasterizer_config)
        renderer = SVRasterRenderer(model, rasterizer, renderer_config)

        start_time = time.time()
        result = renderer.render_image(camera_pose, intrinsics)
        render_time = time.time() - start_time

        results[name] = {
            "render_time": render_time,
            "tile_count": renderer.render_stats["tile_count"],
            "avg_voxels_per_tile": np.mean(renderer.render_stats["voxels_per_tile"]),
        }

        print(f"  {name}: {render_time:.3f}秒, {renderer.render_stats['tile_count']}个分块")

    return results


def demo_frustum_culling():
    """演示视锥剔除功能"""
    print("\n=== 视锥剔除演示 ===")

    model = create_test_model()
    camera_pose, intrinsics = create_test_camera()

    # 测试不同的视锥剔除配置
    culling_configs = [
        ("无剔除", FrustumCullingConfig(enable_frustum_culling=False)),
        ("标准剔除", FrustumCullingConfig(enable_frustum_culling=True, culling_margin=0.1)),
        ("严格剔除", FrustumCullingConfig(enable_frustum_culling=True, culling_margin=0.0)),
    ]

    results = {}

    for name, culling_config in culling_configs:
        print(f"\n测试 {name}...")

        rasterizer_config = VoxelRasterizerConfig()
        renderer_config = SVRasterRendererConfig(
            image_width=800, image_height=600, frustum_config=culling_config, log_render_stats=True
        )

        from nerfs.svraster.voxel_rasterizer import VoxelRasterizer

        rasterizer = VoxelRasterizer(rasterizer_config)
        renderer = SVRasterRenderer(model, rasterizer, renderer_config)

        start_time = time.time()
        result = renderer.render_image(camera_pose, intrinsics)
        render_time = time.time() - start_time

        results[name] = {
            "render_time": render_time,
            "total_voxels": renderer.render_stats["total_voxels"],
            "visible_voxels": renderer.render_stats["visible_voxels"],
            "culled_voxels": renderer.render_stats["culled_voxels"],
            "culling_time": renderer.render_stats["culling_time_ms"],
        }

        print(f"  {name}: {render_time:.3f}秒")
        print(f"    总体素: {renderer.render_stats['total_voxels']}")
        print(f"    可见体素: {renderer.render_stats['visible_voxels']}")
        print(f"    剔除体素: {renderer.render_stats['culled_voxels']}")

    return results


def demo_depth_sorting():
    """演示深度排序功能"""
    print("\n=== 深度排序演示 ===")

    model = create_test_model()
    camera_pose, intrinsics = create_test_camera()

    # 测试不同的深度排序配置
    sorting_configs = [
        ("无排序", DepthSortingConfig(enable_depth_sorting=False)),
        ("后向前", DepthSortingConfig(enable_depth_sorting=True, sort_method="back_to_front")),
        ("前向后", DepthSortingConfig(enable_depth_sorting=True, sort_method="front_to_back")),
    ]

    results = {}

    for name, sorting_config in sorting_configs:
        print(f"\n测试 {name}...")

        rasterizer_config = VoxelRasterizerConfig()
        renderer_config = SVRasterRendererConfig(
            image_width=800, image_height=600, depth_config=sorting_config, log_render_stats=True
        )

        from nerfs.svraster.voxel_rasterizer import VoxelRasterizer

        rasterizer = VoxelRasterizer(rasterizer_config)
        renderer = SVRasterRenderer(model, rasterizer, renderer_config)

        start_time = time.time()
        result = renderer.render_image(camera_pose, intrinsics)
        render_time = time.time() - start_time

        results[name] = {
            "render_time": render_time,
            "sorting_time": renderer.render_stats["sorting_time_ms"],
        }

        print(f"  {name}: {render_time:.3f}秒")
        print(f"    排序时间: {renderer.render_stats['sorting_time_ms']:.2f}ms")

    return results


def demo_performance_analysis():
    """演示性能分析"""
    print("\n=== 性能分析演示 ===")

    model = create_test_model()
    camera_pose, intrinsics = create_test_camera()

    # 创建性能监控配置
    rasterizer_config = VoxelRasterizerConfig()
    renderer_config = SVRasterRendererConfig(
        image_width=800, image_height=600, enable_performance_monitoring=True, log_render_stats=True
    )

    from nerfs.svraster.voxel_rasterizer import VoxelRasterizer

    rasterizer = VoxelRasterizer(rasterizer_config)
    renderer = SVRasterRenderer(model, rasterizer, renderer_config)

    # 多次渲染以获得平均性能
    num_iterations = 5
    times = []

    print(f"进行 {num_iterations} 次渲染测试...")

    for i in range(num_iterations):
        start_time = time.time()
        result = renderer.render_image(camera_pose, intrinsics)
        render_time = time.time() - start_time
        times.append(render_time)

        print(f"  第 {i+1} 次: {render_time:.3f}秒")

    avg_time = np.mean(times)
    std_time = np.std(times)

    print(f"\n性能统计:")
    print(f"  平均渲染时间: {avg_time:.3f} ± {std_time:.3f}秒")
    print(f"  平均 FPS: {1.0/avg_time:.1f}")

    # 详细性能分析
    stats = renderer.render_stats
    print(f"\n详细性能分析:")
    print(
        f"  投影时间: {stats['projection_time_ms']:.2f}ms ({stats['projection_time_ms']/stats['render_time_ms']*100:.1f}%)"
    )
    print(
        f"  剔除时间: {stats['culling_time_ms']:.2f}ms ({stats['culling_time_ms']/stats['render_time_ms']*100:.1f}%)"
    )
    print(
        f"  排序时间: {stats['sorting_time_ms']:.2f}ms ({stats['sorting_time_ms']/stats['render_time_ms']*100:.1f}%)"
    )
    print(
        f"  光栅化时间: {stats['rasterization_time_ms']:.2f}ms ({stats['rasterization_time_ms']/stats['render_time_ms']*100:.1f}%)"
    )

    return {
        "avg_time": avg_time,
        "std_time": std_time,
        "fps": 1.0 / avg_time,
        "detailed_stats": stats,
    }


def demo_adaptive_tiling():
    """演示自适应分块"""
    print("\n=== 自适应分块演示 ===")

    model = create_test_model()
    camera_pose, intrinsics = create_test_camera()

    # 测试自适应分块
    tile_config = TileConfig(
        tile_size=64, overlap=8, use_adaptive_tiling=True, min_tile_size=32, max_tile_size=128
    )

    rasterizer_config = VoxelRasterizerConfig()
    renderer_config = SVRasterRendererConfig(
        image_width=800, image_height=600, tile_config=tile_config, log_render_stats=True
    )

    from nerfs.svraster.voxel_rasterizer import VoxelRasterizer

    rasterizer = VoxelRasterizer(rasterizer_config)
    renderer = SVRasterRenderer(model, rasterizer, renderer_config)

    start_time = time.time()
    result = renderer.render_image(camera_pose, intrinsics)
    render_time = time.time() - start_time

    print(f"自适应分块渲染时间: {render_time:.3f}秒")
    print(f"分块数量: {renderer.render_stats['tile_count']}")
    print(f"平均每分块体素数: {np.mean(renderer.render_stats['voxels_per_tile']):.1f}")

    return {
        "render_time": render_time,
        "tile_count": renderer.render_stats["tile_count"],
        "avg_voxels_per_tile": np.mean(renderer.render_stats["voxels_per_tile"]),
    }


def main():
    """主函数"""
    print("SVRaster 基于投影的光栅化设计演示")
    print("=" * 50)

    # 检查 CUDA 可用性
    if torch.cuda.is_available():
        print(f"CUDA 可用: {torch.cuda.get_device_name()}")
    else:
        print("CUDA 不可用，使用 CPU")

    try:
        # 1. 基本投影光栅化
        basic_result = demo_basic_projection_rasterization()

        # 2. 分块配置测试
        tile_results = demo_tile_configuration()

        # 3. 视锥剔除测试
        culling_results = demo_frustum_culling()

        # 4. 深度排序测试
        sorting_results = demo_depth_sorting()

        # 5. 性能分析
        performance_results = demo_performance_analysis()

        # 6. 自适应分块
        adaptive_results = demo_adaptive_tiling()

        # 总结
        print("\n" + "=" * 50)
        print("演示总结")
        print("=" * 50)

        print(f"基本渲染: {performance_results['avg_time']:.3f}秒")
        print(f"平均 FPS: {performance_results['fps']:.1f}")

        print(f"\n分块配置对比:")
        for name, result in tile_results.items():
            print(f"  {name}: {result['render_time']:.3f}秒, {result['tile_count']}个分块")

        print(f"\n视锥剔除效果:")
        for name, result in culling_results.items():
            culling_ratio = result["culled_voxels"] / result["total_voxels"] * 100
            print(f"  {name}: 剔除率 {culling_ratio:.1f}%, 时间 {result['render_time']:.3f}秒")

        print(f"\n深度排序对比:")
        for name, result in sorting_results.items():
            print(f"  {name}: {result['render_time']:.3f}秒")

        print("\n演示完成！")

    except Exception as e:
        print(f"演示过程中出现错误: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
