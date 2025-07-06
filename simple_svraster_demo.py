"""
SVRaster 简化推理演示

这个脚本展示SVRaster的核心推理功能，不依赖复杂的数据集模块
"""

import torch
import numpy as np
import time
import sys
import os

# 添加项目路径
sys.path.append('/home/xishansnow/3DVision/NeuroCity')

# 只导入核心组件
from src.nerfs.svraster.core import SVRasterModel, SVRasterConfig


def simple_inference_demo():
    """简化的推理演示"""
    
    print("=== SVRaster 简化推理演示 ===\n")
    
    # 1. 创建配置
    print("1. 创建配置...")
    config = SVRasterConfig(
        image_width=400,
        image_height=300,
        scene_bounds=(-1.5, -1.5, -1.5, 1.5, 1.5, 1.5),
        sh_degree=1,
        grid_resolution=64  # 较小的网格用于快速演示
    )
    print(f"   - 图像分辨率: {config.image_width}x{config.image_height}")
    print(f"   - 场景边界: {config.scene_bounds}")
    print(f"   - 网格分辨率: {config.grid_resolution}")
    
    # 2. 初始化模型
    print("\n2. 初始化模型...")
    model = SVRasterModel(config)
    print(f"   - 模型设备: {model.device}")
    print(f"   - 体素网格形状: {model.voxel_grid.shape}")
    print(f"   - 特征网格形状: {model.feature_grid.shape}")
    
    # 3. 创建测试场景
    print("\n3. 创建测试场景...")
    with torch.no_grad():
        # 生成坐标网格
        res = config.grid_resolution
        coords = torch.stack(torch.meshgrid(
            torch.linspace(-1, 1, res),
            torch.linspace(-1, 1, res),
            torch.linspace(-1, 1, res),
            indexing='ij'
        ), dim=-1)
        
        # 创建两个球形对象
        center1 = torch.tensor([0.3, 0.0, 0.0])
        center2 = torch.tensor([-0.3, 0.0, 0.0])
        
        dist1 = torch.norm(coords - center1, dim=-1)
        dist2 = torch.norm(coords - center2, dim=-1)
        
        # 高斯分布密度
        density1 = torch.exp(-dist1 * 4.0) * 0.8
        density2 = torch.exp(-dist2 * 3.0) * 0.6
        
        # 合并密度
        total_density = density1 + density2
        
        # 设置体素网格
        model.voxel_grid = total_density.unsqueeze(-1).to(model.device)
        
        # 创建彩色特征（让两个球有不同颜色）
        features = torch.zeros(*coords.shape[:-1], model.feature_grid.shape[-1])
        
        # 第一个球（红色）
        mask1 = dist1 < 0.5
        features[mask1, 0] = 1.0  # 红色通道
        
        # 第二个球（蓝色）
        mask2 = dist2 < 0.5
        features[mask2, 2] = 1.0  # 蓝色通道
        
        # 添加一些噪声
        features += torch.randn_like(features) * 0.1
        
        model.feature_grid = features.to(model.device)
    
    print(f"   - 创建了两个球形对象")
    print(f"   - 密度范围: [{model.voxel_grid.min():.3f}, {model.voxel_grid.max():.3f}]")
    
    # 4. 生成测试光线
    print("\n4. 生成测试光线...")
    
    # 相机位置
    camera_pos = torch.tensor([0.0, 0.0, 2.0], device=model.device)
    
    # 生成光线（从相机指向场景）
    num_rays = 1000
    
    # 在图像平面上随机采样
    pixel_coords = torch.rand(num_rays, 2, device=model.device) * 2 - 1  # [-1, 1]
    
    # 光线起点（所有光线都从相机位置开始）
    rays_o = camera_pos.unsqueeze(0).expand(num_rays, -1)
    
    # 光线方向（从相机指向场景中的像素）
    rays_d = torch.stack([
        pixel_coords[:, 0] * 0.8,  # x方向
        pixel_coords[:, 1] * 0.6,  # y方向
        torch.full((num_rays,), -1.0, device=model.device)  # z方向（向前）
    ], dim=-1)
    
    # 归一化方向
    rays_d = rays_d / torch.norm(rays_d, dim=1, keepdim=True)
    
    print(f"   - 生成了 {num_rays} 条光线")
    print(f"   - 相机位置: {camera_pos}")
    print(f"   - 光线起点形状: {rays_o.shape}")
    print(f"   - 光线方向形状: {rays_d.shape}")
    
    # 5. 训练模式渲染（体积渲染）
    print("\n5. 训练模式渲染（体积渲染）...")
    
    start_time = time.time()
    with torch.no_grad():
        training_outputs = model(rays_o, rays_d, mode="training")
    training_time = time.time() - start_time
    
    print(f"   - 渲染时间: {training_time:.3f}秒")
    print(f"   - 输出键: {list(training_outputs.keys())}")
    if 'rgb' in training_outputs:
        rgb = training_outputs['rgb']
        print(f"   - RGB形状: {rgb.shape}")
        print(f"   - RGB范围: [{rgb.min():.3f}, {rgb.max():.3f}]")
    if 'depth' in training_outputs:
        depth = training_outputs['depth']
        print(f"   - 深度形状: {depth.shape}")
        print(f"   - 深度范围: [{depth.min():.3f}, {depth.max():.3f}]")
    
    # 6. 推理模式渲染（光栅化）
    print("\n6. 推理模式渲染（光栅化）...")
    
    start_time = time.time()
    with torch.no_grad():
        inference_outputs = model(rays_o, rays_d, mode="inference")
    inference_time = time.time() - start_time
    
    print(f"   - 渲染时间: {inference_time:.3f}秒")
    print(f"   - 输出键: {list(inference_outputs.keys())}")
    if 'rgb' in inference_outputs:
        rgb = inference_outputs['rgb']
        print(f"   - RGB形状: {rgb.shape}")
        print(f"   - RGB范围: [{rgb.min():.3f}, {rgb.max():.3f}]")
    if 'depth' in inference_outputs:
        depth = inference_outputs['depth']
        print(f"   - 深度形状: {depth.shape}")
        print(f"   - 深度范围: [{depth.min():.3f}, {depth.max():.3f}]")
    
    # 7. 比较结果
    print("\n7. 比较两种渲染模式...")
    
    if 'rgb' in training_outputs and 'rgb' in inference_outputs:
        rgb_diff = torch.mean(torch.abs(training_outputs['rgb'] - inference_outputs['rgb']))
        print(f"   - RGB平均差异: {rgb_diff:.6f}")
    
    if 'depth' in training_outputs and 'depth' in inference_outputs:
        depth_diff = torch.mean(torch.abs(training_outputs['depth'] - inference_outputs['depth']))
        print(f"   - 深度平均差异: {depth_diff:.6f}")
    
    if training_time > 0 and inference_time > 0:
        speedup = training_time / inference_time
        print(f"   - 速度提升: {speedup:.2f}x")
    
    # 8. 性能测试
    print("\n8. 性能测试...")
    
    batch_sizes = [100, 500, 1000]
    
    for batch_size in batch_sizes:
        # 生成测试光线
        test_rays_o = torch.randn(batch_size, 3, device=model.device) * 0.1
        test_rays_d = torch.randn(batch_size, 3, device=model.device)
        test_rays_d = test_rays_d / torch.norm(test_rays_d, dim=1, keepdim=True)
        
        # 推理模式性能测试
        start_time = time.time()
        with torch.no_grad():
            _ = model(test_rays_o, test_rays_d, mode="inference")
        elapsed = time.time() - start_time
        
        rays_per_sec = batch_size / elapsed if elapsed > 0 else float('inf')
        print(f"   - 批量大小 {batch_size}: {elapsed:.3f}秒, {rays_per_sec:.0f} 光线/秒")
    
    print("\n=== 演示完成 ===")
    
    return training_outputs, inference_outputs


def demo_usage_guide():
    """演示使用指南"""
    
    print("\n=== SVRaster 使用指南 ===\n")
    
    print("1. 基本使用流程:")
    print("   a) 创建配置: config = SVRasterConfig(...)")
    print("   b) 初始化模型: model = SVRasterModel(config)")
    print("   c) 训练模式: outputs = model(rays_o, rays_d, mode='training')")
    print("   d) 推理模式: outputs = model(rays_o, rays_d, mode='inference')")
    
    print("\n2. 关键参数说明:")
    print("   - image_width/height: 图像分辨率")
    print("   - scene_bounds: 场景边界 (x_min, y_min, z_min, x_max, y_max, z_max)")
    print("   - grid_resolution: 体素网格分辨率")
    print("   - sh_degree: 球谐函数阶数")
    
    print("\n3. 输入数据格式:")
    print("   - rays_o: 光线起点 [N, 3]")
    print("   - rays_d: 光线方向 [N, 3] (需要归一化)")
    
    print("\n4. 输出数据格式:")
    print("   - rgb: 颜色 [N, 3]")
    print("   - depth: 深度 [N]")
    print("   - alpha: 透明度 [N] (如果可用)")
    
    print("\n5. 两种渲染模式:")
    print("   - training: 体积渲染，用于训练，较慢但准确")
    print("   - inference: 光栅化渲染，用于推理，较快")
    
    print("\n6. 性能优化建议:")
    print("   - 使用较小的 grid_resolution 进行快速测试")
    print("   - 批量处理光线以提高效率")
    print("   - 推理时使用 torch.no_grad() 减少内存使用")


if __name__ == "__main__":
    try:
        # 运行简化演示
        training_outputs, inference_outputs = simple_inference_demo()
        
        # 显示使用指南
        demo_usage_guide()
        
        print("\n🎉 所有演示完成!")
        
    except Exception as e:
        print(f"\n❌ 演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
