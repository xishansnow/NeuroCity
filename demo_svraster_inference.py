"""
SVRaster 推理和渲染演示

这个脚本展示如何使用 SVRaster 进行实际的推理和渲染
"""

import torch
import numpy as np
from pathlib import Path
import time

# 导入SVRaster组件
from src.nerfs.svraster import SVRasterModel, SVRasterConfig


def demo_inference():
    """演示SVRaster推理和渲染"""
    
    print("=== SVRaster 推理和渲染演示 ===\n")
    
    # 1. 创建配置
    print("1. 创建配置...")
    config = SVRasterConfig(
        image_width=800,
        image_height=600,
        scene_bounds=(-2.0, -2.0, -2.0, 2.0, 2.0, 2.0),
        sh_degree=2,
        grid_resolution=128
    )
    print(f"   - 图像分辨率: {config.image_width}x{config.image_height}")
    print(f"   - 场景边界: {config.scene_bounds}")
    print(f"   - 体素网格分辨率: {config.grid_resolution}")
    
    # 2. 初始化模型
    print("\n2. 初始化模型...")
    model = SVRasterModel(config)
    print(f"   - 模型设备: {model.device}")
    print(f"   - 体素数量: {model.voxel_grid.shape}")
    
    # 3. 生成一些测试体素数据（模拟训练后的结果）
    print("\n3. 生成测试体素数据...")
    with torch.no_grad():
        # 在场景中心创建一个球形对象
        coords = torch.stack(torch.meshgrid(
            torch.linspace(-1, 1, config.grid_resolution),
            torch.linspace(-1, 1, config.grid_resolution),
            torch.linspace(-1, 1, config.grid_resolution),
            indexing='ij'
        ), dim=-1)
        
        # 创建球形密度分布
        distances = torch.norm(coords, dim=-1)
        densities = torch.exp(-distances * 3.0)  # 高斯球
        
        # 设置体素数据
        model.voxel_grid = densities.unsqueeze(-1).to(model.device)
        model.feature_grid = torch.randn_like(model.feature_grid) * 0.1
        
    print(f"   - 体素网格形状: {model.voxel_grid.shape}")
    print(f"   - 特征网格形状: {model.feature_grid.shape}")
    
    # 4. 生成相机光线
    print("\n4. 生成相机光线...")
    
    # 相机参数
    camera_pos = torch.tensor([0.0, 0.0, 3.0], device=model.device)
    image_center = torch.tensor([0.0, 0.0, 0.0], device=model.device)
    
    # 生成图像平面上的像素坐标
    num_rays = 1000  # 采样1000条光线进行快速演示
    
    # 随机采样像素位置
    pixel_coords = torch.rand(num_rays, 2, device=model.device) * 2 - 1  # [-1, 1]
    
    # 转换为3D光线
    rays_o = camera_pos.unsqueeze(0).expand(num_rays, -1)
    
    # 计算光线方向（从相机指向像素）
    rays_d = torch.stack([
        pixel_coords[:, 0] * 0.5,  # x方向
        pixel_coords[:, 1] * 0.5,  # y方向
        torch.full((num_rays,), -1.0, device=model.device)  # z方向（向前）
    ], dim=-1)
    
    # 归一化光线方向
    rays_d = rays_d / torch.norm(rays_d, dim=1, keepdim=True)
    
    print(f"   - 光线数量: {num_rays}")
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
    print(f"   - RGB形状: {training_outputs['rgb'].shape}")
    print(f"   - 深度形状: {training_outputs['depth'].shape}")
    print(f"   - RGB范围: [{training_outputs['rgb'].min():.3f}, {training_outputs['rgb'].max():.3f}]")
    
    # 6. 推理模式渲染（光栅化）
    print("\n6. 推理模式渲染（光栅化）...")
    start_time = time.time()
    
    with torch.no_grad():
        inference_outputs = model(rays_o, rays_d, mode="inference")
    
    inference_time = time.time() - start_time
    print(f"   - 渲染时间: {inference_time:.3f}秒")
    print(f"   - 输出键: {list(inference_outputs.keys())}")
    print(f"   - RGB形状: {inference_outputs['rgb'].shape}")
    print(f"   - 深度形状: {inference_outputs['depth'].shape}")
    print(f"   - RGB范围: [{inference_outputs['rgb'].min():.3f}, {inference_outputs['rgb'].max():.3f}]")
    
    # 7. 比较两种模式的结果
    print("\n7. 比较两种渲染模式...")
    
    rgb_diff = torch.mean(torch.abs(training_outputs['rgb'] - inference_outputs['rgb']))
    depth_diff = torch.mean(torch.abs(training_outputs['depth'] - inference_outputs['depth']))
    
    print(f"   - RGB平均差异: {rgb_diff:.6f}")
    print(f"   - 深度平均差异: {depth_diff:.6f}")
    print(f"   - 速度提升: {training_time/inference_time:.2f}x")
    
    # 8. 保存结果（可选）
    print("\n8. 保存结果...")
    
    # 将光线结果重塑为图像格式（如果需要）
    if num_rays == config.image_width * config.image_height:
        training_image = training_outputs['rgb'].reshape(config.image_height, config.image_width, 3)
        inference_image = inference_outputs['rgb'].reshape(config.image_height, config.image_width, 3)
        
        # 这里可以保存图像
        print("   - 可以保存为图像格式")
    else:
        print(f"   - 当前是采样渲染（{num_rays}条光线），可扩展为完整图像渲染")
    
    print("\n=== 演示完成 ===")
    
    return training_outputs, inference_outputs


def demo_batch_inference():
    """演示批量推理"""
    
    print("\n=== 批量推理演示 ===\n")
    
    # 创建模型
    config = SVRasterConfig(grid_resolution=64)  # 较小的网格用于快速演示
    model = SVRasterModel(config)
    
    # 生成测试数据
    with torch.no_grad():
        model.voxel_grid = torch.rand_like(model.voxel_grid) * 0.5
        model.feature_grid = torch.randn_like(model.feature_grid) * 0.1
    
    # 测试不同批量大小
    batch_sizes = [100, 500, 1000, 5000]
    
    for batch_size in batch_sizes:
        print(f"批量大小: {batch_size}")
        
        # 生成光线
        rays_o = torch.randn(batch_size, 3, device=model.device) * 0.1
        rays_d = torch.randn(batch_size, 3, device=model.device)
        rays_d = rays_d / torch.norm(rays_d, dim=1, keepdim=True)
        
        # 推理
        start_time = time.time()
        with torch.no_grad():
            outputs = model(rays_o, rays_d, mode="inference")
        
        elapsed = time.time() - start_time
        rays_per_sec = batch_size / elapsed
        
        print(f"   - 渲染时间: {elapsed:.3f}秒")
        print(f"   - 光线/秒: {rays_per_sec:.0f}")
        print()


def demo_camera_path_rendering():
    """演示相机路径渲染"""
    
    print("\n=== 相机路径渲染演示 ===\n")
    
    config = SVRasterConfig(grid_resolution=64)
    model = SVRasterModel(config)
    
    # 生成测试场景
    with torch.no_grad():
        coords = torch.stack(torch.meshgrid(
            torch.linspace(-1, 1, config.grid_resolution),
            torch.linspace(-1, 1, config.grid_resolution),
            torch.linspace(-1, 1, config.grid_resolution),
            indexing='ij'
        ), dim=-1)
        
        # 创建更复杂的场景
        distances = torch.norm(coords, dim=-1)
        densities = torch.exp(-distances * 2.0) + torch.exp(-torch.norm(coords - torch.tensor([0.5, 0, 0]), dim=-1) * 3.0) * 0.5
        
        model.voxel_grid = densities.unsqueeze(-1).to(model.device)
        model.feature_grid = torch.randn_like(model.feature_grid) * 0.1
    
    # 定义相机路径（围绕场景旋转）
    num_frames = 8
    radius = 2.5
    
    for frame in range(num_frames):
        angle = frame * 2 * np.pi / num_frames
        
        # 计算相机位置
        camera_pos = torch.tensor([
            radius * np.cos(angle),
            0.0,
            radius * np.sin(angle)
        ], device=model.device)
        
        # 生成朝向场景中心的光线
        num_rays = 400  # 20x20的小图像
        
        # 创建图像平面网格
        i, j = torch.meshgrid(
            torch.linspace(-0.5, 0.5, 20),
            torch.linspace(-0.5, 0.5, 20),
            indexing='ij'
        )
        
        # 展平为光线
        rays_o = camera_pos.unsqueeze(0).expand(num_rays, -1)
        
        # 计算光线方向
        directions = torch.stack([
            i.flatten(),
            j.flatten(),
            torch.full((num_rays,), -1.0)
        ], dim=-1).to(model.device)
        
        # 将方向转换到世界坐标系
        forward = -camera_pos / torch.norm(camera_pos)
        right = torch.cross(torch.tensor([0., 1., 0.], device=model.device), forward)
        right = right / torch.norm(right)
        up = torch.cross(forward, right)
        
        rays_d = (directions[:, 0:1] * right + 
                 directions[:, 1:2] * up + 
                 directions[:, 2:3] * forward)
        rays_d = rays_d / torch.norm(rays_d, dim=1, keepdim=True)
        
        # 渲染当前帧
        start_time = time.time()
        with torch.no_grad():
            outputs = model(rays_o, rays_d, mode="inference")
        
        elapsed = time.time() - start_time
        
        print(f"帧 {frame+1}/{num_frames}:")
        print(f"   - 相机位置: [{camera_pos[0]:.2f}, {camera_pos[1]:.2f}, {camera_pos[2]:.2f}]")
        print(f"   - 渲染时间: {elapsed:.3f}秒")
        print(f"   - RGB范围: [{outputs['rgb'].min():.3f}, {outputs['rgb'].max():.3f}]")


if __name__ == "__main__":
    # 运行演示
    try:
        # 基础推理演示
        training_outputs, inference_outputs = demo_inference()
        
        # 批量推理演示
        demo_batch_inference()
        
        # 相机路径渲染演示
        demo_camera_path_rendering()
        
        print("\n所有演示完成！")
        
    except Exception as e:
        print(f"演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
