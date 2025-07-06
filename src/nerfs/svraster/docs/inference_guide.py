"""
from __future__ import annotations

SVRaster 推理和渲染使用指南

展示如何使用 SVRaster 进行推理渲染的完整示例
"""

from typing import Optional, List, Dict, Tuple, Union

import torch
import numpy as np
from pathlib import Path

# 导入SVRaster组件
from nerfs.svraster import SVRasterModel, SVRasterConfig
from nerfs.svraster.renderer import SVRasterRenderer, SVRasterRendererConfig


def basic_inference_example():
    """基础推理示例：使用模型直接推理"""
    
    print("=== SVRaster 基础推理示例 ===")
    
    # 1. 创建配置
    config = SVRasterConfig(
        image_width=800,
        image_height=600,
        scene_bounds=(-2.0, -2.0, -2.0, 2.0, 2.0, 2.0),
        sh_degree=2
    )
    
    # 2. 初始化模型
    model = SVRasterModel(config)
    print(f"模型已初始化，设备: {model.device}")
    
    # 3. 生成测试光线
    # 模拟从特定视点看向场景的光线
    num_rays = 1000
    rays_o = torch.randn(num_rays, 3) * 0.1  # 光线起点
    rays_d = torch.randn(num_rays, 3)        # 光线方向
    rays_d = rays_d / torch.norm(rays_d, dim=1, keepdim=True)  # 归一化方向
    
    print(f"生成 {num_rays} 条测试光线")
    
    # 4. 推理渲染（使用光栅化模式）
    with torch.no_grad():
        outputs = model(rays_o, rays_d, mode="inference")
    
    print(f"推理完成！")
    print(f"输出键: {list(outputs.keys())}")
    print(f"RGB形状: {outputs['rgb'].shape}")
    print(f"深度形状: {outputs['depth'].shape}")
    
    return outputs


def advanced_rendering_example():
    """高级渲染示例：使用渲染器进行完整图像渲染"""
    
    print("\n=== SVRaster 高级渲染示例 ===")
    
    # 1. 创建渲染器配置
    renderer_config = SVRasterRendererConfig(
        image_width=800,
        image_height=600,
        quality_level="high",
        use_half_precision=False,
        background_color=(0.0, 0.0, 0.0)
    )
    
    # 2. 初始化渲染器
    renderer = SVRasterRenderer(renderer_config)
    
    # 3. 假设已有训练好的模型（这里用新模型演示）
    model_config = SVRasterConfig()
    model = SVRasterModel(model_config)
    
    # 模拟加载检查点
    print("模拟加载训练好的模型...")
    renderer.model = model
    renderer.model_config = model_config
    renderer.is_model_loaded = True
    
    # 4. 设置相机参数
    # 相机位姿矩阵 (world-to-camera)
    camera_pose = torch.eye(4)
    camera_pose[2, 3] = 3.0  # 相机在z=3位置
    
    # 相机内参
    focal_length = 800.0
    camera_intrinsics = torch.tensor([
        [focal_length, 0, 400],
        [0, focal_length, 300],
        [0, 0, 1]
    ], dtype=torch.float32)
    
    # 5. 渲染完整图像
    print("开始渲染完整图像...")
    try:
        outputs = renderer.render_single_view(
            camera_pose=camera_pose,
            intrinsics=camera_intrinsics,
            image_size=(600, 800)
        )
        
        print(f"渲染完成！")
        print(f"图像形状: {outputs['rgb'].shape}")
        
        # 保存渲染结果
        rgb_image = outputs['rgb'].cpu().numpy()
        rgb_image = np.clip(rgb_image * 255, 0, 255).astype(np.uint8)
        
        # 这里可以使用PIL或其他库保存图像
        print(f"渲染图像范围: [{rgb_image.min()}, {rgb_image.max()}]")
        
    except Exception as e:
        print(f"渲染过程中出现错误: {e}")
        print("这是正常的，因为我们使用的是未训练的模型")
    
    return renderer


def batch_rendering_example():
    """批量渲染示例：渲染多个视点"""
    
    print("\n=== SVRaster 批量渲染示例 ===")
    
    # 初始化
    config = SVRasterConfig()
    model = SVRasterModel(config)
    
    # 生成多个视点的相机位姿
    num_views = 8
    radius = 3.0
    
    rendered_images = []
    
    for i in range(num_views):
        # 圆形轨迹上的相机位置
        angle = 2 * np.pi * i / num_views
        camera_x = radius * np.cos(angle)
        camera_z = radius * np.sin(angle)
        
        # 创建look-at矩阵
        camera_pos = torch.tensor([camera_x, 0.0, camera_z])
        target_pos = torch.tensor([0.0, 0.0, 0.0])
        up_vector = torch.tensor([0.0, 1.0, 0.0])
        
        # 构建相机矩阵
        forward = target_pos - camera_pos
        forward = forward / torch.norm(forward)
        right = torch.cross(forward, up_vector)
        right = right / torch.norm(right)
        up = torch.cross(right, forward)
        
        camera_matrix = torch.eye(4)
        camera_matrix[:3, 0] = right
        camera_matrix[:3, 1] = up
        camera_matrix[:3, 2] = -forward
        camera_matrix[:3, 3] = camera_pos
        
        # 生成该视点的光线
        height, width = 200, 200
        rays_o, rays_d = generate_rays(camera_matrix, width, height)
        
        # 渲染
        with torch.no_grad():
            outputs = model(rays_o.flatten(0, 1), rays_d.flatten(0, 1), mode="inference")
        
        # 重塑为图像
        rgb_image = outputs['rgb'].view(height, width, 3)
        rendered_images.append(rgb_image)
        
        print(f"完成视点 {i+1}/{num_views}")
    
    print(f"批量渲染完成，生成 {len(rendered_images)} 张图像")
    return rendered_images


def generate_rays(camera_matrix, width, height, focal_length=800.0):
    """生成给定相机参数的光线"""
    
    # 生成像素坐标
    i, j = torch.meshgrid(
        torch.arange(width, dtype=torch.float32),
        torch.arange(height, dtype=torch.float32),
        indexing='xy'
    )
    
    # 转换到相机坐标
    dirs = torch.stack([
        (i - width * 0.5) / focal_length,
        -(j - height * 0.5) / focal_length,
        -torch.ones_like(i)
    ], dim=-1)
    
    # 转换到世界坐标
    rays_d = torch.sum(dirs[..., None, :] * camera_matrix[:3, :3], dim=-1)
    rays_o = camera_matrix[:3, 3].expand_as(rays_d)
    
    return rays_o, rays_d


def load_and_render_example(checkpoint_path: Optional[str] = None):
    """从检查点加载模型并渲染"""
    
    print("\n=== 从检查点加载模型渲染示例 ===")
    
    if checkpoint_path and Path(checkpoint_path).exists():
        # 加载配置
        config = SVRasterConfig()
        model = SVRasterModel(config)
        
        # 加载检查点
        checkpoint = torch.load(checkpoint_path, map_location=model.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        print(f"成功加载检查点: {checkpoint_path}")
        print(f"训练轮次: {checkpoint.get('epoch', 'unknown')}")
        
        # 使用加载的模型进行渲染
        return basic_inference_example_with_model(model)
    
    else:
        print("未提供有效的检查点路径，使用随机初始化的模型演示")
        return basic_inference_example()


def basic_inference_example_with_model(model):
    """使用给定模型进行推理"""
    
    # 生成测试数据
    rays_o = torch.randn(100, 3) * 0.1
    rays_d = torch.randn(100, 3)
    rays_d = rays_d / torch.norm(rays_d, dim=1, keepdim=True)
    
    # 推理
    with torch.no_grad():
        outputs = model(rays_o, rays_d, mode="inference")
    
    return outputs


if __name__ == "__main__":
    print("SVRaster 推理和渲染使用指南")
    print("=" * 50)
    
    # 运行示例
    try:
        # 基础推理
        basic_outputs = basic_inference_example()
        
        # 高级渲染
        renderer = advanced_rendering_example()
        
        # 批量渲染
        batch_images = batch_rendering_example()
        
        # 检查点加载（可选）
        # load_and_render_example("path/to/your/checkpoint.pth")
        
        print("\n" + "=" * 50)
        print("所有示例运行完成！")
        print("\n推理模式特点:")
        print("- 使用光栅化渲染（TrueVoxelRasterizer）")
        print("- 比训练时的体积渲染更快")
        print("- 适合实时或交互式应用")
        print("- 支持透明度和深度输出")
        
    except Exception as e:
        print(f"运行示例时出现错误: {e}")
        print("这可能是由于模型未完全训练或依赖项问题")
