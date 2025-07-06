"""
SVRaster 推理快速入门指南

这是使用 SVRaster 进行推理和渲染的最简单方法
"""

import torch
from nerfs.svraster import SVRasterModel, SVRasterConfig


def quick_inference():
    """最简单的推理方法"""
    
    # 1. 创建配置和模型
    config = SVRasterConfig()
    model = SVRasterModel(config)
    
    # 2. 准备输入数据
    # 光线起点 [N, 3] - 通常是相机位置
    rays_o = torch.randn(100, 3) * 0.1
    
    # 光线方向 [N, 3] - 从相机指向场景的方向
    rays_d = torch.randn(100, 3)
    rays_d = rays_d / torch.norm(rays_d, dim=1, keepdim=True)  # 归一化
    
    # 3. 推理渲染
    with torch.no_grad():
        # 使用 mode="inference" 进行光栅化渲染
        outputs = model(rays_o, rays_d, mode="inference")
    
    # 4. 获取结果
    rgb = outputs['rgb']      # 颜色 [N, 3]
    depth = outputs['depth']  # 深度 [N]
    
    print(f"渲染完成！RGB形状: {rgb.shape}, 深度形状: {depth.shape}")
    return rgb, depth


def inference_from_camera():
    """从相机参数生成光线并推理"""
    
    # 相机参数
    camera_pos = torch.tensor([0.0, 0.0, 3.0])  # 相机位置
    look_at = torch.tensor([0.0, 0.0, 0.0])     # 看向的目标点
    up = torch.tensor([0.0, 1.0, 0.0])          # 上方向
    
    # 构建视图矩阵
    forward = look_at - camera_pos
    forward = forward / torch.norm(forward)
    right = torch.cross(forward, up)
    right = right / torch.norm(right)
    up = torch.cross(right, forward)
    
    # 生成图像平面上的光线
    width, height = 64, 64  # 小图像用于演示
    focal_length = 50.0
    
    # 像素坐标
    i, j = torch.meshgrid(
        torch.arange(width, dtype=torch.float32),
        torch.arange(height, dtype=torch.float32),
        indexing='xy'
    )
    
    # 转换为相机坐标系中的方向
    dirs = torch.stack([
        (i - width * 0.5) / focal_length,
        -(j - height * 0.5) / focal_length,
        -torch.ones_like(i)
    ], dim=-1)
    
    # 转换到世界坐标系
    camera_matrix = torch.stack([right, up, -forward], dim=0)
    rays_d = torch.sum(dirs[..., None, :] * camera_matrix, dim=-1)
    rays_o = camera_pos.expand_as(rays_d)
    
    # 展平为光线批次
    rays_o = rays_o.reshape(-1, 3)
    rays_d = rays_d.reshape(-1, 3)
    
    # 推理
    config = SVRasterConfig()
    model = SVRasterModel(config)
    
    with torch.no_grad():
        outputs = model(rays_o, rays_d, mode="inference")
    
    # 重塑为图像
    rgb_image = outputs['rgb'].reshape(height, width, 3)
    depth_image = outputs['depth'].reshape(height, width)
    
    print(f"渲染图像完成！图像形状: {rgb_image.shape}")
    return rgb_image, depth_image


def compare_training_vs_inference():
    """比较训练模式和推理模式的渲染"""
    
    config = SVRasterConfig()
    model = SVRasterModel(config)
    
    # 相同的输入数据
    rays_o = torch.randn(50, 3) * 0.1
    rays_d = torch.randn(50, 3)
    rays_d = rays_d / torch.norm(rays_d, dim=1, keepdim=True)
    
    print("比较训练模式 vs 推理模式:")
    
    # 训练模式（体积渲染）
    with torch.no_grad():
        training_outputs = model(rays_o, rays_d, mode="training")
    
    print(f"训练模式输出: {list(training_outputs.keys())}")
    print(f"  RGB形状: {training_outputs['rgb'].shape}")
    
    # 推理模式（光栅化渲染）
    with torch.no_grad():
        inference_outputs = model(rays_o, rays_d, mode="inference")
    
    print(f"推理模式输出: {list(inference_outputs.keys())}")
    print(f"  RGB形状: {inference_outputs['rgb'].shape}")
    
    return training_outputs, inference_outputs


def load_checkpoint_and_render(checkpoint_path: str):
    """从检查点加载模型并渲染"""
    
    # 加载检查点
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # 创建配置（需要与训练时一致）
    config = SVRasterConfig()
    if 'model_config' in checkpoint:
        config = checkpoint['model_config']
    
    # 创建并加载模型
    model = SVRasterModel(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"已加载检查点，训练轮次: {checkpoint.get('epoch', '未知')}")
    
    # 使用加载的模型进行推理
    return quick_inference_with_model(model)


def quick_inference_with_model(model):
    """使用指定模型进行快速推理"""
    
    rays_o = torch.randn(100, 3) * 0.1
    rays_d = torch.randn(100, 3)
    rays_d = rays_d / torch.norm(rays_d, dim=1, keepdim=True)
    
    with torch.no_grad():
        outputs = model(rays_o, rays_d, mode="inference")
    
    return outputs


if __name__ == "__main__":
    print("SVRaster 推理快速入门")
    print("=" * 40)
    
    print("\n1. 基础推理:")
    rgb, depth = quick_inference()
    
    print("\n2. 从相机参数推理:")
    rgb_img, depth_img = inference_from_camera()
    
    print("\n3. 训练模式 vs 推理模式:")
    training_out, inference_out = compare_training_vs_inference()
    
    print("\n推理要点:")
    print("- 使用 mode='inference' 启用光栅化渲染")
    print("- 光栅化比体积渲染更快，适合实时应用")
    print("- 输入是光线起点和方向 [N, 3]")
    print("- 输出包含 RGB 颜色和深度信息")
    print("- 可以从检查点加载训练好的模型")
