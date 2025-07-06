"""
SVRaster 使用示例和指南

这是一个完整的、实用的 SVRaster 使用示例，展示如何进行推理和渲染。
"""

import torch
import numpy as np
import time
from typing import Dict, Tuple, Optional

def create_test_scene():
    """创建一个简单的测试场景"""
    
    # 场景参数
    resolution = 64
    scene_bounds = (-1.0, -1.0, -1.0, 1.0, 1.0, 1.0)
    
    # 创建坐标网格
    coords = torch.stack(torch.meshgrid(
        torch.linspace(-1, 1, resolution),
        torch.linspace(-1, 1, resolution),
        torch.linspace(-1, 1, resolution),
        indexing='ij'
    ), dim=-1)
    
    # 创建两个球体
    center1 = torch.tensor([0.3, 0.0, 0.0])
    center2 = torch.tensor([-0.3, 0.0, 0.0])
    
    dist1 = torch.norm(coords - center1, dim=-1)
    dist2 = torch.norm(coords - center2, dim=-1)
    
    # 密度分布
    density1 = torch.exp(-dist1 * 4.0) * 0.8
    density2 = torch.exp(-dist2 * 3.0) * 0.6
    densities = density1 + density2
    
    # 颜色特征
    colors = torch.zeros(*coords.shape)  # 与coords相同的形状
    mask1 = dist1 < 0.3
    mask2 = dist2 < 0.3
    colors[mask1, 0] = 1.0  # 红色球
    colors[mask2, 2] = 1.0  # 蓝色球
    
    return {
        'positions': coords.reshape(-1, 3),
        'densities': densities.reshape(-1),
        'colors': colors.reshape(-1, 3),
        'sizes': torch.full((resolution**3,), 0.1),
        'scene_bounds': scene_bounds
    }


def generate_camera_rays(camera_pos: torch.Tensor, 
                        target_pos: torch.Tensor,
                        image_size: Tuple[int, int],
                        num_rays: int = 1000) -> Tuple[torch.Tensor, torch.Tensor]:
    """生成相机光线"""
    
    # 相机方向
    forward = target_pos - camera_pos
    forward = forward / torch.norm(forward)
    
    # 构建相机坐标系
    up = torch.tensor([0.0, 1.0, 0.0])
    right = torch.cross(forward, up)
    right = right / torch.norm(right)
    up = torch.cross(right, forward)
    
    # 随机采样图像像素
    width, height = image_size
    pixel_coords = torch.rand(num_rays, 2) * 2 - 1  # [-1, 1]
    
    # 计算光线方向
    ray_directions = (pixel_coords[:, 0:1] * right * 0.5 + 
                     pixel_coords[:, 1:2] * up * 0.5 + 
                     forward)
    ray_directions = ray_directions / torch.norm(ray_directions, dim=1, keepdim=True)
    
    # 所有光线从相机位置开始
    ray_origins = camera_pos.unsqueeze(0).expand(num_rays, -1)
    
    return ray_origins, ray_directions


def volume_rendering(voxel_data: Dict, 
                    rays_o: torch.Tensor, 
                    rays_d: torch.Tensor,
                    t_near: float = 0.1,
                    t_far: float = 4.0,
                    num_samples: int = 64) -> Dict[str, torch.Tensor]:
    """简化的体积渲染（训练模式）"""
    
    device = rays_o.device
    num_rays = rays_o.shape[0]
    
    # 沿光线采样
    t_vals = torch.linspace(t_near, t_far, num_samples, device=device)
    t_vals = t_vals.unsqueeze(0).expand(num_rays, -1)
    
    # 添加随机扰动
    t_vals = t_vals + torch.rand_like(t_vals) * (t_far - t_near) / num_samples
    
    # 计算采样点
    pts = rays_o.unsqueeze(1) + rays_d.unsqueeze(1) * t_vals.unsqueeze(2)
    
    # 查询体素密度和颜色（简化版）
    densities = torch.zeros(num_rays, num_samples, device=device)
    colors = torch.zeros(num_rays, num_samples, 3, device=device)
    
    # 简化的体素查询
    for i in range(num_rays):
        for j in range(num_samples):
            pt = pts[i, j]
            # 简单的距离查询
            dist = torch.norm(pt)
            if dist < 1.0:
                densities[i, j] = torch.exp(-dist * 2.0)
                colors[i, j] = torch.sigmoid(pt)  # 简单的颜色映射
    
    # 体积渲染积分
    delta = t_vals[:, 1:] - t_vals[:, :-1]
    delta = torch.cat([delta, torch.full_like(delta[:, :1], 1e10)], dim=1)
    
    alpha = 1.0 - torch.exp(-densities * delta)
    transmittance = torch.cumprod(1.0 - alpha + 1e-10, dim=1)
    transmittance = torch.cat([torch.ones_like(transmittance[:, :1]), transmittance[:, :-1]], dim=1)
    
    weights = alpha * transmittance
    
    # 最终颜色和深度
    rgb = torch.sum(weights.unsqueeze(2) * colors, dim=1)
    depth = torch.sum(weights * t_vals, dim=1)
    
    return {
        'rgb': rgb,
        'depth': depth,
        'weights': weights
    }


def rasterization_rendering(voxel_data: Dict,
                           rays_o: torch.Tensor,
                           rays_d: torch.Tensor) -> Dict[str, torch.Tensor]:
    """简化的光栅化渲染（推理模式）"""
    
    device = rays_o.device
    num_rays = rays_o.shape[0]
    
    # 简化的光栅化：直接查询最近的体素
    rgb = torch.zeros(num_rays, 3, device=device)
    depth = torch.full((num_rays,), 4.0, device=device)
    
    # 对每条光线进行简化的光栅化
    for i in range(num_rays):
        ray_o = rays_o[i]
        ray_d = rays_d[i]
        
        # 简单的光线步进
        for t in torch.linspace(0.1, 4.0, 32, device=device):
            pt = ray_o + ray_d * t
            
            # 检查是否在场景内
            if torch.all(torch.abs(pt) < 1.0):
                # 简单的密度查询
                dist = torch.norm(pt)
                if dist < 0.5:
                    density = torch.exp(-dist * 3.0)
                    if density > 0.1:
                        rgb[i] = torch.sigmoid(pt)
                        depth[i] = t
                        break
    
    return {
        'rgb': rgb,
        'depth': depth
    }


def demo_svraster_usage():
    """演示 SVRaster 使用方法"""
    
    print("=== SVRaster 使用演示 ===\n")
    
    # 检查设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 1. 创建测试场景
    print("\n1. 创建测试场景...")
    scene_data = create_test_scene()
    print(f"   - 体素数量: {len(scene_data['positions'])}")
    print(f"   - 密度范围: [{scene_data['densities'].min():.3f}, {scene_data['densities'].max():.3f}]")
    
    # 移动到设备
    for key in scene_data:
        if isinstance(scene_data[key], torch.Tensor):
            scene_data[key] = scene_data[key].to(device)
    
    # 2. 设置相机
    print("\n2. 设置相机...")
    camera_pos = torch.tensor([0.0, 0.0, 2.0], device=device)
    target_pos = torch.tensor([0.0, 0.0, 0.0], device=device)
    image_size = (400, 300)
    num_rays = 500
    
    # 生成光线
    rays_o, rays_d = generate_camera_rays(camera_pos, target_pos, image_size, num_rays)
    rays_o = rays_o.to(device)
    rays_d = rays_d.to(device)
    
    print(f"   - 相机位置: {camera_pos}")
    print(f"   - 目标位置: {target_pos}")
    print(f"   - 光线数量: {num_rays}")
    
    # 3. 训练模式渲染（体积渲染）
    print("\n3. 训练模式渲染（体积渲染）...")
    start_time = time.time()
    training_outputs = volume_rendering(scene_data, rays_o, rays_d)
    training_time = time.time() - start_time
    
    print(f"   - 渲染时间: {training_time:.3f}秒")
    print(f"   - RGB形状: {training_outputs['rgb'].shape}")
    print(f"   - RGB范围: [{training_outputs['rgb'].min():.3f}, {training_outputs['rgb'].max():.3f}]")
    print(f"   - 深度范围: [{training_outputs['depth'].min():.3f}, {training_outputs['depth'].max():.3f}]")
    
    # 4. 推理模式渲染（光栅化）
    print("\n4. 推理模式渲染（光栅化）...")
    start_time = time.time()
    inference_outputs = rasterization_rendering(scene_data, rays_o, rays_d)
    inference_time = time.time() - start_time
    
    print(f"   - 渲染时间: {inference_time:.3f}秒")
    print(f"   - RGB形状: {inference_outputs['rgb'].shape}")
    print(f"   - RGB范围: [{inference_outputs['rgb'].min():.3f}, {inference_outputs['rgb'].max():.3f}]")
    print(f"   - 深度范围: [{inference_outputs['depth'].min():.3f}, {inference_outputs['depth'].max():.3f}]")
    
    # 5. 性能比较
    print("\n5. 性能比较...")
    if training_time > 0 and inference_time > 0:
        speedup = training_time / inference_time
        print(f"   - 速度提升: {speedup:.2f}x")
    
    rays_per_sec_training = num_rays / training_time if training_time > 0 else 0
    rays_per_sec_inference = num_rays / inference_time if inference_time > 0 else 0
    
    print(f"   - 训练模式: {rays_per_sec_training:.0f} 光线/秒")
    print(f"   - 推理模式: {rays_per_sec_inference:.0f} 光线/秒")
    
    # 6. 质量分析
    print("\n6. 质量分析...")
    rgb_diff = torch.mean(torch.abs(training_outputs['rgb'] - inference_outputs['rgb']))
    depth_diff = torch.mean(torch.abs(training_outputs['depth'] - inference_outputs['depth']))
    
    print(f"   - RGB差异: {rgb_diff:.4f}")
    print(f"   - 深度差异: {depth_diff:.4f}")
    
    return training_outputs, inference_outputs


def show_complete_usage_guide():
    """显示完整的使用指南"""
    
    print("\n" + "="*50)
    print("SVRaster 完整使用指南")
    print("="*50)
    
    print("""
🎯 SVRaster 是什么？
SVRaster 是一个高效的神经辐射场渲染系统，结合了两种渲染方式：
- 训练模式：体积渲染，准确但慢
- 推理模式：光栅化，快速但近似

📋 基本使用流程：
1. 创建场景配置
2. 初始化模型
3. 生成相机光线
4. 选择渲染模式
5. 获取渲染结果

💡 实际使用代码：
```python
# 导入必要的模块
from src.nerfs.svraster import SVRasterModel, SVRasterConfig

# 1. 创建配置
config = SVRasterConfig(
    image_width=800,
    image_height=600,
    scene_bounds=(-2.0, -2.0, -2.0, 2.0, 2.0, 2.0),
    grid_resolution=128,
    sh_degree=2
)

# 2. 初始化模型
model = SVRasterModel(config)

# 3. 准备光线数据
# 光线起点（相机位置）
rays_o = torch.tensor([[0.0, 0.0, 3.0]]).expand(1000, -1)

# 光线方向（从相机指向场景）
rays_d = torch.randn(1000, 3)
rays_d = rays_d / torch.norm(rays_d, dim=1, keepdim=True)

# 4. 推理渲染（快速）
with torch.no_grad():
    outputs = model(rays_o, rays_d, mode="inference")

# 5. 训练渲染（精确）
with torch.no_grad():
    outputs = model(rays_o, rays_d, mode="training")

# 6. 获取结果
rgb = outputs['rgb']      # 颜色 [N, 3]
depth = outputs['depth']  # 深度 [N]
```

🔧 关键参数说明：
- image_width/height: 图像分辨率
- scene_bounds: 场景边界 (x_min, y_min, z_min, x_max, y_max, z_max)
- grid_resolution: 体素网格分辨率（影响质量和速度）
- sh_degree: 球谐函数阶数（影响光照质量）

📊 两种渲染模式：

训练模式 (mode="training"):
- 使用体积渲染
- 沿光线积分
- 更准确，质量更高
- 速度较慢
- 适合训练和高质量渲染

推理模式 (mode="inference"):
- 使用光栅化
- 直接投影体素
- 速度更快
- 质量略低
- 适合实时渲染和推理

⚡ 性能优化建议：
1. 使用 torch.no_grad() 进行推理
2. 批量处理光线提高效率
3. 调整 grid_resolution 平衡质量和速度
4. 推理模式用于实时应用
5. 使用 GPU 加速计算

🎨 应用场景：
- 神经辐射场渲染
- 3D 场景重建
- 虚拟现实渲染
- 游戏引擎集成
- 影视特效制作

🔍 调试技巧：
1. 从小的 grid_resolution 开始测试
2. 检查光线数据的合理性
3. 验证场景边界设置
4. 比较两种模式的输出差异
5. 监控内存和计算时间

📈 扩展功能：
- 支持不同的损失函数
- 可自定义采样策略
- 支持多种激活函数
- 可配置的渲染参数
- 灵活的相机模型
""")


if __name__ == "__main__":
    try:
        # 运行演示
        print("🚀 开始 SVRaster 演示...")
        training_outputs, inference_outputs = demo_svraster_usage()
        
        # 显示完整指南
        show_complete_usage_guide()
        
        print("\n🎉 演示完成！")
        print("现在您可以开始使用 SVRaster 进行神经辐射场渲染了！")
        
    except KeyboardInterrupt:
        print("\n⏹️  用户中断演示")
    except Exception as e:
        print(f"\n❌ 演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
