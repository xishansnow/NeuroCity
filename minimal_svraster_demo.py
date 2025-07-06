"""
SVRaster 最小演示

直接使用核心组件，避免复杂的训练器模块
"""

import torch
import numpy as np
import time
import sys
import os

# 添加项目路径
sys.path.append('/home/xishansnow/3DVision/NeuroCity')

# 直接导入核心组件
from src.nerfs.svraster.core import SVRasterModel, SVRasterConfig


def minimal_demo():
    """最小化演示"""
    
    print("=== SVRaster 最小演示 ===\n")
    
    # 检查设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    try:
        # 1. 创建最基本的配置
        print("\n1. 创建配置...")
        config = SVRasterConfig()
        print(f"   - 成功创建配置")
        print(f"   - 图像尺寸: {config.image_width}x{config.image_height}")
        
        # 2. 尝试创建模型
        print("\n2. 创建模型...")
        model = SVRasterModel(config)
        print(f"   - 成功创建模型")
        print(f"   - 模型设备: {model.device}")
        
        # 3. 创建简单的测试数据
        print("\n3. 准备测试数据...")
        num_rays = 10  # 很少的光线用于测试
        
        # 随机光线
        rays_o = torch.randn(num_rays, 3, device=model.device) * 0.1
        rays_d = torch.randn(num_rays, 3, device=model.device)
        rays_d = rays_d / torch.norm(rays_d, dim=1, keepdim=True)
        
        print(f"   - 光线数量: {num_rays}")
        print(f"   - 光线起点形状: {rays_o.shape}")
        print(f"   - 光线方向形状: {rays_d.shape}")
        
        # 4. 测试推理模式
        print("\n4. 测试推理模式...")
        try:
            with torch.no_grad():
                outputs = model(rays_o, rays_d, mode="inference")
            
            print(f"   - ✅ 推理模式成功!")
            print(f"   - 输出键: {list(outputs.keys())}")
            
            for key, value in outputs.items():
                if torch.is_tensor(value):
                    print(f"   - {key}: {value.shape}, 范围 [{value.min():.3f}, {value.max():.3f}]")
                
        except Exception as e:
            print(f"   - ❌ 推理模式失败: {e}")
        
        # 5. 测试训练模式
        print("\n5. 测试训练模式...")
        try:
            with torch.no_grad():
                outputs = model(rays_o, rays_d, mode="training")
            
            print(f"   - ✅ 训练模式成功!")
            print(f"   - 输出键: {list(outputs.keys())}")
            
            for key, value in outputs.items():
                if torch.is_tensor(value):
                    print(f"   - {key}: {value.shape}, 范围 [{value.min():.3f}, {value.max():.3f}]")
                
        except Exception as e:
            print(f"   - ❌ 训练模式失败: {e}")
        
        print(f"\n✅ 基本功能测试完成!")
        
    except Exception as e:
        print(f"❌ 创建模型失败: {e}")
        import traceback
        traceback.print_exc()


def show_usage():
    """显示使用方法"""
    
    print("\n=== SVRaster 使用方法 ===\n")
    
    print("1. 基本使用:")
    print("""
from src.nerfs.svraster.core import SVRasterModel, SVRasterConfig

# 创建配置
config = SVRasterConfig(
    image_width=800,
    image_height=600,
    scene_bounds=(-2, -2, -2, 2, 2, 2),
    grid_resolution=128
)

# 创建模型
model = SVRasterModel(config)

# 准备光线数据
rays_o = torch.randn(1000, 3)  # 光线起点 [N, 3]
rays_d = torch.randn(1000, 3)  # 光线方向 [N, 3]
rays_d = rays_d / torch.norm(rays_d, dim=1, keepdim=True)  # 归一化

# 推理渲染（快速）
with torch.no_grad():
    outputs = model(rays_o, rays_d, mode="inference")

# 训练渲染（准确）
with torch.no_grad():
    outputs = model(rays_o, rays_d, mode="training")
""")
    
    print("2. 重要参数:")
    print("   - mode='inference': 使用光栅化，快速渲染")
    print("   - mode='training': 使用体积渲染，准确渲染")
    print("   - rays_o: 光线起点，形状 [N, 3]")
    print("   - rays_d: 光线方向，形状 [N, 3]，必须归一化")
    
    print("\n3. 输出格式:")
    print("   - rgb: RGB颜色 [N, 3]")
    print("   - depth: 深度值 [N]")
    print("   - alpha: 透明度 [N] (如果可用)")
    
    print("\n4. 性能建议:")
    print("   - 推理时使用 torch.no_grad()")
    print("   - 批量处理光线以提高效率")
    print("   - 调整 grid_resolution 平衡质量和速度")


if __name__ == "__main__":
    try:
        # 运行最小演示
        minimal_demo()
        
        # 显示使用方法
        show_usage()
        
        print("\n🎉 演示完成!")
        
    except Exception as e:
        print(f"\n❌ 演示失败: {e}")
        import traceback
        traceback.print_exc()
