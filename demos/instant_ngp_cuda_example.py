#!/usr/bin/env python3
"""
GTX 1080 Ti 专用 Instant NGP CUDA 实现示例
不依赖 tiny-cuda-nn，完全兼容 PyTorch
"""

import torch
import torch.nn as nn
import numpy as np
import time
import sys
import os

# 添加项目路径
sys.path.append('/home/xishansnow/3DVision/NeuroCity/src')

def main():
    print("🚀 GTX 1080 Ti Instant NGP CUDA 示例")
    print("=" * 50)
    
    # 检查CUDA可用性
    print(f"CUDA 可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU 设备: {torch.cuda.get_device_name()}")
        print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # 导入我们的CUDA实现
    try:
        from nerfs.instant_ngp.cuda_model import InstantNGPModel
        print("✅ 成功导入 Instant NGP CUDA 模型")
    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        return
    
    # 创建模型 - 针对GTX 1080 Ti优化的配置
    print("\n🏗️ 创建模型...")
    model = InstantNGPModel(
        # 哈希编码参数
        num_levels=16,              # 16个分辨率层级
        base_resolution=16,         # 基础分辨率 16³
        finest_resolution=512,      # 最高分辨率 512³
        log2_hashmap_size=19,       # 哈希表大小 2^19 = 524,288
        feature_dim=2,              # 每个特征2维
        
        # 网络参数
        hidden_dim=64,              # 隐藏层维度
        num_layers=2,               # 网络层数
        geo_feature_dim=15,         # 几何特征维度
        num_layers_color=3,         # 颜色网络层数
        hidden_dim_color=64,        # 颜色网络隐藏层维度
        
        # 球面谐波
        sh_degree=4,                # 4阶球面谐波 (25个系数)
        
        # 边界框
        aabb_min=torch.tensor([-1.0, -1.0, -1.0]),
        aabb_max=torch.tensor([1.0, 1.0, 1.0]),
        
        use_cuda=True
    ).cuda()
    
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 生成测试数据
    print("\n📊 生成测试数据...")
    batch_size = 10000
    
    # 随机3D位置 [-1, 1]³
    positions = torch.rand(batch_size, 3).cuda() * 2.0 - 1.0
    
    # 随机方向向量 (归一化)
    directions = torch.randn(batch_size, 3).cuda()
    directions = directions / directions.norm(dim=-1, keepdim=True)
    
    print(f"位置数据: {positions.shape}")
    print(f"方向数据: {directions.shape}")
    
    # 前向传播测试
    print("\n⚡ 性能测试...")
    
    # 预热
    with torch.no_grad():
        for _ in range(5):
            density, color = model(positions, directions)
    
    # 正式测试
    torch.cuda.synchronize()
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(100):
            density, color = model(positions, directions)
    
    torch.cuda.synchronize()
    end_time = time.time()
    
    avg_time = (end_time - start_time) / 100
    points_per_second = batch_size / avg_time
    
    print(f"✅ 前向传播完成")
    print(f"   密度输出: {density.shape}, 范围: [{density.min():.3f}, {density.max():.3f}]")
    print(f"   颜色输出: {color.shape}, 范围: [{color.min():.3f}, {color.max():.3f}]")
    print(f"   平均时间: {avg_time*1000:.2f} ms")
    print(f"   处理速度: {points_per_second:,.0f} 点/秒")
    
    # 梯度测试
    print("\n🔄 梯度测试...")
    positions.requires_grad_(True)
    
    torch.cuda.synchronize()
    start_time = time.time()
    
    density, color = model(positions, directions)
    loss = density.mean() + color.mean()
    loss.backward()
    
    torch.cuda.synchronize()
    end_time = time.time()
    
    print(f"✅ 反向传播完成")
    if positions.grad is not None:
        print(f"   梯度形状: {positions.grad.shape}")
        print(f"   梯度范围: [{positions.grad.min():.3f}, {positions.grad.max():.3f}]")
    print(f"   反向时间: {(end_time - start_time)*1000:.2f} ms")
    
    # 与PyTorch fallback对比
    print("\n🔍 性能对比...")
    
    # 强制使用PyTorch实现
    model_torch = InstantNGPModel(
        num_levels=16,
        base_resolution=16,
        finest_resolution=512,
        log2_hashmap_size=19,
        feature_dim=2,
        hidden_dim=64,
        num_layers=2,
        geo_feature_dim=15,
        num_layers_color=3,
        hidden_dim_color=64,
        sh_degree=4,
        use_cuda=False  # 禁用CUDA
    ).cuda()
    
    # 小批量测试（PyTorch版本较慢）
    small_positions = positions[:1000]
    small_directions = directions[:1000]
    
    # PyTorch版本
    torch.cuda.synchronize()
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(10):
            density_torch, color_torch = model_torch(small_positions, small_directions)
    
    torch.cuda.synchronize()
    end_time = time.time()
    
    torch_time = (end_time - start_time) / 10
    
    # CUDA版本
    torch.cuda.synchronize()
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(10):
            density_cuda, color_cuda = model(small_positions, small_directions)
    
    torch.cuda.synchronize()
    end_time = time.time()
    
    cuda_time = (end_time - start_time) / 10
    
    speedup = torch_time / cuda_time
    
    print(f"   PyTorch 时间: {torch_time*1000:.2f} ms")
    print(f"   CUDA 时间: {cuda_time*1000:.2f} ms")
    print(f"   🚀 加速比: {speedup:.1f}x")
    
    # 内存使用情况
    print(f"\n💾 内存使用:")
    print(f"   已分配: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    print(f"   已缓存: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
    
    print(f"\n🎉 测试完成! GTX 1080 Ti 上的 Instant NGP 运行正常")

if __name__ == "__main__":
    main()
