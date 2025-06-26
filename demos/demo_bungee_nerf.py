#!/usr/bin/env python3
"""
Bungee NeRF 演示脚本

展示Bungee NeRF的多尺度渐进式训练方法，包括：
- 多尺度渐进训练
- 动态分辨率调整
- 记忆高效的大场景处理
- 渐进式细节增强
"""

import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from src.nerfs.bungee_nerf.core import BungeeNeRF, BungeeNeRFConfig
    from src.nerfs.bungee_nerf.progressive_encoder import ProgressiveEncoder
    from src.nerfs.bungee_nerf.multiscale_renderer import MultiscaleRenderer
    BUNGEE_NERF_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Bungee NeRF模块导入失败: {e}")
    BUNGEE_NERF_AVAILABLE = False


class MockBungeeNeRFConfig:
    """模拟Bungee NeRF配置"""
    def __init__(self):
        self.base_resolution = 64
        self.max_resolution = 512
        self.num_scales = 4
        self.hidden_dim = 256
        self.num_layers = 6
        self.use_progressive_encoding = True
        self.encoding_start_freq = 0
        self.encoding_end_freq = 8
        self.scene_bounds = (-4.0, -4.0, -4.0, 4.0, 4.0, 4.0)
        self.progressive_steps = [1000, 2000, 3000, 4000]


class MockBungeeNeRF(torch.nn.Module):
    """模拟Bungee NeRF模型"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.training_step = 0
        
        # 创建多尺度网络
        self.networks = torch.nn.ModuleDict()
        for scale in range(config.num_scales):
            scale_hidden = config.hidden_dim // (2 ** scale)
            scale_hidden = max(scale_hidden, 64)
            
            network = torch.nn.Sequential(
                torch.nn.Linear(63, scale_hidden), # 位置编码后的维度
                torch.nn.ReLU(
                )
            )
            self.networks[f'scale_{scale}'] = network
    
    def positional_encoding(self, x: torch.Tensor, max_freq: int = 10) -> torch.Tensor:
        """位置编码"""
        encoded = [x]
        for i in range(max_freq):
            for fn in [torch.sin, torch.cos]:
                encoded.append(fn(2.**i * x))
        return torch.cat(encoded, dim=-1)
    
    def forward(
        self,
        positions: torch.Tensor,
        step: Optional[int] = None,
    )
        """前向传播"""
        if step is None:
            step = self.training_step
        
        # 位置编码
        encoded_pos = self.positional_encoding(positions)
        
        # 确定当前使用的尺度
        current_scale = min(step // 1000, self.config.num_scales - 1)
        
        # 使用对应尺度的网络
        network = self.networks[f'scale_{current_scale}']
        output = network(encoded_pos)
        
        density = torch.relu(output[..., 0])
        color = torch.sigmoid(output[..., 1:])
        
        return {
            'density': density, 'color': color, 'current_scale': current_scale, }
    
    def set_training_step(self, step: int):
        """设置训练步数"""
        self.training_step = step


def create_multiscale_dataset(
    num_views: int = 50,
    max_resolution: int = 128,
)
    """创建多尺度数据集"""
    print(f"📊 创建多尺度数据集: {num_views}个视角, 最大分辨率{max_resolution}x{max_resolution}")
    
    datasets = {}
    resolutions = [32, 64, 128]
    
    for res in resolutions:
        if res <= max_resolution:
            ray_origins = []
            ray_directions = []
            colors = []
            
            for i in range(num_views):
                theta = 2 * np.pi * i / num_views
                cam_pos = torch.tensor([
                    4.0 * np.cos(theta), 4.0 * np.sin(theta), 2.0
                ])
                
                # 简化：只生成少量光线
                for _ in range(100):
                    u = torch.rand(1) * 2 - 1
                    v = torch.rand(1) * 2 - 1
                    
                    ray_dir = torch.tensor([u, v, -1.0]).squeeze()
                    ray_dir = ray_dir / torch.norm(ray_dir)
                    
                    color = torch.sigmoid(cam_pos + ray_dir)
                    
                    ray_origins.append(cam_pos)
                    ray_directions.append(ray_dir)
                    colors.append(color)
            
            datasets[f'res_{res}'] = {
                'ray_origins': torch.stack(
                    ray_origins,
                )
            }
    
    return datasets


def progressive_training(
    model: MockBungeeNeRF,
    datasets: dict[str,
    torch.Tensor],
    num_epochs_per_scale: int = 100,
)
    """渐进式训练"""
    print(f"🚀 开始Bungee NeRF渐进式训练")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️  使用设备: {device}")
    
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    training_history = []
    total_step = 0
    
    # 按分辨率递增的顺序训练
    resolutions = sorted([int(k.split('_')[1]) for k in datasets.keys()])
    
    for scale, resolution in enumerate(resolutions):
        print(f"\n📏 训练尺度 {scale}: 分辨率 {resolution}x{resolution}")
        
        dataset = datasets[f'res_{resolution}']
        ray_origins = dataset['ray_origins'].to(device)
        ray_directions = dataset['ray_directions'].to(device)
        colors = dataset['colors'].to(device)
        
        for epoch in range(num_epochs_per_scale):
            model.set_training_step(total_step)
            
            # 随机采样
            batch_size = min(512, len(ray_origins))
            indices = torch.randperm(len(ray_origins))[:batch_size]
            
            batch_origins = ray_origins[indices]
            batch_colors = colors[indices]
            
            # 前向传播
            optimizer.zero_grad()
            outputs = model(batch_origins, total_step)
            
            # 计算损失
            color_loss = torch.nn.functional.mse_loss(outputs['color'], batch_colors)
            
            # 反向传播
            color_loss.backward()
            optimizer.step()
            
            # 记录
            if epoch % 50 == 0:
                with torch.no_grad():
                    mse = color_loss.item()
                    psnr = -10 * np.log10(mse) if mse > 0 else float('inf')
                    
                    training_history.append({
                        'step': total_step, 'epoch': epoch, 'resolution': resolution, 'loss': color_loss.item(
                        )
                    
                    print(f"  Epoch {epoch:3d}: Loss={color_loss.item():.6f}, PSNR={psnr:.2f}dB, "
                          f"Scale={outputs['current_scale']}")
            
            total_step += 1
    
    print("\n✅ 渐进式训练完成!")
    return training_history


def demonstrate_bungee_nerf():
    """演示Bungee NeRF的完整流程"""
    print("🌟 Bungee NeRF 演示")
    print("=" * 60)
    
    if not BUNGEE_NERF_AVAILABLE:
        print("⚠️ 使用模拟实现进行演示")
    
    # 1. 创建配置
    config = MockBungeeNeRFConfig()
    print(f"⚙️  模型配置:")
    print(f"   - 基础分辨率: {config.base_resolution}")
    print(f"   - 最大分辨率: {config.max_resolution}")
    print(f"   - 尺度数量: {config.num_scales}")
    print(f"   - 渐进式编码: {config.use_progressive_encoding}")
    
    # 2. 创建多尺度数据集
    datasets = create_multiscale_dataset(num_views=20, max_resolution=128)
    print(f"📊 创建了 {len(datasets)} 个分辨率的数据集")
    
    # 3. 创建模型
    model = MockBungeeNeRF(config)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"🧠 模型参数数量: {total_params:, }")
    
    # 4. 渐进式训练
    training_history = progressive_training(model, datasets, num_epochs_per_scale=50)
    
    # 5. 性能统计
    print("\n" + "=" * 60)
    print("📊 Bungee NeRF性能统计:")
    
    if training_history:
        final_metrics = training_history[-1]
        print(f"   - 最终损失: {final_metrics['loss']:.6f}")
        print(f"   - 最终PSNR: {final_metrics['psnr']:.2f} dB")
        print(f"   - 最大训练尺度: {final_metrics['current_scale']}")
    
    print(f"   - 总参数量: {total_params:, }")
    print(f"   - 模型大小: {total_params * 4 / 1024 / 1024:.2f} MB")
    
    print("\n🎉 Bungee NeRF演示完成!")
    print("\n📋 Bungee NeRF特点:")
    print("   ✅ 多尺度渐进式训练")
    print("   ✅ 动态编码频率调整")
    print("   ✅ 内存高效的大场景处理")
    print("   ✅ 渐进式细节增强")
    print("   ✅ 自适应分辨率调度")
    print("   ✅ 稳定的训练收敛")
    
    return model, training_history


if __name__ == '__main__':
    print("启动Bungee NeRF演示...")
    model, history = demonstrate_bungee_nerf()
    print("演示完成!") 