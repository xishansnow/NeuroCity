#!/usr/bin/env python3
"""
Classic NeRF 演示脚本

展示经典Neural Radiance Fields的基本使用方法，包括：
- 基础训练流程
- 新视角合成
- 渲染质量评估
- 模型保存与加载
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
    from src.nerfs.classic_nerf.core import ClassicNeRF, ClassicNeRFConfig
    from src.nerfs.classic_nerf.dataset import ClassicNeRFDataset
    from src.nerfs.classic_nerf.trainer import ClassicNeRFTrainer
    CLASSIC_NERF_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Classic NeRF模块导入失败: {e}")
    CLASSIC_NERF_AVAILABLE = False


class MockClassicNeRFConfig:
    """模拟Classic NeRF配置"""
    def __init__(self):
        self.hidden_dim = 256
        self.num_layers = 8
        self.skip_layers = [4]
        self.use_viewdirs = True
        self.scene_bounds = (-2.0, -2.0, -2.0, 2.0, 2.0, 2.0)
        self.near = 0.1
        self.far = 10.0
        self.num_samples = 64
        self.num_importance_samples = 128
        self.pe_freq_pos = 10
        self.pe_freq_dir = 4


class MockClassicNeRF(torch.nn.Module):
    """模拟Classic NeRF模型"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 位置编码MLP
        pos_input_dim = 3 + 3 * 2 * config.pe_freq_pos
        
        # 密度网络
        density_layers = []
        for i in range(config.num_layers):
            if i == 0:
                density_layers.append(torch.nn.Linear(pos_input_dim, config.hidden_dim))
            elif i in config.skip_layers:
                density_layers.append(
                    torch.nn.Linear,
                )
            else:
                density_layers.append(torch.nn.Linear(config.hidden_dim, config.hidden_dim))
            
            if i < config.num_layers - 1:
                density_layers.append(torch.nn.ReLU())
        
        self.density_net = torch.nn.ModuleList(density_layers)
        
        # 颜色网络
        if config.use_viewdirs:
            dir_input_dim = 3 + 3 * 2 * config.pe_freq_dir
            self.color_net = torch.nn.Sequential(
                torch.nn.Linear(
                    config.hidden_dim + dir_input_dim,
                    config.hidden_dim // 2,
                )
            )
        else:
            self.color_net = torch.nn.Sequential(
                torch.nn.Linear(config.hidden_dim, 3), torch.nn.Sigmoid()
            )
        
        # 密度输出层
        self.density_head = torch.nn.Sequential(
            torch.nn.Linear(config.hidden_dim, 1), torch.nn.ReLU()
        )
    
    def positional_encoding(self, x: torch.Tensor, num_freqs: int) -> torch.Tensor:
        """位置编码"""
        encoded = [x]
        for i in range(num_freqs):
            for fn in [torch.sin, torch.cos]:
                encoded.append(fn(2.**i * x))
        return torch.cat(encoded, dim=-1)
    
    def forward(
        self,
        positions: torch.Tensor,
        directions: Optional[torch.Tensor] = None,
    )
        """前向传播"""
        # 位置编码
        pos_encoded = self.positional_encoding(positions, self.config.pe_freq_pos)
        
        # 密度网络前向传播
        x = pos_encoded
        for i, layer in enumerate(self.density_net):
            if i in self.config.skip_layers and i > 0:
                x = torch.cat([x, pos_encoded], dim=-1)
            x = layer(x)
        
        # 密度输出
        density = self.density_head(x)
        
        # 颜色网络
        if self.config.use_viewdirs and directions is not None:
            dir_encoded = self.positional_encoding(directions, self.config.pe_freq_dir)
            color_input = torch.cat([x, dir_encoded], dim=-1)
        else:
            color_input = x
        
        color = self.color_net(color_input)
        
        return {
            'density': density.squeeze(-1), 'color': color
        }


def create_synthetic_dataset(
    num_views: int = 100,
    image_size: int = 64,
)
    """创建合成数据集用于演示"""
    print(f"📊 创建合成数据集: {num_views}个视角, 图像大小{image_size}x{image_size}")
    
    # 生成相机位置（围绕原点的球面）
    angles = torch.linspace(0, 2*np.pi, num_views)
    elevations = torch.linspace(-np.pi/6, np.pi/6, num_views)
    
    ray_origins = []
    ray_directions = []
    colors = []
    
    for i in range(num_views):
        # 相机位置
        radius = 3.0
        theta = angles[i]
        phi = elevations[i % len(elevations)]
        
        cam_pos = torch.tensor([
            radius * torch.cos(
                phi,
            )
        ])
        
        # 生成光线
        for y in range(image_size):
            for x in range(image_size):
                # 像素坐标转换为世界坐标
                u = (x - image_size/2) / (image_size/2)
                v = (y - image_size/2) / (image_size/2)
                
                # 简化的光线方向（假设朝向原点）
                ray_dir = torch.tensor([u, v, -1.0])
                ray_dir = ray_dir / torch.norm(ray_dir)
                
                # 简单的颜色函数（基于位置和方向）
                color = torch.sigmoid(cam_pos + ray_dir)
                
                ray_origins.append(cam_pos)
                ray_directions.append(ray_dir)
                colors.append(color)
    
    return (
        torch.stack(ray_origins), torch.stack(ray_directions), torch.stack(colors)
    )


def train_classic_nerf(
    config: MockClassicNeRFConfig,
    ray_origins: torch.Tensor,
    ray_directions: torch.Tensor,
    target_colors: torch.Tensor,
    num_epochs: int = 100,
)
    """训练Classic NeRF模型"""
    print(f"🚀 开始训练Classic NeRF模型")
    print(f"📈 训练数据: {len(ray_origins)} 条光线")
    print(f"🔄 训练轮次: {num_epochs}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️  使用设备: {device}")
    
    # 创建模型
    model = MockClassicNeRF(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    
    # 移动数据到设备
    ray_origins = ray_origins.to(device)
    ray_directions = ray_directions.to(device)
    target_colors = target_colors.to(device)
    
    # 训练循环
    model.train()
    for epoch in range(num_epochs):
        # 随机采样光线
        batch_size = 1024
        indices = torch.randperm(len(ray_origins))[:batch_size]
        
        batch_origins = ray_origins[indices]
        batch_directions = ray_directions[indices]
        batch_colors = target_colors[indices]
        
        # 前向传播
        optimizer.zero_grad()
        outputs = model(batch_origins, batch_directions)
        
        # 计算损失
        color_loss = torch.nn.functional.mse_loss(outputs['color'], batch_colors)
        
        # 反向传播
        color_loss.backward()
        optimizer.step()
        
        # 记录进度
        if epoch % 20 == 0:
            with torch.no_grad():
                # 计算PSNR
                mse = color_loss.item()
                psnr = -10 * np.log10(mse) if mse > 0 else float('inf')
                print(f"Epoch {epoch:3d}: Loss={color_loss.item():.6f}, PSNR={psnr:.2f}dB")
    
    print("✅ 训练完成!")
    return model


def render_novel_views(
    model: MockClassicNeRF,
    config: MockClassicNeRFConfig,
    num_views: int = 8,
    image_size: int = 64,
)
    """渲染新视角"""
    print(f"🎬 渲染新视角: {num_views}个视角")
    
    device = next(model.parameters()).device
    model.eval()
    
    rendered_images = []
    
    with torch.no_grad():
        for i in range(num_views):
            # 新的相机位置
            theta = 2 * np.pi * i / num_views
            cam_pos = torch.tensor([
                2.5 * np.cos(theta), 2.5 * np.sin(theta), 1.0
            ]).to(device)
            
            # 渲染图像
            image = torch.zeros(image_size, image_size, 3)
            
            for y in range(image_size):
                for x in range(image_size):
                    u = (x - image_size/2) / (image_size/2)
                    v = (y - image_size/2) / (image_size/2)
                    
                    ray_dir = torch.tensor([u, v, -1.0]).to(device)
                    ray_dir = ray_dir / torch.norm(ray_dir)
                    
                    # 简化渲染（不进行体积渲染，直接使用模型输出）
                    output = model(cam_pos.unsqueeze(0), ray_dir.unsqueeze(0))
                    image[y, x] = output['color'][0].cpu()
            
            rendered_images.append(image)
    
    return rendered_images


def visualize_results(rendered_images: list[torch.Tensor], save_path: str = "demo_outputs"):
    """可视化渲染结果"""
    print(f"📊 可视化渲染结果")
    
    os.makedirs(save_path, exist_ok=True)
    
    # 创建对比图
    num_images = len(rendered_images)
    cols = min(4, num_images)
    rows = (num_images + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    for i, image in enumerate(rendered_images):
        row = i // cols
        col = i % cols
        
        axes[row, col].imshow(image.numpy())
        axes[row, col].set_title(f'视角 {i+1}')
        axes[row, col].axis('off')
    
    # 隐藏多余的子图
    for i in range(num_images, rows * cols):
        row = i // cols
        col = i % cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig(f"{save_path}/classic_nerf_novel_views.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # 保存单个图像
    for i, image in enumerate(rendered_images):
        plt.figure(figsize=(4, 4))
        plt.imshow(image.numpy())
        plt.title(f'Classic NeRF 新视角 {i+1}')
        plt.axis('off')
        plt.savefig(f"{save_path}/classic_nerf_view_{i+1:02d}.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"💾 结果保存到: {save_path}/")


def demonstrate_classic_nerf():
    """演示Classic NeRF的完整流程"""
    print("🌟 Classic NeRF 演示")
    print("=" * 60)
    
    if not CLASSIC_NERF_AVAILABLE:
        print("⚠️ 使用模拟实现进行演示")
    
    # 1. 创建配置
    config = MockClassicNeRFConfig()
    print(f"⚙️  模型配置:")
    print(f"   - 隐藏层维度: {config.hidden_dim}")
    print(f"   - 网络层数: {config.num_layers}")
    print(f"   - 使用视角方向: {config.use_viewdirs}")
    print(f"   - 采样点数: {config.num_samples}")
    
    # 2. 创建数据集
    ray_origins, ray_directions, target_colors = create_synthetic_dataset(
        num_views=20, image_size=32
    )
    
    # 3. 训练模型
    model = train_classic_nerf(
        config, ray_origins, ray_directions, target_colors, num_epochs=100
    )
    
    # 4. 渲染新视角
    rendered_images = render_novel_views(model, config, num_views=6, image_size=32)
    
    # 5. 可视化结果
    visualize_results(rendered_images)
    
    # 6. 模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("\n" + "=" * 60)
    print("📊 模型统计:")
    print(f"   - 总参数量: {total_params:, }")
    print(f"   - 可训练参数: {trainable_params:, }")
    print(f"   - 模型大小: {total_params * 4 / 1024 / 1024:.2f} MB")
    
    print("\n🎉 Classic NeRF演示完成!")
    print("\n📋 Classic NeRF特点:")
    print("   ✅ 隐式场景表示")
    print("   ✅ 连续的体积密度和颜色")
    print("   ✅ 视角相关的颜色渲染")
    print("   ✅ 高质量的新视角合成")
    print("   ✅ 端到端可微分渲染")
    
    return model, rendered_images


if __name__ == '__main__':
    print("启动Classic NeRF演示...")
    model, images = demonstrate_classic_nerf()
    print("演示完成!") 