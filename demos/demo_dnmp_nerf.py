#!/usr/bin/env python3
"""
DNMP NeRF 演示脚本

展示Differentiable Neural Mesh Primitive (DNMP) NeRF的功能，包括：
- 可微分网格表示
- 网格自动编码器
- 光栅化渲染
- 几何与纹理联合优化
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
    from src.nerfs.dnmp_nerf.core import DNMPNeRF, DNMPNeRFConfig
    from src.nerfs.dnmp_nerf.mesh_autoencoder import MeshAutoencoder
    from src.nerfs.dnmp_nerf.rasterizer import DifferentiableRasterizer
    DNMP_NERF_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ DNMP NeRF模块导入失败: {e}")
    DNMP_NERF_AVAILABLE = False


class MockDNMPNeRFConfig:
    """模拟DNMP NeRF配置"""
    def __init__(self):
        self.mesh_resolution = 128
        self.latent_dim = 256
        self.encoder_layers = [512, 256, 128]
        self.decoder_layers = [128, 256, 512]
        self.texture_dim = 64
        self.use_mesh_autoencoder = True
        self.rasterization_size = 256
        self.scene_bounds = (-2.0, -2.0, -2.0, 2.0, 2.0, 2.0)


class MockMeshAutoencoder(torch.nn.Module):
    """模拟网格自动编码器"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 编码器
        encoder_layers = []
        in_dim = config.mesh_resolution * 3  # 顶点坐标
        for hidden_dim in config.encoder_layers:
            encoder_layers.extend([
                torch.nn.Linear(in_dim, hidden_dim), torch.nn.ReLU()
            ])
            in_dim = hidden_dim
        encoder_layers.append(torch.nn.Linear(in_dim, config.latent_dim))
        self.encoder = torch.nn.Sequential(*encoder_layers)
        
        # 解码器
        decoder_layers = []
        in_dim = config.latent_dim
        for hidden_dim in config.decoder_layers:
            decoder_layers.extend([
                torch.nn.Linear(in_dim, hidden_dim), torch.nn.ReLU()
            ])
            in_dim = hidden_dim
        decoder_layers.append(torch.nn.Linear(in_dim, config.mesh_resolution * 3))
        self.decoder = torch.nn.Sequential(*decoder_layers)
    
    def encode(self, vertices: torch.Tensor) -> torch.Tensor:
        """编码网格顶点到潜在空间"""
        batch_size = vertices.shape[0]
        vertices_flat = vertices.view(batch_size, -1)
        return self.encoder(vertices_flat)
    
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """从潜在空间解码网格顶点"""
        batch_size = latent.shape[0]
        vertices_flat = self.decoder(latent)
        return vertices_flat.view(batch_size, self.config.mesh_resolution, 3)
    
    def forward(self, vertices: torch.Tensor) -> dict[str, torch.Tensor]:
        """自动编码器前向传播"""
        latent = self.encode(vertices)
        reconstructed = self.decode(latent)
        return {
            'latent': latent, 'reconstructed': reconstructed
        }


class MockDNMPNeRF(torch.nn.Module):
    """模拟DNMP NeRF模型"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 网格自动编码器
        if config.use_mesh_autoencoder:
            self.mesh_autoencoder = MockMeshAutoencoder(config)
        
        # 纹理网络
        self.texture_network = torch.nn.Sequential(
            torch.nn.Linear(
                3 + config.latent_dim,
                config.texture_dim,
            )
        )
        
        # 几何网络
        self.geometry_network = torch.nn.Sequential(
            torch.nn.Linear(
                3,
                128,
            )
        )
    
    def generate_mesh(self, batch_size: int = 1) -> torch.Tensor:
        """生成初始网格"""
        # 简化：生成球形网格顶点
        vertices = []
        for i in range(self.config.mesh_resolution):
            theta = 2 * np.pi * i / self.config.mesh_resolution
            vertex = torch.tensor([
                np.cos(theta), np.sin(theta), 0.0
            ])
            vertices.append(vertex)
        
        mesh = torch.stack(vertices).unsqueeze(0).repeat(batch_size, 1, 1)
        return mesh
    
    def forward(
        self,
        positions: torch.Tensor,
        mesh: Optional[torch.Tensor] = None,
    )
        """前向传播"""
        batch_size = positions.shape[0]
        
        if mesh is None:
            mesh = self.generate_mesh(batch_size).to(positions.device)
        
        # 网格编码
        if self.config.use_mesh_autoencoder:
            mesh_output = self.mesh_autoencoder(mesh)
            latent = mesh_output['latent']
            reconstructed_mesh = mesh_output['reconstructed']
        else:
            latent = torch.randn(batch_size, self.config.latent_dim).to(positions.device)
            reconstructed_mesh = mesh
        
        # 几何计算
        sdf_values = self.geometry_network(positions)
        
        # 纹理计算
        latent_expanded = latent.unsqueeze(1).expand(-1, positions.shape[1], -1)
        texture_input = torch.cat([positions, latent_expanded], dim=-1)
        colors = self.texture_network(texture_input)
        
        return {
            'sdf': sdf_values.squeeze(
                -1,
            )
        }


def create_mesh_dataset(
    num_meshes: int = 100,
    mesh_resolution: int = 64,
)
    """创建网格数据集"""
    print(f"📊 创建网格数据集: {num_meshes}个网格, 分辨率{mesh_resolution}")
    
    meshes = []
    colors = []
    
    for i in range(num_meshes):
        # 生成变形的球形网格
        vertices = []
        mesh_colors = []
        
        for j in range(mesh_resolution):
            theta = 2 * np.pi * j / mesh_resolution
            phi = np.pi * (i / num_meshes - 0.5)  # 变化高度
            
            # 添加一些形变
            radius = 1.0 + 0.3 * np.sin(4 * theta)
            
            vertex = torch.tensor([
                radius * np.cos(
                    theta,
                )
            ])
            
            # 基于位置的颜色
            color = torch.sigmoid(vertex + 0.5)
            
            vertices.append(vertex)
            mesh_colors.append(color)
        
        mesh = torch.stack(vertices)
        mesh_color = torch.stack(mesh_colors)
        
        meshes.append(mesh)
        colors.append(mesh_color.mean(0))  # 平均颜色作为整体颜色
    
    return {
        'meshes': torch.stack(
            meshes,
        )
    }


def train_dnmp_nerf(
    model: MockDNMPNeRF,
    dataset: dict[str,
    torch.Tensor],
    num_epochs: int = 200,
)
    """训练DNMP NeRF模型"""
    print(f"🚀 开始训练DNMP NeRF模型")
    print(f"📈 训练数据: {len(dataset['meshes'])} 个网格")
    print(f"🔄 训练轮次: {num_epochs}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️  使用设备: {device}")
    
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    meshes = dataset['meshes'].to(device)
    colors = dataset['colors'].to(device)
    positions = dataset['positions'].to(device)
    
    training_history = []
    
    for epoch in range(num_epochs):
        # 随机采样
        batch_size = 16
        mesh_indices = torch.randperm(len(meshes))[:batch_size]
        pos_indices = torch.randperm(len(positions))[:batch_size * 50]
        
        batch_meshes = meshes[mesh_indices]
        batch_colors = colors[mesh_indices]
        batch_positions = positions[pos_indices].unsqueeze(0).repeat(batch_size, 1, 1)
        
        # 前向传播
        optimizer.zero_grad()
        outputs = model(batch_positions.reshape(-1, 3), batch_meshes)
        
        # 计算损失
        # 1. 重建损失
        if 'mesh' in outputs:
            recon_loss = torch.nn.functional.mse_loss(outputs['mesh'], batch_meshes)
        else:
            recon_loss = 0
        
        # 2. 颜色损失（简化）
        predicted_colors = outputs['color'].reshape(batch_size, -1, 3).mean(1)
        color_loss = torch.nn.functional.mse_loss(predicted_colors, batch_colors)
        
        # 3. SDF正则化
        sdf_loss = torch.mean(torch.abs(outputs['sdf']))
        
        total_loss = color_loss + 0.1 * recon_loss + 0.01 * sdf_loss
        
        # 反向传播
        total_loss.backward()
        optimizer.step()
        
        # 记录
        if epoch % 40 == 0:
            with torch.no_grad():
                mse = color_loss.item()
                psnr = -10 * np.log10(mse) if mse > 0 else float('inf')
                
                training_history.append({
                    'epoch': epoch, 'total_loss': total_loss.item(
                    )
                })
                
                print(f"Epoch {epoch:3d}: Total={total_loss.item():.6f}, "
                      f"Color={color_loss.item():.6f}, PSNR={psnr:.2f}dB")
    
    print("✅ 训练完成!")
    return training_history


def demonstrate_dnmp_nerf():
    """演示DNMP NeRF的完整流程"""
    print("🌟 DNMP NeRF 演示")
    print("=" * 60)
    
    if not DNMP_NERF_AVAILABLE:
        print("⚠️ 使用模拟实现进行演示")
    
    # 1. 创建配置
    config = MockDNMPNeRFConfig()
    print(f"⚙️  模型配置:")
    print(f"   - 网格分辨率: {config.mesh_resolution}")
    print(f"   - 潜在维度: {config.latent_dim}")
    print(f"   - 纹理维度: {config.texture_dim}")
    print(f"   - 使用网格自动编码器: {config.use_mesh_autoencoder}")
    
    # 2. 创建数据集
    dataset = create_mesh_dataset(num_meshes=50, mesh_resolution=32)
    
    # 3. 创建模型
    model = MockDNMPNeRF(config)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"🧠 模型参数数量: {total_params:, }")
    
    # 4. 训练模型
    training_history = train_dnmp_nerf(model, dataset, num_epochs=100)
    
    # 5. 性能统计
    print("\n" + "=" * 60)
    print("📊 DNMP NeRF性能统计:")
    
    if training_history:
        final_metrics = training_history[-1]
        print(f"   - 最终总损失: {final_metrics['total_loss']:.6f}")
        print(f"   - 最终颜色损失: {final_metrics['color_loss']:.6f}")
        print(f"   - 最终PSNR: {final_metrics['psnr']:.2f} dB")
        print(f"   - 重建损失: {final_metrics['recon_loss']:.6f}")
        print(f"   - SDF损失: {final_metrics['sdf_loss']:.6f}")
    
    print(f"   - 总参数量: {total_params:, }")
    print(f"   - 模型大小: {total_params * 4 / 1024 / 1024:.2f} MB")
    
    print("\n🎉 DNMP NeRF演示完成!")
    print("\n📋 DNMP NeRF特点:")
    print("   ✅ 可微分网格表示")
    print("   ✅ 网格自动编码器")
    print("   ✅ SDF几何建模")
    print("   ✅ 纹理与几何联合优化")
    print("   ✅ 高效光栅化渲染")
    print("   ✅ 显式几何控制")
    
    return model, training_history


if __name__ == '__main__':
    print("启动DNMP NeRF演示...")
    model, history = demonstrate_dnmp_nerf()
    print("演示完成!") 