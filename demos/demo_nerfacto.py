from typing import Optional
#!/usr/bin/env python3
"""
Nerfacto 演示脚本

展示Nerfacto的实用化NeRF实现，包括：
- 快速收敛训练
- 高质量渲染
- 相机参数优化
- 实用工具集成
"""

import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from src.nerfs.nerfacto.core import Nerfacto, NeRFactoConfig
    from src.nerfs.nerfacto.trainer import NerfactoTrainer
    NERFACTO_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Nerfacto模块导入失败: {e}")
    NERFACTO_AVAILABLE = False

class MockNerfactoConfig:
    """模拟Nerfacto配置"""
    def __init__(self):
        self.hidden_dim = 256
        self.num_layers = 8
        self.skip_layers = [4]
        self.geo_feat_dim = 256
        self.num_levels = 16
        self.max_res = 2048
        self.base_res = 16
        self.log2_hashmap_size = 19
        self.features_per_level = 2
        self.scene_bounds = (-2.0, -2.0, -2.0, 2.0, 2.0, 2.0)
        self.near = 0.05
        self.far = 1000.0
        self.num_samples = 64
        self.num_importance_samples = 128

class MockNerfacto(torch.nn.Module):
    """模拟Nerfacto模型"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 哈希编码网络（简化）
        self.hash_encoding = torch.nn.Sequential(
            torch.nn.Linear(3, 64), torch.nn.ReLU(), torch.nn.Linear(64, config.geo_feat_dim)
        )
        
        # 几何网络
        geo_layers = []
        in_dim = config.geo_feat_dim
        for i in range(config.num_layers):
            if i in config.skip_layers:
                geo_layers.append(torch.nn.Linear(in_dim + config.geo_feat_dim, config.hidden_dim))
            else:
                geo_layers.append(torch.nn.Linear(in_dim, config.hidden_dim))
            
            if i < config.num_layers - 1:
                geo_layers.append(torch.nn.ReLU())
            in_dim = config.hidden_dim
        
        self.geometry_network = torch.nn.ModuleList(geo_layers)
        
        # 密度头
        self.density_head = torch.nn.Sequential(
            torch.nn.Linear(config.hidden_dim, 1), torch.nn.Softplus()
        )
        
        # 颜色网络
        self.color_network = torch.nn.Sequential(
            torch.nn.Linear(
                config.hidden_dim + 27,
                config.hidden_dim // 2,
            )
            torch.nn.ReLU(), torch.nn.Linear(config.hidden_dim // 2, 3), torch.nn.Sigmoid()
        )
    
    def hash_encode(self, positions: torch.Tensor) -> torch.Tensor:
        """哈希编码（简化版）"""
        return self.hash_encoding(positions)
    
    def spherical_harmonics_encoding(self, directions: torch.Tensor) -> torch.Tensor:
        """球谐编码（简化）"""
        # 简化的方向编码
        x, y, z = directions[..., 0], directions[..., 1], directions[..., 2]
        
        # 一阶球谐函数
        sh_0 = torch.ones_like(x) * 0.28209479177  # Y_0^0
        sh_1 = 0.48860251190 * y                    # Y_1^{-1}
        sh_2 = 0.48860251190 * z                    # Y_1^0
        sh_3 = 0.48860251190 * x                    # Y_1^1
        
        # 二阶球谐函数（部分）
        sh_4 = 1.09254843059 * x * y                # Y_2^{-2}
        sh_5 = 1.09254843059 * y * z                # Y_2^{-1}
        sh_6 = 0.31539156525 * (3*z*z - 1)          # Y_2^0
        sh_7 = 1.09254843059 * x * z                # Y_2^1
        sh_8 = 0.54627421529 * (x*x - y*y)          # Y_2^2
        
        return torch.stack([
            sh_0, sh_1, sh_2, sh_3, sh_4, sh_5, sh_6, sh_7, sh_8, # 添加更多项以达到27维
            x*x, y*y, z*z, x*y, x*z, y*z, x*x*x, y*y*y, z*z*z, x*x*y, x*x*z, y*y*x, y*y*z, z*z*x, z*z*y, x*y*z, x*x*y*y, x*x*z*z, y*y*z*z
        ], dim=-1)
    
    def forward(self, positions: torch.Tensor, directions: torch.Tensor) -> dict[str, torch.Tensor]:
        """前向传播"""
        # 哈希编码
        geo_features = self.hash_encode(positions)
        
        # 几何网络前向传播
        x = geo_features
        for i, layer in enumerate(self.geometry_network):
            if i in self.config.skip_layers and i > 0:
                x = torch.cat([x, geo_features], dim=-1)
            x = layer(x)
        
        # 密度
        density = self.density_head(x)
        
        # 方向编码
        dir_encoded = self.spherical_harmonics_encoding(directions)
        
        # 颜色网络
        color_input = torch.cat([x, dir_encoded], dim=-1)
        color = self.color_network(color_input)
        
        return {
            'density': density.squeeze(-1), 'color': color
        }

def create_realistic_dataset(
    num_views: int = 100,
    image_size: int = 64,
)
    """创建更真实的数据集"""
    print(f"📊 创建真实感数据集: {num_views}个视角, 图像大小{image_size}x{image_size}")
    
    ray_origins = []
    ray_directions = []
    colors = []
    
    for i in range(num_views):
        # 相机轨迹
        theta = 2 * np.pi * i / num_views
        phi = 0.2 * np.sin(4 * theta)  # 高度变化
        
        cam_pos = torch.tensor([
            3.0 * np.cos(theta), 3.0 * np.sin(theta), 2.0 + phi
        ])
        
        # 朝向原点
        look_at = torch.tensor([0.0, 0.0, 0.0])
        forward = look_at - cam_pos
        forward = forward / torch.norm(forward)
        
        # 构建相机坐标系
        up = torch.tensor([0.0, 0.0, 1.0])
        right = torch.cross(forward, up)
        right = right / torch.norm(right)
        up = torch.cross(right, forward)
        
        # 生成光线
        for y in range(image_size):
            for x in range(image_size):
                # 归一化像素坐标
                u = (x + 0.5) / image_size - 0.5
                v = (y + 0.5) / image_size - 0.5
                
                # 光线方向
                ray_dir = forward + u * right + v * up
                ray_dir = ray_dir / torch.norm(ray_dir)
                
                # 复杂的颜色函数
                distance = torch.norm(cam_pos)
                color = torch.sigmoid(torch.tensor([
                    0.8 + 0.2 * ray_dir[0] + 0.1 * np.sin(
                        distance,
                    )
                ]))
                
                ray_origins.append(cam_pos)
                ray_directions.append(ray_dir)
                colors.append(color)
    
    return (
        torch.stack(ray_origins), torch.stack(ray_directions), torch.stack(colors)
    )

def train_nerfacto(
    model: MockNerfacto,
    ray_origins: torch.Tensor,
    ray_directions: torch.Tensor,
    target_colors: torch.Tensor,
    num_epochs: int = 200,
)
    """训练Nerfacto模型"""
    print(f"🚀 开始训练Nerfacto模型")
    print(f"📈 训练数据: {len(ray_origins)} 条光线")
    print(f"🔄 训练轮次: {num_epochs}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️  使用设备: {device}")
    
    model = model.to(device)
    
    # 使用AdamW优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    ray_origins = ray_origins.to(device)
    ray_directions = ray_directions.to(device)
    target_colors = target_colors.to(device)
    
    training_history = []
    
    for epoch in range(num_epochs):
        # 随机采样
        batch_size = 2048
        indices = torch.randperm(len(ray_origins))[:batch_size]
        
        batch_origins = ray_origins[indices]
        batch_directions = ray_directions[indices]
        batch_colors = target_colors[indices]
        
        # 前向传播
        optimizer.zero_grad()
        outputs = model(batch_origins, batch_directions)
        
        # 计算损失
        color_loss = torch.nn.functional.mse_loss(outputs['color'], batch_colors)
        
        # L1正则化
        l1_reg = sum(torch.sum(torch.abs(param)) for param in model.parameters())
        total_loss = color_loss + 1e-6 * l1_reg
        
        # 反向传播
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        # 记录
        if epoch % 40 == 0:
            with torch.no_grad():
                mse = color_loss.item()
                psnr = -10 * np.log10(mse) if mse > 0 else float('inf')
                lr = scheduler.get_last_lr()[0]
                
                training_history.append({
                    'epoch': epoch, 'loss': color_loss.item(
                    )
                })
                
                print(f"Epoch {epoch:3d}: Loss={color_loss.item():.6f}, PSNR={psnr:.2f}dB, "
                      f"LR={lr:.2e}")
    
    print("✅ 训练完成!")
    return training_history

def demonstrate_nerfacto():
    """演示Nerfacto的完整流程"""
    print("🌟 Nerfacto 演示")
    print("=" * 60)
    
    if not NERFACTO_AVAILABLE:
        print("⚠️ 使用模拟实现进行演示")
    
    # 1. 创建配置
    config = MockNerfactoConfig()
    print(f"⚙️  模型配置:")
    print(f"   - 隐藏层维度: {config.hidden_dim}")
    print(f"   - 几何特征维度: {config.geo_feat_dim}")
    print(f"   - 哈希网格层数: {config.num_levels}")
    print(f"   - 最大分辨率: {config.max_res}")
    print(f"   - 基础分辨率: {config.base_res}")
    
    # 2. 创建数据集
    ray_origins, ray_directions, target_colors = create_realistic_dataset(
        num_views=30, image_size=48
    )
    
    # 3. 创建模型
    model = MockNerfacto(config)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"🧠 模型参数数量: {total_params:, }")
    
    # 4. 训练模型
    training_history = train_nerfacto(
        model, ray_origins, ray_directions, target_colors, num_epochs=120
    )
    
    # 5. 性能统计
    print("\n" + "=" * 60)
    print("📊 Nerfacto性能统计:")
    
    if training_history:
        final_metrics = training_history[-1]
        print(f"   - 最终损失: {final_metrics['loss']:.6f}")
        print(f"   - 最终PSNR: {final_metrics['psnr']:.2f} dB")
        print(f"   - 最终学习率: {final_metrics['learning_rate']:.2e}")
    
    print(f"   - 总参数量: {total_params:, }")
    print(f"   - 模型大小: {total_params * 4 / 1024 / 1024:.2f} MB")
    
    print("\n🎉 Nerfacto演示完成!")
    print("\n📋 Nerfacto特点:")
    print("   ✅ 实用化NeRF实现")
    print("   ✅ 快速收敛训练")
    print("   ✅ 高质量渲染")
    print("   ✅ 多层哈希编码")
    print("   ✅ 球谐方向编码")
    print("   ✅ 相机参数优化")
    print("   ✅ 工程化设计")
    
    return model, training_history

if __name__ == '__main__':
    print("启动Nerfacto演示...")
    model, history = demonstrate_nerfacto()
    print("演示完成!") 