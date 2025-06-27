from typing import Optional
#!/usr/bin/env python3
"""
Block NeRF 演示脚本

展示Block NeRF的大规模城市场景建模功能，包括：
- 场景分块管理
- 块级组合渲染
- 可见性网络
- 大规模场景重建
"""

import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from src.nerfs.block_nerf.core import BlockNeRF, BlockNeRFConfig
    from src.nerfs.block_nerf.block_manager import BlockManager
    from src.nerfs.block_nerf.visibility_network import VisibilityNetwork
    BLOCK_NERF_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Block NeRF模块导入失败: {e}")
    BLOCK_NERF_AVAILABLE = False

class MockBlockNeRFConfig:
    """模拟Block NeRF配置"""
    def __init__(self):
        self.scene_bounds = (-50.0, -50.0, -10.0, 50.0, 50.0, 10.0)
        self.block_size = 20.0
        self.overlap_size = 2.0
        self.max_blocks = 16
        self.hidden_dim = 256
        self.num_layers = 8
        self.use_visibility_network = True
        self.visibility_threshold = 0.5

class MockVisibilityNetwork(torch.nn.Module):
    """模拟可见性网络"""
    
    def __init__(self, config):
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(6, 128), # position + direction
            torch.nn.ReLU(
            )
        )
    
    def forward(self, positions: torch.Tensor, directions: torch.Tensor) -> torch.Tensor:
        """计算可见性概率"""
        input_tensor = torch.cat([positions, directions], dim=-1)
        return self.network(input_tensor).squeeze(-1)

class MockBlockManager:
    """模拟块管理器"""
    
    def __init__(self, config):
        self.config = config
        self.blocks = {}
        self._create_blocks()
    
    def _create_blocks(self):
        """创建场景块"""
        scene_bounds = self.config.scene_bounds
        block_size = self.config.block_size
        
        x_min, y_min, z_min, x_max, y_max, z_max = scene_bounds
        
        x_coords = np.arange(x_min, x_max, block_size)
        y_coords = np.arange(y_min, y_max, block_size)
        
        block_id = 0
        for x in x_coords:
            for y in y_coords:
                if block_id < self.config.max_blocks:
                    block_bounds = (
                        x, y, z_min, x + block_size, y + block_size, z_max
                    )
                    self.blocks[block_id] = {
                        'bounds': block_bounds, 'center': torch.tensor([
                            x + block_size/2, y + block_size/2, (z_min + z_max)/2
                        ])
                    }
                    block_id += 1
    
    def get_relevant_blocks(self, position: torch.Tensor) -> list[int]:
        """获取与位置相关的块"""
        relevant_blocks = []
        for block_id, block_info in self.blocks.items():
            bounds = block_info['bounds']
            if (bounds[0] <= position[0] <= bounds[3] and
                bounds[1] <= position[1] <= bounds[4] and
                bounds[2] <= position[2] <= bounds[5]):
                relevant_blocks.append(block_id)
        
        # 如果没有找到，返回最近的块
        if not relevant_blocks:
            distances = []
            for block_id, block_info in self.blocks.items():
                center = block_info['center']
                distance = torch.norm(position - center)
                distances.append((distance, block_id))
            distances.sort()
            relevant_blocks = [distances[0][1]]
        
        return relevant_blocks

class MockBlockNeRF(torch.nn.Module):
    """模拟Block NeRF模型"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.block_manager = MockBlockManager(config)
        
        # 每个块都有自己的NeRF网络
        self.block_networks = torch.nn.ModuleDict()
        for block_id in self.block_manager.blocks.keys():
            network = torch.nn.Sequential(
                torch.nn.Linear(63, config.hidden_dim), # 位置编码后
                torch.nn.ReLU(
                )
            )
            self.block_networks[str(block_id)] = network
        
        # 可见性网络
        if config.use_visibility_network:
            self.visibility_network = MockVisibilityNetwork(config)
    
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
        directions: Optional[torch.Tensor] = None,
    )
        """前向传播"""
        batch_size = positions.shape[0]
        
        # 编码位置
        encoded_pos = self.positional_encoding(positions)
        
        # 为每个位置找到相关的块
        densities = []
        colors = []
        weights = []
        
        for i in range(batch_size):
            pos = positions[i]
            relevant_blocks = self.block_manager.get_relevant_blocks(pos)
            
            block_densities = []
            block_colors = []
            block_weights = []
            
            for block_id in relevant_blocks:
                # 使用对应块的网络
                network = self.block_networks[str(block_id)]
                output = network(encoded_pos[i:i+1])
                
                density = torch.relu(output[0, 0])
                color = torch.sigmoid(output[0, 1:])
                
                # 计算可见性权重
                if self.config.use_visibility_network and directions is not None:
                    visibility = self.visibility_network(pos.unsqueeze(0), directions[i:i+1])
                    weight = visibility[0]
                else:
                    weight = 1.0 / len(relevant_blocks)
                
                block_densities.append(density)
                block_colors.append(color)
                block_weights.append(weight)
            
            # 合并块的输出
            if block_densities:
                block_weights_tensor = torch.stack(block_weights)
                block_weights_tensor = block_weights_tensor / block_weights_tensor.sum()
                
                final_density = sum(d * w for d, w in zip(block_densities, block_weights_tensor))
                final_color = sum(c * w for c, w in zip(block_colors, block_weights_tensor))
            else:
                final_density = torch.tensor(0.0)
                final_color = torch.zeros(3)
            
            densities.append(final_density)
            colors.append(final_color)
        
        return {
            'density': torch.stack(
                densities,
            )
        }

def create_urban_dataset(num_views: int = 100, scene_size: float = 80.0) -> dict[str, torch.Tensor]:
    """创建城市场景数据集"""
    print(f"📊 创建城市场景数据集: {num_views}个视角, 场景大小{scene_size}x{scene_size}")
    
    ray_origins = []
    ray_directions = []
    colors = []
    
    for i in range(num_views):
        # 在城市中的相机位置
        angle = 2 * np.pi * i / num_views
        height = 5.0 + 10.0 * np.random.random()  # 变化的高度
        radius = 30.0 + 20.0 * np.random.random()  # 变化的距离
        
        cam_pos = torch.tensor([
            radius * np.cos(angle), radius * np.sin(angle), height
        ])
        
        # 朝向场景中心的方向
        target = torch.tensor([0.0, 0.0, 0.0])
        forward = target - cam_pos
        forward = forward / torch.norm(forward)
        
        # 生成一些光线（简化）
        for _ in range(50):
            # 添加一些随机偏移
            offset = torch.randn(3) * 0.3
            ray_dir = forward + offset
            ray_dir = ray_dir / torch.norm(ray_dir)
            
            # 基于位置和方向的复杂颜色函数
            distance_factor = torch.norm(cam_pos) / 50.0
            height_factor = cam_pos[2] / 20.0
            color = torch.sigmoid(torch.tensor([
                distance_factor + ray_dir[0], height_factor + ray_dir[1], 0.5 + ray_dir[2]
            ]))
            
            ray_origins.append(cam_pos)
            ray_directions.append(ray_dir)
            colors.append(color)
    
    return {
        'ray_origins': torch.stack(
            ray_origins,
        )
    }

def train_block_nerf(
    model: MockBlockNeRF,
    dataset: dict[str,
    torch.Tensor],
    num_epochs: int = 200,
)
    """训练Block NeRF模型"""
    print(f"🚀 开始训练Block NeRF模型")
    print(f"📈 训练数据: {len(dataset['ray_origins'])} 条光线")
    print(f"🏗️  场景块数: {len(model.block_manager.blocks)}")
    print(f"🔄 训练轮次: {num_epochs}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️  使用设备: {device}")
    
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    ray_origins = dataset['ray_origins'].to(device)
    ray_directions = dataset['ray_directions'].to(device)
    colors = dataset['colors'].to(device)
    
    training_history = []
    
    for epoch in range(num_epochs):
        # 随机采样
        batch_size = 512
        indices = torch.randperm(len(ray_origins))[:batch_size]
        
        batch_origins = ray_origins[indices]
        batch_directions = ray_directions[indices]
        batch_colors = colors[indices]
        
        # 前向传播
        optimizer.zero_grad()
        outputs = model(batch_origins, batch_directions)
        
        # 计算损失
        color_loss = torch.nn.functional.mse_loss(outputs['color'], batch_colors)
        
        # 反向传播
        color_loss.backward()
        optimizer.step()
        
        # 记录
        if epoch % 40 == 0:
            with torch.no_grad():
                mse = color_loss.item()
                psnr = -10 * np.log10(mse) if mse > 0 else float('inf')
                
                training_history.append({
                    'epoch': epoch, 'loss': color_loss.item(
                    )
                })
                
                print(f"Epoch {epoch:3d}: Loss={color_loss.item():.6f}, PSNR={psnr:.2f}dB, "
                      f"Blocks={outputs['num_blocks']}")
    
    print("✅ 训练完成!")
    return training_history

def demonstrate_block_nerf():
    """演示Block NeRF的完整流程"""
    print("🌟 Block NeRF 演示")
    print("=" * 60)
    
    if not BLOCK_NERF_AVAILABLE:
        print("⚠️ 使用模拟实现进行演示")
    
    # 1. 创建配置
    config = MockBlockNeRFConfig()
    print(f"⚙️  模型配置:")
    print(f"   - 场景边界: {config.scene_bounds}")
    print(f"   - 块大小: {config.block_size}")
    print(f"   - 重叠大小: {config.overlap_size}")
    print(f"   - 最大块数: {config.max_blocks}")
    print(f"   - 使用可见性网络: {config.use_visibility_network}")
    
    # 2. 创建数据集
    dataset = create_urban_dataset(num_views=50, scene_size=80.0)
    
    # 3. 创建模型
    model = MockBlockNeRF(config)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"🧠 模型参数数量: {total_params:, }")
    print(f"🏗️  实际创建块数: {len(model.block_manager.blocks)}")
    
    # 4. 训练模型
    training_history = train_block_nerf(model, dataset, num_epochs=100)
    
    # 5. 性能统计
    print("\n" + "=" * 60)
    print("📊 Block NeRF性能统计:")
    
    if training_history:
        final_metrics = training_history[-1]
        print(f"   - 最终损失: {final_metrics['loss']:.6f}")
        print(f"   - 最终PSNR: {final_metrics['psnr']:.2f} dB")
        print(f"   - 使用块数: {final_metrics['num_blocks']}")
    
    print(f"   - 总参数量: {total_params:, }")
    print(f"   - 模型大小: {total_params * 4 / 1024 / 1024:.2f} MB")
    print(f"   - 平均每块参数: {total_params // len(model.block_manager.blocks):, }")
    
    print("\n🎉 Block NeRF演示完成!")
    print("\n📋 Block NeRF特点:")
    print("   ✅ 大规模城市场景建模")
    print("   ✅ 场景分块管理")
    print("   ✅ 块级组合渲染")
    print("   ✅ 可见性网络优化")
    print("   ✅ 内存高效处理")
    print("   ✅ 可扩展架构")
    
    return model, training_history

if __name__ == '__main__':
    print("启动Block NeRF演示...")
    model, history = demonstrate_block_nerf()
    print("演示完成!") 