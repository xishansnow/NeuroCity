# Block-NeRF: 可扩展的大场景神经视图合成

本包实现了Block-NeRF，这是一种通过将大场景分解为独立训练的NeRF块来实现可扩展神经视图合成的方法。

## 概述

Block-NeRF通过以下方式解决了将NeRF应用到大场景的挑战：
- **块分解**：将场景分解为可管理的块
- **外观嵌入**：处理光照和环境变化
- **姿态细化**：改善相机姿态对齐
- **可见性预测**：高效的渲染块选择
- **块合成**：块间的无缝混合

## 特性

### 核心组件

- **BlockNeRF模型**：带有基于块分解的主模型
- **块管理器**：处理块组织和选择
- **可见性网络**：预测块可见性以实现高效渲染
- **外观嵌入**：控制环境变化
- **姿态细化**：改善相机姿态对齐
- **块合成器**：平滑地混合块之间的过渡

### 核心能力

- ✅ 城市规模场景重建
- ✅ 动态外观处理
- ✅ 高效的基于块的渲染
- ✅ 姿态细化和优化
- ✅ 曝光和光照控制
- ✅ 无缝块过渡
- ✅ 多GPU训练支持

## 安装

```bash
# 安装依赖
pip install torch torchvision numpy opencv-python pillow matplotlib tqdm tensorboard

# 包已可从src/nerfs/block_nerf目录使用
```

## 快速开始

### 基本使用

```python
from block_nerf import BlockNeRF, BlockNeRFConfig, BlockManager

# 创建配置
config = BlockNeRFConfig(
    num_blocks=8,
    block_size=100.0,
    overlap_ratio=0.1,
    appearance_dim=32
)

# 创建块管理器
block_manager = BlockManager(
    scene_bounds=(-500, -500, -50, 500, 500, 50),
    block_size=100.0,
    overlap_ratio=0.1
)

# 创建模型
model = BlockNeRF(config, block_manager)
```

### 训练脚本

```bash
# 训练Block-NeRF模型
python train_block_nerf.py \
    --data_dir /path/to/waymo/data \
    --exp_name waymo_block_nerf \
    --num_blocks 8 \
    --block_size 100 \
    --max_steps 200000 \
    --use_appearance_embedding

# 分布式训练
python -m torch.distributed.launch --nproc_per_node=4 \
    train_block_nerf.py \
    --data_dir /path/to/data \
    --exp_name distributed_training \
    --distributed
```

### 渲染

```bash
# 渲染测试视图
python render_block_nerf.py \
    --checkpoint ./checkpoints/waymo_block_nerf/final.pth \
    --data_dir /path/to/waymo/data \
    --output_dir ./renders \
    --render_type test

# 渲染视频
python render_block_nerf.py \
    --checkpoint ./checkpoints/waymo_block_nerf/final.pth \
    --render_type video \
    --camera_path spiral \
    --output_dir ./videos
```

## 配置

### 模型配置

```python
config = BlockNeRFConfig(
    # 块分解
    num_blocks=8,               # 块数量
    block_size=100.0,           # 块大小（米）
    overlap_ratio=0.1,          # 块重叠比例
    
    # 网络架构
    netdepth=8,                 # 网络深度
    netwidth=256,               # 网络宽度
    
    # 外观嵌入
    appearance_dim=32,          # 外观嵌入维度
    use_appearance_embedding=True,
    
    # 姿态细化
    optimize_poses=True,        # 优化相机姿态
    pose_lr=1e-4,              # 姿态学习率
    
    # 训练参数
    batch_size=1024,            # 批量大小
    learning_rate=5e-4,         # 学习率
    max_steps=200000,           # 最大步数
    
    # 块选择
    visibility_threshold=0.1,   # 可见性阈值
    max_blocks_per_ray=3,      # 每条光线最大块数
    
    # 场景边界
    scene_bounds=(-500, -500, -50, 500, 500, 50)
)
```

### 块管理配置

```python
from block_nerf import BlockManager

block_manager = BlockManager(
    scene_bounds=(-500, -500, -50, 500, 500, 50),
    block_size=100.0,
    overlap_ratio=0.1,
    min_images_per_block=50,    # 每块最少图像数
    max_blocks=16,              # 最大块数
    adaptive_subdivision=True   # 自适应细分
)
```

## 核心算法

### 块分解策略

1. **均匀网格分解**：将场景划分为规则网格
2. **自适应分解**：基于图像密度的动态分解
3. **重叠区域**：确保块间平滑过渡

### 外观嵌入

```python
class AppearanceEmbedding(nn.Module):
    def __init__(self, num_images, embedding_dim=32):
        super().__init__()
        self.embedding = nn.Embedding(num_images, embedding_dim)
    
    def forward(self, image_indices):
        return self.embedding(image_indices)
```

### 可见性网络

```python
class VisibilityNetwork(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, camera_pos, block_centers):
        # 预测相机-块可见性
        features = torch.cat([camera_pos, block_centers], dim=-1)
        return self.mlp(features)
```

## 数据集支持

### Waymo开放数据集

```python
from block_nerf.dataset import WaymoDataset

dataset = WaymoDataset(
    data_root="/path/to/waymo",
    split="train",
    sequence_id="segment-xxx",
    camera_names=["FRONT", "FRONT_LEFT", "FRONT_RIGHT"],
    image_scale=0.5
)
```

### 自定义数据集

```python
from block_nerf.dataset import CustomDataset

dataset = CustomDataset(
    data_root="/path/to/data",
    poses_file="poses.txt",
    images_dir="images",
    intrinsics_file="intrinsics.txt"
)
```

## 训练流程

### 1. 数据预处理

```python
# 块分配
block_assignment = block_manager.assign_images_to_blocks(
    camera_positions=camera_positions,
    image_paths=image_paths
)

# 外观编码
appearance_codes = appearance_encoder.encode_images(images)
```

### 2. 分布式训练

```python
from block_nerf.trainer import DistributedTrainer

trainer = DistributedTrainer(
    model=model,
    config=config,
    rank=rank,
    world_size=world_size
)

trainer.train(
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    num_epochs=100
)
```

### 3. 姿态优化

```python
# 联合优化网络参数和相机姿态
optimizer_nerf = torch.optim.Adam(model.parameters(), lr=5e-4)
optimizer_poses = torch.optim.Adam(poses, lr=1e-4)

for step in range(max_steps):
    # NeRF前向传播
    rgb_pred = model(rays_o, rays_d, block_ids, appearance_codes)
    
    # 计算损失
    loss = F.mse_loss(rgb_pred, rgb_target)
    
    # 反向传播
    loss.backward()
    optimizer_nerf.step()
    optimizer_poses.step()
```

## 渲染流程

### 块选择

```python
def select_visible_blocks(camera_pos, block_centers, visibility_net):
    """选择可见的块"""
    visibility_scores = visibility_net(camera_pos, block_centers)
    visible_blocks = torch.where(visibility_scores > visibility_threshold)[0]
    return visible_blocks
```

### 块合成

```python
def composite_blocks(block_colors, block_weights, block_depths):
    """合成多个块的颜色"""
    # 深度排序
    sorted_indices = torch.argsort(block_depths)
    
    # Alpha合成
    final_color = torch.zeros_like(block_colors[0])
    accumulated_alpha = torch.zeros(block_colors.shape[:-1])
    
    for idx in sorted_indices:
        alpha = block_weights[idx]
        color = block_colors[idx]
        
        final_color += (1 - accumulated_alpha) * alpha * color
        accumulated_alpha += (1 - accumulated_alpha) * alpha
    
    return final_color
```

## 性能优化

### 内存管理

```python
# 动态块加载
class DynamicBlockLoader:
    def __init__(self, max_cached_blocks=4):
        self.max_cached_blocks = max_cached_blocks
        self.cached_blocks = {}
    
    def load_block(self, block_id):
        if block_id not in self.cached_blocks:
            if len(self.cached_blocks) >= self.max_cached_blocks:
                # 移除最久未使用的块
                oldest_block = min(self.cached_blocks.keys())
                del self.cached_blocks[oldest_block]
            
            # 加载新块
            self.cached_blocks[block_id] = self._load_block_data(block_id)
        
        return self.cached_blocks[block_id]
```

### 并行渲染

```python
def parallel_render_blocks(rays, block_ids, models):
    """并行渲染多个块"""
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for block_id in block_ids:
            future = executor.submit(
                models[block_id].render_rays,
                rays
            )
            futures.append(future)
        
        results = [future.result() for future in futures]
    
    return results
```

## 评估指标

### 渲染质量

```python
from block_nerf.metrics import compute_metrics

metrics = compute_metrics(
    pred_images=rendered_images,
    target_images=ground_truth_images
)

print(f"PSNR: {metrics['psnr']:.2f}")
print(f"SSIM: {metrics['ssim']:.3f}")
print(f"LPIPS: {metrics['lpips']:.3f}")
```

### 系统性能

```python
# 渲染速度测试
import time

start_time = time.time()
rendered_image = model.render_image(H, W, K, c2w)
render_time = time.time() - start_time

print(f"渲染时间: {render_time:.2f}秒")
print(f"渲染速度: {1/render_time:.2f} FPS")
```

## 故障排除

### 常见问题

**块边界伪影**
```python
# 增加重叠比例
config.overlap_ratio = 0.2

# 使用更平滑的权重函数
config.blending_function = 'gaussian'
```

**内存不足**
```python
# 减少同时加载的块数量
config.max_blocks_per_ray = 2
config.max_cached_blocks = 2
```

**训练不稳定**
```python
# 调整学习率
config.learning_rate = 1e-4
config.pose_lr = 5e-5

# 使用梯度裁剪
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

## 应用场景

### 自动驾驶场景

```python
# Waymo数据集训练
python train_block_nerf.py \
    --dataset waymo \
    --data_dir /path/to/waymo \
    --sequence_length 200 \
    --use_lidar_supervision \
    --appearance_regularization 0.01
```

### 城市重建

```python
# 大规模城市数据
python train_block_nerf.py \
    --dataset custom \
    --data_dir /path/to/city_data \
    --num_blocks 64 \
    --block_size 50 \
    --distributed
```

## 许可证

Apache 2.0许可证

## 引用

```bibtex
@article{tancik2022blocknerf,
  title={Block-NeRF: Scalable Large Scene Neural View Synthesis},
  author={Tancik, Matthew and others},
  journal={CVPR},
  year={2022}
}
```

## 贡献

欢迎贡献代码和报告问题！请查看CONTRIBUTING.md了解详细信息。 