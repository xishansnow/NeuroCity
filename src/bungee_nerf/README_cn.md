# BungeeNeRF: 极端多尺度场景渲染的渐进式神经辐射场

本包实现了BungeeNeRF，这是一个专为极端多尺度场景渲染设计的渐进式神经辐射场，基于Xiangli等人的论文《BungeeNeRF: Progressive Neural Radiance Field for Extreme Multi-scale Scene Rendering》。

## 特性

- **渐进式训练**：多阶段训练，逐步增加细节级别
- **多尺度渲染**：跨大幅度变化尺度的细节级别渲染
- **Google Earth Studio支持**：原生支持Google Earth Studio数据
- **自适应采样**：基于距离的自适应采样以实现高效渲染
- **灵活架构**：可配置的模型架构和训练参数

## 架构概览

BungeeNeRF包含几个关键组件：

1. **渐进式位置编码器**：在训练期间逐步激活高频通道
2. **多尺度编码器**：基于观看距离处理不同细节级别
3. **渐进式块**：在训练阶段添加的额外细化块
4. **多尺度渲染器**：细节级别体积渲染
5. **渐进式训练器**：实现渐进式训练策略

## 安装

```bash
# 安装所需依赖
pip install torch torchvision tqdm tensorboard pillow opencv-python scipy

# 包已在您的环境中可用
# 如果您使用提供的代码，无需额外安装
```

## 快速开始

### 基本使用

```python
from bungee_nerf import BungeeNeRF, BungeeNeRFConfig

# 创建配置
config = BungeeNeRFConfig(
    num_stages=4,
    base_resolution=16,
    max_resolution=2048,
    hidden_dim=256,
    num_layers=8
)

# 创建模型
model = BungeeNeRF(config)

# 设置渐进式阶段
model.set_current_stage(2)

# 前向传播
outputs = model(rays_o, rays_d, bounds, distances)
```

### 训练

```python
# 命令行训练
python -m bungee_nerf.train_bungee_nerf \
    --data_dir /path/to/dataset \
    --trainer_type progressive \
    --num_epochs 100 \
    --num_stages 4 \
    --log_dir ./logs \
    --save_dir ./checkpoints
```

### 渲染

```python
# 命令行渲染
python -m bungee_nerf.render_bungee_nerf \
    --checkpoint ./checkpoints/best.pth \
    --data_dir /path/to/dataset \
    --render_type test \
    --output_dir ./renders
```

## 配置选项

### 模型配置

```python
config = BungeeNeRFConfig(
    # 渐进式结构
    num_stages=4,              # 渐进式阶段数
    base_resolution=16,        # 编码基础分辨率
    max_resolution=2048,       # 编码最大分辨率
    scale_factor=4.0,          # 阶段间缩放因子
    
    # 位置编码
    num_freqs_base=4,          # 基础频带数
    num_freqs_max=10,          # 最大频带数
    include_input=True,        # 包含输入坐标
    
    # MLP架构
    hidden_dim=256,            # 隐藏层维度
    num_layers=8,              # MLP层数
    skip_layers=[4],           # 跳跃连接层
    
    # 渐进式块
    block_hidden_dim=128,      # 渐进式块隐藏维度
    block_num_layers=4,        # 渐进式块层数
    
    # 训练参数
    batch_size=4096,           # 批量大小
    learning_rate=5e-4,        # 学习率
    max_steps=200000,          # 最大训练步数
    
    # 采样
    num_samples=64,            # 每光线采样数
    num_importance=128,        # 重要性采样数
    perturb=True,              # 添加采样扰动
    
    # 多尺度参数
    scale_weights=[1.0, 0.8, 0.6, 0.4],           # 尺度权重
    distance_thresholds=[100.0, 50.0, 25.0, 10.0], # 距离阈值
    
    # 损失权重
    color_loss_weight=1.0,     # 颜色损失权重
    depth_loss_weight=0.1,     # 深度损失权重
    progressive_loss_weight=0.05 # 渐进式损失权重
)
```

### 训练选项

包支持三种类型的训练器：

1. **BungeeNeRFTrainer**：用于标准NeRF训练的基础训练器
2. **ProgressiveTrainer**：基于阶段渐进的渐进式训练
3. **MultiScaleTrainer**：带自适应采样的多尺度训练

## 数据集支持

### 支持的格式

- **NeRF合成**：Blender渲染的合成场景
- **LLFF**：真实世界前向面对场景
- **Google Earth Studio**：具有极端尺度变化的航空/卫星图像

### Google Earth Studio数据

对于Google Earth Studio数据，包期望：

```
data_dir/
├── metadata.json          # 来自GES的相机元数据
├── images/               # 渲染图像
│   ├── frame_000000.jpg
│   ├── frame_000001.jpg
│   └── ...
```

metadata.json应包含带有位置、旋转和FOV信息的相机帧。

## 渐进式训练

BungeeNeRF使用渐进式训练策略：

1. **阶段0**：使用浅基础块训练远距离视图
2. **阶段1-N**：为近距离视图逐步添加细节块
3. **频率激活**：逐步激活高频位置编码

### 训练调度

```python
from bungee_nerf.utils import create_progressive_schedule

schedule = create_progressive_schedule(
    num_stages=4,
    steps_per_stage=50000,
    warmup_steps=1000
)
```

## 多尺度渲染

包实现细节级别渲染：

- **基于距离的LOD**：基于相机距离的不同细节级别
- **自适应采样**：为较近对象提供更多采样
- **渐进式块**：为高细节区域提供额外细化

## API参考

### 核心类

#### BungeeNeRF
实现BungeeNeRF架构的主模型类。

```python
model = BungeeNeRF(config)
model.set_current_stage(stage)
outputs = model(rays_o, rays_d, bounds, distances)
```

#### ProgressivePositionalEncoder
实现渐进式位置编码的编码器。

```python
encoder = ProgressivePositionalEncoder(
    num_freqs_base=4,
    num_freqs_max=10,
    include_input=True
)

# 设置当前激活频率
encoder.set_current_freqs(current_freqs)
encoded = encoder(positions)
```

#### MultiScaleRenderer
支持多尺度渲染的渲染器。

```python
renderer = MultiScaleRenderer(config)
results = renderer.render_rays(
    model, rays_o, rays_d, bounds, distances
)
```

## 训练和评估

### 训练脚本

```bash
# 基础训练
python train_bungee_nerf.py \
    --data_dir data/google_earth_studio/scene \
    --trainer_type progressive \
    --num_stages 4 \
    --num_epochs 200

# 多尺度训练
python train_bungee_nerf.py \
    --data_dir data/extreme_scale_scene \
    --trainer_type multiscale \
    --distance_thresholds 1000,500,100,50 \
    --scale_weights 1.0,0.8,0.6,0.4
```

### 评估指标

```python
from bungee_nerf.evaluation import evaluate_model

metrics = evaluate_model(
    model=model,
    test_dataset=test_dataset,
    output_dir="evaluation_results"
)

print(f"PSNR: {metrics['psnr']:.2f}")
print(f"SSIM: {metrics['ssim']:.3f}")
print(f"LPIPS: {metrics['lpips']:.3f}")
```

## 渐进式训练实现

### 频率调度

```python
def frequency_schedule(step, total_steps, num_freqs_base, num_freqs_max):
    """计算当前步骤的激活频率数"""
    progress = step / total_steps
    num_freqs = num_freqs_base + progress * (num_freqs_max - num_freqs_base)
    return int(num_freqs)
```

### 渐进式损失

```python
def progressive_loss(coarse_rgb, fine_rgb, target_rgb, stage_weights):
    """计算渐进式训练损失"""
    coarse_loss = F.mse_loss(coarse_rgb, target_rgb)
    fine_loss = F.mse_loss(fine_rgb, target_rgb)
    
    # 基于阶段的权重
    total_loss = stage_weights[0] * coarse_loss + stage_weights[1] * fine_loss
    
    return total_loss
```

## 多尺度渲染实现

### 距离基础采样

```python
def distance_based_sampling(rays_o, rays_d, distances, thresholds):
    """基于距离的自适应采样"""
    num_samples_list = []
    
    for i, threshold in enumerate(thresholds):
        mask = distances < threshold
        if i == 0:
            num_samples = torch.where(mask, 128, 64)
        else:
            prev_mask = distances < thresholds[i-1]
            current_mask = mask & ~prev_mask
            num_samples = torch.where(current_mask, 64, 32)
        
        num_samples_list.append(num_samples)
    
    return num_samples_list
```

### 多尺度特征融合

```python
def multiscale_feature_fusion(features_list, distances, scale_weights):
    """融合多尺度特征"""
    weighted_features = []
    
    for features, weight in zip(features_list, scale_weights):
        # 基于距离的权重调制
        distance_weight = torch.exp(-distances / 100.0)
        final_weight = weight * distance_weight
        
        weighted_features.append(features * final_weight)
    
    return torch.stack(weighted_features).sum(dim=0)
```

## 性能优化

### 内存管理

```python
# 梯度检查点以节省内存
def forward_with_checkpointing(model, x):
    return torch.utils.checkpoint.checkpoint(model, x)

# 分块渲染大图像
def render_image_in_chunks(model, rays, chunk_size=1024):
    results = []
    for i in range(0, rays.shape[0], chunk_size):
        chunk_rays = rays[i:i+chunk_size]
        chunk_result = model(chunk_rays)
        results.append(chunk_result)
    return torch.cat(results, dim=0)
```

### 训练加速

```python
# 混合精度训练
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    outputs = model(rays_o, rays_d, bounds, distances)
    loss = compute_loss(outputs, targets)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

## 故障排除

### 常见问题

**渐进式训练不稳定**
```python
# 调整阶段转换
config.stage_transition_smoothness = 0.1
config.frequency_annealing_rate = 0.001
```

**多尺度渲染质量差**
```python
# 调整距离阈值
config.distance_thresholds = [200.0, 100.0, 50.0, 25.0]
config.scale_weights = [1.0, 0.9, 0.7, 0.5]
```

**内存不足**
```python
# 减少批量大小和采样数
config.batch_size = 2048
config.num_samples = 32
config.num_importance = 64
```

## 应用案例

### Google Earth Studio数据训练

```python
# 为极端尺度场景配置
config = BungeeNeRFConfig(
    num_stages=6,
    base_resolution=8,
    max_resolution=4096,
    distance_thresholds=[10000, 5000, 1000, 500, 100, 50],
    progressive_loss_weight=0.1
)
```

### 城市场景渲染

```python
# 城市多尺度配置
config = BungeeNeRFConfig(
    num_stages=4,
    scale_weights=[1.0, 0.8, 0.6, 0.4],
    distance_thresholds=[500, 200, 100, 50],
    use_appearance_embedding=True
)
```

## 许可证

MIT许可证

## 引用

```bibtex
@article{xiangli2022bungeenerf,
  title={BungeeNeRF: Progressive Neural Radiance Field for Extreme Multi-scale Scene Rendering},
  author={Xiangli, Yuanbo and others},
  journal={ECCV},
  year={2022}
}
```

## 贡献

欢迎贡献！请查看项目的贡献指南以获取更多信息。 