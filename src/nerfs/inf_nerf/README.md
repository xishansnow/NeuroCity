# InfNeRF: Infinite Scale NeRF with Decoupled Architecture

InfNeRF 是一个具有 O(log n) 空间复杂度的无限尺度神经辐射场实现，采用训练器和渲染器解耦的架构设计。

## 架构特点

### 🔧 解耦设计
- **训练器 (Trainer)**: 专门负责训练循环、损失计算和模型优化
- **渲染器 (Renderer)**: 专门负责推理渲染、图像生成和视频制作
- **核心模型 (Core)**: 包含 octree 结构、LoD 感知 NeRF 等核心组件
- **体积渲染器 (Volume Renderer)**: 提供训练阶段的体积渲染算法

### 🌳 Octree 结构
- 基于八叉树的层次细节 (LoD) 结构
- 自适应深度和节点剪枝
- 内存高效的场景表示

### 🎯 主要功能
- 无限尺度场景渲染
- 抗锯齿渲染
- 分布式训练支持
- 高质量图像和视频生成

## 快速开始

### 安装依赖

```bash
pip install torch torchvision
pip install imageio  # 用于视频渲染
```

### 基本使用

```python
from src.nerfs.inf_nerf import (
    InfNeRF, InfNeRFConfig,
    InfNeRFTrainer, InfNeRFTrainerConfig,
    InfNeRFRenderer, InfNeRFRendererConfig
)

# 1. 创建模型
config = InfNeRFConfig(
    max_depth=8,
    hidden_dim=64,
    num_samples=64
)
model = InfNeRF(config)

# 2. 训练模型
trainer_config = InfNeRFTrainerConfig(
    num_epochs=100,
    lr_init=1e-2,
    rays_batch_size=4096
)
trainer = InfNeRFTrainer(model, train_dataset, trainer_config)
trainer.train()

# 3. 渲染图像
renderer_config = InfNeRFRendererConfig(
    image_width=800,
    image_height=600
)
renderer = InfNeRFRenderer(model, renderer_config)

# 渲染单张图像
result = renderer.render_image(camera_pose, intrinsics)

# 渲染视频
renderer.render_spiral_video(
    center=torch.tensor([0, 0, 0]),
    radius=2.0,
    num_frames=100,
    intrinsics=intrinsics,
    output_path="output.mp4"
)
```

### 从检查点加载

```python
# 从检查点加载渲染器
renderer = InfNeRFRenderer.from_checkpoint(
    "checkpoints/best.pth",
    renderer_config
)

# 渲染演示图像
from src.nerfs.inf_nerf import render_demo_images
render_demo_images(renderer, num_views=8, output_dir="demo_renders")
```

## 核心组件

### InfNeRF 模型
```python
class InfNeRF(nn.Module):
    """主模型，包含 octree 结构和 LoD 感知 NeRF"""
    
    def __init__(self, config: InfNeRFConfig):
        # 初始化八叉树结构
        # 设置 LoD 感知 NeRF
        
    def forward(self, rays_o, rays_d, near, far, focal_length, pixel_width):
        # 前向传播，使用 octree 进行层次渲染
```

### 训练器
```python
class InfNeRFTrainer:
    """训练器，负责训练循环和优化"""
    
    def train(self):
        # 主训练循环
        
    def train_step(self, batch):
        # 单步训练
        
    def validate(self):
        # 验证
```

### 渲染器
```python
class InfNeRFRenderer:
    """渲染器，负责推理和图像生成"""
    
    def render_image(self, camera_pose, intrinsics):
        # 渲染单张图像
        
    def render_video(self, camera_trajectory, intrinsics, output_path):
        # 渲染视频
        
    def render_spiral_video(self, center, radius, num_frames, ...):
        # 渲染螺旋轨迹视频
```

### 体积渲染器
```python
class VolumeRenderer:
    """体积渲染器，提供训练阶段的渲染算法"""
    
    def volume_render(self, colors, densities, z_vals, rays_d):
        # 体积渲染
        
    def compute_losses(self, outputs, targets):
        # 计算训练损失
```

## 配置说明

### InfNeRFConfig
- `max_depth`: 八叉树最大深度
- `hidden_dim`: 隐藏层维度
- `num_samples`: 采样点数量
- `scene_bound`: 场景边界

### InfNeRFTrainerConfig
- `num_epochs`: 训练轮数
- `lr_init`: 初始学习率
- `rays_batch_size`: 光线批次大小
- `mixed_precision`: 是否使用混合精度

### InfNeRFRendererConfig
- `image_width/height`: 图像尺寸
- `render_chunk_size`: 渲染块大小
- `save_depth/alpha`: 是否保存深度/透明度

## 高级功能

### 分布式训练
```python
# 支持多 GPU 训练
trainer_config = InfNeRFTrainerConfig(
    distributed=True,
    local_rank=0
)
```

### 自定义损失
```python
# 在 VolumeRenderer 中自定义损失权重
volume_config = VolumeRendererConfig(
    lambda_rgb=1.0,
    lambda_depth=0.1,
    lambda_distortion=0.01
)
```

### 内存优化
```python
# 分块渲染以节省内存
renderer_config = InfNeRFRendererConfig(
    render_chunk_size=1024,
    max_rays_per_batch=8192
)
```

## 性能优化

1. **内存管理**: 使用分块渲染和梯度检查点
2. **混合精度**: 支持 AMP 训练
3. **八叉树剪枝**: 自动剪枝低密度节点
4. **层次采样**: LoD 感知的采样策略

## 扩展性

- 支持自定义 NeRF 网络结构
- 可扩展的损失函数
- 灵活的相机轨迹生成
- 模块化的渲染管线

## 贡献

欢迎提交 Issue 和 Pull Request！

## 许可证

MIT License 