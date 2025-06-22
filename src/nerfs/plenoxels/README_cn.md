# Plenoxels: 无神经网络的辐射场

本软件包实现了 **Plenoxels**，这是一种革命性的神经辐射场方法，使用稀疏体素网格和球谐函数替代神经网络。基于 Alex Yu 等人的论文 "Plenoxels: Radiance Fields without Neural Networks"。

## 概述

Plenoxels 代表了神经辐射场(NeRF)方法的范式转变：

- **无神经网络**：使用稀疏体素网格替代 MLP
- **球谐函数**：用 SH 系数表示视角相关的颜色
- **快速训练**：相比 vanilla NeRF 实现 100 倍加速
- **高质量**：保持相当或更好的渲染质量
- **内存高效**：稀疏表示减少内存使用

## 主要特性

### 🚀 快速训练
- 直接优化体素参数
- 无需通过神经网络的前向/后向传播
- 从粗到细的训练策略

### 🎯 高质量渲染
- 三线性插值实现平滑采样
- 球谐函数处理视角相关外观
- 正确的 alpha 合成体积渲染

### 💾 内存高效
- 稀疏体素网格表示
- 自动修剪低密度体素
- 可配置的分辨率等级

### 🔧 灵活配置
- 支持多种数据集格式
- 可定制的训练参数
- 易于集成到现有流水线

## 架构

```
Plenoxels架构:
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   输入光线      │ -> │   体素网格       │ -> │   体积渲染      │
│  (origin,dirs)  │    │ (密度 + SH)      │    │   (RGB, 深度)   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │ 三线性插值 +     │
                    │ SH计算           │
                    └──────────────────┘
```

## 安装

```bash
# 安装依赖
pip install torch torchvision numpy opencv-python imageio tqdm tensorboard

# 安装包
cd NeuroCity/src/plenoxels
python -m pip install -e .
```

## 快速开始

### 基础使用

```python
from src.plenoxels import PlenoxelConfig, PlenoxelModel

# 创建模型配置
config = PlenoxelConfig(
    grid_resolution=(256, 256, 256),
    scene_bounds=(-1.0, -1.0, -1.0, 1.0, 1.0, 1.0),
    sh_degree=2,
    use_coarse_to_fine=True
)

# 初始化模型
model = PlenoxelModel(config)

# 前向传播
outputs = model(ray_origins, ray_directions)
rgb = outputs['rgb']      # 渲染颜色
depth = outputs['depth']  # 深度值
```

### 训练示例

```python
from src.plenoxels import (
    PlenoxelConfig, PlenoxelDatasetConfig, PlenoxelTrainerConfig,
    create_plenoxel_trainer
)

# 配置
model_config = PlenoxelConfig(
    grid_resolution=(256, 256, 256),
    sh_degree=2,
    use_coarse_to_fine=True
)

dataset_config = PlenoxelDatasetConfig(
    data_dir="data/nerf_synthetic/lego",
    dataset_type="blender",
    num_rays_train=1024
)

trainer_config = PlenoxelTrainerConfig(
    max_epochs=10000,
    learning_rate=0.1,
    experiment_name="plenoxel_lego"
)

# 训练模型
trainer = create_plenoxel_trainer(model_config, trainer_config, dataset_config)
trainer.train()
```

## 数据集支持

### Blender 合成数据集
```python
dataset_config = PlenoxelDatasetConfig(
    data_dir="path/to/nerf_synthetic/scene",
    dataset_type="blender",
    white_background=True,
    downsample_factor=1
)
```

### COLMAP 真实数据集
```python
dataset_config = PlenoxelDatasetConfig(
    data_dir="path/to/colmap/scene",
    dataset_type="colmap",
    downsample_factor=4
)
```

### LLFF 前向数据集
```python
dataset_config = PlenoxelDatasetConfig(
    data_dir="path/to/llff/scene",
    dataset_type="llff",
    downsample_factor=8
)
```

## 配置选项

### 模型配置

```python
@dataclass
class PlenoxelConfig:
    # 体素网格设置
    grid_resolution: Tuple[int, int, int] = (256, 256, 256)
    scene_bounds: Tuple[float, ...] = (-1.0, -1.0, -1.0, 1.0, 1.0, 1.0)
    
    # 球谐函数
    sh_degree: int = 2  # 0-3，越高视角相关效果越明显
    
    # 从粗到细训练
    use_coarse_to_fine: bool = True
    coarse_resolutions: List = [(64, 64, 64), (128, 128, 128), (256, 256, 256)]
    coarse_epochs: List[int] = [2000, 5000, 10000]
    
    # 正则化
    sparsity_threshold: float = 0.01
    tv_lambda: float = 1e-6      # 总变分
    l1_lambda: float = 1e-8      # L1 稀疏性
    
    # 渲染
    near_plane: float = 0.1
    far_plane: float = 10.0
```

### 训练配置

```python
@dataclass
class PlenoxelTrainerConfig:
    # 训练
    max_epochs: int = 10000
    learning_rate: float = 0.1
    weight_decay: float = 0.0
    
    # 损失权重
    color_loss_weight: float = 1.0
    tv_loss_weight: float = 1e-6
    l1_loss_weight: float = 1e-8
    
    # 修剪
    pruning_threshold: float = 0.01
    pruning_interval: int = 1000
    
    # 日志和评估
    eval_interval: int = 1000
    save_interval: int = 5000
    log_interval: int = 100
    use_tensorboard: bool = True
```

## 高级功能

### 从粗到细训练

Plenoxels 支持逐步增加体素分辨率的渐进训练：

```python
config = PlenoxelConfig(
    use_coarse_to_fine=True,
    coarse_resolutions=[(64, 64, 64), (128, 128, 128), (256, 256, 256)],
    coarse_epochs=[2000, 5000, 10000]
)
```

### 稀疏性正则化

自动修剪低密度体素：

```python
# 训练期间，低于阈值的体素会被修剪
model.prune_voxels(threshold=0.01)

# 获取占用统计
stats = model.get_occupancy_stats()
print(f"稀疏度: {stats['sparsity_ratio']:.2%}")
```

### 自定义损失函数

```python
from src.plenoxels import PlenoxelLoss

class CustomPlenoxelLoss(PlenoxelLoss):
    def forward(self, outputs, targets):
        losses = super().forward(outputs, targets)
        
        # 添加自定义损失
        if 'depth' in outputs and 'depths' in targets:
            depth_loss = F.mse_loss(outputs['depth'], targets['depths'])
            losses['depth_loss'] = depth_loss
        
        return losses
```

## 工具函数

### 体素网格操作

```python
from src.plenoxels.utils import (
    create_voxel_grid,
    prune_voxel_grid,
    compute_voxel_occupancy_stats
)

# 创建体素网格
grid_info = create_voxel_grid(
    resolution=(128, 128, 128),
    scene_bounds=(-1, -1, -1, 1, 1, 1)
)

# 计算统计信息
stats = compute_voxel_occupancy_stats(density_grid)
```

### 渲染工具

```python
from src.plenoxels.utils import (
    generate_rays,
    sample_points_along_rays,
    volume_render
)

# 从相机姿态生成光线
rays_o, rays_d = generate_rays(poses, focal, H, W)

# 沿光线采样点
points, t_vals = sample_points_along_rays(
    rays_o, rays_d, near=0.1, far=10.0, num_samples=192
)
```

## 性能优化

### GPU 内存管理

```python
# 对高分辨率场景使用较小的批处理大小
dataset_config.num_rays_train = 512  # GPU 内存受限时减少

# 使用混合精度训练
trainer_config.use_amp = True
```

### 训练速度

```python
# 从粗分辨率开始
config.grid_resolution = (128, 128, 128)  # 更快的初始训练

# 减少采样数以提高速度
model(rays_o, rays_d, num_samples=64)  # vs 质量优先的 192
```

## 评估指标

```python
# 计算 PSNR
mse = torch.mean((pred_rgb - gt_rgb) ** 2)
psnr = -10.0 * torch.log10(mse)

# 计算 SSIM (需要额外依赖)
from skimage.metrics import structural_similarity as ssim
ssim_val = ssim(pred_img, gt_img, multichannel=True)
```

## 故障排除

### 常见问题

1. **内存不足**
   - 减少 `grid_resolution`
   - 降低 `num_rays_train`
   - 使用梯度检查点

2. **训练缓慢**
   - 启用从粗到细训练
   - 初始使用较低的 SH 阶数
   - 更频繁地修剪体素

3. **质量差**
   - 增加 `grid_resolution`
   - 更高的 `sh_degree` 用于视角相关效果
   - 正确调整场景边界

### 调试模式

```python
# 启用调试日志
import logging
logging.basicConfig(level=logging.DEBUG)

# 检查占用统计
stats = model.get_occupancy_stats()
print(f"占用体素: {stats['occupied_voxels']}/{stats['total_voxels']}")
```

## 示例

查看 `example_usage.py` 文件获取完整示例：

```bash
# 运行演示
python -m src.plenoxels.example_usage --mode demo

# 在 Blender 数据集上训练
python -m src.plenoxels.example_usage --mode train \
    --data_dir data/nerf_synthetic/lego \
    --dataset_type blender \
    --max_epochs 10000

# 渲染新视角
python -m src.plenoxels.example_usage --mode render \
    --checkpoint outputs/plenoxel_exp/best.pth \
    --num_renders 40
```

## 测试

运行测试套件：

```bash
python -m src.plenoxels.test_plenoxels
```

## 引用

如果您使用此实现，请引用原始论文：

```bibtex
@article{yu2021plenoxels,
  title={Plenoxels: Radiance fields without neural networks},
  author={Yu, Alex and Fridovich-Keil, Sara and Tancik, Matthew and Chen, Qinhong and Recht, Benjamin and Kanazawa, Angjoo},
  journal={arXiv preprint arXiv:2112.05131},
  year={2021}
}
```

## 许可证

此实现仅供研究和教育目的。许可证详情请参考原始论文和代码。

## 贡献

欢迎贡献！请随时提交问题和拉取请求。

## 致谢

此实现基于 Yu 等人在 Plenoxels 方面的优秀工作。特别感谢原作者在神经辐射场方面的开创性研究。 