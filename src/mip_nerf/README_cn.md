# Mip-NeRF: 多尺度表示用于抗锯齿神经辐射场

本目录包含Mip-NeRF的完整实现，该方法通过将每个像素表示为锥体而非光线，并使用积分位置编码来解决神经辐射场中的锯齿伪影问题。

## 概述

Mip-NeRF在原始NeRF基础上的改进：

1. **积分位置编码（IPE）**：IPE不是编码单独的点，而是通过对锥台体积进行位置编码积分来编码整个锥形锥台。

2. **锥形锥台表示**：每个像素在3D空间中表示为锥形锥台，考虑像素的有限大小和抗锯齿。

3. **多尺度渲染**：该方法自然处理不同观看距离下的不同细节级别。

4. **抗锯齿**：减少了原始NeRF中常见的锯齿伪影，特别是在不同分辨率下渲染时。

## 关键特性

- ✅ **积分位置编码**：处理锥台体积而非点样本
- ✅ **锥形锥台采样**：正确的像素足迹建模
- ✅ **分层采样**：粗到精采样策略
- ✅ **多尺度损失**：多分辨率级别的训练
- ✅ **全面训练流水线**：完整的训练、验证和测试
- ✅ **多数据集支持**：Blender合成和LLFF真实场景
- ✅ **可视化工具**：训练曲线、渲染图像和视频生成

## 架构

### 核心组件

- **`MipNeRF`**：结合粗略和精细网络的主模型类
- **`IntegratedPositionalEncoder`**：用于编码锥台的IPE实现
- **`ConicalFrustum`**：3D空间中像素锥的表示
- **`MipNeRFMLP`**：带有积分位置编码的神经网络
- **`MipNeRFRenderer`**：带抗锯齿的体积渲染

### 训练组件

- **`MipNeRFTrainer`**：带验证和测试的完整训练流水线
- **`MipNeRFLoss`**：带粗略和精细网络组件的损失函数
- **数据集类**：支持Blender和LLFF数据集

## 安装

该实现需要以下依赖：

```bash
torch >= 1.9.0
torchvision
numpy
opencv-python
imageio
matplotlib
tqdm
tensorboard
PIL
pathlib
```

## 使用方法

### 基本训练

```python
from src.mip_nerf import MipNeRF, MipNeRFConfig, MipNeRFTrainer
from src.mip_nerf.dataset import create_mip_nerf_dataset

# 配置
config = MipNeRFConfig(
    netdepth=8,
    netwidth=256,
    num_samples=64,
    num_importance=128,
    use_viewdirs=True,
    lr_init=5e-4,
    lr_final=5e-6
)

# 加载数据集
train_dataset = create_mip_nerf_dataset(
    data_dir="path/to/dataset",
    dataset_type='blender',
    split='train'
)

val_dataset = create_mip_nerf_dataset(
    data_dir="path/to/dataset", 
    dataset_type='blender',
    split='val'
)

# 创建并训练模型
model = MipNeRF(config)
trainer = MipNeRFTrainer(config, model, train_dataset, val_dataset)
trainer.train(num_epochs=100)
```

### 推理

```python
# 加载训练好的模型
model = MipNeRF(config)
checkpoint = torch.load('path/to/checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])

# 渲染光线
with torch.no_grad():
    results = model(origins, directions, viewdirs, near=2.0, far=6.0)
    rgb = results['fine']['rgb']  # 或 results['coarse']['rgb']
```

## 数据集格式

### Blender合成数据集

该实现支持标准NeRF Blender数据集格式：

```
dataset/
├── transforms_train.json
├── transforms_val.json  
├── transforms_test.json
├── train/
│   ├── r_0.png
│   ├── r_1.png
│   └── ...
├── val/
└── test/
```

### LLFF真实场景

也支持LLFF格式数据集：

```
dataset/
├── poses_bounds.npy
└── images/
    ├── IMG_0001.jpg
    ├── IMG_0002.jpg
    └── ...
```

## 配置选项

`MipNeRFConfig`中的关键配置参数：

- **`netdepth`**：MLP中的层数（默认：8）
- **`netwidth`**：每层神经元数（默认：256）
- **`num_samples`**：每光线粗采样数（默认：64）
- **`num_importance`**：每光线精细采样数（默认：128）
- **`min_deg_point`**：位置编码最小度数（默认：0）
- **`max_deg_point`**：位置编码最大度数（默认：12）
- **`use_viewdirs`**：是否使用视角方向（默认：True）
- **`lr_init`**：初始学习率（默认：5e-4）
- **`lr_final`**：最终学习率（默认：5e-6）

## 训练技巧

1. **从较低分辨率开始**进行快速实验
2. **对Blender场景中的透明对象使用白色背景**
3. **训练期间监控PSNR** - 应该稳步增加
4. **根据收敛情况调整学习率调度**
5. **使用验证集**防止过拟合

## 与原始NeRF的差异

| 特性 | 原始NeRF | Mip-NeRF |
|------|----------|----------|
| 位置编码 | 点位置编码 | 积分位置编码（IPE） |
| 光线表示 | 无穷小光线 | 锥形锥台 |
| 抗锯齿 | 无 | 通过IPE内置 |
| 多尺度 | 手动技巧 | 自然处理 |
| 像素足迹 | 忽略 | 正确建模 |

## 数学基础

### 积分位置编码

对于均值为μ、协方差为Σ的多元高斯分布：

```
IPE(μ, Σ) = [E[sin(2^j μ)], E[cos(2^j μ)]] for j = 0, ..., L-1
```

其中：
```
E[sin(x)] = exp(-σ²/2) * sin(μ)
E[cos(x)] = exp(-σ²/2) * cos(μ)
```

### 锥形锥台到高斯

每个像素锥近似为具有以下特性的3D高斯：
- **均值**：锥台的中心
- **协方差**：结合轴向（沿光线）和径向（垂直）方差

## 性能

标准数据集上的预期性能：

### NeRF合成数据集（PSNR）

| 场景 | 原始NeRF | Mip-NeRF | 改进 |
|------|----------|----------|------|
| Chair | 33.00 | 34.84 | +1.84 |
| Drums | 25.01 | 25.75 | +0.74 |
| Ficus | 30.13 | 33.90 | +3.77 |
| Hotdog | 36.18 | 37.40 | +1.22 |
| Lego | 32.54 | 36.39 | +3.85 |

### LLFF数据集（PSNR）

| 场景 | 原始NeRF | Mip-NeRF | 改进 |
|------|----------|----------|------|
| Fern | 25.17 | 25.68 | +0.51 |
| Flower | 27.40 | 28.03 | +0.63 |
| Fortress | 31.16 | 31.82 | +0.66 |
| Horns | 27.45 | 28.20 | +0.75 |
| Leaves | 20.92 | 21.74 | +0.82 |

## 训练命令

### 在Blender数据集上训练

```bash
python -m src.mip_nerf.train_mip_nerf \
    --data_dir data/nerf_synthetic/lego \
    --dataset_type blender \
    --exp_name lego_mipnerf \
    --num_epochs 100 \
    --white_background
```

### 在LLFF数据集上训练

```bash
python -m src.mip_nerf.train_mip_nerf \
    --data_dir data/nerf_llff_data/fern \
    --dataset_type llff \
    --exp_name fern_mipnerf \
    --num_epochs 100 \
    --factor 8
```

## 测试和评估

```bash
# 测试训练好的模型
python -m src.mip_nerf.test_mip_nerf \
    --checkpoint path/to/checkpoint.pth \
    --data_dir data/nerf_synthetic/lego \
    --dataset_type blender \
    --compute_metrics

# 渲染视频
python -m src.mip_nerf.render_video \
    --checkpoint path/to/checkpoint.pth \
    --data_dir data/nerf_synthetic/lego \
    --video_dir videos/lego_spiral
```

## 实现细节

### 积分位置编码实现

```python
def integrated_pos_enc(means, covs, min_deg, max_deg):
    """积分位置编码函数"""
    scales = 2.0 ** torch.arange(min_deg, max_deg)
    
    # 计算每个频率的方差
    scaled_means = means[..., None, :] * scales[:, None]
    scaled_vars = covs[..., None, :] * (scales[:, None] ** 2)
    
    # 计算积分编码
    sin_vals = torch.sin(scaled_means) * torch.exp(-0.5 * scaled_vars)
    cos_vals = torch.cos(scaled_means) * torch.exp(-0.5 * scaled_vars)
    
    return torch.cat([sin_vals, cos_vals], dim=-1)
```

### 锥形锥台计算

```python
def cast_rays(origins, directions, radii, near, far, num_samples):
    """投射锥形光线"""
    # 沿光线采样距离
    t_vals = torch.linspace(near, far, num_samples)
    
    # 计算采样点
    means = origins[..., None, :] + directions[..., None, :] * t_vals[..., :, None]
    
    # 计算锥的半径
    radii_at_t = radii[..., None] * t_vals
    
    # 计算协方差矩阵
    # 轴向方差（沿光线方向）
    dt = t_vals[..., 1:] - t_vals[..., :-1]
    dt = torch.cat([dt, dt[..., -1:]], dim=-1)
    axial_var = (dt / 3.0) ** 2
    
    # 径向方差（垂直于光线）
    radial_var = radii_at_t ** 2 / 4.0
    
    return means, axial_var, radial_var
```

## 故障排除

### 常见问题

**训练不稳定**
```python
# 降低学习率
config.lr_init = 1e-4
config.lr_final = 1e-6

# 调整位置编码频率
config.max_deg_point = 10
```

**内存不足**
```python
# 减少批量大小
config.batch_size = 1024

# 减少采样数
config.num_samples = 32
config.num_importance = 64
```

**渲染质量差**
```python
# 增加网络容量
config.netwidth = 512
config.netdepth = 10

# 增加采样数
config.num_samples = 128
config.num_importance = 256
```

## 许可证

MIT许可证

## 引用

```bibtex
@article{barron2021mipnerf,
  title={Mip-NeRF: A Multiscale Representation for Anti-Aliasing Neural Radiance Fields},
  author={Barron, Jonathan T and Mildenhall, Ben and Tancik, Matthew and Hedman, Peter and Martin-Brualla, Ricardo and Srinivasan, Pratul P},
  journal={ICCV},
  year={2021}
}
```

## 致谢

感谢原始Mip-NeRF作者的出色工作，以及NeRF社区的持续贡献。 