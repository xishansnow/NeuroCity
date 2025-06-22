# 经典 NeRF: 视图合成的神经辐射场

本包实现了原始 NeRF 模型，基于开创性论文：

**"NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis"**  

*Ben Mildenhall, Pratul P. Srinivasan, Matthew Tancik, Jonathan T. Barron, Ravi Ramamoorthi, Ren Ng*  
ECCV 2020

## 概述

神经辐射场（NeRF）是一种通过使用稀疏输入视图优化底层连续体积场景函数来合成复杂场景新视图的方法。该方法使用全连接深度网络将场景表示为连续 5D 函数，输出空间中任意点的体积密度和视角相关发射辐射。

### 关键特性

- **位置编码**：通过正弦位置编码实现高频细节
- **视角相关渲染**：逼真的镜面反射和视角相关效果  
- **分层体积采样**：提高效率的粗到精采样策略
- **体积渲染**：带有神经辐射场的可微分体积渲染
- **多数据集支持**：Blender 合成场景和真实世界数据集

## 安装

```bash
# 安装依赖
pip install torch torchvision numpy imageio opencv-python tqdm tensorboard

# 安装包
cd src/classic_nerf
pip install -e .
```

## 快速开始

### 基本使用

```python
from classic_nerf import NeRFConfig, NeRF, NeRFTrainer
from classic_nerf.dataset import create_nerf_dataloader

# 创建配置
config = NeRFConfig(
    netdepth=8,
    netwidth=256,
    N_samples=64,
    N_importance=128,
    learning_rate=5e-4
)

# 加载数据集
train_loader = create_nerf_dataloader(
    'blender', 
    'path/to/blender/scene', 
    split='train',
    batch_size=1024
)

# 创建并训练模型
trainer = NeRFTrainer(config)
trainer.train(train_loader, num_epochs=100)
```

### 训练模型

```python
import torch
from classic_nerf import NeRFConfig, NeRFTrainer
from classic_nerf.dataset import create_nerf_dataloader

# 配置模型
config = NeRFConfig(
    # 网络架构
    netdepth=8,              # MLP 深度
    netwidth=256,            # MLP 宽度
    netdepth_fine=8,         # 精细网络深度
    netwidth_fine=256,       # 精细网络宽度
    
    # 位置编码
    multires=10,             # 坐标编码级别
    multires_views=4,        # 方向编码级别
    
    # 采样
    N_samples=64,            # 每光线粗采样数
    N_importance=128,        # 每光线精细采样数
    perturb=True,            # 分层采样
    
    # 训练
    learning_rate=5e-4,
    lrate_decay=250,
    
    # 场景边界
    near=2.0,
    far=6.0
)

# 创建数据加载器
train_loader = create_nerf_dataloader(
    dataset_type='blender',
    basedir='data/nerf_synthetic/lego',
    split='train',
    batch_size=1024,
    white_bkgd=True
)

val_loader = create_nerf_dataloader(
    dataset_type='blender',
    basedir='data/nerf_synthetic/lego', 
    split='val',
    batch_size=1024,
    white_bkgd=True
)

# 创建训练器
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
trainer = NeRFTrainer(config, device=device)

# 训练模型
trainer.train(
    train_loader=train_loader,
    val_loader=val_loader,
    num_epochs=200,
    log_dir='logs/classic_nerf',
    ckpt_dir='checkpoints/classic_nerf',
    val_interval=10,
    save_interval=50
)
```

### 渲染新视图

```python
import numpy as np
from classic_nerf.utils import create_spherical_poses, to8b
import imageio

# 加载训练好的模型
trainer.load_checkpoint('checkpoints/classic_nerf/final.pth')

# 创建螺旋相机路径
render_poses = create_spherical_poses(radius=4.0, n_poses=40)

# 相机参数
H, W = 800, 800
focal = 1111.1
K = np.array([[focal, 0, W/2], [0, focal, H/2], [0, 0, 1]])

# 渲染视频
frames = []
for i, c2w in enumerate(render_poses):
    print(f"正在渲染第{i+1}/{len(render_poses)}帧")
    rgb = trainer.render_test_image(H, W, K, c2w.numpy())
    frames.append(to8b(rgb))

# 保存视频
imageio.mimwrite('spiral_render.mp4', frames, fps=30, quality=8)
```

## 模型架构

### NeRF 网络

NeRF 模型包含：

1. **位置编码**：将 3D 坐标和 2D 视角方向映射到高维空间
2. **MLP 网络**：8 层全连接网络，使用 ReLU 激活
3. **跳跃连接**：在第 4 层连接输入坐标
4. **视角相关输出**：分离的密度和颜色分支

```
输入 (x, y, z, θ, φ) → 位置编码 → MLP → (RGB, σ)
                                  ↑
                              跳跃连接
```

### 体积渲染

渲染方程沿光线积分：

```
C(r) = ∫ T(t) σ(r(t)) c(r(t), d) dt

其中 T(t) = exp(-∫ σ(r(s)) ds)
```

## 配置选项

### 网络架构
- `netdepth`: MLP 深度 (默认: 8)  
- `netwidth`: MLP 宽度 (默认: 256)
- `netdepth_fine`: 精细网络深度 (默认: 8)
- `netwidth_fine`: 精细网络宽度 (默认: 256)

### 位置编码
- `multires`: 坐标编码级别 (默认: 10)
- `multires_views`: 方向编码级别 (默认: 4)

### 采样
- `N_samples`: 每光线粗采样数 (默认: 64)
- `N_importance`: 每光线精细采样数 (默认: 128)
- `perturb`: 启用分层采样 (默认: True)

### 训练
- `learning_rate`: 学习率 (默认: 5e-4)
- `lrate_decay`: 学习率衰减 (默认: 250)
- `raw_noise_std`: 密度噪声标准差 (默认: 1.0)

## 数据集格式

### Blender 合成数据集

```
scene_name/
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

### LLFF 真实场景数据集

```
scene_name/
├── poses_bounds.npy
└── images/
    ├── IMG_0001.jpg
    ├── IMG_0002.jpg
    └── ...
```

## 性能基准

### 训练时间（单个场景）
- GPU: NVIDIA RTX 3080
- 图像分辨率: 800x800
- 典型训练时间: 12-24 小时

### 质量指标（NeRF 合成数据集）

| 场景 | PSNR | SSIM | LPIPS |
|------|------|------|-------|
| Chair | 33.00 | 0.967 | 0.046 |
| Drums | 25.01 | 0.925 | 0.091 |
| Ficus | 30.13 | 0.964 | 0.044 |
| Hotdog | 36.18 | 0.974 | 0.121 |
| Lego | 32.54 | 0.961 | 0.050 |

## 工程实现

### 体积渲染实现

```python
def volume_render(rgb, sigma, z_vals, rays_d, noise_std=0.0):
    """体积渲染函数"""
    # 计算距离
    dists = z_vals[..., 1:] - z_vals[..., :-1]
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[..., :1].shape)], -1)
    
    # 乘以光线方向的模长
    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)
    
    # 添加噪声防止浮点精度问题
    if noise_std > 0.:
        noise = torch.randn(sigma.shape) * noise_std
        sigma = sigma + noise
    
    # 计算 alpha 值
    alpha = 1. - torch.exp(-sigma * dists)
    
    # 计算透射率
    T = torch.cumprod(torch.cat([torch.ones(alpha.shape[:-1] + [1]), 1. - alpha + 1e-10], -1), -1)[..., :-1]
    
    # 计算权重
    weights = alpha * T
    
    # 渲染 RGB
    rgb_map = torch.sum(weights[..., None] * rgb, -2)
    
    return rgb_map, weights
```

### 位置编码实现

```python
def positional_encoding(x, L):
    """位置编码函数"""
    encoding = [x]
    for i in range(L):
        for fn in [torch.sin, torch.cos]:
            encoding.append(fn(2.**i * x))
    return torch.cat(encoding, -1)
```

## 训练技巧

### 优化策略
1. **学习率调度**：使用指数衰减
2. **权重初始化**：Xavier 初始化
3. **批量大小**：建议 1024-4096 光线
4. **数据增强**：随机光线采样

### 内存优化
```python
# 分块处理大图像
def render_image_in_chunks(model, H, W, K, c2w, chunk=1024):
    """分块渲染大图像"""
    rays_o, rays_d = get_rays(H, W, K, c2w)
    rays_o = rays_o.reshape(-1, 3)
    rays_d = rays_d.reshape(-1, 3)
    
    all_rgb = []
    for i in range(0, rays_o.shape[0], chunk):
        rgb_chunk = model(rays_o[i:i+chunk], rays_d[i:i+chunk])
        all_rgb.append(rgb_chunk)
    
    return torch.cat(all_rgb, 0).reshape(H, W, 3)
```

## 故障排除

### 常见问题及解决方案

**训练不收敛**
- 检查学习率设置
- 确认数据集格式正确
- 调整位置编码级别

**内存不足**
- 减少批量大小
- 使用梯度检查点
- 降低图像分辨率

**渲染结果模糊**
- 增加采样点数量
- 提高位置编码级别
- 检查相机参数

## 扩展功能

### 自定义损失函数
```python
class NeRFLoss(nn.Module):
    def __init__(self, lambda_rgb=1.0, lambda_depth=0.1):
        super().__init__()
        self.lambda_rgb = lambda_rgb
        self.lambda_depth = lambda_depth
    
    def forward(self, pred_rgb, target_rgb, pred_depth=None, target_depth=None):
        rgb_loss = F.mse_loss(pred_rgb, target_rgb)
        loss = self.lambda_rgb * rgb_loss
        
        if pred_depth is not None and target_depth is not None:
            depth_loss = F.mse_loss(pred_depth, target_depth)
            loss += self.lambda_depth * depth_loss
        
        return loss
```

## 许可证

MIT 许可证

## 引用

```bibtex
@inproceedings{mildenhall2020nerf,
  title={NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis},
  author={Mildenhall, Ben and Srinivasan, Pratul P and Tancik, Matthew and Barron, Jonathan T and Ramamoorthi, Ravi and Ng, Ren},
  booktitle={European conference on computer vision},
  pages={405--421},
  year={2020},
  organization={Springer}
}
```

## 致谢

感谢原始 NeRF 作者的开创性工作，以及 PyTorch 和相关开源库的支持。 