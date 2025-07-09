# SVRaster One - 可微分光栅化渲染器

## 概述

SVRaster One 是一个基于 SVRaster 原始论文实现的可微分光栅化渲染器，支持端到端的梯度传播和训练。与传统的体积渲染不同，SVRaster One 使用基于投影的光栅化方法，实现了高效的推理和可微分的训练。

## 核心特性

### 🎯 可微分光栅化
- **软光栅化**：使用高斯核实现可微分的像素分配
- **软深度排序**：基于 softmax 的可微分深度排序
- **Alpha 混合**：可微分的透明度合成
- **梯度传播**：支持端到端的反向传播

### 🧊 稀疏体素表示
- **动态细分**：基于梯度幅度的自适应体素细分
- **自适应剪枝**：移除低密度体素，保持稀疏性
- **Morton 排序**：空间局部性优化
- **内存优化**：智能内存管理

### ⚡ 高效渲染
- **投影式渲染**：直接投影体素到屏幕空间
- **视锥剔除**：移除不可见体素
- **分块处理**：支持图像分块并行渲染
- **CUDA 加速**：GPU 并行计算

### 🔄 端到端训练
- **混合精度训练**：支持 AMP 加速
- **自适应优化**：动态调整体素结构
- **多种损失函数**：RGB、深度、感知、SSIM 损失
- **梯度裁剪**：防止梯度爆炸

## 架构设计

### 1. 核心组件

```
SVRasterOne
├── SparseVoxelGrid          # 稀疏体素网格
│   ├── MortonCode          # Morton 编码
│   ├── adaptive_subdivision # 自适应细分
│   └── adaptive_pruning    # 自适应剪枝
├── DifferentiableVoxelRasterizer  # 可微分光栅化器
│   ├── soft_rasterization  # 软光栅化
│   ├── soft_depth_sorting  # 软深度排序
│   └── alpha_blending      # Alpha 混合
├── SVRasterOneLoss         # 损失函数
│   ├── rgb_loss           # RGB 重建损失
│   ├── depth_loss         # 深度损失
│   ├── density_reg_loss   # 密度正则化
│   └── sparsity_loss      # 稀疏性损失
└── SVRasterOneTrainer     # 训练器
    ├── adaptive_optimization # 自适应优化
    ├── mixed_precision     # 混合精度训练
    └── checkpoint_management # 检查点管理
```

### 2. 可微分光栅化流程

```python
# 1. 体素投影（可微分）
screen_voxels = project_voxels_to_screen(voxels, camera_matrix, intrinsics)

# 2. 视锥剔除
visible_voxels = frustum_culling(screen_voxels, viewport_size)

# 3. 软深度排序（可微分）
if use_soft_sorting:
    sorted_voxels = soft_depth_sort(visible_voxels, temperature)
else:
    sorted_voxels = hard_depth_sort(visible_voxels)

# 4. 软光栅化（可微分）
if soft_rasterization:
    framebuffer = soft_rasterize_voxels(sorted_voxels, sigma)
else:
    framebuffer = hard_rasterize_voxels(sorted_voxels)
```

### 3. 梯度传播机制

```python
# 前向传播
rendered_output = model.forward(camera_matrix, intrinsics, mode="training")

# 计算损失
losses = model.compute_loss(rendered_output, target_data)

# 反向传播（梯度自动传播到体素参数）
losses["total_loss"].backward()

# 自适应优化
gradient_magnitudes = torch.abs(model.voxel_grid.voxel_features.grad[:, 0])
model.adaptive_optimization(gradient_magnitudes)
```

## 安装和使用

### 1. 环境要求

```bash
# Python 3.8+
# PyTorch 1.9+
# CUDA 11.0+ (可选，用于 GPU 加速)
```

### 2. 基本使用

```python
from svraster_one import SVRasterOne, SVRasterOneConfig

# 创建配置
config = SVRasterOneConfig()
config.rendering.image_width = 800
config.rendering.image_height = 600
config.voxel.grid_resolution = 256
config.voxel.max_voxels = 1000000

# 创建模型
model = SVRasterOne(config)

# 渲染
camera_matrix = torch.eye(4)  # 相机变换矩阵
intrinsics = torch.tensor([[800, 0, 400], [0, 800, 300], [0, 0, 1]])  # 相机内参

rendered_output = model.forward(camera_matrix, intrinsics, mode="inference")
rgb_image = rendered_output["rgb"]
depth_map = rendered_output["depth"]
```

### 3. 训练示例

```python
from svraster_one import SVRasterOneTrainer

# 创建训练器
trainer = SVRasterOneTrainer(model, config)

# 训练
trainer.train(
    train_loader=train_loader,
    val_loader=val_loader,
    num_epochs=1000,
    save_dir="checkpoints"
)
```

### 4. 自适应优化

```python
# 在训练过程中自动触发
for epoch in range(num_epochs):
    for batch in train_loader:
        # 前向传播和反向传播
        result = model.training_step_forward(...)
        result["total_loss"].backward()
        
        # 自适应优化（每100步）
        if step % 100 == 0:
            gradient_magnitudes = torch.abs(model.voxel_grid.voxel_features.grad[:, 0])
            model.adaptive_optimization(gradient_magnitudes)
```

## 配置说明

### 体素配置 (VoxelConfig)

```python
@dataclass
class VoxelConfig:
    grid_resolution: int = 256      # 体素网格分辨率
    voxel_size: float = 0.01       # 体素大小
    max_voxels: int = 1000000      # 最大体素数量
    sparsity_threshold: float = 0.01  # 稀疏性阈值
    adaptive_subdivision: bool = True  # 自适应细分
    subdivision_threshold: float = 0.1  # 细分阈值
    use_morton_ordering: bool = True   # Morton 排序
```

### 渲染配置 (RenderingConfig)

```python
@dataclass
class RenderingConfig:
    image_width: int = 800         # 图像宽度
    image_height: int = 600        # 图像高度
    soft_rasterization: bool = True  # 软光栅化
    temperature: float = 0.1       # 软光栅化温度
    sigma: float = 1.0            # 高斯核标准差
    depth_sorting: str = "back_to_front"  # 深度排序方式
    use_soft_sorting: bool = True  # 软排序
    alpha_blending: bool = True    # Alpha 混合
```

### 训练配置 (TrainingConfig)

```python
@dataclass
class TrainingConfig:
    rgb_loss_weight: float = 1.0   # RGB 损失权重
    depth_loss_weight: float = 0.1  # 深度损失权重
    density_reg_weight: float = 0.01  # 密度正则化权重
    sparsity_weight: float = 0.001   # 稀疏性权重
    learning_rate: float = 1e-3    # 学习率
    batch_size: int = 4096         # 批次大小
    use_amp: bool = True           # 混合精度训练
    grad_clip: float = 1.0         # 梯度裁剪
```

## 性能优化

### 1. 内存优化

```python
# 获取内存使用情况
memory_usage = model.get_memory_usage()
print(f"内存使用: {memory_usage['total_memory_mb']:.2f} MB")

# 优化内存使用
model.optimize_memory(target_memory_mb=1000.0)
```

### 2. 渲染性能

```python
# 推理模式（硬光栅化）
rendered_output = model.forward(camera_matrix, intrinsics, mode="inference")

# 训练模式（软光栅化）
rendered_output = model.forward(camera_matrix, intrinsics, mode="training")
```

### 3. 自适应优化

```python
# 基于梯度幅度的体素细分
model.voxel_grid.adaptive_subdivision(gradient_magnitudes)

# 基于密度的体素剪枝
model.voxel_grid.adaptive_pruning()

# Morton 排序优化
model.voxel_grid.sort_by_morton()
```

## 与 SVRaster 论文的对应关系

### 1. 核心思想
- **投影式渲染**：将体素直接投影到屏幕空间，避免沿光线积分
- **可微分光栅化**：使用软光栅化实现梯度传播
- **稀疏体素表示**：动态调整体素结构，保持稀疏性

### 2. 技术实现
- **软光栅化**：使用高斯核替代硬光栅化
- **软深度排序**：基于 softmax 的可微分排序
- **自适应优化**：基于梯度的体素细分和剪枝

### 3. 性能优势
- **高效推理**：投影式渲染比体积渲染更快
- **内存效率**：稀疏体素表示减少内存使用
- **可微分训练**：支持端到端优化

## 扩展功能

### 1. 自定义损失函数

```python
from svraster_one.losses import CombinedLoss

# 组合损失函数
combined_loss = CombinedLoss(config)
combined_loss.use_perceptual = True
combined_loss.use_ssim = True

losses = combined_loss(rendered_output, target_data, voxel_data)
```

### 2. 序列渲染

```python
# 渲染相机序列
camera_matrices = torch.stack([cam1, cam2, cam3, ...])  # [N, 4, 4]
sequence_output = model.render_sequence(camera_matrices, intrinsics)
```

### 3. 体素导入导出

```python
# 导出体素数据
model.export_voxels("voxels.pth")

# 导入体素数据
model.import_voxels("voxels.pth")
```

## 故障排除

### 1. 常见问题

**Q: 梯度计算失败**
A: 确保使用 `mode="training"` 和 `soft_rasterization=True`

**Q: 内存不足**
A: 减少 `max_voxels` 或使用 `optimize_memory()`

**Q: 渲染速度慢**
A: 使用 `mode="inference"` 或减少体素数量

### 2. 性能调优

```python
# 减少体素数量
config.voxel.max_voxels = 100000

# 降低图像分辨率
config.rendering.image_width = 400
config.rendering.image_height = 300

# 调整软光栅化参数
config.rendering.temperature = 0.05
config.rendering.sigma = 0.5
```

## 贡献指南

欢迎贡献代码！请遵循以下步骤：

1. Fork 项目
2. 创建功能分支
3. 提交更改
4. 推送到分支
5. 创建 Pull Request

## 许可证

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。

## 引用

如果您在研究中使用了 SVRaster One，请引用：

```bibtex
@article{svraster2023,
  title={SVRaster: Efficient Neural Radiance Fields via Sparse Voxel Rasterization},
  author={...},
  journal={...},
  year={2023}
}
```

## 联系方式

如有问题或建议，请通过以下方式联系：

- 提交 Issue
- 发送邮件
- 参与讨论

---

**SVRaster One** - 让可微分光栅化渲染更简单、更高效！ 