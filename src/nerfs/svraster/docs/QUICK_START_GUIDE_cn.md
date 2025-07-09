# SVRaster 快速开始指南（重构更新版）

## 🚀 快速上手 SVRaster 1.0.0

本指南基于最新的双渲染器架构，帮助您快速上手 SVRaster 的训练和推理。

## 🎯 核心概念

### 双渲染器架构
- **训练**: `VolumeRenderer` - 体积渲染支持梯度传播
- **推理**: `VoxelRasterizer` - 光栅化实现快速渲染

### 紧密耦合设计
- `SVRasterTrainer` + `VolumeRenderer` = 训练管线
- `SVRasterRenderer` + `VoxelRasterizer` = 推理管线

## 📦 安装

```bash
# 克隆仓库
git clone https://github.com/YourRepo/NeuroCity.git
cd NeuroCity

# 安装依赖
pip install -r requirements.txt

# 构建 CUDA 扩展（可选，用于加速）
cd src/nerfs/svraster/cuda
python setup.py build_ext --inplace
```

## 🔧 系统检查

```python
import src.nerfs.svraster as svraster

# 检查系统兼容性
svraster.check_compatibility()

# 查看可用组件
print(f"SVRaster 版本: {svraster.__version__}")
print(f"CUDA 可用: {svraster.CUDA_AVAILABLE}")
print(f"组件数量: {len(svraster.__all__)}")
```

## 🎓 训练流程

### 1. 基础配置

```python
import src.nerfs.svraster as svraster

# 模型配置
model_config = svraster.SVRasterConfig(
    max_octree_levels=8,        # 八叉树层数
    base_resolution=128,        # 基础分辨率
    scene_bounds=(-1, -1, -1, 1, 1, 1),  # 场景边界
    sh_degree=2,               # 球谐函数阶数
    learning_rate=1e-3,        # 学习率
    weight_decay=1e-6          # 权重衰减
)

# 数据集配置
dataset_config = svraster.SVRasterDatasetConfig(
    data_dir="data/nerf_synthetic/lego",  # 数据路径
    image_width=800,
    image_height=800,
    downscale_factor=2,        # 降采样因子
    num_rays_train=1024,       # 训练光线数
    num_rays_val=512           # 验证光线数
)

# 训练器配置
trainer_config = svraster.SVRasterTrainerConfig(
    num_epochs=100,            # 训练轮数
    batch_size=1,
    learning_rate=1e-3,
    save_every=10,             # 保存间隔
    validate_every=5,          # 验证间隔
    use_amp=True,              # 混合精度训练
    log_dir="logs/training"    # 日志目录
)
```

### 2. 创建训练组件

```python
# 创建模型
model = svraster.SVRasterModel(model_config)

# 创建数据集
dataset = svraster.SVRasterDataset(dataset_config)

# 创建体积渲染器（训练专用）
volume_renderer = svraster.VolumeRenderer(model_config)

# 创建训练器（紧密耦合）
trainer = svraster.SVRasterTrainer(model, volume_renderer, trainer_config)

print(f"模型参数数量: {sum(p.numel() for p in model.parameters())}")
print(f"训练器类型: {type(trainer).__name__}")
print(f"渲染器类型: {type(volume_renderer).__name__}")
```

### 3. 开始训练

```python
# 启动训练
trainer.train(dataset)

# 保存模型
trainer.save_checkpoint("checkpoints/model_epoch_100.pth")
```

## 🎨 推理流程

### 1. 加载模型

```python
# 加载训练好的模型
model = svraster.SVRasterModel.load_checkpoint("checkpoints/model_epoch_100.pth")
print("模型加载完成")
```

### 2. 创建推理组件

```python
# 光栅化器配置
raster_config = svraster.VoxelRasterizerConfig(
    background_color=(1.0, 1.0, 1.0),  # 白色背景
    near_plane=0.1,
    far_plane=100.0,
    density_activation="exp",
    color_activation="sigmoid"
)

# 渲染器配置
renderer_config = svraster.SVRasterRendererConfig(
    image_width=800,
    image_height=800,
    render_batch_size=4096,
    background_color=(1.0, 1.0, 1.0),
    output_format="png"
)

# 创建真实体素光栅化器（推理专用）
rasterizer = svraster.VoxelRasterizer(raster_config)

# 创建渲染器（紧密耦合）
renderer = svraster.SVRasterRenderer(model, rasterizer, renderer_config)

print(f"渲染器类型: {type(renderer).__name__}")
print(f"光栅化器类型: {type(rasterizer).__name__}")
```

### 3. 渲染图像

```python
import torch

# 定义相机位姿
camera_pose = torch.eye(4)  # 单位矩阵表示相机位姿
camera_pose[2, 3] = 2.0     # 相机距离场景2个单位

# 渲染图像
image = renderer.render(camera_pose, image_size=(800, 800))

# 保存结果
import torchvision.utils as vutils
vutils.save_image(image, "rendered_image.png")
print("渲染完成: rendered_image.png")
```

## 🚀 GPU 加速

如果您的系统支持 CUDA，可以使用 GPU 加速组件：

```python
if svraster.CUDA_AVAILABLE:
    # GPU 模型
    gpu_model = svraster.SVRasterGPU(model_config)
    
    # GPU 训练器
    gpu_trainer = svraster.SVRasterGPUTrainer(gpu_model, model_config)
    
    # EMA 模型（指数移动平均）
    ema_model = svraster.EMAModel(model, decay=0.999)
    
    print("GPU 加速组件已启用")
else:
    print("GPU 加速不可用，使用 CPU 版本")
```

## 🔧 高级功能

### 1. 自定义损失函数

```python
# 创建损失函数
loss_fn = svraster.SVRasterLoss(model_config)

# 计算各种损失
rgb_loss = loss_fn.compute_rgb_loss(pred_rgb, gt_rgb)
depth_loss = loss_fn.compute_depth_loss(pred_depth, gt_depth)
sparsity_loss = loss_fn.compute_sparsity_loss(voxel_densities)
```

### 2. 球谐函数

```python
# 计算球谐基函数
view_dirs = torch.randn(1000, 3)
view_dirs = view_dirs / torch.norm(view_dirs, dim=-1, keepdim=True)

sh_values = svraster.eval_sh_basis(degree=2, dirs=view_dirs)
print(f"球谐函数输出: {sh_values.shape}")
```

### 3. 工具函数

```python
# Morton 编码
morton_code = svraster.morton_encode_3d(1, 2, 3)
x, y, z = svraster.morton_decode_3d(morton_code)

# 八叉树操作
subdivided = svraster.octree_subdivision(octree_nodes)
pruned = svraster.octree_pruning(octree_nodes, threshold=0.01)
```

## 📊 性能优化建议

### 1. 训练优化
- 使用 `use_amp=True` 启用混合精度训练
- 调整 `num_rays_train` 平衡速度和质量
- 使用 `morton_ordering=True` 提高缓存效率

### 2. 推理优化
- 使用 `VoxelRasterizer` 而非 `VolumeRenderer`
- 调整 `render_batch_size` 优化GPU利用率
- 启用 `use_cached_features=True` 缓存特征

### 3. 内存优化
- 使用 `gradient_checkpointing` 减少显存使用
- 调整 `downscale_factor` 降低分辨率
- 分批处理大场景

## 🐛 常见问题

### Q: 训练器初始化失败
```python
# 错误方式
trainer = svraster.SVRasterTrainer(model, trainer_config)

# 正确方式
volume_renderer = svraster.VolumeRenderer(model_config)
trainer = svraster.SVRasterTrainer(model, volume_renderer, trainer_config)
```

### Q: 渲染器初始化失败
```python
# 错误方式
renderer = svraster.SVRasterRenderer(model, renderer_config)

# 正确方式
rasterizer = svraster.VoxelRasterizer(raster_config)
renderer = svraster.SVRasterRenderer(model, rasterizer, renderer_config)
```

### Q: CUDA 扩展不可用
```bash
# 构建 CUDA 扩展
cd src/nerfs/svraster/cuda
python setup.py build_ext --inplace

# 检查 CUDA 环境
python -c "import torch; print(torch.cuda.is_available())"
```

## 📚 进阶学习

- 查看 `examples/` 目录的完整示例
- 阅读 `docs/` 目录的详细文档
- 参考 `demos/` 目录的演示代码

## 🔗 相关资源

- [API 参考文档](./API_REFERENCE_UPDATED_cn.md)
- [训练与推理对比](./TRAINING_VS_INFERENCE_RENDERING_cn.md)
- [完整文档索引](./COMPLETE_DOCUMENTATION_INDEX_cn.md)

---

恭喜！您已经掌握了 SVRaster 1.0.0 的基本使用方法。现在可以开始您的稀疏体素渲染之旅了！
