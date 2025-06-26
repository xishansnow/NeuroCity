# SVRaster: 稀疏体素光栅化

SVRaster 是一个高性能的稀疏体素光栅化实现，用于实时高保真度辐射场渲染。该软件包实现了论文 "Sparse Voxels Rasterization: Real-time High-fidelity Radiance Field Rendering" 中描述的方法，无需神经网络或 3D 高斯。

## 主要特性

- **自适应稀疏体素**: 基于八叉树的层次化体素分配与细节层次
- **射线方向相关的莫顿排序**: 正确的深度排序，避免闪烁伪影
- **实时性能**: 高效光栅化，实现高帧率渲染
- **高保真度**: 支持高达 65536³ 的网格分辨率
- **网格兼容性**: 与体积融合、体素池化和行进立方体无缝集成

## 架构设计

### 核心组件

1. **AdaptiveSparseVoxels**: 管理基于八叉树 LOD 的稀疏体素表示
2. **VoxelRasterizer**: 高效稀疏体素渲染的定制光栅化器
3. **SVRasterModel**: 结合稀疏体素和光栅化的主模型
4. **SVRasterLoss**: 训练用损失函数

### 关键创新

- **自适应分配**: 显式地将稀疏体素分配到不同的细节层次
- **莫顿排序**: 使用射线方向相关的莫顿排序确保正确的图元混合
- **无神经网络**: 直接体素表示，无需 MLP 或 3D 高斯
- **高效存储**: 仅保留叶节点，无需完整八叉树结构

## 安装

```bash
# 安装依赖
pip install torch torchvision numpy pillow tqdm tensorboard

# 安装 SVRaster (如果已打包)
pip install svraster

# 或直接从源码使用
python -m src.svraster.example_usage --help
```

## 快速开始

### 基本用法

```python
from src.svraster import SVRasterConfig, SVRasterModel
from src.svraster.dataset import SVRasterDatasetConfig, create_svraster_dataset
from src.svraster.trainer import SVRasterTrainerConfig, create_svraster_trainer

# 创建模型配置
model_config = SVRasterConfig(
    max_octree_levels=12,
    base_resolution=64,
    scene_bounds=(-2.0, -2.0, -2.0, 2.0, 2.0, 2.0),
    density_activation="exp",
    color_activation="sigmoid"
)

# 创建模型
model = SVRasterModel(model_config)

# 渲染射线
ray_origins = torch.randn(1000, 3)
ray_directions = torch.randn(1000, 3)
outputs = model(ray_origins, ray_directions)
```

### 训练

```python
# 数据集配置
dataset_config = SVRasterDatasetConfig(
    data_dir="./data/nerf_synthetic/lego",
    dataset_type="blender",
    image_height=800,
    image_width=800,
    white_background=True
)

# 训练器配置
trainer_config = SVRasterTrainerConfig(
    num_epochs=100,
    learning_rate=1e-3,
    enable_subdivision=True,
    enable_pruning=True
)

# 创建数据集
train_dataset = create_svraster_dataset(dataset_config, split="train")
val_dataset = create_svraster_dataset(dataset_config, split="val")

# 创建并运行训练器
trainer = create_svraster_trainer(model_config, trainer_config, train_dataset, val_dataset)
trainer.train()
```

### 命令行使用

```bash
# 训练
python -m src.svraster.example_usage --mode train \
    --data_dir ./data/nerf_synthetic/lego \
    --output_dir ./outputs/svraster_lego

# 渲染
python -m src.svraster.example_usage --mode render \
    --data_dir ./data/nerf_synthetic/lego \
    --checkpoint ./outputs/svraster_lego/checkpoints/best_model.pth \
    --output_dir ./outputs/svraster_lego/renders
```

## 配置选项

### 模型配置

```python
SVRasterConfig(
    # 场景表示
    max_octree_levels=16,        # 最大八叉树层数 (65536³ 分辨率)
    base_resolution=64,          # 基础网格分辨率
    scene_bounds=(-1, -1, -1, 1, 1, 1),  # 场景边界框
    
    # 体素属性
    density_activation="exp",     # 密度激活函数
    color_activation="sigmoid",   # 颜色激活函数
    sh_degree=2,                 # 球谐函数度数
    
    # 自适应分配
    subdivision_threshold=0.01,   # 体素细分阈值
    pruning_threshold=0.001,     # 体素剪枝阈值
    
    # 光栅化
    ray_samples_per_voxel=8,     # 每个体素沿射线的采样数
    morton_ordering=True,        # 使用莫顿排序
    
    # 渲染
    background_color=(0, 0, 0),  # 背景颜色
    use_view_dependent_color=True,
    use_opacity_regularization=True
)
```

### 数据集配置

```python
SVRasterDatasetConfig(
    # 数据路径
    data_dir="./data",
    images_dir="images",
    
    # 数据格式
    dataset_type="blender",      # blender, colmap
    image_height=800,
    image_width=800,
    
    # 数据划分
    train_split=0.8,
    val_split=0.1,
    test_split=0.1,
    
    # 射线采样
    num_rays_train=1024,
    num_rays_val=512,
    
    # 背景处理
    white_background=False,
    black_background=False
)
```

### 训练器配置

```python
SVRasterTrainerConfig(
    # 训练参数
    num_epochs=100,
    batch_size=1,
    learning_rate=1e-3,
    
    # 自适应细分
    enable_subdivision=True,
    subdivision_start_epoch=10,
    subdivision_interval=5,
    
    # 剪枝
    enable_pruning=True,
    pruning_start_epoch=20,
    pruning_interval=10,
    
    # 日志和保存
    val_interval=5,
    log_interval=100,
    save_interval=1000,
    
    # 硬件
    device="cuda",
    use_mixed_precision=True
)
```

## 数据格式

### 支持的数据集类型

1. **Blender 合成数据**: NeRF 合成数据集格式
2. **COLMAP**: 使用 COLMAP 处理的真实世界捕获数据

### 目录结构

```
data/
├── images/              # 输入图像
│   ├── image_001.png
│   └── ...
├── transforms_train.json # 相机位姿 (Blender格式)
├── transforms_val.json
└── transforms_test.json
```

## 高级功能

### 自适应细分

SVRaster 根据渲染梯度自动细分体素：

```python
# 启用自适应细分
trainer_config.enable_subdivision = True
trainer_config.subdivision_start_epoch = 10
trainer_config.subdivision_interval = 5
trainer_config.subdivision_threshold = 0.01
```

### 体素剪枝

移除低密度体素以保持效率：

```python
# 启用剪枝
trainer_config.enable_pruning = True
trainer_config.pruning_start_epoch = 20
trainer_config.pruning_interval = 10
trainer_config.pruning_threshold = 0.001
```

### 莫顿排序

射线方向相关的莫顿排序防止闪烁伪影：

```python
# 莫顿排序默认启用
model_config.morton_ordering = True
```

## 性能优化

### 内存效率

- 对大图像使用分块渲染
- 启用混合精度训练
- 根据 GPU 内存调整 render_chunk_size

```python
trainer_config.use_mixed_precision = True
trainer_config.render_chunk_size = 1024  # 根据 GPU 内存调整
```

### 速度优化

- 为场景使用适当的八叉树层数
- 启用体素剪枝以移除不必要的体素
- 如果内存允许，使用更大的批次大小

## 评估指标

SVRaster 支持标准的 NeRF 评估指标：

- **PSNR**: 峰值信噪比
- **SSIM**: 结构相似性指数
- **LPIPS**: 学习感知图像块相似性

## 故障排除

### 常见问题

1. **内存不足**: 减少 render_chunk_size 或图像分辨率
2. **训练缓慢**: 启用混合精度并检查 GPU 利用率
3. **质量差**: 增加 max_octree_levels 或调整细分阈值
4. **伪影**: 确保启用莫顿排序

### 性能提示

- 从较低分辨率开始进行快速原型设计
- 使用自适应细分以更好地保留细节
- 启用剪枝以在训练期间保持效率
- 监控体素数量以防止过度内存使用

## 引用

如果您在研究中使用 SVRaster，请引用：

```bibtex
@article{sun2024svraster,
  title={Sparse Voxels Rasterization: Real-time High-fidelity Radiance Field Rendering},
  author={Sun, Cheng and Choe, Jaesung and Loop, Charles and Ma, Wei-Chiu and Wang, Yu-Chiang Frank},
  journal={arXiv preprint arXiv:2412.04459},
  year={2024}
}
```

## 许可证

此实现仅供研究使用。有关许可详情，请参考原始论文和官方实现。

## 贡献

欢迎贡献！请随时提交问题和拉取请求。

## 致谢

此实现基于 Sun 等人的论文 "Sparse Voxels Rasterization: Real-time High-fidelity Radiance Field Rendering"。我们感谢作者的出色工作以及研究社区对神经辐射场的推进。 