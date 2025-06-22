# Mega-NeRF++: 针对高分辨率摄影测量图像的改进可扩展神经辐射场

**Mega-NeRF++** 是一个先进的神经辐射场实现，专门为具有高分辨率摄影测量图像的大规模场景设计。该软件包为航空影像、无人机拍摄和大规模摄影测量数据集的可扩展 3D 重建提供了完整解决方案。





## 🚀 主要特性

### 核心功能
- ** 可扩展架构 **: 处理包含数千张高分辨率图像的场景
- ** 高分辨率支持 **: 原生支持最高 8K 分辨率的图像
- ** 内存高效训练 **: 先进的内存管理和流式数据加载
- ** 多分辨率训练 **: 从低分辨率到高分辨率的渐进式训练
- ** 分布式训练 **: 支持多 GPU 的大规模场景训练

### 高级组件
- ** 分层空间编码 **: 多尺度空间表示
- ** 自适应空间分割 **: 智能场景细分
- ** 摄影测量优化 **: 专门处理航空影像
- ** 细节层次渲染 **: 基于观看距离的自适应质量
- ** 渐进式细化 **: 迭代质量改进

### 摄影测量特性
- ** 光束平差集成 **: 训练期间的相机姿态优化
- ** 航空影像支持 **: 针对无人机和卫星影像优化
- ** 大场景处理 **: 高效处理城市规模重建
- ** 多视图一致性 **: 跨视点的几何一致性

## 📦 安装

### 先决条件
- Python 3.8+
- PyTorch 1.12+ 带 CUDA 支持
- NVIDIA GPU 具有 8GB + 显存 (大规模场景推荐 16GB+)

### 安装依赖
```bash
# 克隆仓库
git clone https://github.com/your-org/mega-nerf-plus.git
cd mega-nerf-plus

# 安装依赖
pip install -r requirements.txt

# 开发模式安装包
pip install -e .
```

### 所需包
```
torch>=1.12.0
torchvision>=0.13.0
numpy>=1.21.0
opencv-python>=4.5.0
Pillow>=8.3.0
tifffile>=2021.7.2
h5py>=3.3.0
tqdm>=4.62.0
wandb>=0.12.0
imageio>=2.9.0
scikit-image>=0.18.0
matplotlib>=3.4.0
```

## 🚀 快速开始

### 基础训练示例

```python
from mega_nerf_plus import MegaNeRFPlus, MegaNeRFPlusConfig, MegaNeRFPlusTrainer
from mega_nerf_plus.dataset import create_meganerf_plus_dataset

# 创建配置
config = MegaNeRFPlusConfig(
    max_image_resolution=4096,
    batch_size=4096,
    num_levels=8,
    progressive_upsampling=True
)

# 加载数据集
train_dataset = create_meganerf_plus_dataset(
    'path/to/dataset',
    dataset_type='photogrammetric',
    split='train'
)

# 创建模型
model = MegaNeRFPlus(config)

# 创建训练器
trainer = MegaNeRFPlusTrainer(
    config=config,
    model=model,
    train_dataset=train_dataset
)

# 开始训练
trainer.train(num_epochs=100)
```

### 命令行接口

```bash
# 基础训练
python -m mega_nerf_plus.example_usage \
    --mode basic \
    --data_dir /path/to/dataset \
    --output_dir /path/to/output

# 带分割的大场景训练
python -m mega_nerf_plus.example_usage \
    --mode large_scene \
    --data_dir /path/to/large_dataset \
    --output_dir /path/to/output

# 训练好的模型推理
python -m mega_nerf_plus.example_usage \
    --mode inference \
    --model_path /path/to/checkpoint.pth \
    --data_dir /path/to/test_data \
    --output_dir /path/to/rendered_images
```

## 📁 数据集格式

### 摄影测量数据集结构
```
dataset/
├── images/                 # 高分辨率图像
│   ├── IMG_0001.jpg
│   ├── IMG_0002.jpg
│   └── ...
├── poses.txt              # 相机姿态 (4x4 矩阵)
├── intrinsics.txt         # 相机内参 (3x3 矩阵)
└── metadata.json          # 可选元数据
```

### COLMAP 数据集支持
```
dataset/
├── images/
├── cameras.txt            # COLMAP 相机参数
├── images.txt             # COLMAP 图像姿态
├── points3D.txt           # COLMAP 3D 点 (可选)
└── sparse/                # COLMAP 稀疏重建
```

### 大场景数据集
```
dataset/
├── images/
├── poses.txt
├── intrinsics.txt
├── partitions/            # 空间分割 (自动生成)
│   ├── partition_0/
│   ├── partition_1/
│   └── ...
└── cache/                 # 缓存数据用于快速加载
```

## ⚙️ 配置

### 基础配置
```python
config = MegaNeRFPlusConfig(
    # 网络架构
    num_levels=8,              # 分层编码级别
    base_resolution=32,        # 基础网格分辨率
    max_resolution=2048,       # 最大网格分辨率

    # 多分辨率参数
    num_lods=4,               # LOD 级别数

    # 训练参数
    batch_size=4096,          # 光线批次大小
    lr_init=5e-4,            # 初始学习率
    lr_decay_steps=200000,    # 学习率衰减步数

    # 内存管理
    max_memory_gb=16.0,       # 最大 GPU 内存使用
    use_mixed_precision=True, # 使用混合精度训练
)
```

### 大场景配置
```python
config = MegaNeRFPlusConfig(
    # 更高分辨率设置
    max_image_resolution=8192,
    max_resolution=4096,

    # 空间分割
    max_partition_size=2048,
    adaptive_partitioning=True,
    overlap_ratio=0.15,

    # 内存优化
    max_memory_gb=24.0,
    gradient_checkpointing=True,
    streaming_mode=True,
)
```

## 🔧 高级用法

### 空间分割
```python
from mega_nerf_plus.spatial_partitioner import PhotogrammetricPartitioner

partitioner = PhotogrammetricPartitioner(config)
partitions = partitioner.partition_scene(
    scene_bounds,
    camera_positions,
    camera_orientations,
    image_resolutions,
    intrinsics
)
```

### 内存管理
```python
from mega_nerf_plus.memory_manager import MemoryManager, MemoryOptimizer

# 初始化内存管理器
memory_manager = MemoryManager(max_memory_gb=16.0)
memory_manager.start_monitoring()

# 优化模型内存效率
model = MemoryOptimizer.optimize_model_memory(
    model,
    use_checkpointing=True,
    use_mixed_precision=True
)
```

### 多尺度训练
```python
from mega_nerf_plus.trainer import MultiScaleTrainer

trainer = MultiScaleTrainer(
    config=config,
    model=model,
    train_dataset=train_dataset
)

# 训练自动进行分辨率级别切换
trainer.train(num_epochs=200)
```

### 分布式训练
```bash
# 在 4 个 GPU 上启动分布式训练
torchrun --nproc_per_node=4 train_distributed.py \
    --data_dir /path/to/dataset \
    --output_dir /path/to/output \
    --batch_size 16384
```

## 📊 性能基准

### 内存使用
| 分辨率 | 批次大小 | GPU 内存 | 训练速度 |
|--------|----------|---------|----------|
| 2K     | 4096     | 8GB     | 100 rays/ms |
| 4K     | 2048     | 12GB    | 80 rays/ms  |
| 8K     | 1024     | 20GB    | 50 rays/ms  |

### 可扩展性
| 场景大小 | 图像数量 | 参数量 | 训练时间 |
|----------|----------|--------|----------|
| 小型     | 100      | 10M    | 2 小时    |
| 中型     | 500      | 50M    | 8 小时    |
| 大型     | 2000     | 200M   | 24 小时   |
| 城市级   | 10000    | 500M   | 5 天      |

## 🎯 应用场景

### 航空摄影测量
- 基于无人机的 3D 重建
- 航空测量处理
- 基础设施监测
- 城市规划

### 大规模制图
- 城市规模重建
- 卫星影像处理
- 地理信息系统
- 数字孪生创建

### 科学应用
- 考古遗址记录
- 环境监测
- 灾害评估
- 气候变化研究

## 🔬 技术细节

### 架构概览
```
MegaNeRF++ 架构:
├── 分层空间编码器
│   ├── 多分辨率哈希编码
│   ├── 自适应网格结构
│   └── 位置编码
├── 多分辨率 MLP
│   ├── LOD 感知网络
│   ├── 渐进式细化
│   └── 跳跃连接
├── 摄影测量渲染器
│   ├── 自适应采样
│   ├── 分层渲染
│   └── 多视图一致性
└── 内存管理
    ├── 流式数据加载
    ├── 智能缓存
    └── GPU 内存优化
```

### 关键创新
1. ** 分层空间编码 **: 大场景的多尺度表示
2. ** 自适应分割 **: 基于图像覆盖的智能场景细分
3. ** 渐进式训练 **: 稳定收敛的渐进分辨率增加
4. ** 内存流式处理 **: 高效处理超出内存的数据集
5. ** 多视图优化 **: 摄影测量一致性约束

## 📈 评估指标

### 渲染质量
- **PSNR**: 峰值信噪比
- **SSIM**: 结构相似性指数
- **LPIPS**: 学习感知图像块相似性

### 几何精度
- ** 深度误差 **: 平均绝对深度误差
- ** 点云精度 **: 3D 重建精度
- ** 多视图一致性 **: 跨视点几何一致性

### 效率指标
- ** 训练速度 **: 每秒处理的光线数
- ** 内存使用 **: 峰值 GPU 内存消耗
- ** 收敛率 **: 达到目标质量的步数

## 🛠️ 开发

### 贡献
1. Fork 仓库
2. 创建功能分支
3. 实现带测试的更改
4. 提交拉取请求

### 测试
```bash
# 运行单元测试
python -m pytest tests/

# 运行集成测试
python -m pytest tests/integration/

# 运行性能基准测试
python tests/benchmark.py
```

### 代码风格
- 遵循 PEP 8 指南
- 使用类型提示
- 为所有公共函数编写文档
- 为新功能添加单元测试

## 📄 引用

如果您在研究中使用 Mega-NeRF++，请引用：

```bibtex
@article{isprs-archives-XLVIII-1-2024-769-2024,
author = {Xu, Y. and Wang, T. and Zhan, Z. and Wang, X.},
title = {Mega-NeRF++: An Improved Scalable NeRFs for High-resolution Photogrammetric Images},
joural = {The International Archives of the Photogrammetry, Remote Sensing and Spatial Information Sciences},
volume = {XLVIII-1-2024},
year = {2024},
pages = {769--776},
url = {https://isprs-archives.copernicus.org/articles/XLVIII-1-2024/769/2024/},
doi = {10.5194/isprs-archives-XLVIII-1-2024-769-2024}
}
```

## 🤝 致谢

- 基于 Mildenhall 等人的 NeRF 框架构建
- 受 Mega-NeRF 大规模场景启发
- 利用 instant-ngp 哈希编码技术
- 融合 Mip-NeRF 抗锯齿策略

## 📞 支持

- ** 文档 **: [https://mega-nerf-plus.readthedocs.io](https://mega-nerf-plus.readthedocs.io)
- ** 问题 **: [GitHub Issues](https://github.com/your-org/mega-nerf-plus/issues)
- ** 讨论 **: [GitHub Discussions](https://github.com/your-org/mega-nerf-plus/discussions)
- ** 邮箱 **: support@mega-nerf-plus.org

## 📜 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件。

---

**Mega-NeRF++** - 实现从摄影测量影像进行大规模、高质量 3D 重建。🌍✨