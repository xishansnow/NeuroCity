# InfNeRF: 具有 O(log n) 空间复杂度的无限尺度 NeRF 渲染

本模块实现了论文 "InfNeRF: Towards Infinite Scale NeRF Rendering with O(log n) Space Complexity" by Jiabin Liang et al. (SIGGRAPH Asia 2024) 中描述的 InfNeRF。

## 概述

InfNeRF 扩展了神经辐射场 (NeRF) 以处理具有对数空间复杂度的无限尺度场景渲染。关键创新是使用基于八叉树的细节级别 (LoD) 结构，该结构在空间和尺度维度上对场景进行分区。

### 核心特性

- **🌲 基于八叉树的 LoD 结构**: 具有自动级别选择的分层场景表示
- **📐 O(log n) 空间复杂度**: 渲染期间的对数内存使用
- **🎯 抗锯齿渲染**: 通过分层采样内置抗锯齿
- **⚡ 可扩展训练**: 具有金字塔监督的分布式训练
- **🔧 内存高效**: 智能八叉树剪枝和内存管理
- **🎨 大规模场景**: 支持城市规模和地球规模重建

## 架构

### 核心组件

1. **OctreeNode**: 分层结构中的单个节点，每个节点都有自己的 NeRF
2. **LoDAwareNeRF**: 具有自适应复杂性的细节级别感知神经网络
3. **InfNeRFRenderer**: 具有基于八叉树采样和抗锯齿的渲染器
4. **InfNeRF**: 将八叉树结构与体积渲染相结合的主模型

### 细节级别管理

- **地面采样距离 (GSD)**: 基于八叉树级别的自动计算
- **自适应采样**: 基于像素足迹的适当 LoD 级别动态选择
- **半径扰动**: 随机抗锯齿以平滑级别过渡

## 安装

InfNeRF 是 NeuroCity 项目的一部分。确保您具有以下依赖项：

```bash
pip install torch torchvision numpy matplotlib opencv-python pillow
pip install wandb  # 可选，用于实验跟踪
```

## 快速开始

### 基本用法

```python
from src.nerfs.inf_nerf import InfNeRF, InfNeRFConfig, demo_inf_nerf

# 运行完整演示
demo_inf_nerf(
    data_path="path/to/your/dataset",
    output_path="outputs/inf_nerf_results"
)
```

### 自定义配置

```python
from src.nerfs.inf_nerf import InfNeRF, InfNeRFConfig

# 创建配置
config = InfNeRFConfig(
    max_depth=8,                    # 最大八叉树深度
    grid_size=2048,                 # 每个节点的网格分辨率
    max_gsd=1.0,                    # 最粗细节级别 (米)
    min_gsd=0.01,                   # 最细细节级别 (米)
    scene_bound=100.0,              # 场景大小
    use_pruning=True,               # 启用八叉树剪枝
    distributed_training=False      # 单 GPU 训练
)

# 创建模型
model = InfNeRF(config)

# 从稀疏点构建八叉树
sparse_points = load_sparse_points("sparse_points.ply")
model.build_octree(sparse_points)
```

### 训练

```python
from src.nerfs.inf_nerf import InfNeRFTrainer, InfNeRFTrainerConfig
from src.nerfs.inf_nerf import InfNeRFDataset, InfNeRFDatasetConfig

# 设置数据集
dataset_config = InfNeRFDatasetConfig(
    data_root="path/to/dataset",
    num_pyramid_levels=4,           # 多尺度监督
    rays_per_image=1024,
    batch_size=4096
)

train_dataset = InfNeRFDataset(dataset_config, split='train')
val_dataset = InfNeRFDataset(dataset_config, split='val')

# 设置训练器
trainer_config = InfNeRFTrainerConfig(
    num_epochs=100,
    lr_init=1e-2,
    lambda_rgb=1.0,
    lambda_regularization=1e-4,     # 级别一致性
    use_wandb=True                  # 实验跟踪
)

trainer = InfNeRFTrainer(model, train_dataset, trainer_config, val_dataset)

# 训练
trainer.train()
```

### 渲染

```python
# 内存高效渲染
from src.nerfs.inf_nerf.utils import memory_efficient_rendering

rendered = memory_efficient_rendering(
    model=model,
    rays_o=rays_o,                  # [N, 3] 光线起点
    rays_d=rays_d,                  # [N, 3] 光线方向
    near=0.1,
    far=100.0,
    focal_length=focal_length,
    pixel_width=1.0,
    max_memory_gb=8.0
)

rgb = rendered['rgb']               # [N, 3] 渲染颜色
depth = rendered['depth']           # [N] 渲染深度
```

## 数据集格式

InfNeRF 期望数据集采用以下结构：

```
dataset/
├── images/                 # 输入图像
│   ├── image_001.jpg
│   ├── image_002.jpg
│   └── ...
├── cameras.json           # 相机参数
└── sparse_points.ply      # SfM 稀疏点
```

### 相机格式

```json
{
  "image_001.jpg": {
    "intrinsic": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
    "extrinsic": [[r11, r12, r13, tx], [r21, r22, r23, ty], 
                  [r31, r32, r33, tz], [0, 0, 0, 1]]
  }
}
```

### 数据准备

从 COLMAP 或 NeRFStudio 格式转换：

```python
from src.nerfs.inf_nerf.dataset import prepare_colmap_data

prepare_colmap_data(
    colmap_dir="path/to/colmap/reconstruction",
    output_dir="path/to/inf_nerf/dataset"
)
```

## 关键算法

### 八叉树构建

InfNeRF 基于运动结构的稀疏点自适应构建八叉树：

1. **空间分区**: 基于点密度的递归细分
2. **级别分配**: 每个节点的自动 GSD 计算
3. **剪枝**: 移除数据不足的节点

### 级别选择

对于沿光线的每个采样球：

```python
# 论文中的方程 5
level = floor(log2(root_gsd / sample_radius))
```

### 抗锯齿

通过以下方式内置抗锯齿：

1. **分层采样**: 父节点提供平滑的低通滤波版本
2. **半径扰动**: 随机扰动以平滑过渡
3. **多尺度训练**: 跨分辨率级别的金字塔监督

## 性能

### 内存复杂度

- **传统 NeRF**: O(n) - 需要所有参数
- **Block-NeRF/Mega-NeRF**: O(n) 用于鸟瞰视图
- **InfNeRF**: O(log n) - 仅八叉树节点子集

### 实际结果

来自论文：
- **17% 参数使用量** 用于渲染 vs 传统方法
- **2.4 dB PSNR 改进** 超过 Mega-NeRF
- **3.46x 吞吐量改进** 在大规模场景中

## 实用工具

### 八叉树分析

```python
from src.nerfs.inf_nerf.utils import visualize_octree, analyze_octree_memory

# 可视化八叉树结构
visualize_octree(model.root_node, max_depth=6, save_path="octree.png")

# 分析内存使用
stats = analyze_octree_memory(model.root_node)
print(f"总内存: {stats['total_memory_mb']:.1f} MB")
print(f"按级别的节点数: {stats['nodes_by_level']}")
```

### 性能分析

```python
from src.nerfs.inf_nerf.utils.rendering_utils import rendering_profiler

with rendering_profiler.profile("my_render_pass"):
    result = model.render(...)

rendering_profiler.print_summary()
```

## 高级特性

### 分布式训练

```python
trainer_config = InfNeRFTrainerConfig(
    distributed=True,
    world_size=4,               # 4 个 GPU
    local_rank=0,               # 当前 GPU
    octree_growth_schedule=[1000, 5000, 10000]  # 何时增长八叉树
)
```

### 自定义 LoD 策略

```python
from src.nerfs.inf_nerf.utils.lod_utils import LoDManager

lod_manager = LoDManager(config)
level = lod_manager.determine_lod_level(sample_radius, max_level)
```

### 内存高效渲染

```python
from src.nerfs.inf_nerf.utils.rendering_utils import MemoryEfficientRenderer

renderer = MemoryEfficientRenderer(model, max_memory_gb=4.0)
result = renderer.render_memory_efficient(rays_o, rays_d, ...)
```

## 示例

查看 `example_usage.py` 中的完整示例：

- **基本演示**: 简单合成场景
- **大规模训练**: 城市规模重建
- **性能分析**: 内存和时间分析
- **自定义数据集**: 数据准备工作流

## 限制

- **训练时间**: 由于八叉树构建比传统 NeRF 更长
- **稀疏点依赖**: 需要良好的 SfM 重建
- **GPU 内存**: 训练仍需要大量内存
- **实现**: 论文中的一些优化未完全实现

## 未来工作

- **CUDA 优化**: 更快的哈希编码和八叉树遍历
- **动态八叉树**: 运行时八叉树修改
- **时间一致性**: 扩展到动态场景
- **压缩**: 进一步的内存减少技术

## 引用

```bibtex
@article{liang2024infnerf,
  title={InfNeRF: Towards Infinite Scale NeRF Rendering with O(log n) Space Complexity},
  author={Liang, Jiabin and Zhang, Lanqing and Zhao, Zhuoran and Xu, Xiangyu},
  journal={arXiv preprint arXiv:2403.14376},
  year={2024}
}
```

## 参考文献

- [InfNeRF 论文](https://arxiv.org/abs/2403.14376)
- [项目主页](https://jiabinliang.github.io/InfNeRF.io/)
- [NeRF: Representing Scenes as Neural Radiance Fields](https://arxiv.org/abs/2003.08934)
- [Instant Neural Graphics Primitives](https://arxiv.org/abs/2201.05989)
- [Mega-NeRF: Scalable Construction of Large-Scale NeRFs](https://arxiv.org/abs/2112.10703) 