# Occupancy Networks (OccupancyNet)

基于论文 [Occupancy Networks: Learning 3D Reconstruction in Function Space](https://arxiv.org/abs/1812.03828) 的PyTorch实现。

## 概述

占用网络学习一个连续函数 `f: R³ → [0,1]`，将3D空间中的点映射为占用概率。这种表示方法可以处理任意拓扑的3D形状，并支持高分辨率的几何重建。

## 主要特性

- **连续3D表示**: 学习空间中任意点的占用概率
- **任意拓扑**: 支持复杂拓扑结构的3D形状
- **高分辨率重建**: 通过查询任意分辨率的点来重建几何
- **条件生成**: 支持基于特征编码的条件生成
- **网格提取**: 使用Marching Cubes算法提取三角网格

## 架构组件

### 核心模型 (`core.py`)

#### `OccupancyNetwork`
- 基础占用网络模型
- 使用残差连接的全连接网络
- 支持批标准化和Dropout
- 可配置的网络深度和宽度

#### `ConditionalOccupancyNetwork`
- 条件占用网络
- 支持形状编码条件生成
- 可用于形状插值和变形

### 数据处理 (`dataset.py`)

#### `OccupancyDataset`
- 从3D网格生成训练数据
- 支持表面采样和均匀空间采样
- 自动计算占用标签
- 支持数据增强

#### `SyntheticOccupancyDataset`
- 合成几何形状数据集
- 支持球体、立方体、圆柱体等基本形状
- 用于快速原型验证

### 训练器 (`trainer.py`)

#### `OccupancyTrainer`
- 完整的训练管道
- 支持验证和模型检查点
- TensorBoard日志记录
- 自动网格提取评估

### 工具函数 (`utils/`)

- **网格处理**: 网格标准化、采样、网格提取
- **可视化**: 占用场可视化、训练曲线绘制
- **评估指标**: IoU、Chamfer距离等

## 快速开始

### 1. 安装依赖

```bash
pip install torch torchvision
pip install trimesh scikit-image
pip install tensorboard
```

### 2. 基本使用

```python
from src.models.occupancy_net import OccupancyNetwork, OccupancyDataset, OccupancyTrainer

# 创建模型
model = OccupancyNetwork(
    dim_input=3,
    dim_hidden=128,
    num_layers=5
)

# 创建数据集
dataset = OccupancyDataset(
    data_root='path/to/data',
    split='train',
    num_points=100000
)

# 创建训练器
trainer = OccupancyTrainer(
    model=model,
    train_dataloader=dataloader,
    device='cuda'
)

# 开始训练
trainer.train(num_epochs=100)
```

### 3. 网格重建

```python
# 训练后进行网格重建
mesh_data = model.extract_mesh(
    resolution=128,
    threshold=0.5
)

# 保存网格
import trimesh
mesh = trimesh.Trimesh(
    vertices=mesh_data['vertices'],
    faces=mesh_data['faces']
)
mesh.export('reconstructed_mesh.obj')
```

## 配置示例

```python
config = {
    'model': {
        'dim_input': 3,
        'dim_hidden': 256,
        'num_layers': 8,
        'use_batch_norm': True,
        'dropout_prob': 0.1
    },
    'dataset': {
        'data_root': 'data/shapenet',
        'num_points': 100000,
        'surface_sampling': 0.5,
        'uniform_sampling': 0.5
    },
    'optimizer': {
        'type': 'adam',
        'lr': 1e-4,
        'weight_decay': 1e-5
    },
    'trainer': {
        'device': 'cuda',
        'log_dir': 'logs/occupancy_net',
        'checkpoint_dir': 'checkpoints/occupancy_net'
    }
}

trainer = create_trainer_from_config(config)
```

## 数据格式

### 输入数据结构
```
data_root/
├── meshes/
│   ├── shape1.obj
│   ├── shape2.obj
│   └── ...
├── train.json
├── val.json
└── test.json
```

### 训练样本格式
```python
sample = {
    'points': torch.Tensor,      # [N, 3] 查询点坐标
    'occupancy': torch.Tensor,   # [N, 1] 占用标签 (0或1)
    'shape_id': str             # 形状标识符
}
```

## 评估指标

- **IoU (Intersection over Union)**: 重建形状与真实形状的交并比
- **Chamfer Distance**: 点云之间的双向最近邻距离
- **网格质量**: 顶点数、面数、流形性等

## 高级功能

### 条件生成
```python
conditional_model = ConditionalOccupancyNetwork(
    dim_condition=128
)

# 编码形状特征
shape_code = conditional_model.encode_shape(points, occupancy)

# 条件生成
pred_occupancy = conditional_model(query_points, condition=shape_code)
```

### 多尺度训练
```python
# 在训练过程中使用不同分辨率的采样点
dataset = OccupancyDataset(
    num_points=[50000, 100000, 200000],  # 多尺度采样
    multiscale_training=True
)
```

### 自定义损失函数
```python
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, pred, target):
        ce_loss = F.binary_cross_entropy(pred, target, reduction='none')
        pt = torch.where(target == 1, pred, 1 - pred)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

# 在训练器中使用自定义损失
model.compute_loss = FocalLoss()
```

## 性能优化

- **分块推理**: 支持大批量点的内存高效推理
- **梯度累积**: 处理大批量训练数据
- **混合精度**: 使用AMP加速训练
- **数据并行**: 多GPU训练支持

## 引用

如果使用本实现，请引用原始论文：

```bibtex
@inproceedings{mescheder2019occupancy,
  title={Occupancy networks: Learning 3d reconstruction in function space},
  author={Mescheder, Lars and Oechsle, Michael and Niemeyer, Michael and Nowozin, Sebastian and Geiger, Andreas},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={4460--4470},
  year={2019}
}
```

## 许可证

本实现遵循原项目的许可证条款。 