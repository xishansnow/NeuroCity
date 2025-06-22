# NeuralVDB: Efficient Sparse Volumetric Neural Representations

NeuralVDB是一个高效的稀疏体素神经表示库，基于八叉树数据结构和神经网络编码，专为大规模城市场景和复杂几何体的建模而设计。

## 特性

### 核心功能
- **稀疏体素表示**: 基于八叉树的高效存储结构
- **神经网络编码**: 使用MLP网络进行特征提取和占用预测
- **分层数据结构**: 支持自适应分辨率和多尺度处理
- **内存优化**: 相比传统体素化方法显著减少内存使用

### 高级功能
- **自适应分辨率**: 根据场景复杂度动态调整细分级别
- **多尺度特征**: 支持不同尺度的特征提取和融合
- **渐进式训练**: 逐步增加训练复杂度以获得更好的收敛
- **特征压缩**: 使用量化和聚类技术减少存储需求

### 应用场景
- **城市场景建模**: 大规模城市环境的稀疏表示
- **SDF/占用场建模**: 支持有符号距离场和占用场预测
- **3D重建**: 从点云数据重建三维场景
- **实时渲染**: 高效的体积渲染和可视化

## 安装

### 依赖要求
```bash
torch>=1.9.0
numpy>=1.19.0
scipy>=1.6.0
scikit-learn>=0.24.0
matplotlib>=3.3.0
tqdm>=4.60.0
```

### 安装方法
```bash
# 从源码安装
git clone <repository-url>
cd NeuroCity
pip install -e .

# 或者直接安装依赖
pip install -r requirements.txt
```

## 快速开始

### 基础使用

```python
import numpy as np
from neuralvdb import NeuralVDB, NeuralVDBConfig, create_sample_data

# 创建配置
config = NeuralVDBConfig(
    feature_dim=32,
    max_depth=8,
    learning_rate=1e-3
)

# 创建模型
model = NeuralVDB(config)

# 生成示例数据
points, occupancies = create_sample_data(n_points=10000, scene_type='mixed')

# 训练模型
model.fit(points, occupancies, num_epochs=100)

# 预测
test_points = np.random.rand(1000, 3) * 100
predictions = model.predict(test_points)
```

### 高级使用

```python
from neuralvdb import AdvancedNeuralVDB, AdvancedNeuralVDBConfig

# 创建高级配置
config = AdvancedNeuralVDBConfig(
    feature_dim=64,
    max_depth=10,
    adaptive_resolution=True,
    multi_scale_features=True,
    progressive_training=True,
    feature_compression=True
)

# 创建高级模型
model = AdvancedNeuralVDB(config)

# 训练（支持更多高级功能）
model.fit(points, occupancies, num_epochs=200)
```

### 瓦片数据处理

```python
from neuralvdb import TileCityGenerator, TileDataset

# 生成城市瓦片数据
generator = TileCityGenerator(
    city_size=(10000, 10000, 100),
    tile_size=(1000, 1000),
    output_dir="city_tiles"
)
generator.generate_and_save_all_tiles(density='medium')

# 加载瓦片数据进行训练
dataset = TileDataset("city_tiles")
# ... 训练代码
```

## 命令行工具

### 训练脚本

```bash
# 基础训练
python -m neuralvdb.train_neuralvdb \
    --model-type basic \
    --data-type synthetic \
    --scene-type urban \
    --epochs 100 \
    --output-dir ./outputs

# 高级训练
python -m neuralvdb.train_neuralvdb \
    --model-type advanced \
    --data-type tiles \
    --data-path ./city_tiles \
    --epochs 200 \
    --feature-dim 64 \
    --save-visualizations
```

### 数据生成

```python
from neuralvdb import generate_sample_dataset

# 生成示例数据集
generate_sample_dataset(
    output_dir="./dataset",
    num_samples=1000,
    grid_size=(64, 64, 64),
    scene_types=['mixed', 'architectural', 'organic']
)
```

## API 参考

### 核心类

#### NeuralVDB
基础NeuralVDB模型类。

**方法:**
- `fit(points, occupancies, ...)`: 训练模型
- `predict(points)`: 预测占用概率
- `save(path)`: 保存模型
- `load(path)`: 加载模型
- `visualize_octree(...)`: 可视化八叉树结构

#### AdvancedNeuralVDB
高级NeuralVDB模型类，包含更多功能。

**额外方法:**
- `get_memory_usage()`: 获取内存使用统计

### 配置类

#### NeuralVDBConfig
```python
@dataclass
class NeuralVDBConfig:
    voxel_size: float = 1.0
    max_depth: int = 8
    min_depth: int = 3
    feature_dim: int = 32
    hidden_dims: List[int] = None
    activation: str = 'relu'
    dropout: float = 0.1
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 1024
    sparsity_threshold: float = 0.01
    occupancy_threshold: float = 0.5
```

#### AdvancedNeuralVDBConfig

继承自NeuralVDBConfig，增加高级参数：

```python
adaptive_resolution: bool = True
multi_scale_features: bool = True
progressive_training: bool = True
feature_compression: bool = True
quantization_bits: int = 8
occupancy_weight: float = 1.0
smoothness_weight: float = 0.1
sparsity_weight: float = 0.01
consistency_weight: float = 0.1
```

### 数据处理

#### 数据集类
- `NeuralVDBDataset`: 基础点云数据集
- `VoxelDataset`: 体素数据集，支持SDF和占用场
- `TileDataset`: 瓦片数据集，用于大规模场景
- `MultiScaleDataset`: 多尺度数据集

#### 生成器类
- `TileCityGenerator`: 城市瓦片生成器
- `SimpleVDBGenerator`: 简单几何体生成器

### 可视化

#### VDBViewer

VDB数据查看器，支持：
- 2D切片显示
- 3D等值面渲染
- 统计信息仪表板
- 交互式查看

```python
from neuralvdb import VDBViewer

viewer = VDBViewer("data.npy")
viewer.plot_2d_slice(axis=2)
viewer.plot_3d_isosurface(threshold=0.5)
viewer.plot_statistics_dashboard()
```

#### 可视化函数
- `visualize_training_data()`: 训练数据可视化
- `visualize_predictions()`: 预测结果可视化
- `compare_predictions()`: 模型对比可视化

## 示例

### 示例1: 城市场景建模

```python
import numpy as np
from neuralvdb import *

# 创建城市瓦片数据
generator = TileCityGenerator(
    city_size=(5000, 5000, 100),
    tile_size=(500, 500),
    voxel_size=1.0
)
generator.generate_and_save_all_tiles(density='high')

# 训练模型
config = AdvancedNeuralVDBConfig(
    feature_dim=64,
    max_depth=10,
    adaptive_resolution=True
)

model = AdvancedNeuralVDB(config)

# 加载瓦片数据
dataset = TileDataset("tiles")
train_loader, val_loader = create_data_loaders(dataset)

# 训练
model.fit(train_loader, val_loader, num_epochs=200)

# 可视化结果
model.visualize_octree(max_depth=6, save_path="octree.png")
```

### 示例2: SDF建模

```python
from neuralvdb import NeuralSDFTrainer, MLP, VoxelDataset

# 创建SDF网络
model = MLP(
    input_dim=3,
    hidden_dims=[256, 256, 256],
    output_dim=1,
    activation='relu'
)

# 创建训练器
trainer = NeuralSDFTrainer(model)

# 加载SDF数据
dataset = VoxelDataset(coords, sdf_values=sdf_values, task_type='sdf')
train_loader, val_loader = create_data_loaders(dataset)

# 训练
trainer.train(train_loader, val_loader, num_epochs=100)
```

### 示例3: 数据可视化

```python
from neuralvdb import VDBViewer, visualize_predictions

# 查看VDB数据
viewer = VDBViewer("scene.npy")
viewer.print_statistics()
viewer.plot_statistics_dashboard(save_path="dashboard.png")

# 可视化预测结果
points = np.random.rand(5000, 3) * 100
predictions = model.predict(points)
visualize_predictions(points, predictions, save_path="predictions.png")
```

## 性能优化

### 内存优化
- 使用稀疏表示减少内存占用
- 特征压缩降低存储需求
- 动态加载大规模数据集

### 训练优化
- 渐进式训练策略
- 自适应学习率调度
- 梯度裁剪防止爆炸

### 推理优化
- 八叉树加速查询
- 批量预测
- GPU加速计算

## 故障排除

### 常见问题

**Q: 训练时内存不足怎么办？**
A: 
- 减小batch_size
- 使用feature_compression=True
- 降低max_depth
- 使用TileDataset的load_in_memory=False

**Q: 训练收敛慢怎么办？**
A:
- 使用progressive_training=True
- 调整学习率
- 增加网络容量
- 检查数据质量

**Q: 预测结果不准确怎么办？**
A:
- 增加训练数据
- 调整sparsity_threshold
- 使用更深的网络
- 检查数据标注

### 调试技巧

```python
# 检查模型信息
info = model.get_model_info()
print(info)

# 检查数据统计
stats = dataset.get_statistics()
print(stats)

# 可视化训练数据
visualize_training_data(points, occupancies)

# 监控训练过程
training_stats = model.fit(...)
print(f"Train loss: {training_stats['train_losses']}")
```

## 贡献

欢迎贡献代码和报告问题！

### 开发环境设置
```bash
git clone <repository-url>
cd NeuroCity
pip install -e ".[dev]"
```

### 运行测试
```bash
python -m pytest tests/
```

## 许可证

[许可证信息]

## 引用

如果您在研究中使用了NeuralVDB，请引用：

```bibtex
@article{neuralvdb2024,
  title={NeuralVDB: Efficient Sparse Volumetric Neural Representations},
  author={[作者信息]},
  journal={[期刊信息]},
  year={2024}
}
```

## 相关链接

- [论文链接]
- [演示视频]
- [在线文档]
- [问题反馈] 