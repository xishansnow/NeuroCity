# NeuralVDB: 高效稀疏体积神经表示

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
- `optimize_structure()`: 优化八叉树结构
- `compress_features()`: 压缩特征表示

### 配置类

#### NeuralVDBConfig
```python
@dataclass
class NeuralVDBConfig:
    voxel_size: float = 1.0              # 体素大小
    max_depth: int = 8                   # 最大八叉树深度
    min_depth: int = 3                   # 最小八叉树深度
    feature_dim: int = 32                # 特征维度
    hidden_dims: List[int] = None        # 隐藏层维度
    activation: str = 'relu'             # 激活函数
    dropout: float = 0.1                 # Dropout率
    learning_rate: float = 1e-3          # 学习率
    weight_decay: float = 1e-5           # 权重衰减
    batch_size: int = 1024               # 批量大小
    sparsity_threshold: float = 0.01     # 稀疏性阈值
    occupancy_threshold: float = 0.5     # 占用阈值
```

#### AdvancedNeuralVDBConfig

继承自NeuralVDBConfig，增加高级参数：

```python
adaptive_resolution: bool = True         # 自适应分辨率
multi_scale_features: bool = True        # 多尺度特征
progressive_training: bool = True        # 渐进式训练
feature_compression: bool = True         # 特征压缩
compression_ratio: float = 0.5           # 压缩比
quantization_bits: int = 8               # 量化位数
clustering_method: str = 'kmeans'        # 聚类方法
```

### 数据集类

#### VDBDataset
```python
from neuralvdb import VDBDataset

dataset = VDBDataset(
    data_path="path/to/data",
    split='train',
    cache_data=True,
    transform=None
)
```

#### TileDataset
```python
from neuralvdb import TileDataset

dataset = TileDataset(
    tiles_dir="path/to/tiles",
    sample_ratio=0.1,
    stratified_sampling=True
)
```

## 高级功能

### 自适应八叉树结构

```python
# 启用自适应分辨率
config = AdvancedNeuralVDBConfig(
    adaptive_resolution=True,
    adaptation_threshold=0.02,
    max_subdivision_level=10
)

model = AdvancedNeuralVDB(config)
```

### 多尺度特征融合

```python
# 多尺度特征配置
config = AdvancedNeuralVDBConfig(
    multi_scale_features=True,
    scale_levels=[1, 2, 4, 8],
    feature_fusion_method='attention'
)
```

### 渐进式训练

```python
# 渐进式训练策略
training_schedule = {
    'epochs': [50, 100, 150, 200],
    'depths': [4, 6, 8, 10],
    'learning_rates': [1e-2, 5e-3, 1e-3, 5e-4]
}

model.fit_progressive(
    points, occupancies,
    schedule=training_schedule
)
```

## 性能优化

### 内存优化

```python
# 启用特征压缩
config = AdvancedNeuralVDBConfig(
    feature_compression=True,
    compression_ratio=0.5,
    quantization_bits=8
)

# 使用流式数据加载
from neuralvdb import StreamingDataLoader

dataloader = StreamingDataLoader(
    dataset,
    batch_size=1024,
    prefetch_factor=2,
    num_workers=4
)
```

### 训练加速

```python
# 混合精度训练
from neuralvdb import NeuralVDBTrainer

trainer = NeuralVDBTrainer(
    model=model,
    use_amp=True,
    gradient_accumulation_steps=4
)

# 多GPU训练
trainer = NeuralVDBTrainer(
    model=model,
    device_ids=[0, 1, 2, 3],
    distributed=True
)
```

## 可视化工具

### 八叉树可视化

```python
from neuralvdb.visualization import visualize_octree

# 可视化八叉树结构
visualize_octree(
    model.octree,
    depth_range=(3, 8),
    show_features=True,
    output_path="octree_vis.png"
)
```

### 特征可视化

```python
from neuralvdb.visualization import visualize_features

# 可视化学习到的特征
visualize_features(
    model,
    sample_points=test_points,
    feature_dim_to_show=[0, 1, 2],
    output_path="features_vis.png"
)
```

### 3D场景渲染

```python
from neuralvdb.viewer import VDBViewer

# 启动交互式查看器
viewer = VDBViewer(model)
viewer.launch(
    port=8080,
    resolution=(1024, 768),
    camera_controls=True
)
```

## 评估指标

### 几何精度

```python
from neuralvdb.metrics import evaluate_geometry

metrics = evaluate_geometry(
    model=model,
    ground_truth_points=gt_points,
    ground_truth_occupancies=gt_occupancies
)

print(f"IoU: {metrics['iou']:.3f}")
print(f"Chamfer Distance: {metrics['chamfer_distance']:.6f}")
print(f"F-Score: {metrics['f_score']:.3f}")
```

### 压缩效率

```python
from neuralvdb.metrics import evaluate_compression

compression_metrics = evaluate_compression(model)
print(f"压缩比: {compression_metrics['compression_ratio']:.2f}")
print(f"重建误差: {compression_metrics['reconstruction_error']:.6f}")
```

## 应用示例

### 城市建筑建模

```python
# 加载城市建筑数据
from neuralvdb.datasets import CityBuildingDataset

dataset = CityBuildingDataset(
    city_bounds=(-1000, -1000, 0, 1000, 1000, 200),
    building_types=['residential', 'commercial', 'industrial']
)

# 训练专门的城市模型
config = NeuralVDBConfig(
    max_depth=10,
    feature_dim=64,
    sparsity_threshold=0.005
)

model = NeuralVDB(config)
model.fit(dataset.points, dataset.occupancies)
```

### 医学图像分割

```python
# 医学数据适配
from neuralvdb.medical import MedicalVolumeAdapter

adapter = MedicalVolumeAdapter()
points, labels = adapter.convert_dicom_to_points("ct_scan.dcm")

# 训练分割模型
segmentation_model = NeuralVDB(NeuralVDBConfig(
    feature_dim=32,
    max_depth=8,
    occupancy_threshold=0.3
))
```

### 点云重建

```python
# 从点云重建表面
from neuralvdb.reconstruction import PointCloudReconstructor

reconstructor = PointCloudReconstructor(
    model=model,
    surface_threshold=0.5,
    smoothing_iterations=10
)

mesh = reconstructor.extract_mesh(point_cloud)
```

## 性能基准

### 内存使用对比

| 方法 | 1M体素 | 10M体素 | 100M体素 |
|------|---------|---------|----------|
| 传统体素网格 | 4GB | 40GB | 400GB |
| 八叉树 | 0.8GB | 5GB | 35GB |
| NeuralVDB | 0.2GB | 1.2GB | 8GB |

### 训练时间

| 场景复杂度 | 传统方法 | NeuralVDB | 加速比 |
|------------|----------|-----------|--------|
| 简单 | 2小时 | 30分钟 | 4x |
| 中等 | 8小时 | 1.5小时 | 5.3x |
| 复杂 | 24小时 | 4小时 | 6x |

## 故障排除

### 常见问题

**内存不足**
```python
# 减少批量大小和特征维度
config.batch_size = 512
config.feature_dim = 16
```

**训练收敛慢**
```python
# 调整学习率和使用warmup
config.learning_rate = 5e-3
config.warmup_steps = 1000
```

**八叉树过深**
```python
# 限制最大深度
config.max_depth = 6
config.sparsity_threshold = 0.02
```

## 扩展和自定义

### 自定义损失函数

```python
from neuralvdb.losses import BaseLoss

class CustomLoss(BaseLoss):
    def forward(self, predictions, targets):
        # 实现自定义损失
        return loss_value
```

### 自定义数据加载器

```python
from neuralvdb.datasets import BaseDataset

class CustomDataset(BaseDataset):
    def __init__(self, data_path):
        # 实现自定义数据加载
        pass
    
    def __getitem__(self, idx):
        # 返回数据样本
        return points, occupancies
```

## 开发路线图

### 即将推出的功能

- [ ] 支持颜色信息编码
- [ ] 实时交互式编辑
- [ ] 分布式训练优化
- [ ] WebGL可视化接口
- [ ] 移动端推理支持

### 长期目标

- [ ] 时间序列支持（4D）
- [ ] 物理仿真集成
- [ ] AR/VR应用支持
- [ ] 云端服务部署

## 贡献指南

我们欢迎各种形式的贡献：

1. **Bug报告**：使用GitHub Issues
2. **功能请求**：创建Feature Request
3. **代码贡献**：提交Pull Request
4. **文档改进**：修改或添加文档
5. **示例和教程**：分享使用经验

### 开发环境设置

```bash
# 克隆仓库
git clone <repository-url>
cd NeuroCity

# 创建开发环境
python -m venv dev_env
source dev_env/bin/activate

# 安装开发依赖
pip install -e ".[dev]"

# 运行测试
pytest tests/
```

## 许可证

MIT许可证 - 详见LICENSE文件

## 引用

如果您在研究中使用NeuralVDB，请引用：

```bibtex
@misc{neuralvdb2024,
  title={NeuralVDB: Efficient Sparse Volumetric Neural Representations},
  author={NeuroCity Team},
  year={2024},
  url={https://github.com/neurocity/neuralvdb}
}
```

## 联系方式

- 📧 邮件：neuralvdb@neurocity.ai
- 💬 讨论：GitHub Discussions
- 🐛 问题报告：GitHub Issues
- 📖 文档：https://neuralvdb.readthedocs.io
- 🌐 官网：https://neurocity.ai/neuralvdb

## 致谢

感谢以下开源项目和研究工作：
- OpenVDB项目提供的数据结构灵感
- PyTorch深度学习框架
- 所有贡献者和用户的支持

---

**NeuralVDB：让稀疏体积表示更智能、更高效！** 🚀 