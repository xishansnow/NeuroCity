# GFV (Global Feature Vector) Library

A high-performance global feature vector library based on multi-resolution hash encoding for neural graphics primitives and spatial data processing.

## 特性 (Features)

- 🌍 **全球地理坐标支持** - Global geographic coordinate support
- 🔥 **多分辨率哈希编码** - Multi-resolution hash encoding
- 📊 **分层特征表示** - Hierarchical feature representation
- ⚡ **高效查询和更新** - Efficient query and update operations
- 💾 **分布式存储支持** - Distributed storage support
- 🎯 **PyTorch Lightning 集成** - PyTorch Lightning integration
- 📈 **丰富的可视化工具** - Rich visualization tools

## 安装 (Installation)

```bash
# 从源码安装
cd NeuroCity
pip install -e .

# 或直接导入使用
from src.gfv import GlobalHashConfig, GlobalFeatureLibrary
```

## 依赖 (Dependencies)

```
torch >= 1.9.0
numpy >= 1.20.0
matplotlib >= 3.3.0
mercantile >= 1.2.0
pyproj >= 3.0.0
sqlite3 (built-in)
tqdm >= 4.62.0
scipy >= 1.7.0
seaborn >= 0.11.0
plotly >= 5.0.0 (optional, for interactive visualizations)
pytorch-lightning >= 1.5.0 (optional, for Lightning training)
h5py >= 3.0.0 (optional, for HDF5 support)
```

## 快速开始 (Quick Start)

### 基础使用

```python
from src.gfv import GlobalHashConfig, GlobalFeatureLibrary

# 1. 创建配置
config = GlobalHashConfig(
    num_levels=16,
    max_hash=2**14,
    base_resolution=16,
    finest_resolution=512,
    feature_dim=2,
    db_path="global_features.db"
)

# 2. 创建全球特征库
gfv_library = GlobalFeatureLibrary(config)

# 3. 查询特征
beijing_features = gfv_library.get_feature_vector(39.9042, 116.4074, zoom=10)
print(f"北京特征维度: {beijing_features.shape}")

# 4. 获取区域特征
bounds = (116.0, 39.5, 117.0, 40.5)  # 北京区域
region_features = gfv_library.get_region_features(bounds, zoom=8)
print(f"区域包含 {len(region_features)} 个瓦片")
```

### 训练示例

```python
from src.gfv import GlobalFeatureDataset, GFVTrainer

# 1. 准备训练数据
coords = [(39.9042, 116.4074), (31.2304, 121.4737)]  # 北京, 上海
features = [np.random.randn(64), np.random.randn(64)]  # 示例特征

# 2. 创建数据集
dataset = GlobalFeatureDataset(coords, features)

# 3. 创建训练器
trainer_config = {
    'learning_rate': 1e-3,
    'num_epochs': 100,
    'batch_size': 32
}
trainer = GFVTrainer(gfv_library, trainer_config)

# 4. 训练模型
results = trainer.train(dataset, save_path="gfv_model.pth")
```

### PyTorch Lightning 训练

```python
from src.gfv import GFVLightningModule
import pytorch_lightning as pl

# 1. 创建 Lightning 模块
lightning_module = GFVLightningModule(
    config=config,
    learning_rate=1e-3
)

# 2. 创建训练器
trainer = pl.Trainer(
    max_epochs=100,
    accelerator='gpu',
    devices=1
)

# 3. 训练
trainer.fit(lightning_module, train_dataloader, val_dataloader)
```

## 核心组件 (Core Components)

### 1. GlobalHashConfig
全球哈希编码配置类，包含所有必要的参数设置。

```python
config = GlobalHashConfig(
    num_levels=16,              # 哈希表层数
    max_hash=2**14,            # 最大哈希值
    base_resolution=16,         # 基础分辨率
    finest_resolution=512,      # 最细分辨率
    feature_dim=2,             # 每层特征维度
    global_bounds=(-180, -90, 180, 90),  # 全球边界
    tile_size=256,             # 瓦片大小
    max_zoom=18,               # 最大缩放级别
    db_path="global_features.db",  # 数据库路径
    cache_size=10000           # 缓存大小
)
```

### 2. MultiResolutionHashEncoding
多分辨率哈希编码网络，支持空间坐标到特征向量的映射。

```python
from src.gfv.core import MultiResolutionHashEncoding

encoder = MultiResolutionHashEncoding(config)
coords = torch.randn(100, 3)  # [N, 3] 坐标
features = encoder(coords)    # [N, total_feature_dim] 特征
```

### 3. GlobalFeatureDatabase
全球特征数据库，提供高效的存储和查询功能。

```python
from src.gfv.core import GlobalFeatureDatabase

database = GlobalFeatureDatabase(config)

# 存储特征
database.store_features(x=100, y=200, zoom=10, features=features)

# 查询特征  
features = database.query_features(lat=39.9042, lon=116.4074, zoom=10)

# 获取统计信息
stats = database.get_database_stats()
```

### 4. GlobalFeatureLibrary
全球特征库主类，提供完整的特征管理功能。

```python
from src.gfv.core import GlobalFeatureLibrary

library = GlobalFeatureLibrary(config)

# 训练
training_data = [(lat, lon, features), ...]
library.train_on_global_data(training_data, num_epochs=100)

# 查询
features = library.get_feature_vector(lat, lon, zoom=10)

# 保存/加载模型
library.save_model("model.pth")
library.load_model("model.pth")
```

## 数据集类 (Dataset Classes)

### SDFDataset
```python
from src.gfv.dataset import SDFDataset

dataset = SDFDataset("sdf_data.npy")
coords, sdf_values = dataset[0]
```

### GlobalFeatureDataset
```python
from src.gfv.dataset import GlobalFeatureDataset

dataset = GlobalFeatureDataset(coords, features, zoom_levels)
sample = dataset[0]  # {'coords': ..., 'features': ..., 'zoom': ...}
```

### MultiScaleDataset
```python
from src.gfv.dataset import MultiScaleDataset

dataset = MultiScaleDataset(base_coords, zoom_levels=[8, 10, 12, 14])
```

## 工具函数 (Utilities)

### 坐标转换
```python
from src.gfv.utils import lat_lon_to_tile, calculate_distance

# 经纬度转瓦片坐标
tile_x, tile_y = lat_lon_to_tile(39.9042, 116.4074, zoom=10)

# 计算距离
distance = calculate_distance(39.9042, 116.4074, 31.2304, 121.4737)
```

### 可视化
```python
from src.gfv.utils import plot_coverage_map, visualize_global_features

# 绘制覆盖图
plot_coverage_map(database_stats, save_path="coverage.png")

# 可视化全球特征
visualize_global_features(coords, features, save_path="features.png")
```

### 数据处理
```python
from src.gfv.utils import load_sdf_data, save_feature_cache

# 加载 SDF 数据
coords, sdf_values = load_sdf_data("data.npy")

# 保存特征缓存
save_feature_cache(features_dict, "cache.npz")
```

## 高级功能 (Advanced Features)

### 多尺度训练
```python
from src.gfv.trainer import GFVMultiScaleTrainer

multiscale_trainer = GFVMultiScaleTrainer(
    model=gfv_library,
    config={'progressive_training': True}
)

results = multiscale_trainer.train_multiscale(
    train_dataset=multiscale_dataset,
    save_path="multiscale_model.pth"
)
```

### 交互式可视化
```python
from src.gfv.utils import plot_interactive_map, create_dashboard

# 创建交互式地图
fig = plot_interactive_map(coords, features)
fig.show()

# 创建综合仪表板
dashboard = create_dashboard(
    database_stats=stats,
    training_history=results,
    coords=coords,
    features=features
)
dashboard.show()
```

## 性能优化 (Performance Optimization)

### 批量查询
```python
# 批量查询比单次查询更高效
batch_features = database.batch_query_features(coords_list, zoom=10)
```

### 缓存策略
```python
# 调整缓存大小以优化内存使用
config = GlobalHashConfig(cache_size=50000)  # 增大缓存
```

### GPU 加速
```python
# 使用 GPU 进行哈希编码计算
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
encoder = encoder.to(device)
```

## 示例项目 (Example Projects)

查看 `src/gfv/example_usage.py` 获取完整的使用示例，包括：

- 基础使用示例
- 训练示例
- 多尺度特征处理
- 可视化示例
- 性能分析

运行示例：
```bash
cd src/gfv
python example_usage.py
```

## API 参考 (API Reference)

详细的 API 文档请参考各模块的 docstring：

- `core.py` - 核心组件
- `dataset.py` - 数据集类
- `trainer.py` - 训练器组件
- `utils/` - 工具函数包

## 配置选项 (Configuration Options)

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `num_levels` | int | 16 | 哈希表层数 |
| `max_hash` | int | 16384 | 最大哈希值 |
| `base_resolution` | int | 16 | 基础分辨率 |
| `finest_resolution` | int | 512 | 最细分辨率 |
| `feature_dim` | int | 2 | 每层特征维度 |
| `global_bounds` | tuple | (-180, -90, 180, 90) | 全球边界 |
| `tile_size` | int | 256 | 瓦片大小 |
| `max_zoom` | int | 18 | 最大缩放级别 |
| `db_path` | str | "global_features.db" | 数据库路径 |
| `cache_size` | int | 10000 | 缓存大小 |

## 常见问题 (FAQ)

### Q: 如何选择合适的哈希表层数？
A: 哈希表层数决定了特征的表示能力。一般建议：
- 小规模场景：8-12 层
- 中等规模：12-16 层  
- 大规模全球场景：16-20 层

### Q: 如何优化查询性能？
A: 几个优化建议：
1. 使用批量查询而非单次查询
2. 增大缓存大小
3. 使用适当的缩放级别
4. 考虑使用 GPU 加速

### Q: 数据库文件过大怎么办？
A: 可以：
1. 使用更小的特征维度
2. 定期清理不需要的缓存
3. 使用压缩存储格式（HDF5）
4. 分布式存储

## 许可证 (License)

本项目采用 MIT 许可证。详见 LICENSE 文件。

## 贡献 (Contributing)

欢迎提交 Issues 和 Pull Requests！

## 更新日志 (Changelog)

### v1.0.0
- 初始发布
- 基础全球特征向量功能
- 多分辨率哈希编码
- PyTorch Lightning 支持
- 丰富的可视化工具 