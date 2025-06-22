# NeuroCity - 神经辐射场城市建模平台

NeuroCity 是一个综合性的神经辐射场（Neural Radiance Fields）平台，专门用于大规模城市场景的建模、渲染和数据生成。该平台集成了多种最先进的 NeRF 实现和数据生成工具。

## 🌟 功能特性

### 核心软件包

- **NeRFs Package** (`src/nerfs/`): 集成 13 种 NeRF 实现的统一软件包
- **DataGen Package** (`src/datagen/`): 专业的数据生成和采样工具
- **GFV Package** (`src/gfv/`): 几何特征向量处理
- **NeuralVDB Package** (`src/neuralvdb/`): 神经体素数据库

### NeRF 实现

- **Classic NeRF**: 原始神经辐射场实现
- **Instant-NGP**: 即时神经图形基元，极快训练速度
- **Mip-NeRF**: 具有抗锯齿功能的多尺度 NeRF
- **Block-NeRF**: 大规模场景分块表示
- **Mega-NeRF**: 大规模户外场景重建
- **Grid-NeRF**: 基于网格的高效渲染
- **Plenoxels**: 无神经网络的稀疏体素表示
- **SVRaster**: 稀疏体素光栅化
- **Bungee-NeRF**: 渐进式训练策略
- **Pyramid-NeRF**: 多尺度金字塔表示
- **DNMP-NeRF**: 可微分神经网格基元
- **Nerfacto**: 实用的 NeRF 实现
- **Mega-NeRF Plus**: 增强版大规模 NeRF

### 数据生成能力

- **多种数据源**：支持生成合成数据或从 OSM 下载真实数据
- **灵活配置**：可调整城市尺寸、体素大小、建筑类型等
- **多种格式**：支持 VDB 和 numpy 格式输出
- **完整元数据**：包含建筑信息、坐标、类型等详细数据
- **体素采样**：支持多种采样策略（均匀、分层、表面采样）
- **神经网络训练**：支持 SDF 和 Occupancy 网络训练
- **完整流水线**：从数据生成到模型训练的端到端解决方案

## 快速开始

### 1. 安装依赖

```bash
# 方法 1：使用安装脚本（推荐）
chmod +x install_dependencies.sh
./install_dependencies.sh

# 方法 2：手动安装
pip3 install -r requirements.txt
```

### 2. 生成测试数据

#### 方法 1：生成合成城市数据（推荐）

```bash
# 生成 10km x 10km 的 tile 化城市数据
python3 simple_vdb_generator.py

# 生成复杂城市（需要 OpenVDB）
python3 generate_test_vdb.py
```

#### 方法 2：从 OSM 下载真实数据

```bash
# 下载北京天安门附近的建筑物
python3 osm_to_vdb.py
```

### 3. 体素采样

```bash
# 运行采样示例
python3 example_usage.py
# 选择选项 1 进行体素采样

# 或直接运行采样器
python3 sampler.py
```

### 4. 神经网络训练

```bash
# 运行训练示例
python3 example_usage.py
# 选择选项 2 进行占用网络训练
# 选择选项 3 进行 SDF 网络训练

# 或运行完整流水线
python3 train_pipeline.py --task occupancy --epochs 50
```

## 文件说明

### 核心文件

- `simple_vdb_generator.py` - 大规模城市 tile 体素生成器
- `generate_test_vdb.py` - 完整的城市 VDB 生成器
- `osm_to_vdb.py` - 从 OSM 下载建筑物并转换为 VDB
- `sampler.py` - 体素采样器模块
- `neural_sdf.py` - SDF/Occupancy 神经网络训练模块
- `train_pipeline.py` - 完整训练流水线
- `example_usage.py` - 使用示例脚本
- `requirements.txt` - Python 依赖列表
- `install_dependencies.sh` - 自动安装脚本

### 输出文件

- `tiles/tile_x_y.npy` - tile 体素数据
- `tiles/tile_x_y.json` - tile 元数据
- `samples/coords_x_y.npy` - 采样坐标
- `samples/labels_x_y.npy` - 占用标签
- `samples/sdf_x_y.npy` - SDF 值
- `model_occupancy.pth` - 占用网络模型
- `model_sdf.pth` - SDF 网络模型

## 使用示例

### 1. 生成 tile 化城市数据

```python
from simple_vdb_generator import TileCityGenerator

# 创建生成器
generator = TileCityGenerator(
    city_size=(10000, 10000, 100),  # 10km x 10km x 100m
    tile_size=(1000, 1000),         # 1km x 1km tiles
    voxel_size=1.0,                 # 1 米体素
    output_dir="tiles"
)

# 生成所有 tiles
generator.generate_and_save_all_tiles(n_per_tile=20)
```

### 2. 体素采样

```python
from sampler import VoxelSampler

# 创建采样器
sampler = VoxelSampler(
    tiles_dir="tiles",
    voxel_size=1.0,
    sample_ratio=0.1
)

# 分层采样
samples = sampler.sample_stratified(0, 0, n_samples=10000)

# 对所有 tiles 采样
all_samples = sampler.sample_all_tiles(
    sampling_method='stratified',
    n_samples_per_tile=10000
)

# 保存采样数据
sampler.save_samples(all_samples, "samples")
```

### 3. 神经网络训练

```python
from neural_sdf import MLP, NeuralSDFTrainer, load_training_data

# 加载训练数据
train_dataloader, val_dataloader = load_training_data(
    samples_dir="samples",
    task_type='occupancy',  # 或 'sdf'
    train_ratio=0.8
)

# 创建模型
model = MLP(
    input_dim=3,
    hidden_dims=[256, 512, 512, 256, 128],
    output_dim=1,
    activation='relu'
)

# 创建训练器
trainer = NeuralSDFTrainer(
    model=model,
    learning_rate=1e-3,
    weight_decay=1e-5
)

# 训练模型
trainer.train(
    train_dataloader=train_dataloader,
    val_dataloader=val_dataloader,
    num_epochs=50,
    save_path='model.pth'
)

# 预测
test_coords = np.array([[100, 100, 10], [200, 200, 20]])
predictions = trainer.predict(test_coords)
```

### 4. 完整流水线

```python
from train_pipeline import TrainingPipeline, get_default_config

# 获取配置
config = get_default_config()
config['training']['task_type'] = 'occupancy'
config['training']['num_epochs'] = 50

# 创建流水线
pipeline = TrainingPipeline(config)

# 运行完整流水线
pipeline.run_full_pipeline()
```

## 配置参数

### 城市参数

- `city_size`: 城市尺寸 (x, y, z) 单位：米
- `tile_size`: tile 尺寸 (x, y) 单位：米
- `voxel_size`: 体素大小（米），影响精度和文件大小
- `max_height`: 最大建筑高度（米）

### 采样参数

- `sampling_method`: 采样方法 ('uniform', 'stratified', 'near_surface')
- `n_samples_per_tile`: 每个 tile 的采样数量
- `occupied_ratio`: 占用体素比例（分层采样）
- `surface_threshold`: 表面阈值（表面采样）
- `noise_std`: 噪声标准差（表面采样）

### 模型参数

- `input_dim`: 输入维度（通常为 3）
- `hidden_dims`: 隐藏层维度列表
- `output_dim`: 输出维度（通常为 1）
- `activation`: 激活函数 ('relu', 'leaky_relu', 'tanh', 'sigmoid')
- `dropout`: dropout 比例

### 训练参数

- `task_type`: 任务类型 ('occupancy' 或 'sdf')
- `learning_rate`: 学习率
- `weight_decay`: 权重衰减
- `num_epochs`: 训练轮数
- `train_ratio`: 训练集比例
- `early_stopping_patience`: 早停耐心值

## 数据格式

### Tile 数据格式

```
tiles/
├── tile_0_0.npy          # tile体素数据 (1000, 1000, 100)
├── tile_0_0.json         # tile元数据
├── tile_0_1.npy
├── tile_0_1.json
└── ...
```

### 采样数据格式

```
samples/
├── coords_0_0.npy        # 采样坐标 (N, 3)
├── labels_0_0.npy        # 占用标签 (N,) - occupancy任务
├── sdf_0_0.npy           # SDF值 (N,) - sdf任务
├── coords_0_1.npy
├── labels_0_1.npy
└── ...
```

### 模型文件格式

```
model_occupancy.pth        # PyTorch模型文件
model_sdf.pth             # PyTorch模型文件
training_history.png      # 训练历史图
```

## 性能优化

### 内存使用

- 小规模测试：使用较小的 tile 尺寸
- 大规模数据：调整采样比例和 batch size
- 内存不足：使用数据流式处理

### 处理速度

- 合成数据：几秒到几分钟
- 采样：取决于 tile 数量和采样策略
- 训练：取决于数据量和模型复杂度

## 故障排除

### 常见问题

1. **OpenVDB 安装失败**
   ```bash
   # 使用简化版本
   python3 simple_vdb_generator.py
   ```

2. **内存不足**
   ```python
   # 减少采样数量或 tile 尺寸
   config['sampling']['n_samples_per_tile'] = 5000
   ```

3. **训练收敛慢**
   ```python
   # 调整学习率和网络结构
   config['training']['learning_rate'] = 5e-4
   config['model']['hidden_dims'] = [512, 1024, 1024, 512]
   ```

4. **采样数据不平衡**
   ```python
   # 使用分层采样
   sampler.sample_stratified(tile_x, tile_y, n_samples=10000, occupied_ratio=0.3)
   ```

### 调试模式

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 扩展功能

### 添加新的采样策略

```python
def custom_sampling_method(self, tile_x, tile_y, **kwargs):
    # 实现自定义采样逻辑
    pass
```

### 添加新的网络架构

```python
class CustomNetwork(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        # 实现自定义网络架构
        pass
```

### 支持更多数据源

- 支持从其他 GIS 数据源导入
- 支持从 3D 模型文件转换
- 支持从点云数据生成

## 许可证

MIT License

## 贡献

欢迎提交 Issue 和 Pull Request！

## 联系方式

如有问题，请创建 GitHub Issue。 