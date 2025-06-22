# DataGen 数据生成软件包

DataGen是一个专门用于生成仿真数据的Python软件包，特别适用于神经辐射场（Neural Radiance Fields）和3D场景建模的训练数据生成。

## 功能特点

### 🎯 核心功能
- **占用网格生成**: 支持球体、立方体、圆柱体等基础几何体
- **SDF数据生成**: 生成有向距离场（Signed Distance Field）数据
- **表面采样**: 精确的表面附近采样，适用于SDF训练
- **点云采样**: 多种点云处理和采样策略
- **复合场景**: 支持多个几何体的组合和复杂场景生成

### 🔧 技术特性
- **模块化设计**: 清晰的功能分离，易于扩展
- **GPU支持**: 自动设备检测和GPU加速
- **多种采样策略**: 均匀采样、分层采样、表面采样、密度采样等
- **数据验证**: 完整的配置参数验证机制
- **统计跟踪**: 生成过程的性能监控

## 软件包结构

```
src/datagen/
├── __init__.py              # 主初始化文件
├── core.py                  # 核心配置和管道
├── samplers/               # 采样器模块
│   ├── __init__.py
│   ├── voxel_sampler.py    # 体素采样器
│   ├── surface_sampler.py  # 表面采样器
│   └── point_cloud_sampler.py # 点云采样器
├── generators/             # 生成器模块
│   ├── __init__.py
│   ├── sdf_generator.py    # SDF生成器
│   ├── occupancy_generator.py # 占用网格生成器
│   └── synthetic_scene_generator.py # 合成场景生成器
├── datasets/              # 数据集类
│   ├── __init__.py
│   └── synthetic_datasets.py
└── utils/                 # 工具函数
    ├── __init__.py
    ├── geometry_utils.py
    └── io_utils.py
```

## 快速开始

### 基础占用网格生成

```python
from src.datagen.generators import OccupancyGenerator

# 创建生成器
generator = OccupancyGenerator(
    voxel_size=1.0,
    grid_bounds=(-50, -50, 0, 50, 50, 50)
)

# 生成球体
sphere_occupancy = generator.generate_sphere_occupancy(
    center=(0, 0, 25),
    radius=15.0,
    filled=True
)

# 生成复合场景
objects = [
    {'type': 'sphere', 'params': {'center': (0, 0, 20), 'radius': 10}},
    {'type': 'box', 'params': {'center': (20, 0, 20), 'size': (10, 10, 10)}}
]
scene = generator.generate_complex_scene(objects, 'union')
```

### SDF数据生成

```python
from src.datagen.generators import SDFGenerator

# 创建SDF生成器
sdf_gen = SDFGenerator()

# 生成几何体SDF
coords = np.random.uniform(-30, 30, (10000, 3))
sdf_values = sdf_gen.generate_geometric_sdf(
    coords, 
    'sphere', 
    {'center': [0, 0, 0], 'radius': 15.0}
)
```

### 表面采样

```python
from src.datagen.samplers import SurfaceSampler

# 创建表面采样器
surface_sampler = SurfaceSampler(
    surface_threshold=0.5,
    sampling_radius=3.0,
    adaptive_sampling=True
)

# 表面采样
surface_samples = surface_sampler.sample_near_surface(
    coordinates, occupancy,
    n_samples=5000,
    noise_std=1.0
)
```

### 点云采样

```python
from src.datagen.samplers import PointCloudSampler

# 创建点云采样器
pc_sampler = PointCloudSampler(
    downsample_ratio=0.1,
    noise_level=0.05,
    normal_estimation=True
)

# 点云采样
pc_samples = pc_sampler.sample_from_point_cloud(
    points,
    n_samples=3000
)
```

## 配置系统

DataGen使用配置类来管理参数：

```python
from src.datagen import DataGenConfig, DataGenPipeline

config = DataGenConfig(
    output_dir="outputs",
    voxel_size=0.5,
    grid_size=(200, 200, 100),
    n_samples_per_tile=5000,
    sampling_strategy='stratified',
    use_gpu=True
)

pipeline = DataGenPipeline(config)
```

## 演示示例

运行演示来了解DataGen的功能：

```bash
cd demos
python demo_datagen_simple.py
```

演示包括：
- 占用网格生成（球体、立方体、复合场景）
- 表面采样演示
- 点云采样演示

## 支持的几何体

### 基础几何体
- **球体**: 中心点 + 半径
- **立方体**: 中心点 + 尺寸
- **圆柱体**: 中心点 + 半径 + 高度 + 轴向

### 复合场景
- **并集**: 多个几何体的合并
- **交集**: 多个几何体的交集
- **差集**: 几何体的布尔减法

## 采样策略

### 体素采样
- 均匀采样：在体素网格中均匀采样
- 分层采样：按占用状态分层采样

### 表面采样
- 表面检测：基于阈值和梯度的表面检测
- 自适应采样：根据局部密度调整采样
- 多分辨率采样：不同分辨率级别的采样

### 点云采样
- 密度采样：基于局部点密度的采样
- 曲率采样：基于曲率的特征采样
- 多尺度采样：不同尺度的采样

## 数据格式

### 输出数据
- **坐标**: numpy数组 [N, 3]
- **SDF值**: numpy数组 [N]
- **占用值**: numpy数组 [N] 或 [nx, ny, nz]
- **法向量**: numpy数组 [N, 3]（可选）

### 元数据
- JSON格式的配置和统计信息
- 包含生成参数、数据统计等

## 性能优化

### 内存管理
- 分块处理大型数据集
- 可配置的批次大小
- 自动内存清理

### 计算优化
- 向量化操作
- GPU加速（CUDA支持）
- 并行采样

## 扩展指南

### 添加新的几何体
1. 在 `geometry_utils.py` 中添加SDF函数
2. 在 `OccupancyGenerator` 中添加生成方法
3. 在 `_generate_single_object` 中添加类型支持

### 添加新的采样策略
1. 在相应的采样器类中添加方法
2. 更新配置类以支持新参数
3. 添加相应的文档和测试

## 依赖项

```
numpy >= 1.20.0
scipy >= 1.7.0
torch >= 1.9.0 (用于SDF生成器)
```

## 版本信息

- **当前版本**: v1.0.0
- **作者**: NeuroCity Team
- **许可证**: MIT

## 未来计划

- [ ] 网格导入和SDF转换
- [ ] 更多几何体类型支持
- [ ] 实时可视化界面
- [ ] 分布式计算支持
- [ ] 与NeRF训练的无缝集成

## 贡献指南

欢迎贡献代码和建议！请遵循项目的编码规范，并在提交前运行测试。

## 支持

如有问题或建议，请在项目仓库中提交Issue。 