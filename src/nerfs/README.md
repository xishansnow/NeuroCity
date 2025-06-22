# NeRFs Package - Neural Radiance Fields Collection

NeRFs 软件包是 NeuroCity 项目中的核心组件，集成了多种最先进的神经辐射场（Neural Radiance Fields）实现和相关方法，用于神经场景表示和渲染。

## 🌟 功能特点

- **多种 NeRF 实现**：包含 13 种不同的 NeRF 变体和改进方法
- **统一接口**：提供一致的 API 接口访问所有 NeRF 实现
- **模块化设计**：每个 NeRF 实现都是独立的模块，可单独使用
- **完整功能**：涵盖训练、推理、渲染等完整流程
- **高性能**：支持 GPU 加速和各种优化技术
- **易于扩展**：可轻松添加新的 NeRF 实现

## 📦 包含的 NeRF 实现

### 1. **Classic NeRF** (`classic_nerf`)
- **描述**：原始的神经辐射场实现
- **特点**：基础的体积渲染和神经网络表示
- **适用场景**：小规模场景，学习和研究

### 2. **Block-NeRF** (`block_nerf`) 
- **描述**：大规模场景的分块表示方法
- **特点**：空间分区、块级管理、可扩展性
- **适用场景**：大规模城市场景建模

### 3. **Mega-NeRF** (`mega_nerf`)
- **描述**：大规模户外场景重建
- **特点**：空间分区、多尺度表示
- **适用场景**：大规模户外环境

### 4. **Mega-NeRF Plus** (`mega_nerf_plus`)
- **描述**：增强版 Mega-NeRF
- **特点**：改进的内存管理、多分辨率渲染
- **适用场景**：超大规模场景处理

### 5. **Instant-NGP** (`instant_ngp`)
- **描述**：即时神经图形基元
- **特点**：哈希编码、极快训练速度
- **适用场景**：实时应用、快速原型

### 6. **Mip-NeRF** (`mip_nerf`)
- **描述**：具有抗锯齿功能的 NeRF
- **特点**：多尺度表示、抗锯齿、锥形投射
- **适用场景**：高质量渲染、多尺度场景

### 7. **Grid-NeRF** (`grid_nerf`)
- **描述**：基于网格的 NeRF 变体
- **特点**：网格加速、高效渲染
- **适用场景**：需要快速渲染的应用

### 8. **Plenoxels** (`plenoxels`)
- **描述**：无神经网络的稀疏体素表示
- **特点**：直接体素优化、无需神经网络
- **适用场景**：简单场景、快速训练

### 9. **SVRaster** (`svraster`)
- **描述**：稀疏体素光栅化
- **特点**：自适应稀疏体素、高效光栅化
- **适用场景**：实时渲染、内存受限环境

### 10. **Bungee-NeRF** (`bungee_nerf`)
- **描述**：渐进式训练的 NeRF
- **特点**：多尺度渐进训练、稳定优化
- **适用场景**：复杂场景、稳定训练

### 11. **Pyramid-NeRF** (`pyramid_nerf`)
- **描述**：多尺度金字塔表示
- **特点**：层次化表示、多分辨率
- **适用场景**：多尺度场景分析

### 12. **DNMP-NeRF** (`dnmp_nerf`)
- **描述**：可微分神经网格基元
- **特点**：网格基础表示、几何先验
- **适用场景**：几何约束场景

### 13. **Nerfacto** (`nerfacto`)
- **描述**：实用的 NeRF 实现
- **特点**：平衡性能和质量、易于使用
- **适用场景**：通用场景、实际应用

### 14. **Occupancy Networks** (`occupancy_net`)
- **描述**：占用网络用于 3D 重建
- **特点**：学习点到占用概率的映射、支持网格提取
- **适用场景**：3D 形状重建、点云处理

### 15. **SDF Networks** (`sdf_net`)
- **描述**：有符号距离函数网络用于形状表示
- **特点**：连续形状表示、几何约束、潜在编码
- **适用场景**：形状生成、几何建模、形状插值

## 🚀 快速开始

### 基本用法

```python
from src.nerfs import get_nerf_module, list_available_nerfs, get_nerf_info

# 查看所有可用的 NeRF 实现
print("可用的 NeRF 实现:")
for nerf_name in list_available_nerfs():
    print(f"- {nerf_name}")

# 获取 NeRF 信息
nerf_info = get_nerf_info()
print(f"\\nInstant-NGP: {nerf_info['instant_ngp']}")

# 加载特定的 NeRF 模块
instant_ngp = get_nerf_module('instant_ngp')
```

### 使用特定 NeRF 实现

```python
# 使用 Instant-NGP
from src.nerfs.instant_ngp import InstantNGPModel, InstantNGPConfig

config = InstantNGPConfig()
model = InstantNGPModel(config)

# 使用 Classic NeRF
from src.nerfs.classic_nerf import ClassicNeRFModel, ClassicNeRFConfig

config = ClassicNeRFConfig()
model = ClassicNeRFModel(config)

# 使用 Occupancy Networks
from src.nerfs.occupancy_net import OccupancyNetwork, OccupancyTrainer

# 创建占用网络
occupancy_net = OccupancyNetwork(
    input_dim=3,
    hidden_dim=256,
    num_layers=5
)

# 使用 SDF Networks
from src.nerfs.sdf_net import SDFNetwork, SDFTrainer

# 创建 SDF 网络
sdf_net = SDFNetwork(
    input_dim=3,
    output_dim=1,
    hidden_dim=256,
    num_layers=8,
    skip_layers=[4]
)
```

### 训练示例

```python
from src.nerfs.instant_ngp import InstantNGPTrainer, InstantNGPConfig
from src.nerfs.instant_ngp.dataset import InstantNGPDataset

# 配置
config = InstantNGPConfig(
    scene_bounds=(-1, -1, -1, 1, 1, 1),
    hash_levels=16,
    hash_size=2**19
)

# 数据集
dataset = InstantNGPDataset("path/to/data")

# 训练器
trainer = InstantNGPTrainer(config)
trainer.train(dataset, num_epochs=1000)
```

## 📁 软件包结构

```
src/nerfs/
├── __init__.py                 # 主初始化文件
├── README.md                   # 本文档
├── example_usage.py            # 使用示例
├── block_nerf/                 # Block-NeRF 实现
├── classic_nerf/               # Classic NeRF 实现
├── dnmp_nerf/                  # DNMP-NeRF 实现
├── grid_nerf/                  # Grid-NeRF 实现
├── mega_nerf/                  # Mega-NeRF 实现
├── mip_nerf/                   # Mip-NeRF 实现
├── nerfacto/                   # Nerfacto 实现
├── plenoxels/                  # Plenoxels 实现
├── svraster/                   # SVRaster 实现
├── bungee_nerf/                # Bungee-NeRF 实现
├── instant_ngp/                # Instant-NGP 实现
├── mega_nerf_plus/             # Mega-NeRF Plus 实现
├── pyramid_nerf/               # Pyramid-NeRF 实现
├── occupancy_net/              # Occupancy Networks 实现
└── sdf_net/                    # SDF Networks 实现
```

每个子模块通常包含：
- `core.py` - 核心模型实现
- `dataset.py` - 数据集处理
- `trainer.py` - 训练逻辑
- `utils/` - 工具函数
- `README.md` - 模块文档

## 🔧 API 参考

### 主要函数

#### `list_available_nerfs()`
返回所有可用的 NeRF 实现列表。

**返回值：**
- `List[str]`：NeRF 实现名称列表

#### `get_nerf_info()`
获取所有 NeRF 实现的详细信息。

**返回值：**
- `Dict[str, str]`：NeRF 名称到描述的映射

#### `get_nerf_module(name: str)`
根据名称获取特定的 NeRF 模块。

**参数：**
- `name`：NeRF 实现名称

**返回值：**
- 对应的 NeRF 模块

**异常：**
- `ValueError`：如果指定的 NeRF 不存在

## 🎯 选择指南

### 根据场景规模选择

- **小规模场景**：`classic_nerf`, `nerfacto`
- **中等规模场景**：`instant_ngp`, `mip_nerf`, `grid_nerf`
- **大规模场景**：`block_nerf`, `mega_nerf`, `mega_nerf_plus`

### 根据性能要求选择

- **实时渲染**：`instant_ngp`, `svraster`, `plenoxels`
- **高质量渲染**：`mip_nerf`, `pyramid_nerf`
- **平衡性能**：`nerfacto`, `grid_nerf`

### 根据特殊需求选择

- **几何约束**：`dnmp_nerf`
- **渐进训练**：`bungee_nerf`
- **多尺度表示**：`mip_nerf`, `pyramid_nerf`
- **无神经网络**：`plenoxels`

## 🔬 技术特性

### 渲染技术
- **体积渲染**：所有实现支持
- **光栅化**：`svraster` 特有
- **多尺度渲染**：`mip_nerf`, `pyramid_nerf`

### 场景表示
- **神经网络**：大多数实现
- **体素网格**：`plenoxels`, `svraster`
- **哈希编码**：`instant_ngp`
- **网格基元**：`dnmp_nerf`

### 优化技术
- **空间分区**：`block_nerf`, `mega_nerf`
- **渐进训练**：`bungee_nerf`
- **自适应采样**：多种实现
- **内存优化**：`mega_nerf_plus`

## 📊 性能对比

| NeRF 实现 | 训练速度 | 渲染速度 | 内存使用 | 质量 | 适用场景 |
|-----------|----------|----------|----------|------|----------|
| Classic NeRF | ⭐⭐ | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | 小规模 |
| Instant-NGP | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | 实时应用 |
| Mip-NeRF | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 高质量 |
| Block-NeRF | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ | 大规模 |
| Plenoxels | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | 快速训练 |

## 🛠️ 扩展开发

### 添加新的 NeRF 实现

1. 在 `src/nerfs/` 下创建新的模块目录
2. 实现必要的组件（core, dataset, trainer）
3. 更新 `__init__.py` 中的导入和列表
4. 添加相应的文档

### 模块接口规范

每个 NeRF 模块应该提供：
- 配置类（`*Config`）
- 模型类（`*Model`）
- 数据集类（`*Dataset`）
- 训练器类（`*Trainer`）

## 📚 依赖项

- PyTorch >= 1.9.0
- NumPy >= 1.19.0
- OpenCV >= 4.5.0
- PyTorch Lightning >= 1.5.0（部分模块）
- 其他特定依赖项见各模块文档

## 🤝 贡献指南

1. Fork 项目
2. 创建特性分支
3. 提交更改
4. 推送到分支
5. 创建 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 详见 LICENSE 文件。

## 👥 作者

NeuroCity Team

## 🙏 致谢

感谢所有 NeRF 研究者和开源贡献者的杰出工作，使得这个综合性的 NeRF 软件包成为可能。

---

*最后更新：2024年* 