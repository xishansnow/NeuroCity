# NeuralVDB Integration for Plenoxels

## 概述

NeuralVDB 集成为 Plenoxels 提供了高效的外部存储功能，支持稀疏体素数据的压缩存储和快速访问。

## 功能特性

### 🗃️ 高效存储
- **稀疏表示**: 仅存储有意义的体素数据
- **压缩算法**: 支持多级压缩（1-9级）
- **半精度**: 可选择16位浮点数存储以减少空间
- **元数据**: 自动保存训练配置和模型参数

### 📊 分层细节层次 (LOD)
- **多分辨率**: 自动生成多个分辨率级别
- **渐进加载**: 支持从低分辨率到高分辨率的渐进加载
- **内存优化**: 根据需要动态加载不同级别的数据

### 🔧 优化功能
- **存储优化**: 自动修剪无效体素和压缩数据
- **统计信息**: 提供详细的存储使用统计
- **批处理**: 支持大规模数据的分块处理

## 安装依赖

```bash
# 安装 OpenVDB (必需)
pip install openvdb

# 或者使用 conda
conda install -c conda-forge openvdb
```

## 基本用法

### 保存模型为 NeuralVDB

```python
from src.plenoxels import (
    VoxelGrid, PlenoxelConfig,
    save_plenoxel_as_neuralvdb,
    NeuralVDBConfig
)

# 创建体素网格
voxel_grid = VoxelGrid(
    resolution=(128, 128, 128),
    scene_bounds=(-1, -1, -1, 1, 1, 1),
    sh_degree=2
)

# 配置存储选项
vdb_config = NeuralVDBConfig(
    compression_level=8,      # 高压缩
    half_precision=True,      # 使用半精度
    tolerance=1e-5,          # 稀疏化阈值
    include_metadata=True     # 包含元数据
)

# 保存为 VDB 文件
success = save_plenoxel_as_neuralvdb(
    voxel_grid=voxel_grid,
    output_path="model.vdb",
    model_config=model_config,
    vdb_config=vdb_config
)
```

### 从 NeuralVDB 加载模型

```python
from src.plenoxels import load_plenoxel_from_neuralvdb

# 从 VDB 文件加载
voxel_grid, model_config = load_plenoxel_from_neuralvdb(
    vdb_path="model.vdb",
    device=torch.device("cuda")
)

print(f"加载的网格分辨率: {voxel_grid.resolution}")
print(f"场景边界: {voxel_grid.scene_bounds}")
print(f"球谐阶数: {voxel_grid.sh_degree}")
```

### 高级存储管理

```python
from src.plenoxels import NeuralVDBManager

# 创建存储管理器
manager = NeuralVDBManager(vdb_config)

# 创建分层细节层次
lod_files = manager.create_hierarchical_lod(
    voxel_grid=voxel_grid,
    output_dir="lod_output",
    levels=4
)

# 优化存储
manager.optimize_vdb_storage(
    vdb_path="model.vdb",
    output_path="model_optimized.vdb"
)

# 获取存储统计
stats = manager.get_storage_stats("model.vdb")
print(f"文件大小: {stats['file_size_mb']:.2f} MB")
print(f"活跃体素数: {stats['total_active_voxels']:,}")
```

## 训练集成

### 使用 NeuralVDB 训练器

```python
from src.plenoxels import (
    NeuralVDBPlenoxelTrainer,
    NeuralVDBTrainerConfig,
    NeuralVDBConfig
)

# 配置 NeuralVDB 训练器
trainer_config = NeuralVDBTrainerConfig(
    # 基础训练设置
    max_epochs=10000,
    learning_rate=0.1,
    
    # NeuralVDB 设置
    save_neuralvdb=True,
    neuralvdb_save_interval=5000,
    neuralvdb_compression_level=8,
    
    # 分层细节层次
    create_lod=True,
    lod_levels=3,
    lod_save_interval=20000,
    
    # 存储优化
    optimize_storage=True,
    storage_stats_interval=1000
)

# 创建训练器
trainer = NeuralVDBPlenoxelTrainer(
    model_config=model_config,
    trainer_config=trainer_config,
    dataset_config=dataset_config
)

# 开始训练
trainer.train()

# 导出最终模型
export_info = trainer.export_final_vdb(
    output_path="final_model.vdb",
    create_lod=True
)
```

### 从 VDB 检查点恢复训练

```python
# 从 VDB 检查点恢复
trainer_config.resume_from = "checkpoint_epoch_5000.vdb"

trainer = NeuralVDBPlenoxelTrainer(
    model_config=model_config,
    trainer_config=trainer_config,
    dataset_config=dataset_config
)

# 继续训练
trainer.train()
```

## 命令行工具

### 基本示例

```bash
# 保存演示数据
python -m src.plenoxels.neuralvdb_example --mode save --vdb_path demo.vdb

# 加载并显示统计信息
python -m src.plenoxels.neuralvdb_example --mode load --vdb_path demo.vdb

# 显示详细统计信息
python -m src.plenoxels.neuralvdb_example --mode stats --vdb_path demo.vdb

# 创建分层细节层次
python -m src.plenoxels.neuralvdb_example --mode lod --lod_dir lod_output

# 优化存储
python -m src.plenoxels.neuralvdb_example --mode optimize \
    --vdb_path input.vdb --output_path optimized.vdb
```

## 配置选项

### NeuralVDBConfig 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `compression_level` | 6 | 压缩级别 (0-9) |
| `half_precision` | True | 使用半精度存储 |
| `chunk_size` | (64,64,64) | 数据块大小 |
| `tolerance` | 1e-4 | 稀疏化阈值 |
| `background_value` | 0.0 | 背景值 |
| `include_metadata` | True | 包含元数据 |

### NeuralVDBTrainerConfig 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `save_neuralvdb` | True | 启用 VDB 保存 |
| `neuralvdb_save_interval` | 10000 | VDB 保存间隔 |
| `create_lod` | False | 创建 LOD |
| `lod_levels` | 3 | LOD 级别数 |
| `optimize_storage` | True | 优化存储 |

## 性能优化

### 内存管理

```python
# 对于大型场景，使用较小的块大小
vdb_config = NeuralVDBConfig(
    chunk_size=(32, 32, 32),  # 减少内存使用
    half_precision=True,       # 使用半精度
    tolerance=1e-3            # 更激进的稀疏化
)
```

### 存储优化

```python
# 高压缩设置
vdb_config = NeuralVDBConfig(
    compression_level=9,       # 最大压缩
    tolerance=1e-5,           # 精确稀疏化
    optimize_storage=True      # 启用存储优化
)
```

### 批处理加载

```python
# 对于大型数据集，使用分块加载
manager = NeuralVDBManager(vdb_config)

# 分块处理大型体素网格
for chunk in large_voxel_grid.chunks():
    chunk_path = f"chunk_{chunk.id}.vdb"
    manager.export_plenoxel_to_vdb(chunk, chunk_path)
```

## 故障排除

### 常见问题

1. **OpenVDB 未安装**
   ```
   ImportError: OpenVDB not available
   ```
   **解决方案**: 安装 OpenVDB
   ```bash
   pip install openvdb
   ```

2. **内存不足**
   ```
   CUDA out of memory
   ```
   **解决方案**: 减少块大小或使用半精度
   ```python
   vdb_config.chunk_size = (32, 32, 32)
   vdb_config.half_precision = True
   ```

3. **文件过大**
   **解决方案**: 增加压缩级别和稀疏化阈值
   ```python
   vdb_config.compression_level = 9
   vdb_config.tolerance = 1e-3
   ```

## 最佳实践

### 1. 存储配置
- 对于测试使用 `compression_level=6`
- 对于生产使用 `compression_level=8-9`
- 大型场景使用 `half_precision=True`

### 2. 训练策略  
- 定期保存 VDB 检查点 (`neuralvdb_save_interval=5000`)
- 在训练后期创建 LOD (`lod_save_interval=20000`)
- 启用存储优化 (`optimize_storage=True`)

### 3. 内存管理
- 监控存储统计信息
- 使用适当的块大小
- 定期清理临时文件

## 示例项目

完整的示例项目请参考 `neuralvdb_example.py` 文件，包含：
- 基本保存和加载操作
- 分层细节层次创建
- 存储优化示例
- 统计信息显示
- 错误处理示例

## 技术细节

### 数据格式
- **密度数据**: 存储为 FloatGrid
- **球谐系数**: 每个系数存储为 Vec3fGrid
- **元数据**: 存储为 StringGrid (JSON格式)

### 坐标系统
- VDB 使用世界坐标系
- 自动处理体素网格到世界坐标的转换
- 支持任意场景边界

### 压缩算法
- 使用 OpenVDB 内置压缩
- 支持有损和无损压缩
- 自动稀疏表示优化 