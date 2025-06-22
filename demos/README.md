# NeuroCity 演示和示例代码

本目录包含 NeuroCity 项目的所有演示代码和使用示例，帮助用户了解和使用项目的各种功能。

## 📁 目录结构

### 🎯 GFV (Global Feature Vector) 相关演示

#### `demo_gfv_usage.py`
**GFV 包完整功能演示**
- 演示 GFV 包从 `global_ngp.py` 迁移后的新功能
- 展示模块化架构的优势
- 包含性能对比和扩展性演示
- 与 NeuroCity 项目集成展示

```bash
cd demos
python demo_gfv_usage.py
```

#### `test_gfv_basic.py`
**GFV 基础功能测试**
- 简化的功能验证脚本
- 测试核心组件导入和基本操作
- 验证配置创建、库初始化等
- 适合快速功能验证

```bash
cd demos
python test_gfv_basic.py
```

### ⚡ PyTorch Lightning 相关演示

#### `simple_lightning_demo.py`
**简化的 Lightning 演示**
- 展示 PyTorch Lightning 基础用法
- 简单的 NeRF 模型训练演示
- 自动混合精度、检查点保存等功能
- 适合初学者理解 Lightning 工作流程

```bash
cd demos
python simple_lightning_demo.py
```

#### `example_lightning_usage.py`
**Lightning 详细使用示例**
- 更详细的 PyTorch Lightning 使用演示
- 包含数据模块、训练器配置等
- 展示高级 Lightning 功能
- 完整的训练流程示例

```bash
cd demos
python example_lightning_usage.py
```

#### `example_multi_model_lightning.py`
**多模型 Lightning 训练**
- 同时训练多种 NeRF 模型的演示
- SVRaster、Grid-NeRF、Instant-NGP、MIP-NeRF 等
- 模型性能对比功能
- Lightning 高级功能演示

```bash
cd demos
python example_multi_model_lightning.py
```

### 🏗️ NeRF 模型演示

#### `demo_mega_nerf_plus.py`
**Mega-NeRF Plus 演示**
- Mega-NeRF 模型的增强版本演示
- 大规模城市场景渲染
- 分布式训练支持
- 高性能优化示例

```bash
cd demos
python demo_mega_nerf_plus.py
```

#### `example_usage.py`
**通用 NeRF 使用示例**
- 基础的 NeRF 模型使用演示
- 场景表示和渲染流程
- 适合了解 NeRF 基本概念

```bash
cd demos
python example_usage.py
```

#### `train_pipeline.py`
**完整训练流水线**
- 端到端的模型训练流程
- 数据加载、模型训练、结果保存
- 包含各种训练策略和优化技巧

```bash
cd demos
python train_pipeline.py
```

### 🗄️ VDB (VoxelDB) 相关工具

#### `simple_vdb_generator.py`
**简单 VDB 生成器**
- 生成基础 VoxelDB 数据结构
- 用于测试和演示目的
- 支持简单几何体生成

```bash
cd demos
python simple_vdb_generator.py
```

#### `generate_test_vdb.py`
**测试 VDB 数据生成**
- 生成用于测试的 VDB 数据
- 包含复杂几何结构
- 支持多种数据格式

```bash
cd demos
python generate_test_vdb.py
```

#### `vdb_viewer.py`
**VDB 数据查看器**
- 可视化 VoxelDB 数据
- 交互式 3D 查看界面
- 支持多种显示模式

```bash
cd demos
python vdb_viewer.py
```

#### `osm_to_vdb.py`
**OSM 数据转换工具**
- 将 OpenStreetMap 数据转换为 VDB 格式
- 支持大规模地理数据处理
- 城市建模数据流水线

```bash
cd demos
python osm_to_vdb.py
```

### 🧪 测试和实验

#### `quick_test.py`
**快速功能测试**
- 项目核心功能的快速验证
- 适合开发过程中的功能检查
- 轻量级测试脚本

```bash
cd demos
python quick_test.py
```

## 🚀 使用指南

### 环境要求

确保已安装所有必要的依赖：

```bash
# 安装基础依赖
pip install torch torchvision numpy matplotlib

# 安装 Lightning 相关
pip install pytorch-lightning

# 安装 GFV 包依赖
pip install mercantile pyproj seaborn plotly h5py

# 安装 VDB 相关依赖
pip install openvdb trimesh
```

### 运行演示

1. **从项目根目录运行**：
   ```bash
   # 在项目根目录下
   python demos/demo_name.py
   ```

2. **从 demos 目录运行**：
   ```bash
   cd demos
   python demo_name.py
   ```

### 常见问题

#### Q: 导入错误 "No module named 'src'"
A: 确保从项目根目录运行脚本，或将项目根目录添加到 Python 路径：
```python
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
```

#### Q: CUDA 相关错误
A: 确保安装了正确版本的 PyTorch 并且 GPU 驱动正常：
```bash
# 检查 CUDA 可用性
python -c "import torch; print(torch.cuda.is_available())"
```

#### Q: 依赖缺失错误
A: 根据错误信息安装对应的依赖包：
```bash
pip install missing_package_name
```

## 📚 演示分类

### 按难度分类

**初学者级别**：
- `test_gfv_basic.py` - GFV 基础测试
- `simple_lightning_demo.py` - Lightning 简单演示
- `quick_test.py` - 快速测试
- `example_usage.py` - 基础使用示例

**中级**：
- `demo_gfv_usage.py` - GFV 完整演示
- `example_lightning_usage.py` - Lightning 详细示例
- `train_pipeline.py` - 训练流水线
- `simple_vdb_generator.py` - VDB 生成器

**高级**：
- `example_multi_model_lightning.py` - 多模型训练
- `demo_mega_nerf_plus.py` - Mega-NeRF Plus
- `osm_to_vdb.py` - OSM 数据处理
- `vdb_viewer.py` - VDB 可视化

### 按功能分类

**模型训练**：
- `simple_lightning_demo.py`
- `example_lightning_usage.py`
- `example_multi_model_lightning.py`
- `train_pipeline.py`

**数据处理**：
- `osm_to_vdb.py`
- `simple_vdb_generator.py`
- `generate_test_vdb.py`

**可视化**：
- `vdb_viewer.py`
- `demo_gfv_usage.py` (包含可视化)

**测试验证**：
- `test_gfv_basic.py`
- `quick_test.py`

## 🔧 开发指南

### 添加新的演示

1. 创建新的演示文件
2. 添加详细的文档字符串
3. 包含使用示例和错误处理
4. 更新本 README 文件
5. 测试演示在不同环境下的运行

### 演示文件命名规范

- `demo_*.py` - 功能演示
- `example_*.py` - 使用示例
- `test_*.py` - 测试脚本
- `*_viewer.py` - 可视化工具
- `*_generator.py` - 数据生成工具

## 📝 更新日志

### 最新更新
- ✅ 创建 demos 文件夹并迁移所有演示代码
- ✅ 重新组织文件结构，提高可维护性
- ✅ 添加详细的 README 文档
- ✅ 按功能和难度对演示进行分类

### 历史版本
- v1.0 - 初始演示代码创建
- v1.1 - 添加 PyTorch Lightning 支持
- v1.2 - 集成 GFV 包演示
- v2.0 - 重构为 demos 文件夹结构

---

**注意**: 所有演示代码仅供学习和参考使用。在生产环境中使用前，请根据具体需求进行适当的修改和优化。 