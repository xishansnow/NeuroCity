# NeuroCity Test Suite

本目录包含 NeuroCity 项目的所有测试文件，按模块组织。

## 📁 目录结构

```
tests/
├── __init__.py                 # 测试包初始化
├── README.md                   # 本文档
├── run_tests.py               # 测试运行器
├── fix_imports.py             # 导入修复脚本
├── nerfs/                     # NeRF 模块测试
│   ├── __init__.py
│   ├── test_classic_nerf.py   # Classic NeRF 测试
│   ├── test_instant_ngp.py    # Instant-NGP 测试
│   ├── test_mip_nerf.py       # Mip-NeRF 测试
│   ├── test_grid_nerf.py      # Grid-NeRF 测试
│   ├── test_svraster.py       # SVRaster 测试
│   ├── test_plenoxels.py      # Plenoxels 测试
│   ├── test_bungee_nerf.py    # Bungee-NeRF 测试
│   ├── test_pyramid_nerf.py   # Pyramid-NeRF 测试
│   ├── test_nerfacto.py       # Nerfacto 测试
│   └── test_mega_nerf_plus.py # Mega-NeRF Plus 测试
├── demos/                     # 演示和示例测试
│   └── __init__.py
├── datagen/                   # 数据生成测试
│   ├── __init__.py
│   └── test_sampler.py        # 采样器功能测试
├── gfv/                       # GFV 模块测试
│   ├── __init__.py
│   └── test_gfv_basic.py      # GFV 基础功能测试
└── neuralvdb/                 # NeuralVDB 测试
    └── __init__.py
```

## 🚀 运行测试

### 运行所有测试
```bash
python tests/run_tests.py
```

### 运行特定模块的测试
```bash
# 运行所有 NeRF 测试
python tests/run_tests.py nerfs

# 运行 GFV 测试
python tests/run_tests.py gfv

# 运行数据生成测试
python tests/run_tests.py datagen

# 运行演示测试
python tests/run_tests.py demos
```

### 运行单个测试文件
```bash
python tests/nerfs/test_instant_ngp.py
python tests/gfv/test_gfv_basic.py
python tests/datagen/test_sampler.py
```

### 列出所有可用测试
```bash
python tests/run_tests.py --list
```

## 🧪 测试内容

### NeRF 模块测试 (`tests/nerfs/`)

每个 NeRF 实现都有对应的测试文件，测试内容包括：

- **模型初始化测试**: 验证模型能正确创建和配置
- **前向传播测试**: 验证模型的前向计算正确性
- **训练测试**: 验证训练流程正常运行
- **数据集测试**: 验证数据加载和处理功能
- **工具函数测试**: 验证各种辅助函数

#### 主要测试模块

1. **Classic NeRF** (`test_classic_nerf.py`)
   - 基础 NeRF 架构测试
   - 体积渲染测试
   - 位置编码测试

2. **Instant-NGP** (`test_instant_ngp.py`)
   - 哈希编码测试
   - 快速训练测试
   - 多分辨率网格测试

3. **Mip-NeRF** (`test_mip_nerf.py`)
   - 抗锯齿测试
   - 多尺度表示测试
   - 锥形投射测试

4. **Grid-NeRF** (`test_grid_nerf.py`)
   - 网格存储测试
   - 网格采样测试
   - 高效渲染测试

5. **SVRaster** (`test_svraster.py`)
   - 稀疏体素测试
   - 光栅化测试
   - 自适应细分测试

### GFV 模块测试 (`tests/gfv/`)

- **GFV 基础测试** (`test_gfv_basic.py`): 几何特征向量基本功能测试
  - 基本导入测试
  - 配置创建测试
  - 库创建测试
  - 坐标工具函数测试
  - 数据集创建测试

### 数据生成测试 (`tests/datagen/`)

- **采样器测试** (`test_sampler.py`): 数据采样和神经网络训练测试
  - 体素采样器测试
  - 神经网络基础功能测试
  - SDF 训练流程测试
  - 数据加载和处理测试

### 演示测试 (`tests/demos/`)

该目录为演示相关的测试预留，当前为空。

## 🔧 测试工具

### 测试运行器 (`run_tests.py`)

主要的测试运行工具，提供以下功能：

- 运行所有测试或特定模块测试
- 显示详细的测试结果
- 统计通过/失败的测试数量
- 支持命令行参数

### 导入修复脚本 (`fix_imports.py`)

用于修复测试文件中的导入路径，在测试文件迁移后确保导入正确。

## 📋 编写测试

### 测试文件命名规范

- 测试文件以 `test_` 开头，如 `test_module_name.py`
- 测试类以 `Test` 开头，如 `class TestModuleName:`
- 测试方法以 `test_` 开头，如 `def test_function_name():`

### 测试文件模板

```python
#!/usr/bin/env python3
"""
Test suite for [Module Name].

Description of what this test module covers.
"""

import sys
import os
# Add the src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import pytest
import torch
import numpy as np

from nerfs.module_name import (
    ConfigClass, ModelClass, TrainerClass
)

class TestModuleName:
    """Test cases for module functionality."""
    
    def test_initialization(self):
        """Test module initialization."""
        config = ConfigClass()
        model = ModelClass(config)
        assert model is not None
    
    def test_forward_pass(self):
        """Test forward pass."""
        # Test implementation here
        pass

def main():
    """Run all tests in this module."""
    pytest.main([__file__])

if __name__ == '__main__':
    main()
```

### 导入规范

测试文件应该：

1. 添加 src 目录到 Python 路径
2. 使用绝对导入引用源模块
3. 从 `nerfs.module_name` 导入所需的类和函数

```python
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from nerfs.instant_ngp import InstantNGPConfig, InstantNGP
from nerfs.instant_ngp.utils import hash_encode
```

## 🏃‍♂️ 持续集成

测试可以集成到 CI/CD 流水线中：

```bash
# 在 CI 环境中运行所有测试
python tests/run_tests.py

# 检查测试返回码
if [ $? -eq 0 ]; then
    echo "All tests passed"
else
    echo "Some tests failed"
    exit 1
fi
```

## 📊 测试覆盖率

建议使用 `pytest-cov` 来检查测试覆盖率：

```bash
pip install pytest-cov
pytest tests/ --cov=src --cov-report=html
```

## ⚠️ 注意事项

1. **路径问题**: 确保测试文件正确添加了 src 目录到 Python 路径
2. **依赖管理**: 测试依赖的外部库需要在运行环境中安装
3. **GPU 测试**: 某些测试需要 GPU 支持，在没有 GPU 的环境中会跳过
4. **数据文件**: 某些测试需要测试数据文件，确保路径正确

## 🤝 贡献指南

添加新测试时请：

1. 遵循命名规范
2. 编写清晰的测试用例
3. 添加必要的文档字符串
4. 确保测试可以独立运行
5. 更新本 README 文档

---

*最后更新: 2024年6月22日* 