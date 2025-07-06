# NeuroCity 项目架构概览

## 🏗️ 项目整体结构

NeuroCity 是一个专注于神经辐射场（NeRF）技术的 3D 场景重建项目，提供了多种先进的 NeRF 实现和实用工具。

```
NeuroCity/
├── 📁 src/                          # 核心源代码
│   ├── 📁 nerfs/                    # NeRF 实现集合
│   │   ├── 📁 svraster/             # SVRaster (主要实现)
│   │   ├── 📁 mega_nerf/            # Mega-NeRF 实现
│   │   ├── 📁 instant_ngp/          # Instant NGP 实现
│   │   └── 📁 ...                   # 其他 NeRF 变体
│   ├── 📁 utils/                    # 通用工具模块
│   ├── 📁 data/                     # 数据处理模块
│   └── 📁 visualization/            # 可视化工具
├── 📁 demos/                        # 演示脚本
├── 📁 examples/                     # 使用示例
├── 📁 tests/                        # 测试代码
├── 📁 data/                         # 数据集
├── 📁 outputs/                      # 输出结果
├── 📁 checkpoints/                  # 模型检查点
└── 📁 docs/                         # 文档（如果有）
```

## 🎯 核心模块：SVRaster

SVRaster (Sparse Voxel Rasterization) 是本项目的主要实现，采用稀疏体素光栅化技术，实现高效的 3D 场景渲染。

### SVRaster 架构设计

```
src/nerfs/svraster/
├── 🔧 core.py                       # 核心模型实现
├── 🏃 trainer.py                    # 训练器（纯 PyTorch）
├── 🎨 renderer.py                   # 渲染器（推理专用）
├── 📊 __init__.py                   # 模块导出
├── 📚 完整文档索引.md                 # 技术文档入口
├── 🏗️ 渲染机制文档/                  # 渲染原理详解
├── 🎓 训练机制文档/                  # 训练机制详解
├── 📋 兼容性与分析文档/               # 技术分析报告
└── ⚙️ 配置与工具文档/                # 实用指南
```

### 设计原则

1. **职责分离**：训练器专注训练，渲染器专注推理
2. **模块化设计**：清晰的模块边界，便于维护和扩展
3. **无依赖耦合**：移除 Lightning 依赖，使用纯 PyTorch
4. **兼容性优先**：确保 Python 3.10 完全兼容
5. **文档驱动**：完善的技术文档支持学习和开发

## 🚀 快速开始

### 1. 环境配置

```bash
# 创建 Python 3.10 虚拟环境
python3.10 -m venv venv
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt

# 可选：安装 OpenVDB 支持
bash install_dependencies.sh
```

### 2. 训练模型

```python
from src.nerfs.svraster import SVRasterConfig, SVRasterTrainer

# 配置训练参数
config = SVRasterConfig(
    scene_name="my_scene",
    max_iterations=50000,
    learning_rate=0.001
)

# 创建训练器
trainer = SVRasterTrainer(config)

# 开始训练
trainer.train()
```

### 3. 渲染推理

```python
from src.nerfs.svraster import SVRasterRenderer, SVRasterRendererConfig

# 配置渲染器
render_config = SVRasterRendererConfig(
    image_width=800,
    image_height=600,
    quality_level="high"
)

# 创建渲染器
renderer = SVRasterRenderer(render_config)

# 加载训练好的模型
renderer.load_model("checkpoints/my_scene/latest.pth")

# 渲染单个视角
result = renderer.render_single_view(camera_pose, intrinsics)
```

## 📚 学习路径

### 初学者路径

1. **基础概念**：阅读 `src/nerfs/svraster/COMPLETE_DOCUMENTATION_INDEX_cn.md`
2. **快速上手**：运行 `demos/demo_svraster_renderer.py`
3. **理解原理**：学习渲染机制文档系列
4. **实践训练**：使用训练机制文档指导

### 开发者路径

1. **架构理解**：阅读 `DOCUMENTATION_VS_SOURCE_ANALYSIS_cn.md`
2. **代码兼容性**：参考 `PYTHON310_COMPATIBILITY_cn.md`
3. **渲染器设计**：深入学习 `RENDERER_DESIGN_cn.md`
4. **扩展开发**：基于现有架构添加新功能

### 研究者路径

1. **技术原理**：深入渲染机制文档的三个部分
2. **训练策略**：研究训练机制文档的细节
3. **性能优化**：分析 CUDA 优化和性能调优
4. **创新实验**：使用提供的工具进行算法改进

## 🔍 主要特性

### SVRaster 核心特性

- **稀疏体素表示**：高效的3D场景表示方法
- **自适应细分**：智能的体素细分策略
- **CUDA 优化**：高性能并行渲染
- **渐进式训练**：多尺度训练策略
- **实时渲染**：支持交互式渲染应用

### 工程特性

- **纯 PyTorch 实现**：无额外框架依赖
- **模块化架构**：清晰的代码组织
- **完整文档**：详细的技术文档支持
- **Python 3.10 兼容**：稳定的环境支持
- **丰富示例**：多种使用场景演示

## 🛠️ 开发指南

### 代码规范

- 使用 `typing` 模块进行类型注解
- 添加 `from __future__ import annotations` 支持前向引用
- 遵循 PEP 8 代码风格
- 编写清晰的文档字符串

### 测试要求

- 运行兼容性测试：`python test_python310_compatibility.py`
- 执行单元测试：`pytest tests/`
- 类型检查：`mypy src/`

### 贡献流程

1. Fork 项目并创建特性分支
2. 实现功能并添加测试
3. 确保通过所有测试和检查
4. 提交 Pull Request 并描述改动

## 📖 文档资源

### 核心技术文档

- [SVRaster 完整文档索引](src/nerfs/svraster/COMPLETE_DOCUMENTATION_INDEX_cn.md)
- [渲染机制详解系列](src/nerfs/svraster/RENDERING_INDEX_cn.md)
- [训练机制详解系列](src/nerfs/svraster/TRAINING_INDEX_cn.md)

### 设计与分析

- [渲染器设计说明](src/nerfs/svraster/RENDERER_DESIGN_cn.md)
- [文档与源码分析](src/nerfs/svraster/DOCUMENTATION_VS_SOURCE_ANALYSIS_cn.md)
- [Python 3.10 兼容性报告](src/nerfs/svraster/PYTHON310_COMPATIBILITY_cn.md)

### 实用工具

- [演示脚本集合](demos/README.md)
- [使用示例](examples/README.md)
- [性能分析工具](demos/performance_comparison.py)

## ❓ 常见问题

### Q: 为什么选择纯 PyTorch 而不是 Lightning？

A: 为了减少依赖复杂性，提高代码可控性，并便于在不同环境中部署。详见文档分析。

### Q: 如何确保 Python 3.10 兼容性？

A: 使用 `typing` 模块语法，避免 3.9+ 新特性，运行兼容性测试脚本验证。

### Q: 渲染器和训练器的区别是什么？

A: 训练器专注模型训练和优化，渲染器专注模型加载和推理渲染，职责清晰分离。

### Q: 如何选择质量级别？

A: 根据硬件性能和质量需求选择：low（快速预览）、medium（平衡）、high（推荐）、ultra（最高质量）。

## 📞 支持与反馈

如有问题或建议，请：

1. 查阅相关技术文档
2. 运行示例脚本验证
3. 检查兼容性和测试结果
4. 提交 Issue 或 Pull Request

---

**项目愿景**：成为最易用、最完整、最高性能的 NeRF 技术实现，为 3D 场景重建和渲染提供强大支持。
