# Mega-NeRF 中文文档

Mega-NeRF 是面向大规模神经场景重建与渲染的高效、可扩展实现，采用模块化设计，适合科研与工程应用。

## 简介

Mega-NeRF 通过空间分块、分布式训练等机制，实现对超大场景的高效神经渲染。该实现结构清晰，易于扩展和二次开发。

## 主要特性
- **模块化架构**：核心、训练器、渲染器、数据集、工具函数完全解耦，便于维护和扩展。
- **高效分块**：支持空间分区与分布式训练，适应大规模场景。
- **快速渲染**：支持批量与视频渲染，分块处理大幅提升效率。
- **灵活数据支持**：兼容 NeRF 标准数据集及自定义数据。
- **混合精度与多卡**：支持 AMP 及多 GPU 分布式训练。
- **完备测试**：全模块测试用例，保障代码质量。

## 目录结构

```
src/nerfs/mega_nerf/
├── __init__.py           # 主 API 与导出
├── core.py               # 核心模型与配置
├── trainer.py            # 训练逻辑与配置
├── renderer.py           # 渲染逻辑与配置
├── dataset.py            # 数据集加载与相机处理
├── utils/                # 工具函数（分区、评测等）
```

## 快速上手

### 安装依赖
```bash
conda activate neurocity
pip install -r requirements.txt
```

### 训练示例
```python
from src.nerfs.mega_nerf import MegaNeRF, MegaNeRFConfig, MegaNeRFTrainer, MegaNeRFTrainerConfig

model_config = MegaNeRFConfig()
trainer_config = MegaNeRFTrainerConfig()

model = MegaNeRF(model_config)
trainer = MegaNeRFTrainer(model, trainer_config)

trainer.train()
```

### 渲染示例
```python
from src.nerfs.mega_nerf import MegaNeRFRenderer, MegaNeRFRendererConfig

renderer_config = MegaNeRFRendererConfig()
renderer = MegaNeRFRenderer(model, renderer_config)

image = renderer.render_image(camera_pose, intrinsics)
```

### 测试方法
```bash
python tests/nerfs/mega_nerf/run_tests.py
```

## 依赖环境
- Python 3.10 及以上
- PyTorch 2.0 及以上
- NumPy 1.20 及以上
- pytest 7.0 及以上
- CUDA（可选，支持 GPU）

## 贡献指南
- 遵循模块化与类型注解规范。
- 新增功能需补充/更新测试。
- 所有公开 API 需完善文档。
- 测试规范详见 `tests/nerfs/mega_nerf/README.md`。

## 引用
如在学术或工程项目中使用本代码，请引用原 Mega-NeRF 论文及本仓库：

```
@article{turki2022megenerf,
  title={Mega-NeRF: Scalable Construction of Large-Scale NeRFs for Virtual Fly-Throughs},
  author={Turki, H and others},
  journal={arXiv preprint arXiv:2112.10703},
  year={2022}
}
```

## 许可证
本项目为 NeuroCity 套件的一部分，遵循主仓库相同的开源协议。 