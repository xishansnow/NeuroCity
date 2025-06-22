# PyTorch Lightning 在 NeuroCity 项目中的使用指南

## 概述

PyTorch Lightning 是一个轻量级的 PyTorch 包装器，可以帮助研究人员和工程师更高效地构建和训练深度学习模型。在这个 NeRF 项目中，我们集成了 PyTorch Lightning 来简化训练过程并提供更好的实验管理。

## 🚀 主要优势

1. **简化的训练循环** - 自动处理前向传播、反向传播、优化器步进等
2. **分布式训练支持** - 轻松扩展到多 GPU/多节点训练
3. **自动日志记录** - 与 TensorBoard、W&B 等无缝集成
4. **检查点管理** - 自动保存和恢复训练状态
5. **早停和调度** - 内置早停、学习率调度等功能
6. **实验管理** - 便于超参数调优和实验对比
7. **混合精度训练** - 自动 FP16 训练以提高效率
8. **自定义回调** - 灵活的扩展机制

## 📦 安装依赖

```bash
pip install pytorch-lightning torchmetrics tensorboard wandb
```

或者使用项目的 requirements.txt：

```bash
pip install -r requirements.txt
```

## 🏗️ 架构设计

### 1. Lightning Module (`SVRasterLightningModule`)

继承自 `pl.LightningModule`，包含：
- 模型定义和初始化
- 训练/验证步骤
- 损失计算和指标追踪
- 优化器和调度器配置
- 自定义训练逻辑（如体素细分、剪枝）

### 2. 配置系统 (`SVRasterLightningConfig`)

统一管理所有训练相关的超参数：
```python
@dataclass
class SVRasterLightningConfig:
    learning_rate: float = 1e-3
    optimizer_type: str = "adamw"
    scheduler_type: str = "cosine"
    enable_subdivision: bool = True
    enable_pruning: bool = True
    use_ema: bool = True
    # ... 更多配置选项
```

### 3. 工厂函数

提供便捷的创建和训练函数：
- `create_lightning_trainer()` - 创建配置好的训练器
- `train_svraster_lightning()` - 一键启动训练

## 🎯 使用示例

### 基础训练

```python
from src.svraster.core import SVRasterConfig
from src.svraster.lightning_trainer import (
    SVRasterLightningConfig, 
    train_svraster_lightning
)

# 模型配置
model_config = SVRasterConfig(
    max_octree_levels=12,
    base_resolution=64,
    scene_bounds=(-2.0, -2.0, -2.0, 2.0, 2.0, 2.0)
)

# Lightning 配置
lightning_config = SVRasterLightningConfig(
    model_config=model_config,
    learning_rate=1e-3,
    optimizer_type="adamw",
    scheduler_type="cosine",
    enable_subdivision=True,
    enable_pruning=True,
    use_ema=True
)

# 开始训练
trained_model = train_svraster_lightning(
    model_config=model_config,
    lightning_config=lightning_config,
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    max_epochs=100,
    gpus=1
)
```

### 高级配置

```python
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

# 高级配置
lightning_config = SVRasterLightningConfig(
    learning_rate=5e-4,
    weight_decay=1e-5,
    optimizer_type="adamw",
    scheduler_type="cosine",
    use_ema=True,
    ema_decay=0.9999,
    gradient_clip_val=0.5
)

# 自定义回调
callbacks = [
    ModelCheckpoint(
        monitor="val/psnr",
        mode="max",
        save_top_k=3,
        filename="svraster-{epoch:02d}-{val/psnr:.3f}"
    ),
    EarlyStopping(
        monitor="val/psnr",
        patience=20,
        mode="max"
    )
]

# 创建训练器
trainer = pl.Trainer(
    max_epochs=200,
    devices=2,  # 多 GPU 训练
    precision="16-mixed",
    callbacks=callbacks,
    logger=TensorBoardLogger("logs", name="svraster_experiment")
)
```

### 分布式训练

```python
# 多 GPU 分布式训练
trainer = pl.Trainer(
    max_epochs=100,
    devices=4,  # 使用 4 个 GPU
    strategy="ddp",  # 分布式数据并行
    precision="16-mixed",
    sync_batchnorm=True
)

# 多节点训练
trainer = pl.Trainer(
    max_epochs=100,
    devices=8,
    num_nodes=2,  # 2 个节点
    strategy="ddp"
)
```

## 🎛️ 核心功能

### 1. 自动混合精度训练

```python
trainer = pl.Trainer(
    precision="16-mixed",  # 自动 FP16 训练
    # 或者
    # precision = "bf16-mixed",  # BFloat16 训练（更稳定）
)
```

### 2. 梯度裁剪

```python
lightning_config = SVRasterLightningConfig(
    gradient_clip_val=1.0,  # 梯度范数裁剪
    gradient_clip_algorithm="norm"
)
```

### 3. EMA (指数移动平均)

```python
lightning_config = SVRasterLightningConfig(
    use_ema=True,
    ema_decay=0.999  # EMA 衰减率
)
```

### 4. 自适应体素细分和剪枝

```python
lightning_config = SVRasterLightningConfig(
    enable_subdivision=True,
    subdivision_start_epoch=10,
    subdivision_interval=5,
    subdivision_threshold=0.01,
    
    enable_pruning=True,
    pruning_start_epoch=20,
    pruning_interval=10,
    pruning_threshold=0.001
)
```

## 📊 日志记录和监控

### TensorBoard 集成

```python
from pytorch_lightning.loggers import TensorBoardLogger

logger = TensorBoardLogger(
    save_dir="logs",
    name="svraster_experiment",
    version="v1.0"
)

trainer = pl.Trainer(logger=logger)
```

查看日志：
```bash
tensorboard --logdir logs
```

### W&B 集成

```python
from pytorch_lightning.loggers import WandbLogger

logger = WandbLogger(
    project="neurocity-svraster",
    name="experiment_1",
    tags=["svraster", "nerf", "voxels"]
)
```

### 自动记录的指标

- 训练/验证损失
- PSNR、SSIM、LPIPS
- 学习率变化
- 体素统计信息
- 梯度范数
- 训练时间

## 🔧 自定义回调

### 体素统计回调

```python
class VoxelStatisticsCallback(pl.Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % 10 == 0:
            stats = pl_module.model.get_voxel_statistics()
            for key, value in stats.items():
                trainer.logger.experiment.add_scalar(
                    f"voxel_stats/{key}", value, trainer.current_epoch
                )
```

### 渲染回调

```python
class RenderingCallback(pl.Callback):
    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % 20 == 0:
            # 渲染测试图像并保存
            self.render_test_images(pl_module, trainer.current_epoch)
```

## 🧪 实验管理

### 超参数扫描

```python
# 定义实验配置
experiments = [
    {"lr": 1e-3, "optimizer": "adam", "subdivision_thresh": 0.01},
    {"lr": 5e-4, "optimizer": "adamw", "subdivision_thresh": 0.005},
    {"lr": 1e-3, "optimizer": "adamw", "subdivision_thresh": 0.001},
]

# 运行所有实验
for i, exp in enumerate(experiments):
    config = SVRasterLightningConfig(
        learning_rate=exp["lr"],
        optimizer_type=exp["optimizer"],
        subdivision_threshold=exp["subdivision_thresh"]
    )
    
    trainer = create_lightning_trainer(
        config, train_dataset, val_dataset,
        experiment_name=f"exp_{i}",
        max_epochs=100
    )
```

### 检查点管理

```python
# 自动保存最佳模型
checkpoint_callback = ModelCheckpoint(
    monitor="val/psnr",
    mode="max",
    save_top_k=3,
    filename="best-{epoch:02d}-{val/psnr:.3f}",
    save_last=True
)

# 从检查点恢复训练
trainer = pl.Trainer(resume_from_checkpoint="checkpoints/last.ckpt")
```

## 🚀 性能优化

### 1. 数据加载优化

```python
train_loader = DataLoader(
    train_dataset,
    batch_size=1,
    num_workers=8,  # 增加工作进程
    pin_memory=True,  # 固定内存
    persistent_workers=True  # 持久化工作进程
)
```

### 2. 编译优化

```python
# PyTorch 2.0 编译加速
lightning_module = torch.compile(lightning_module)
```

### 3. 混合精度训练

```python
trainer = pl.Trainer(
    precision="16-mixed",
    # 或者更激进的设置
    # precision = "bf16-mixed"
)
```

## 📈 监控和调试

### 1. 性能分析

```python
from pytorch_lightning.profilers import PyTorchProfiler

profiler = PyTorchProfiler(
    dirpath="profiler_logs",
    filename="profile",
    export_to_chrome=True
)

trainer = pl.Trainer(profiler=profiler)
```

### 2. 模型摘要

```python
trainer = pl.Trainer(
    enable_model_summary=True,
    max_epochs=1,
    limit_train_batches=1
)
```

### 3. 调试模式

```python
# 快速调试
trainer = pl.Trainer(
    fast_dev_run=True,  # 只运行一个 batch
    # 或者
    limit_train_batches=0.1,  # 只使用 10%的训练数据
    limit_val_batches=0.1
)
```

## 🔄 模型部署

### 推理模式

```python
# 从检查点加载模型
model = SVRasterLightningModule.load_from_checkpoint("checkpoints/best.ckpt")
model.eval()

# 提取核心模型用于部署
core_model = model.model

# 推理
with torch.no_grad():
    outputs = core_model(ray_origins, ray_directions)
```

### 导出模型

```python
# 导出为 TorchScript
scripted_model = torch.jit.script(model.model)
scripted_model.save("svraster_model.pt")

# 导出为 ONNX
torch.onnx.export(
    model.model,
    (ray_origins, ray_directions),
    "svraster_model.onnx"
)
```

## 🤝 与现有代码集成

### 1. 渐进式迁移

可以逐步将现有的训练器迁移到 Lightning：

```python
# 保留现有的 SVRasterTrainer 用于向后兼容
from src.svraster.trainer import SVRasterTrainer  # 原始训练器
from src.svraster.lightning_trainer import SVRasterLightningModule  # Lightning 版本

# 根据需求选择使用哪个训练器
use_lightning = True
if use_lightning:
    trainer = create_lightning_trainer(...)
else:
    trainer = SVRasterTrainer(...)
```

### 2. 共享配置

可以在 Lightning 和传统训练器之间共享模型配置：

```python
# 共享的模型配置
model_config = SVRasterConfig(...)

# Lightning 训练
lightning_config = SVRasterLightningConfig(model_config=model_config)

# 传统训练
trainer_config = SVRasterTrainerConfig(...)
```

## 📚 最佳实践

1. **模型验证**：使用 `fast_dev_run=True` 快速验证代码
2. **内存管理**：合理设置 `batch_size` 和 `num_workers`
3. **日志管理**：定期清理日志文件，避免磁盘空间不足
4. **检查点策略**：保存多个最佳模型，防止意外丢失
5. **实验命名**：使用有意义的实验名称和标签
6. **代码版本控制**：将超参数配置纳入版本控制

## 🐛 常见问题

### Q: 为什么训练比原来慢了？
A: 检查是否启用了不必要的日志记录或回调，适当调整 `log_every_n_steps`。

### Q: 如何在多 GPU 上正确使用 BatchNorm？
A: 使用 `sync_batchnorm=True` 确保 BatchNorm 统计量在 GPU 间同步。

### Q: 检查点文件太大怎么办？
A: 只保存必要的状态，或者调整 `save_top_k` 参数。

### Q: 如何调试分布式训练？
A: 先在单 GPU 上验证，再扩展到多 GPU；使用 `strategy="dp"` 而不是 `"ddp"` 进行调试。

## 📖 相关资源

- [PyTorch Lightning 官方文档](https://pytorch-lightning.readthedocs.io/)
- [Lightning Bolts 模型库](https://pytorch-lightning.readthedocs.io/en/stable/ecosystem/bolts.html)
- [W&B + Lightning 集成指南](https://docs.wandb.ai/guides/integrations/lightning)
- [TensorBoard 使用指南](https://pytorch.org/docs/stable/tensorboard.html)

## 🌟 支持的模型

本项目已为多个 NeRF 模型创建了 PyTorch Lightning 版本：

### 1. SVRaster - 自适应稀疏体素光栅化
```python
from src.svraster.lightning_trainer import train_svraster_lightning
trained_model = train_svraster_lightning(model_config, lightning_config, ...)
```
- ✅ 自适应体素细分和剪枝
- ✅ EMA 模型更新
- ✅ 稀疏体素优化

### 2. Grid-NeRF - 大规模城市场景
```python
from src.grid_nerf.lightning_trainer import train_grid_nerf_lightning
trained_model = train_grid_nerf_lightning(model_config, lightning_config, ...)
```
- ✅ 多分辨率网格表示
- ✅ 分层网格特征管理
- ✅ 大规模场景优化

### 3. Instant-NGP - 快速哈希编码
```python
from src.instant_ngp.lightning_trainer import train_instant_ngp_lightning
trained_model = train_instant_ngp_lightning(model_config, lightning_config, ...)
```
- ✅ 多分辨率哈希编码
- ✅ 自适应射线采样
- ✅ 哈希网格优化

### 4. MIP-NeRF - 抗锯齿积分位置编码
```python
from src.mip_nerf.lightning_trainer import train_mip_nerf_lightning
trained_model = train_mip_nerf_lightning(model_config, lightning_config, ...)
```
- ✅ 圆锥视锥表示
- ✅ 积分位置编码
- ✅ 分层采样优化

## 🔄 模型对比和选择

### 性能特点对比

| 模型 | 训练速度 | 渲染质量 | 内存使用 | 适用场景 |
|------|----------|----------|----------|----------|
| **Instant-NGP** | ⚡⚡⚡⚡ | ⭐⭐⭐ | 💾💾 | 快速原型、实时应用 |
| **MIP-NeRF** | ⚡⚡ | ⭐⭐⭐⭐ | 💾💾💾 | 高质量渲染、抗锯齿 |
| **Grid-NeRF** | ⚡⚡⚡ | ⭐⭐⭐⭐ | 💾💾💾💾 | 大规模城市场景 |
| **SVRaster** | ⚡⚡ | ⭐⭐⭐⭐⭐ | 💾💾 | 自适应场景、高精度 |

### 使用建议

- **快速实验和原型** → Instant-NGP
- **高质量渲染需求** → MIP-NeRF 或 SVRaster
- **大规模城市场景** → Grid-NeRF
- **内存受限环境** → Instant-NGP 或 SVRaster
- **需要抗锯齿效果** → MIP-NeRF

## 🎯 统一训练接口

```python
# 统一的训练示例
def train_any_model(model_type: str):
    if model_type == "svraster":
        from src.svraster.lightning_trainer import train_svraster_lightning
        return train_svraster_lightning(...)
    elif model_type == "grid_nerf":
        from src.grid_nerf.lightning_trainer import train_grid_nerf_lightning
        return train_grid_nerf_lightning(...)
    elif model_type == "instant_ngp":
        from src.instant_ngp.lightning_trainer import train_instant_ngp_lightning
        return train_instant_ngp_lightning(...)
    elif model_type == "mip_nerf":
        from src.mip_nerf.lightning_trainer import train_mip_nerf_lightning
        return train_mip_nerf_lightning(...)
```

## 📝 总结

通过使用 PyTorch Lightning，您可以专注于模型的核心逻辑，而将训练的基础设施交给 Lightning 处理，从而提高开发效率和代码质量。

主要优势：
- 🚀 **多模型支持** - 统一接口支持多种 NeRF 变体
- ⚡ **自动优化** - 混合精度、分布式训练等自动处理
- 📊 **丰富监控** - 集成各种日志记录和可视化工具
- 🔧 **灵活配置** - 支持复杂的超参数调优和实验管理
- 🎯 **专注算法** - 让研究人员专注于模型创新

通过 Lightning，NeuroCity 项目具备了产业级的训练能力，为大规模 NeRF 应用奠定了坚实基础。 