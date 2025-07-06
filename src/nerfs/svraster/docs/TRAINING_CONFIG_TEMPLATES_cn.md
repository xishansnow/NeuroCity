# SVRaster 训练配置模板与实例

## 概述

本文档提供了 SVRaster 训练的完整配置模板和实际使用示例，帮助用户快速设置和启动训练任务。包含了不同场景下的最佳实践配置、常见问题解决方案以及性能优化建议。

## 🎯 快速配置模板

### 1. 基础训练配置

```yaml
# basic_training_config.yaml
# 适用于初学者和快速验证

model:
  # 基础模型参数
  voxel_resolution: 256                    # 体素网格分辨率
  max_voxel_count: 1000000                # 最大体素数量
  feature_channels: 16                     # 特征通道数
  color_channels: 3                        # 颜色通道数
  opacity_channels: 1                      # 透明度通道数
  
  # 体素细分参数
  enable_subdivision: true                 # 启用自适应细分
  max_subdivision_level: 6                 # 最大细分层级
  subdivision_threshold: 0.01              # 细分阈值
  
  # 球谐函数参数
  sh_degree: 2                            # 球谐函数阶数
  enable_sh: true                         # 启用球谐函数

training:
  # 基础训练参数
  epochs: 500                             # 训练轮数
  batch_size: 2                           # 批次大小
  learning_rate: 1e-3                     # 学习率
  device: "cuda"                          # 训练设备
  
  # 优化器设置
  optimizer: "adam"                       # 优化器类型
  weight_decay: 1e-6                      # 权重衰减
  eps: 1e-8                              # 数值稳定性参数
  
  # 学习率调度
  scheduler: "cosine"                     # 调度器类型
  warmup_epochs: 50                       # 预热轮数
  min_lr: 1e-6                           # 最小学习率

dataset:
  # 数据集参数
  data_dir: "./data/nerf_synthetic/lego"  # 数据目录
  image_width: 800                        # 图像宽度
  image_height: 800                       # 图像高度
  camera_angle_x: 0.6911112               # 相机水平视角
  
  # 数据分割
  train_split: 0.8                        # 训练集比例
  val_split: 0.1                          # 验证集比例
  test_split: 0.1                         # 测试集比例
  
  # 数据增强
  enable_augmentation: false              # 启用数据增强
  background_color: [1.0, 1.0, 1.0]      # 背景颜色

loss:
  # 损失函数权重
  rgb_weight: 1.0                         # RGB 重建损失权重
  depth_weight: 0.1                       # 深度损失权重
  ssim_weight: 0.1                        # SSIM 损失权重
  opacity_reg_weight: 0.01                # 透明度正则化权重
  spatial_reg_weight: 0.1                 # 空间正则化权重

monitoring:
  # 监控设置
  log_interval: 10                        # 日志输出间隔
  save_interval: 100                      # 模型保存间隔
  eval_interval: 50                       # 评估间隔
  enable_tensorboard: true                # 启用 TensorBoard
  checkpoint_dir: "./checkpoints"         # 检查点目录
```

### 2. 高性能训练配置

```yaml
# high_performance_config.yaml
# 适用于高端 GPU 和大规模场景

model:
  # 高分辨率设置
  voxel_resolution: 1024                  # 高分辨率体素网格
  max_voxel_count: 10000000               # 更多体素支持
  feature_channels: 32                     # 更多特征通道
  
  # 高级细分策略
  enable_subdivision: true
  max_subdivision_level: 10               # 更深层级细分
  subdivision_threshold: 0.005            # 更严格的细分阈值
  subdivision_interval: 20                # 细分检查间隔
  
  # 高阶球谐函数
  sh_degree: 4                            # 高阶球谐函数
  enable_view_dependent: true             # 启用视角相关渲染

training:
  # 高性能训练设置
  epochs: 2000                            # 更多训练轮数
  batch_size: 8                           # 更大批次
  learning_rate: 5e-4                     # 调优学习率
  
  # 混合精度训练
  enable_mixed_precision: true            # 启用混合精度
  gradient_clip_norm: 1.0                 # 梯度裁剪
  
  # 多 GPU 支持
  distributed: true                       # 分布式训练
  num_gpus: 4                            # GPU 数量
  sync_batchnorm: true                   # 同步批归一化

dataset:
  # 高分辨率数据
  image_width: 1920                       # 高分辨率图像
  image_height: 1080
  
  # 数据加载优化
  num_workers: 8                          # 更多数据加载进程
  pin_memory: true                        # 内存固定
  prefetch_factor: 4                      # 预取因子

loss:
  # 增强损失函数
  rgb_weight: 1.0
  depth_weight: 0.2                       # 增强深度约束
  ssim_weight: 0.2                        # 增强结构相似性
  perceptual_weight: 0.1                  # 感知损失
  distortion_weight: 0.01                 # 畸变损失
  
  # 自适应权重
  enable_adaptive_weights: true           # 启用自适应权重调整
  weight_update_interval: 100             # 权重更新间隔

monitoring:
  # 详细监控
  log_interval: 5                         # 更频繁的日志
  detailed_metrics: true                  # 详细指标记录
  memory_monitoring: true                 # 内存监控
  profile_training: true                  # 性能分析
```

### 3. 内存优化配置

```yaml
# memory_optimized_config.yaml
# 适用于 GPU 内存有限的环境

model:
  # 内存友好设置
  voxel_resolution: 128                   # 降低分辨率
  max_voxel_count: 500000                 # 限制体素数量
  feature_channels: 8                     # 减少特征通道
  
  # 渐进式细分
  enable_subdivision: true
  max_subdivision_level: 4                # 限制细分层级
  progressive_subdivision: true           # 渐进式细分
  
  # 内存优化策略
  enable_voxel_pruning: true             # 启用体素剪枝
  pruning_threshold: 0.01                # 剪枝阈值
  memory_efficient_mode: true           # 内存高效模式

training:
  # 小批次训练
  batch_size: 1                          # 最小批次
  gradient_accumulation_steps: 4         # 梯度累积
  
  # 检查点策略
  checkpoint_every_n_epochs: 25          # 频繁保存检查点
  keep_only_latest_checkpoint: true      # 只保留最新检查点
  
  # 内存管理
  empty_cache_interval: 10               # 清理缓存间隔
  max_memory_usage: 0.8                  # 最大内存使用率

dataset:
  # 内存友好数据加载
  image_width: 400                       # 降低图像分辨率
  image_height: 400
  num_workers: 2                         # 减少数据加载进程
  pin_memory: false                      # 关闭内存固定

loss:
  # 简化损失函数
  rgb_weight: 1.0
  ssim_weight: 0.1                       # 保持关键损失
  disable_heavy_losses: true             # 禁用计算密集的损失
```

## 🛠️ 实际使用示例

### 1. Python API 使用示例

```python
# train_svraster_example.py
import torch
import yaml
from pathlib import Path

from src.nerfs.svraster import (
    SVRasterModel, SVRasterConfig, 
    SVRasterTrainer, SVRasterTrainerConfig,
    SVRasterDataset, SVRasterDatasetConfig
)

def load_config(config_path: str) -> dict:
    """加载 YAML 配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def create_configs(config_dict: dict):
    """从配置字典创建配置对象"""
    
    # 模型配置
    model_config = SVRasterConfig(
        voxel_resolution=config_dict['model']['voxel_resolution'],
        max_voxel_count=config_dict['model']['max_voxel_count'],
        feature_channels=config_dict['model']['feature_channels'],
        enable_subdivision=config_dict['model']['enable_subdivision'],
        max_subdivision_level=config_dict['model']['max_subdivision_level'],
        sh_degree=config_dict['model']['sh_degree']
    )
    
    # 数据集配置
    dataset_config = SVRasterDatasetConfig(
        data_dir=config_dict['dataset']['data_dir'],
        image_width=config_dict['dataset']['image_width'],
        image_height=config_dict['dataset']['image_height'],
        camera_angle_x=config_dict['dataset']['camera_angle_x'],
        train_split=config_dict['dataset']['train_split'],
        val_split=config_dict['dataset']['val_split']
    )
    
    # 训练器配置
    trainer_config = SVRasterTrainerConfig(
        epochs=config_dict['training']['epochs'],
        batch_size=config_dict['training']['batch_size'],
        learning_rate=config_dict['training']['learning_rate'],
        device=config_dict['training']['device'],
        optimizer=config_dict['training']['optimizer'],
        scheduler=config_dict['training']['scheduler'],
        loss_config=config_dict['loss'],
        monitoring_config=config_dict['monitoring']
    )
    
    return model_config, dataset_config, trainer_config

def main():
    # 1. 加载配置
    config_path = "basic_training_config.yaml"
    config_dict = load_config(config_path)
    model_config, dataset_config, trainer_config = create_configs(config_dict)
    
    # 2. 创建数据集
    train_dataset = SVRasterDataset(dataset_config, split="train")
    val_dataset = SVRasterDataset(dataset_config, split="val")
    
    print(f"训练集大小：{len(train_dataset)}")
    print(f"验证集大小：{len(val_dataset)}")
    
    # 3. 创建训练器
    trainer = SVRasterTrainer(
        model_config=model_config,
        trainer_config=trainer_config,
        train_dataset=train_dataset,
        val_dataset=val_dataset
    )
    
    # 4. 开始训练
    print("开始训练。..")
    trainer.train()
    
    # 5. 保存模型
    trainer.save_checkpoint("final_model.pth")
    print("训练完成，模型已保存")

if __name__ == "__main__":
    main()
```

### 2. 命令行训练示例

```bash
#!/bin/bash
# train_svraster.sh

# 基础训练
python train_svraster.py \
    --config basic_training_config.yaml \
    --output_dir ./outputs/basic_training \
    --log_level INFO

# 高性能训练（多 GPU）
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    train_svraster.py \
    --config high_performance_config.yaml \
    --output_dir ./outputs/high_performance_training \
    --distributed

# 恢复训练
python train_svraster.py \
    --config basic_training_config.yaml \
    --resume_from ./checkpoints/svraster-epoch-100.pth \
    --output_dir ./outputs/resumed_training

# 内存优化训练
python train_svraster.py \
    --config memory_optimized_config.yaml \
    --output_dir ./outputs/memory_optimized_training \
    --enable_memory_profiling
```

## 🔧 常见配置问题解决方案

### 1. GPU 内存不足

**问题症状**：
```
RuntimeError: CUDA out of memory. Tried to allocate X MB
```

**解决方案**：
```yaml
# 调整以下参数
model:
  voxel_resolution: 128        # 降低分辨率 (512 -> 128)
  max_voxel_count: 500000      # 减少体素数量
  feature_channels: 8          # 减少特征通道

training:
  batch_size: 1                # 减小批次大小
  gradient_accumulation_steps: 4  # 使用梯度累积
```

### 2. 训练不收敛

**问题症状**：
- 损失不下降或震荡
- 渲染质量差

**解决方案**：
```yaml
training:
  learning_rate: 1e-4          # 降低学习率
  warmup_epochs: 100           # 增加预热

loss:
  rgb_weight: 1.0              # 保持 RGB 损失权重
  ssim_weight: 0.2             # 增加 SSIM 权重
  spatial_reg_weight: 0.2      # 增加空间正则化

model:
  subdivision_threshold: 0.001  # 降低细分阈值
```

### 3. 训练速度慢

**问题症状**：
- 每个 epoch 耗时过长
- GPU 利用率低

**解决方案**：
```yaml
dataset:
  num_workers: 8               # 增加数据加载进程
  pin_memory: true             # 启用内存固定
  prefetch_factor: 4           # 增加预取

training:
  enable_mixed_precision: true  # 启用混合精度
  
model:
  memory_efficient_mode: false # 关闭内存高效模式（如果内存充足）
```

### 4. 渲染质量不理想

**问题症状**：
- 图像模糊
- 细节丢失
- 伪影明显

**解决方案**：
```yaml
model:
  voxel_resolution: 512        # 提高分辨率
  max_subdivision_level: 8     # 增加细分层级
  sh_degree: 3                 # 提高球谐函数阶数

loss:
  perceptual_weight: 0.1       # 添加感知损失
  distortion_weight: 0.01      # 添加畸变损失

training:
  epochs: 1500                 # 增加训练轮数
```

## 📊 性能基准和建议

### 硬件配置建议

| 场景类型 | GPU 内存 | 推荐配置 | 预期性能 |
|----------|---------|----------|----------|
| **学习测试** | 8GB | 基础配置 | 2-4 小时 |
| **研究开发** | 16GB | 高性能配置 | 4-8 小时 |
| **生产部署** | 24GB+ | 高性能+多 GPU | 1-2 小时 |

### 配置优化建议

1. **首次使用**：从基础配置开始，逐步调优
2. **内存有限**：使用内存优化配置，启用梯度累积
3. **追求质量**：使用高性能配置，增加训练轮数
4. **快速迭代**：降低分辨率，减少训练轮数

## 🔗 相关文档

- **[训练机制详解索引](./TRAINING_INDEX_cn.md)** - 完整的训练文档导航
- **[训练机制详解 - 第一部分](./TRAINING_DETAILS_PART1_cn.md)** - 基础训练架构
- **[训练机制详解 - 第二部分](./TRAINING_DETAILS_PART2_cn.md)** - 自适应技术
- **[训练机制详解 - 第三部分](./TRAINING_DETAILS_PART3_cn.md)** - 损失函数与监控

---

**提示**：建议根据实际硬件条件和需求选择合适的配置模板，并根据训练效果逐步调优参数。
