# Block-NeRF 训练文档索引

**版本**: 1.0  
**日期**: 2025年7月5日  
**概述**: Block-NeRF 训练相关的完整文档集合

---

## 📚 文档结构

本 Block-NeRF 训练文档包含以下部分，建议按顺序阅读：

### 🔰 入门文档
- **[README_cn.md](./README_cn.md)** - Block-NeRF 总体介绍和快速开始
  - 概述和特性
  - 安装指南
  - 基本使用示例
  - 模型架构说明

### 📖 核心训练文档

#### 第一部分：训练机制基础
- **[TRAINING_MECHANISM_cn.md](./TRAINING_MECHANISM_cn.md)** - 训练机制详解
  - 训练架构概述
  - 场景分解策略
  - 块级 NeRF 训练
  - 外观嵌入机制
  - 姿态细化算法
  - 可见性网络训练

#### 第二部分：损失函数与优化
- **[TRAINING_DETAILS_PART2_cn.md](./TRAINING_DETAILS_PART2_cn.md)** - 损失函数与优化策略
  - 损失函数设计详解
  - 优化算法选择
  - 学习率调度策略
  - 多GPU训练配置
  - 训练监控与评估
  - 超参数调优指南

#### 第三部分：实际训练脚本
- **[TRAINING_DETAILS_PART3_cn.md](./TRAINING_DETAILS_PART3_cn.md)** - 实际训练实现
  - 主训练脚本解析
  - 训练器类实现
  - 配置文件详解
  - 调试技巧与工具
  - 日志分析方法

#### 第四部分：应用案例与最佳实践
- **[TRAINING_DETAILS_PART4_cn.md](./TRAINING_DETAILS_PART4_cn.md)** - 案例与最佳实践
  - 城市场景训练案例
  - 室内场景训练案例
  - 数据预处理流程
  - 训练最佳实践
  - 常见问题与解决方案
  - 性能优化技巧
  - 部署指南

---

## 🎯 学习路径建议

### 初学者路径
```
README_cn.md → TRAINING_MECHANISM_cn.md → TRAINING_DETAILS_PART2_cn.md
```

### 进阶开发者路径
```
TRAINING_MECHANISM_cn.md → TRAINING_DETAILS_PART2_cn.md → 
TRAINING_DETAILS_PART3_cn.md → TRAINING_DETAILS_PART4_cn.md
```

### 问题排查路径
```
TRAINING_DETAILS_PART4_cn.md (常见问题部分) → 
TRAINING_DETAILS_PART3_cn.md (调试技巧) → 
TRAINING_DETAILS_PART2_cn.md (监控评估)
```

---

## 🔧 实际使用指南

### 训练前准备清单

1. **环境检查**
   - [ ] 确认 GPU 内存 >= 8GB
   - [ ] 安装 CUDA 11.0+
   - [ ] 检查磁盘空间充足
   - [ ] 验证数据集格式

2. **数据准备**
   - [ ] SfM 重建完成
   - [ ] 图像预处理完成
   - [ ] 场景分解参数确定
   - [ ] 训练/验证集划分

3. **配置文件准备**
   - [ ] 模型配置文件
   - [ ] 训练超参数配置
   - [ ] 数据路径配置
   - [ ] 输出路径配置

### 训练监控要点

- **损失曲线监控**：RGB损失、深度损失、正则化损失
- **指标监控**：PSNR、SSIM、LPIPS
- **资源监控**：GPU利用率、内存使用、训练速度
- **可视化监控**：渲染质量、块边界、外观一致性

### 常用训练命令

```bash
# 基础训练
python train_block_nerf.py --config configs/city_scene.yaml

# 多GPU训练
python -m torch.distributed.launch --nproc_per_node=4 train_block_nerf.py --config configs/city_scene.yaml

# 从检查点恢复
python train_block_nerf.py --config configs/city_scene.yaml --resume checkpoints/latest.pth

# 调试模式
python train_block_nerf.py --config configs/debug.yaml --debug --log_level DEBUG
```

---

## 📊 文档内容概览

| 文档部分 | 主要内容 | 阅读时间 | 难度等级 |
|---------|---------|---------|---------|
| README | 总体介绍、快速开始 | 15分钟 | ⭐ |
| 训练机制 | 核心算法、训练流程 | 45分钟 | ⭐⭐⭐ |
| 损失与优化 | 数学细节、优化策略 | 30分钟 | ⭐⭐⭐⭐ |
| 实际训练 | 代码实现、调试技巧 | 60分钟 | ⭐⭐⭐⭐ |
| 案例实践 | 实际应用、最佳实践 | 40分钟 | ⭐⭐⭐ |

---

## 🔗 相关资源

### 论文与参考资料
- [Block-NeRF: Scalable Large Scene Neural View Synthesis](https://waymo.com/research/block-nerf/)
- [NeRF: Representing Scenes as Neural Radiance Fields](https://www.matthewtancik.com/nerf)
- [Mip-NeRF 360: Unbounded Anti-Aliased Neural Radiance Fields](https://jonbarron.info/mipnerf360/)

### 数据集
- [Waymo Open Dataset](https://waymo.com/open)
- [KITTI Dataset](http://www.cvlibs.net/datasets/kitti/)
- [NeRF Synthetic Dataset](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1)

### 工具与软件
- [COLMAP](https://colmap.github.io/) - SfM重建
- [OpenMVG](https://openmvg.readthedocs.io/) - 多视图几何
- [Instant-NGP](https://github.com/NVlabs/instant-ngp) - 快速NeRF训练

---

## 💡 贡献指南

如果您想为此文档做出贡献：

1. **内容改进**：发现错误或需要补充的内容
2. **案例分享**：分享您的训练经验和技巧
3. **代码示例**：提供更好的代码示例
4. **翻译工作**：帮助翻译为其他语言

请通过 GitHub Issues 或 Pull Requests 参与贡献。

---

## 📝 更新日志

| 版本 | 日期 | 更新内容 |
|------|------|----------|
| 1.0 | 2025-07-05 | 初始版本，包含完整训练文档集合 |

---

**注意**: 这是一个活跃维护的文档集合。建议定期查看更新，获取最新的训练技巧和最佳实践。

---

*Happy Training! 🚀*
