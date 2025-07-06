# 🚀 NeuroCity CUDA 快速开始指南

恭喜！您已经成功设置了 NeuroCity 项目的 CUDA 支持。本指南将帮助您快速开始使用 CUDA 加速功能。

## ✅ 当前状态

根据验证结果，您的系统已经具备以下条件：

- **CUDA 环境**: ✅ 可用 (版本 12.6)
- **GPU 设备**: ✅ NVIDIA GeForce GTX 1080 Ti (11.70 GB)
- **计算能力**: ✅ 6.1 (支持大多数 CUDA 功能)
- **文档完整性**: ✅ 100% (所有 CUDA 文档已准备就绪)

## 🎯 推荐使用流程

### 1. 选择合适的 NeRF 模块

根据您的需求和硬件配置，推荐使用顺序：

| 模块 | 推荐指数 | 适用场景 | 文档链接 |
|------|----------|----------|----------|
| **SVRaster** | ⭐⭐⭐⭐⭐ | 实时渲染、VR/AR 应用 | [SVRaster CUDA 指南](src/nerfs/svraster/README_cn.md#cuda-核函数使用指南) |
| **Plenoxels** | ⭐⭐⭐⭐ | 快速训练、移动应用 | [Plenoxels CUDA 指南](src/nerfs/plenoxels/README_cn.md#cuda-核函数使用指南) |
| **InfNeRF** | ⭐⭐⭐ | 大规模场景、城市重建 | [InfNeRF CUDA 指南](src/nerfs/inf_nerf/README_cn.md#cuda-核函数使用指南) |

### 2. 环境准备

```bash
# 检查 CUDA 环境
python verify_cuda_docs.py

# 编译 CUDA 扩展 (如果需要)
python build_cuda_extensions.py --force

# 验证所有功能
python verify_cuda_functionality.py
```

### 3. 开始使用

#### 🌟 SVRaster (推荐新手)

```python
from src.nerfs.svraster import SVRasterConfig, SVRasterModel

# 创建配置
config = SVRasterConfig(
    max_octree_levels=12,
    base_resolution=64,
    scene_bounds=(-2.0, -2.0, -2.0, 2.0, 2.0, 2.0)
)

# 创建模型并移到 GPU
model = SVRasterModel(config).cuda()

# 渲染
ray_origins = torch.randn(1000, 3, device='cuda')
ray_directions = torch.randn(1000, 3, device='cuda')
outputs = model(ray_origins, ray_directions)
```

#### 🔥 Plenoxels (快速训练)

```python
from src.nerfs.plenoxels import PlenoxelConfig, PlenoxelModel

# 创建配置
config = PlenoxelConfig(
    grid_shape=[128, 128, 128],
    sh_degree=3,
    bbox_min=[-1, -1, -1],
    bbox_max=[1, 1, 1]
)

# 创建模型并移到 GPU
model = PlenoxelModel(config).cuda()

# 渲染
outputs = model.render(ray_origins, ray_directions)
```

#### 🏗️ InfNeRF (大规模场景)

```python
from src.nerfs.inf_nerf import InfNeRF, InfNeRFConfig

# 创建配置
config = InfNeRFConfig(
    max_depth=8,
    grid_size=256,
    scene_bound=50.0
)

# 创建模型并移到 GPU
model = InfNeRF(config).cuda()

# 构建八叉树
sparse_points = torch.randn(10000, 3, device='cuda')
model.build_octree(sparse_points)

# 渲染
outputs = model.render(ray_origins, ray_directions, near=0.1, far=100.0)
```

## 🔧 性能优化建议

### 针对您的 GTX 1080 Ti

```python
# 推荐配置
recommended_config = {
    'batch_size': 4096,          # 适合 11GB 显存
    'chunk_size': 8192,          # 分块渲染
    'use_mixed_precision': True,  # 启用混合精度
    'max_resolution': 512,       # 最大体素分辨率
}

# 内存优化
torch.cuda.empty_cache()
torch.backends.cudnn.benchmark = True
```

### 推荐的体素分辨率

| 应用场景 | 推荐分辨率 | 预期内存使用 | 渲染速度 |
|----------|------------|--------------|----------|
| 实时预览 | 64³ | ~1GB | >60 FPS |
| 高质量渲染 | 256³ | ~6GB | 20-30 FPS |
| 最高质量 | 512³ | ~10GB | 5-10 FPS |

## 📚 详细学习路径

### 第一阶段: 基础使用 (1-2 天)
1. 阅读 [主要 CUDA 指南](CUDA_USAGE_GUIDE.md)
2. 尝试 SVRaster 基础示例
3. 学习基本的 CUDA 内存管理

### 第二阶段: 进阶优化 (1 周)
1. 深入学习各模块的 CUDA 文档
2. 尝试性能调优
3. 学习多 GPU 支持

### 第三阶段: 高级应用 (2-4 周)
1. 自定义 CUDA 核函数
2. 大规模场景处理
3. 生产环境部署

## 🔍 故障排除

### 常见问题

1. **内存不足**
   ```python
   # 减少批次大小
   config.batch_size = 2048
   
   # 使用梯度检查点
   torch.utils.checkpoint.checkpoint_sequential
   ```

2. **渲染速度慢**
   ```python
   # 启用所有优化
   torch.backends.cudnn.benchmark = True
   torch.backends.cuda.matmul.allow_tf32 = True
   ```

3. **CUDA 版本不匹配**
   ```bash
   # 重新安装 PyTorch
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

### 获取帮助

如果遇到问题，请按顺序尝试：

1. 查看相关模块的 CUDA 文档
2. 运行验证脚本诊断问题
3. 查看 [故障排除指南](CUDA_USAGE_GUIDE.md#故障排除)
4. 在 GitHub Issues 中提问

## 🎯 下一步行动

选择一个模块开始您的 CUDA 加速之旅：

- **想要快速看到效果**: 从 SVRaster 开始 → [SVRaster CUDA 指南](src/nerfs/svraster/README_cn.md#cuda-核函数使用指南)
- **需要快速训练**: 从 Plenoxels 开始 → [Plenoxels CUDA 指南](src/nerfs/plenoxels/README_cn.md#cuda-核函数使用指南)
- **处理大规模场景**: 从 InfNeRF 开始 → [InfNeRF CUDA 指南](src/nerfs/inf_nerf/README_cn.md#cuda-核函数使用指南)

## 📊 预期性能

基于您的 GTX 1080 Ti，预期性能：

| 场景类型 | 图像分辨率 | 预期 FPS | 训练时间 |
|----------|------------|----------|----------|
| 简单室内 | 512x512 | 30-45 | 15-30 分钟 |
| 复杂室内 | 512x512 | 20-30 | 30-60 分钟 |
| 户外场景 | 512x512 | 15-25 | 1-2 小时 |
| 大规模场景 | 512x512 | 10-20 | 2-4 小时 |

---

🎉 **祝您使用愉快！** 

*如果这个快速指南对您有帮助，请考虑为项目 ⭐ Star！*

*最后更新: 2024年12月*
