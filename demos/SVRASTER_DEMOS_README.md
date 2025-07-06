# SVRaster 演示脚本

这里包含了两个 SVRaster 的完整演示脚本，展示训练和渲染的不同方面。

## 演示脚本列表

### 1. `demo_svraster_training.py` - 训练演示

**用途**: 展示如何使用 SVRaster 进行神经辐射场训练

**特点**:
- ✅ 使用 `VolumeRenderer` 进行体积渲染训练
- ✅ 自适应稀疏体素表示
- ✅ 球谐函数视角相关颜色
- ✅ 现代 PyTorch 训练循环
- ✅ 实时损失监控和验证
- ✅ 自动检查点保存

**运行方法**:
```bash
cd /home/xishansnow/3DVision/NeuroCity
python demos/demo_svraster_training.py
```

**输出**:
- 训练进度监控
- 损失曲线和 PSNR 指标
- 模型检查点保存到 `demos/checkpoints/svraster_training/`
- 训练历史 JSON 文件

### 2. `demo_svraster_rendering.py` - 高效渲染演示

**用途**: 展示 SVRaster 的实时高效渲染能力

**特点**:
- ✅ 使用 `TrueVoxelRasterizer` 进行快速光栅化
- ✅ GPU 加速实时渲染
- ✅ 多分辨率性能基准测试
- ✅ 训练/推理模式质量对比
- ✅ 动画序列渲染
- ✅ 详细性能报告

**运行方法**:
```bash
cd /home/xishansnow/3DVision/NeuroCity
python demos/demo_svraster_rendering.py
```

**输出**:
- `demos/demo_outputs/svraster_rendering/`
  - `volume_rendering.png` - 体积渲染结果
  - `raster_rendering.png` - 光栅化渲染结果
  - `*_depth.png` - 深度图
  - `*_difference.png` - 差异图
  - `animation/frame_*.png` - 动画序列
  - `performance_report.json` - 性能报告

## 技术对比

| 特性 | 训练演示 | 渲染演示 |
|------|----------|----------|
| **主要用途** | 模型训练 | 实时渲染 |
| **渲染器** | VolumeRenderer | TrueVoxelRasterizer |
| **渲染模式** | mode="training" | mode="inference" |
| **性能** | 精确但慢 | 快速实时 |
| **质量** | 高质量 | 近似高质量 |
| **GPU 利用** | 中等 | 最大化 |

## 系统要求

### 最低要求
- Python 3.10+
- PyTorch 1.12+
- CUDA 11.0+ (推荐)
- GPU 内存: 4GB+

### 推荐配置
- Python 3.10/3.11
- PyTorch 2.0+
- CUDA 11.8+
- GPU: RTX 3080/4080+ 或同等 
- GPU 内存: 8GB+
- CPU: 8+ 核心

## 配置参数

### 训练演示主要参数

```python
SVRasterConfig(
    image_width=400,           # 图像宽度
    image_height=300,          # 图像高度
    base_resolution=64,        # 基础体素分辨率
    max_octree_levels=8,       # 最大八叉树层级
    ray_samples_per_voxel=8,   # 每体素采样数
    sh_degree=2,               # 球谐函数阶数
)

SVRasterTrainerConfig(
    learning_rate=1e-3,        # 学习率
    batch_size=1024,           # 批次大小
    max_epochs=100,            # 最大训练轮数
    use_amp=True,              # 混合精度训练
)
```

### 渲染演示主要参数

```python
SVRasterConfig(
    image_width=800,           # 高分辨率渲染
    image_height=600,
    base_resolution=128,       # 高质量体素网格
    ray_samples_per_voxel=4,   # 推理时减少采样
)

SVRasterRendererConfig(
    render_batch_size=8192,    # 大批次提高效率
    max_rays_per_batch=16384,  # 最大光线数
    use_cached_features=True,  # 缓存优化
)
```

## 性能指标

### 训练性能（典型值）
- **训练速度**: 1-5 秒/epoch (取决于批次大小)
- **内存使用**: 2-6GB GPU 内存
- **收敛时间**: 50-200 epochs

### 渲染性能（典型值）
- **全分辨率**: 10-50 FPS (800x600)
- **半分辨率**: 30-120 FPS (400x300)
- **加速比**: 5-20x (相对于体积渲染)

## 故障排除

### 常见问题

1. **CUDA 内存不足**
   ```bash
   # 减少批次大小或分辨率
   config.batch_size = 512
   config.base_resolution = 32
   ```

2. **导入错误**
   ```bash
   # 确保路径正确
   export PYTHONPATH=/home/xishansnow/3DVision/NeuroCity:$PYTHONPATH
   ```

3. **性能较慢**
   ```bash
   # 确保使用 GPU
   torch.cuda.is_available()  # 应该返回 True
   ```

### 调试模式

可以在脚本中启用调试模式：
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 扩展使用

### 自定义场景

修改 `_initialize_demo_scene()` 函数来创建自定义场景：

```python
def create_custom_scene(self):
    # 你的自定义场景代码
    pass
```

### 不同数据集

修改 `_create_synthetic_dataset()` 来使用真实数据：

```python
dataset = SVRasterDataset.from_colmap("path/to/colmap/data")
```

### 高级渲染

尝试不同的渲染配置：

```python
config.sh_degree = 3           # 更高阶球谐函数
config.morton_ordering = True  # Morton 码排序优化
config.depth_peeling_layers = 8  # 更多深度层
```

---

**注意**: 这些演示脚本已经过测试，与 Python 3.10+ 完全兼容。如有问题，请检查系统要求和依赖安装。
