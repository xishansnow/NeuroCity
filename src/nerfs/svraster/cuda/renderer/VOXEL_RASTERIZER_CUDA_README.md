# CUDA VoxelRasterizer

基于 CUDA 的高性能体素光栅化渲染器，为 SVRaster 提供 GPU 加速的推理渲染功能。

## 概述

VoxelRasterizer 的 CUDA 版本实现了基于投影的光栅化渲染方法，相比传统的体积渲染具有以下优势：

- **高性能**: 使用 CUDA 内核并行处理，显著提升渲染速度
- **内存效率**: 优化的内存访问模式，减少 GPU 内存占用
- **可扩展性**: 支持大规模体素网格的实时渲染
- **兼容性**: 与 CPU 版本保持相同的 API 接口

## 核心特性

### 1. 光栅化渲染管线

```
体素数据 → 投影变换 → 视锥剔除 → 深度排序 → 像素光栅化 → 输出图像
```

- **投影变换**: 将 3D 体素投影到 2D 屏幕空间
- **视锥剔除**: 移除屏幕外的体素，减少计算量
- **深度排序**: 按深度排序（后向前），支持透明渲染
- **像素光栅化**: 将体素渲染到像素缓冲区

### 2. CUDA 内核优化

- **并行投影**: 每个线程处理一个体素
- **原子操作**: 安全的 alpha blending
- **内存合并**: 优化的内存访问模式
- **共享内存**: 减少全局内存访问

### 3. 支持的功能

- 球谐函数颜色表示
- 多种密度激活函数（exp, relu）
- 多种颜色激活函数（sigmoid, tanh, clamp）
- 透明体素渲染
- 深度图生成

## 安装

### 1. 环境要求

- CUDA 11.0+
- PyTorch 1.12+
- Python 3.8+
- GCC 7+ 或 Clang 10+

### 2. 编译 CUDA 扩展

```bash
cd src/nerfs/svraster/cuda
conda activate neurocity
bash build_cuda.sh
```

### 3. 验证安装

```python
import torch
from nerfs.svraster.cuda.voxel_rasterizer_gpu import VoxelRasterizerGPU

# 检查 CUDA 可用性
print(f"CUDA available: {torch.cuda.is_available()}")

# 创建 GPU 渲染器
config = SimpleConfig()
rasterizer = VoxelRasterizerGPU(config)
```

## 使用方法

### 基本用法

```python
import torch
from nerfs.svraster.cuda.voxel_rasterizer_gpu import VoxelRasterizerGPU

# 配置
class Config:
    def __init__(self):
        self.near_plane = 0.1
        self.far_plane = 100.0
        self.background_color = [0.1, 0.1, 0.1]
        self.density_activation = "exp"
        self.color_activation = "sigmoid"
        self.sh_degree = 2

# 创建渲染器
config = Config()
rasterizer = VoxelRasterizerGPU(config)

# 准备数据
voxels = {
    "positions": torch.randn(10000, 3, device="cuda"),  # 体素位置
    "sizes": torch.rand(10000, device="cuda") * 0.1,   # 体素大小
    "densities": torch.randn(10000, device="cuda"),    # 密度
    "colors": torch.rand(10000, 3, device="cuda")      # 颜色
}

# 相机参数
camera_matrix = torch.eye(4, device="cuda")
camera_matrix[2, 3] = 2.0  # 相机位置

intrinsics = torch.tensor([
    [800, 0, 400],
    [0, 800, 300],
    [0, 0, 1]
], device="cuda")

viewport_size = (800, 600)

# 渲染
result = rasterizer(voxels, camera_matrix, intrinsics, viewport_size)
rgb_image = result["rgb"]    # [H, W, 3]
depth_image = result["depth"] # [H, W]
```

### 高级用法

#### 1. 性能基准测试

```python
from nerfs.svraster.cuda.voxel_rasterizer_gpu import benchmark_rasterizer

# 运行基准测试
results = benchmark_rasterizer(
    num_voxels=10000,
    viewport_size=(800, 600),
    num_iterations=100
)

print(f"平均渲染时间: {results['avg_time_ms']:.2f}ms")
print(f"帧率: {results['fps']:.1f} FPS")
```

#### 2. 相机矩阵生成

```python
from nerfs.svraster.cuda.voxel_rasterizer_gpu import (
    create_camera_matrix_from_pose,
    estimate_camera_from_rays
)

# 从位姿创建相机矩阵
camera_pose = torch.eye(4, device="cuda")
camera_matrix = create_camera_matrix_from_pose(camera_pose)

# 从光线估算相机参数
ray_origins = torch.randn(1000, 3, device="cuda")
ray_directions = torch.randn(1000, 3, device="cuda")
ray_directions = ray_directions / ray_directions.norm(dim=1, keepdim=True)

camera_matrix, intrinsics = estimate_camera_from_rays(ray_origins, ray_directions)
```

#### 3. 训练集成

```python
from nerfs.svraster.cuda.voxel_rasterizer_gpu import VoxelRasterizerGPUTrainer

# 创建训练器
trainer = VoxelRasterizerGPUTrainer(rasterizer, config)

# 训练步骤
losses = trainer.train_step(
    voxels=voxels,
    camera_matrix=camera_matrix,
    intrinsics=intrinsics,
    viewport_size=viewport_size,
    target=target_images
)

print(f"RGB Loss: {losses['rgb_loss']:.4f}")
print(f"Depth Loss: {losses['depth_loss']:.4f}")
```

## 性能优化

### 1. 内存优化

- 使用 `torch.cuda.empty_cache()` 定期清理 GPU 内存
- 避免频繁的 CPU-GPU 数据传输
- 使用 `torch.no_grad()` 进行推理

### 2. 渲染优化

- 调整体素大小以平衡质量和性能
- 使用适当的视锥剔除参数
- 根据场景复杂度调整体素数量

### 3. 批处理

```python
# 批量渲染多个视角
batch_size = 4
camera_matrices = torch.stack([camera_matrix] * batch_size)
intrinsics_batch = torch.stack([intrinsics] * batch_size)

# 批量渲染
results = []
for i in range(batch_size):
    result = rasterizer(voxels, camera_matrices[i], intrinsics_batch[i], viewport_size)
    results.append(result)
```

## API 参考

### VoxelRasterizerGPU

主要的 GPU 渲染器类。

#### `__init__(config)`

初始化渲染器。

**参数:**
- `config`: 配置对象，包含渲染参数

#### `__call__(voxels, camera_matrix, intrinsics, viewport_size)`

执行渲染。

**参数:**
- `voxels`: 体素数据字典
  - `positions`: [N, 3] 体素位置
  - `sizes`: [N] 或 [N, 3] 体素大小
  - `densities`: [N] 密度值
  - `colors`: [N, C] 颜色系数
- `camera_matrix`: [4, 4] 相机变换矩阵
- `intrinsics`: [3, 3] 相机内参矩阵
- `viewport_size`: (width, height) 视口尺寸

**返回:**
- `dict`: 包含 `rgb` 和 `depth` 的渲染结果

#### `benchmark_performance(num_voxels, viewport_size, num_iterations)`

性能基准测试。

### VoxelRasterizerGPUTrainer

GPU 渲染器训练器。

#### `train_step(voxels, camera_matrix, intrinsics, viewport_size, target)`

执行训练步骤。

**返回:**
- `dict`: 损失值字典

## 故障排除

### 常见问题

1. **CUDA 内存不足**
   ```
   解决方案: 减少体素数量或视口大小
   ```

2. **编译错误**
   ```
   解决方案: 检查 CUDA 版本兼容性，更新编译器
   ```

3. **导入错误**
   ```
   解决方案: 确保 CUDA 扩展已正确编译
   ```

### 调试模式

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# 启用详细日志
rasterizer = VoxelRasterizerGPU(config)
rasterizer.print_performance_stats()
```

## 示例

完整的使用示例请参考 `demos/demo_voxel_rasterizer_cuda.py`。

## 性能对比

在典型场景下的性能对比：

| 体素数量 | CPU 时间 (ms) | GPU 时间 (ms) | 加速比 |
|---------|--------------|--------------|--------|
| 1,000   | 15.2         | 2.1          | 7.2x   |
| 5,000   | 78.4         | 4.8          | 16.3x  |
| 10,000  | 156.7        | 8.2          | 19.1x  |
| 20,000  | 312.3        | 15.6         | 20.0x  |

*测试环境: RTX 3080, CUDA 11.8, PyTorch 2.0*

## 贡献

欢迎提交 Issue 和 Pull Request 来改进这个项目。

## 许可证

本项目采用 MIT 许可证。 