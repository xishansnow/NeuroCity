# VoxelRasterization CUDA 扩展使用指南

## 📋 概述

`src/nerfs/svraster/cuda/renderer/` 目录包含了 CUDA 加速的体素光栅化渲染器。该扩展提供了高性能的 GPU 渲染功能，支持 SVRaster 论文中描述的体素投影光栅化方法。

## 🏗️ 构建状态

✅ **构建成功**
- CUDA 扩展已成功编译
- 所有核心函数可用
- 测试通过

## 📁 文件结构

```
src/nerfs/svraster/cuda/renderer/
├── voxel_rasterizer_cuda_kernel.h      # CUDA 内核头文件
├── voxel_rasterizer_cuda_kernel.cu     # CUDA 内核实现
├── voxel_rasterizer_cuda.cpp           # C++ 绑定代码
├── voxel_rasterizer_gpu.py             # Python 包装器
└── voxel_rasterizer_cuda.cpython-310-x86_64-linux-gnu.so  # 编译后的扩展
```

## 🔧 可用函数

### 1. voxel_rasterization
**主要渲染函数**

```python
import nerfs.svraster.cuda.renderer.voxel_rasterizer_gpu as vrg

# 获取函数
voxel_rasterization_func = vrg.get_voxel_rasterization_function()

# 调用函数
rgb, depth = voxel_rasterization_func(
    voxel_positions,      # [N, 3] 体素位置
    voxel_sizes,          # [N] 体素尺寸
    voxel_densities,      # [N] 密度值
    voxel_colors,         # [N, C] 颜色系数
    camera_matrix,        # [4, 4] 相机变换矩阵
    intrinsics,           # [3, 3] 相机内参
    viewport_size,        # [2] 视口尺寸
    near_plane,           # 近平面
    far_plane,            # 远平面
    background_color,     # [3] 背景颜色
    density_activation,   # 密度激活函数 ("exp", "relu")
    color_activation,     # 颜色激活函数 ("sigmoid", "tanh", "clamp")
    sh_degree             # 球谐函数度数
)
```

### 2. create_camera_matrix
**相机矩阵创建函数**

```python
create_camera_matrix_func = vrg.get_create_camera_matrix_function()
camera_matrix = create_camera_matrix_func(camera_pose)
```

### 3. rays_to_camera_matrix
**从光线估算相机参数**

```python
rays_to_camera_matrix_func = vrg.get_rays_to_camera_matrix_function()
camera_matrix, intrinsics = rays_to_camera_matrix_func(ray_origins, ray_directions)
```

### 4. benchmark
**性能基准测试**

```python
benchmark_func = vrg.get_benchmark_function()
timings = benchmark_func(
    voxel_positions, voxel_sizes, voxel_densities, voxel_colors,
    camera_matrix, intrinsics, viewport_size, num_iterations
)
```

## 🚀 使用方法

### 方法 1: 使用 VoxelRasterizerGPU 类

```python
import torch
from nerfs.svraster.cuda.renderer.voxel_rasterizer_gpu import VoxelRasterizerGPU

# 创建配置
class Config:
    def __init__(self):
        self.near_plane = 0.1
        self.far_plane = 100.0
        self.background_color = [0.0, 0.0, 0.0]
        self.density_activation = "exp"
        self.color_activation = "sigmoid"

config = Config()

# 创建渲染器
rasterizer = VoxelRasterizerGPU(config)

# 准备数据
voxels = {
    "positions": torch.rand(1000, 3, device="cuda"),
    "sizes": torch.rand(1000, device="cuda") * 0.1,
    "densities": torch.randn(1000, device="cuda"),
    "colors": torch.rand(1000, 3, device="cuda")
}

camera_matrix = torch.eye(4, device="cuda")
intrinsics = torch.tensor([[800, 0, 400], [0, 800, 300], [0, 0, 1]], device="cuda")
viewport_size = (800, 600)

# 渲染
result = rasterizer(voxels, camera_matrix, intrinsics, viewport_size)
rgb = result["rgb"]  # [H, W, 3]
depth = result["depth"]  # [H, W]
```

### 方法 2: 直接调用 CUDA 函数

```python
import torch
import nerfs.svraster.cuda.renderer.voxel_rasterizer_gpu as vrg

# 获取函数
voxel_rasterization_func = vrg.get_voxel_rasterization_function()

if voxel_rasterization_func is not None:
    # 准备数据
    voxel_positions = torch.rand(1000, 3, device="cuda")
    voxel_sizes = torch.rand(1000, device="cuda") * 0.1
    voxel_densities = torch.randn(1000, device="cuda")
    voxel_colors = torch.rand(1000, 3, device="cuda")
    
    camera_matrix = torch.eye(4, device="cuda")
    intrinsics = torch.tensor([[800, 0, 400], [0, 800, 300], [0, 0, 1]], device="cuda")
    viewport_size = torch.tensor([800, 600], dtype=torch.int32, device="cuda")
    
    # 调用渲染函数
    rgb, depth = voxel_rasterization_func(
        voxel_positions,
        voxel_sizes,
        voxel_densities,
        voxel_colors,
        camera_matrix,
        intrinsics,
        viewport_size,
        0.1,  # near_plane
        100.0,  # far_plane
        torch.tensor([0.0, 0.0, 0.0], device="cuda"),  # background_color
        "exp",  # density_activation
        "sigmoid",  # color_activation
        2  # sh_degree
    )
```

## ⚙️ 环境设置

### 设置库路径

```bash
# 临时设置
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/xishansnow/anaconda3/envs/neurocity/lib/python3.10/site-packages/torch/lib

# 永久设置（添加到 ~/.bashrc）
echo 'export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/home/xishansnow/anaconda3/envs/neurocity/lib/python3.10/site-packages/torch/lib"' >> ~/.bashrc
source ~/.bashrc
```

### 使用环境设置脚本

```bash
# 使用提供的设置脚本
source /home/xishansnow/3DVision/NeuroCity/src/nerfs/svraster/cuda/setup_env.sh
```

## 🧪 测试

### 运行完整测试

```bash
cd src/nerfs/svraster/cuda
python test_cuda_extension.py
```

### 运行函数访问测试

```bash
cd src/nerfs/svraster/cuda
python test_voxel_rasterization.py
```

## 📊 性能特点

- **GPU 加速**: 利用 CUDA 并行计算
- **内存优化**: 高效的 GPU 内存管理
- **可扩展性**: 支持大量体素的实时渲染
- **精度**: 支持浮点精度计算

## 🔍 故障排除

### 常见问题

1. **ImportError: libc10.so: cannot open shared object file**
   - 解决方案: 设置正确的 LD_LIBRARY_PATH

2. **CUDA extension not available**
   - 检查 CUDA 是否可用
   - 确认扩展已正确编译

3. **Function not found**
   - 确认环境变量设置正确
   - 重新编译扩展

### 调试命令

```bash
# 检查 CUDA 扩展
python -c "import voxel_rasterizer_cuda; print('Extension loaded')"

# 检查函数可用性
python -c "import voxel_rasterizer_cuda; print(dir(voxel_rasterizer_cuda))"

# 检查模块导入
python -c "import nerfs.svraster.cuda.renderer.voxel_rasterizer_gpu as vrg; print(vrg.CUDA_AVAILABLE)"
```

## 📝 注意事项

1. **设备一致性**: 确保所有张量都在同一设备上（CPU 或 GPU）
2. **内存管理**: 大型体素网格可能需要大量 GPU 内存
3. **精度**: 使用 float32 精度以获得最佳性能
4. **线程安全**: 当前实现不是线程安全的

## 🎯 总结

`voxel_rasterization` 函数已经成功构建并可以通过以下方式访问：

1. **直接访问**: `voxel_rasterizer_cuda.voxel_rasterization`
2. **通过模块**: `vrg.get_voxel_rasterization_function()`
3. **通过类**: `VoxelRasterizerGPU` 类

所有函数都已正确导出并可以正常使用！ 