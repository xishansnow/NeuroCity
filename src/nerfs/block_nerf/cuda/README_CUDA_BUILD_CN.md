# Block-NeRF CUDA扩展构建指南

## 概述

这是一个基于CUDA加速的Block-NeRF实现，针对NVIDIA GTX 1080 Ti (计算能力6.1) 进行了优化。

## 系统要求

- NVIDIA GPU (GTX 1080 Ti 或更高)
- CUDA 12.x 或兼容版本
- PyTorch 带 CUDA 支持
- Python 3.8+
- GCC/G++ 编译器

## 构建步骤

### 1. 环境准备

确保你的系统已安装以下依赖：

```bash
# 检查CUDA版本
nvcc --version

# 检查PyTorch CUDA支持
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### 2. 构建CUDA扩展

使用修复版本的构建脚本：

```bash
# 进入CUDA扩展目录
cd src/nerfs/block_nerf/cuda

# 运行修复版本的构建脚本
./build_fixed.sh
```

### 3. 验证安装

```bash
# 运行快速测试
python quick_test.py

# 运行完整测试
python run_tests.py
```

## 项目结构

```
src/nerfs/block_nerf/cuda/
├── block_nerf_cuda_kernels_fixed.cu    # CUDA内核实现 (修复版本)
├── block_nerf_cuda_fixed.cpp           # C++/PyBind11绑定 (修复版本)
├── setup_fixed.py                      # 构建脚本 (修复版本)
├── build_fixed.sh                      # 构建脚本 (修复版本)
├── test_unit.py                         # 单元测试
├── test_functional.py                   # 功能测试
├── test_benchmark.py                    # 性能基准测试
├── integration_example.py               # 集成示例
├── verify_environment.py                # 环境验证
├── quick_test.py                        # 快速测试
└── run_tests.py                         # 测试运行器
```

## 已实现功能

### CUDA内核
- 内存带宽测试
- 块可见性计算
- 射线-块相交选择
- 体积渲染积分
- 块插值权重计算

### PyTorch集成
- 张量输入/输出
- GPU内存管理
- 错误检查和同步
- 自动梯度支持准备

## 测试状态

✅ **工作正常的文件:**
- `block_nerf_cuda_kernels_fixed.cu` - CUDA内核实现
- `block_nerf_cuda_fixed.cpp` - C++绑定
- `setup_fixed.py` - 构建脚本
- `build_fixed.sh` - 构建脚本
- `test_unit.py` - 单元测试
- `test_functional.py` - 功能测试
- `test_benchmark.py` - 基准测试
- `integration_example.py` - 集成示例
- `verify_environment.py` - 环境验证
- `quick_test.py` - 快速测试
- `run_tests.py` - 测试运行器

## 性能优化

### GTX 1080 Ti 特定优化
- 计算能力 6.1 目标架构
- 优化的线程块大小 (256)
- 内存合并访问模式
- 寄存器使用优化

### 内存管理
- GPU内存池化
- 批处理操作
- 异步内存传输准备

## 使用示例

### 基础使用
```python
import torch
import block_nerf_cuda

# 创建测试数据
camera_pos = torch.randn(10, 3, device='cuda')
block_centers = torch.randn(50, 3, device='cuda')
block_radii = torch.ones(50, device='cuda') * 2.0
block_active = torch.ones(50, dtype=torch.int32, device='cuda')

# 计算块可见性
visibility = block_nerf_cuda.block_visibility(
    camera_pos, block_centers, block_radii, block_active, 0.1
)
```

### 渲染管道
```python
# 射线生成
ray_origins = torch.randn(1000, 3, device='cuda')
ray_directions = torch.randn(1000, 3, device='cuda')
ray_near = torch.ones(1000, device='cuda') * 0.1
ray_far = torch.ones(1000, device='cuda') * 100.0

# 块选择
selected_blocks, num_selected = block_nerf_cuda.block_selection(
    ray_origins, ray_directions, ray_near, ray_far,
    block_centers, block_radii, block_active, 32
)
```

## 故障排除

### 常见问题

1. **CUDA版本不匹配**
   ```
   Warning: CUDA version mismatch
   ```
   - 解决方案：确保CUDA工具包和PyTorch版本兼容

2. **编译错误**
   ```
   error: no operator "=" matches these operands
   ```
   - 已在修复版本中解决：使用 `*_fixed.*` 文件

3. **GPU内存不足**
   ```
   CUDA out of memory
   ```
   - 解决方案：减少批处理大小或使用数据分块

### 调试提示

- 使用 `CUDA_LAUNCH_BLOCKING=1` 进行同步调试
- 检查GPU内存使用：`nvidia-smi`
- 验证CUDA内核启动：添加 `cudaGetLastError()` 检查

## 开发指南

### 添加新内核
1. 在 `block_nerf_cuda_kernels_fixed.cu` 中实现CUDA内核
2. 在 `block_nerf_cuda_fixed.cpp` 中添加launcher函数
3. 更新PyBind11绑定
4. 添加相应测试

### 性能分析
- 使用 `nvprof` 或 `nsight` 进行性能分析
- 监控GPU利用率和内存带宽
- 优化内存访问模式

## 许可证

基于 "Block-NeRF: Scalable Large Scene Neural View Synthesis" (CVPR 2022)
针对NeuroCity项目进行了适配和优化。

## 联系方式

如有问题或建议，请通过项目仓库提交issue。
