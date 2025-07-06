#!/bin/bash

# 检查 CUDA 工具链
if ! command -v nvcc &> /dev/null; then
    echo "Error: CUDA toolkit not found. Please install CUDA toolkit first."
    exit 1
fi

# 检查 PyTorch
python3 -c "import torch" &> /dev/null
if [ $? -ne 0 ]; then
    echo "Error: PyTorch not found. Please install PyTorch with CUDA support first."
    exit 1
fi

# 设置 CUDA 架构（适用于 GTX 1080 Ti）
export TORCH_CUDA_ARCH_LIST="6.1"

# 清理之前的构建文件
echo "Cleaning previous build files..."
rm -rf build/ *.so

# 构建 CUDA 扩展
echo "Building CUDA extensions..."
python3 setup.py build_ext --inplace

# 检查构建结果
if [ $? -eq 0 ]; then
    echo "CUDA extensions built successfully!"
    echo "Running tests..."
    cd /home/xishansnow/3DVision/NeuroCity
    python tests/nerfs/test_plenoxels_cuda.py
else
    echo "Error: Failed to build CUDA extensions."
    exit 1
fi 