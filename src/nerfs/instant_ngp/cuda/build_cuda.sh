#!/bin/bash

# Build Instant NGP CUDA Extension for GTX 1080 Ti

# Check CUDA availability
if ! command -v nvcc &> /dev/null; then
    echo "Error: CUDA toolkit not found. Please install CUDA toolkit first."
    exit 1
fi

# Check PyTorch
python3 -c "import torch" &> /dev/null
if [ $? -ne 0 ]; then
    echo "Error: PyTorch not found. Please install PyTorch with CUDA support first."
    exit 1
fi

# Set CUDA architecture for GTX 1080 Ti
export TORCH_CUDA_ARCH_LIST="6.1"

# Clean previous builds
echo "Cleaning previous build files..."
rm -rf build/ *.so

# Build CUDA extension
echo "Building Instant NGP CUDA extensions..."
python3 setup.py build_ext --inplace

# Check build result
if [ $? -eq 0 ]; then
    echo "✅ Instant NGP CUDA extensions built successfully!"
    echo "Running tests..."
    cd /home/xishansnow/3DVision/NeuroCity
    python tests/nerfs/test_instant_ngp_cuda.py
else
    echo "❌ Failed to build CUDA extensions."
    exit 1
fi
