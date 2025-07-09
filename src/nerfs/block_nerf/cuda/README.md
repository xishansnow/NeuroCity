# Block-NeRF CUDA Extension

High-performance CUDA acceleration for Block-NeRF operations.

## Features

- Memory bandwidth testing and optimization
- GPU-accelerated vector operations (addition, multiplication)
- Block visibility computation for view-dependent culling
- Ray-block selection for efficient rendering
- Optimized for NVIDIA GPUs with compute capability 6.1+

## Quick Start

### 1. Build and Install

```bash
# Set up environment
source env_setup.sh

# Install extension
python3 install.py
```

### 2. Verify Installation

```bash
python3 test_cuda.py
```

### 3. Usage Example

```python
import block_nerf_cuda_simple as cuda_ext

# Memory bandwidth test
result = cuda_ext.memory_bandwidth_test(tensor)

# Block visibility computation
visibility = cuda_ext.block_visibility(
    camera_positions, block_centers, block_radii, view_directions
)

# Block selection for rays
selected_blocks, num_selected = cuda_ext.block_selection(
    rays_o, rays_d, block_centers, block_radii
)
```

## Files

- `kernels.cu` - CUDA kernel implementations
- `bindings.cpp` - Python bindings  
- `setup.py` - Build configuration
- `build.sh` - Build script
- `env_setup.sh` - Environment setup script
- `install.py` - Automated installation
- `test_cuda.py` - Comprehensive functionality tests
- `demo.py` - Usage examples and demos
- `check_env.py` - Environment verification

## Requirements

- CUDA 11.0+ (tested with 12.6)
- PyTorch 1.12+ with CUDA support
- NVIDIA GPU with compute capability 6.0+
- C++17 compatible compiler

## Performance

- Memory bandwidth: 200+ GB/s
- Block visibility: 10,000+ operations/ms
- Ray-block selection: 2,000+ rays/ms
