"""
Block-NeRF CUDA Extension Setup Script - Fixed Version

Based on "Block-NeRF: Scalable Large Scene Neural View Synthesis" (CVPR 2022)
Optimized for GTX 1080 Ti (Compute Capability 6.1)
"""

from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import torch
import os

# Check CUDA availability
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available. Please install CUDA and PyTorch with CUDA support.")

# Get CUDA version
try:
    import torch._C
    if hasattr(torch._C, '_cuda_getCompiledVersion'):
        cuda_version = torch._C._cuda_getCompiledVersion()
        print(f"Found CUDA version: {cuda_version}")
    else:
        print("Could not determine CUDA version")
except:
    print("Could not determine CUDA version from PyTorch")
print(f"PyTorch CUDA available: {torch.cuda.is_available()}")

# GTX 1080 Ti specific settings
compute_capability = "6.1"
arch_flags = ['-gencode', f'arch=compute_{compute_capability.replace(".", "")},code=sm_{compute_capability.replace(".", "")}']

# Compiler flags
cxx_flags = [
    '-O3',
    '-std=c++17',
]

nvcc_flags = [
    '-O3',
    '-std=c++17',
    '--expt-relaxed-constexpr',
    '--use_fast_math',
] + arch_flags

# Define the extension
ext_modules = [
    CUDAExtension(
        name='block_nerf_cuda',
        sources=[
            'block_nerf_cuda_fixed.cpp',
            'block_nerf_cuda_kernels_fixed.cu',
        ],
        include_dirs=[
            # Add any additional include directories here
        ],
        extra_compile_args={
            'cxx': cxx_flags,
            'nvcc': nvcc_flags
        },
        extra_link_args=[],
    )
]

setup(
    name='block_nerf_cuda',
    version='1.0.0',
    author='NeuroCity Project',
    description='Block-NeRF CUDA Extensions for PyTorch',
    long_description='CUDA-accelerated Block-NeRF implementation optimized for GTX 1080 Ti',
    ext_modules=ext_modules,
    cmdclass={
        'build_ext': BuildExtension.with_options(no_python_abi_suffix=True, use_ninja=True)
    },
    zip_safe=False,
    python_requires='>=3.8',
)
