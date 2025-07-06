"""
Simple Setup Script for Block-NeRF CUDA Extension

This is a simplified setup script that can be used as an alternative 
to the main setup.py for quick testing and development.
"""

import os
import sys
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Simple setup without complex configurations
setup(
    name='block_nerf_cuda_simple',
    ext_modules=[
        CUDAExtension(
            name='block_nerf_cuda_simple',
            sources=[
                'simple_bindings.cpp',
                'simple_kernels.cu',
            ],
            extra_compile_args={
                'cxx': ['-O3', '-std=c++17'],
                'nvcc': ['-O3', '-std=c++17', '--expt-relaxed-constexpr', '--use_fast_math']
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
    zip_safe=False,
)
