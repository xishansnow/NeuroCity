"""
Simple Setup Script for Block-NeRF CUDA Extension

This is a simplified setup script that can be used as an alternative 
to the main setup.py for quick testing and development.
"""

import os
import sys
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import torch

def get_cuda_flags():
    """Get CUDA compilation flags optimized for GTX 1080 Ti"""
    cuda_flags = [
        '-O3',
        '-std=c++17',
        '--expt-relaxed-constexpr',
        '--use_fast_math',
        '-gencode', 'arch=compute_61,code=sm_61',  # GTX 1080 Ti specific
        '-DCUDA_HAS_FP16=1',
        '-D__CUDA_NO_HALF_OPERATORS__',
        '-D__CUDA_NO_HALF_CONVERSIONS__',
        '-D__CUDA_NO_HALF2_OPERATORS__',
    ]
    return cuda_flags

def get_cxx_flags():
    """Get C++ compilation flags"""
    cxx_flags = [
        '-O3',
        '-std=c++17',
        '-DWITH_CUDA',
        '-DCUDA_HAS_FP16=1',
    ]
    return cxx_flags

if __name__ == '__main__':
    # Set CUDA architecture for GTX 1080 Ti
    os.environ['TORCH_CUDA_ARCH_LIST'] = '6.1'
    
    # Simple setup without complex configurations
    setup(
        name='block_nerf_cuda_simple',
        ext_modules=[
            CUDAExtension(
                name='block_nerf_cuda_simple',
                sources=[
                    'bindings.cpp',
                    'kernels.cu',
                ],
                extra_compile_args={
                    'cxx': get_cxx_flags(),
                    'nvcc': get_cuda_flags()
                },
                libraries=['cudart'],
                library_dirs=[os.path.join(torch.utils.cpp_extension.CUDA_HOME, 'lib64')],
                include_dirs=[
                    os.path.join(torch.utils.cpp_extension.CUDA_HOME, 'include'),
                ],
            )
        ],
        cmdclass={
            'build_ext': BuildExtension.with_options(use_ninja=False)
        },
        zip_safe=False,
    )
