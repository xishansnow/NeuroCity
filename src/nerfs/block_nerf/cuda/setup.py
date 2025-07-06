#!/usr/bin/env python3

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import torch
import os

def get_extensions():
    """Get the extension modules"""
    
    # CUDA flags optimized for GTX 1080 Ti
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
    
    # C++ flags
    cxx_flags = [
        '-O3',
        '-std=c++17',
        '-DWITH_CUDA',
        '-DCUDA_HAS_FP16=1',
    ]
    
    # CUDA extension
    extension = CUDAExtension(
        name='block_nerf_cuda',
        sources=[
            'block_nerf_cuda.cpp',
            'block_nerf_cuda_kernels.cu',
        ],
        extra_compile_args={
            'cxx': cxx_flags,
            'nvcc': cuda_flags
        },
        libraries=['cudart', 'curand'],
    )
    
    return [extension]

if __name__ == '__main__':
    # Set CUDA architecture for GTX 1080 Ti
    os.environ['TORCH_CUDA_ARCH_LIST'] = '6.1'
    
    setup(
        name='block_nerf_cuda',
        ext_modules=get_extensions(),
        cmdclass={'build_ext': BuildExtension},
        zip_safe=False,
        python_requires='>=3.7',
    )
