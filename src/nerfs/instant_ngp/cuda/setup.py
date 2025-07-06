from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

# Set CUDA architecture for GTX 1080 Ti (compute capability 6.1)
os.environ['TORCH_CUDA_ARCH_LIST'] = '6.1'

setup(
    name='instant_ngp_cuda',
    ext_modules=[
        CUDAExtension('instant_ngp_cuda', [
            'instant_ngp_cuda.cpp',
            'hash_encoding_kernel.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
