from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

# Set CUDA architecture for GTX 1080 Ti (compute capability 6.1)
os.environ['TORCH_CUDA_ARCH_LIST'] = '6.1'

setup(
    name='plenoxels_cuda',
    ext_modules=[
        CUDAExtension('plenoxels_cuda', [
            'plenoxels_cuda.cpp',
            'ray_voxel_intersect_cuda.cu',
            'volume_rendering_cuda.cu',
            'feature_interpolation_cuda.cu'
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
) 