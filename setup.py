from setuptools import setup, find_packages, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext
import pybind11

# CUDA extensions for Plenoxels
cuda_extensions = []

try:
    import torch
    from torch.utils.cpp_extension import CUDAExtension
    
    cuda_extensions = [
        CUDAExtension(
            name='plenoxels_cuda',
            sources=[
                'src/nerfs/plenoxels/cuda/plenoxels_cuda.cpp',
                'src/nerfs/plenoxels/cuda/volume_rendering_cuda.cu',
                'src/nerfs/plenoxels/cuda/feature_interpolation_cuda.cu',
                'src/nerfs/plenoxels/cuda/ray_voxel_intersect_cuda.cu',
            ],
            include_dirs=[
                'src/nerfs/plenoxels/cuda',
            ],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': ['-O3', '--use_fast_math']
            }
        )
    ]
except ImportError:
    print("Warning: PyTorch not found. CUDA extensions will not be built.")

setup(
    name="neurocity",
    version="2.0.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.1.0",
        "torchvision>=0.16.0",
        "numpy>=1.26.4",
        "scipy>=1.11.3",
        "opencv-python>=4.8.1",
        "pillow>=10.1.0",
        "matplotlib>=3.8.2",
        "tqdm>=4.66.1",
        "wandb>=0.15.12",
        "tensorboard>=2.14.0",
        "pytorch-lightning>=2.0.0",
        "torchmetrics>=1.2.0",
        "pyyaml>=6.0.1",
        "rich>=13.7.0",
        "tyro>=0.5.12",
        "open3d>=0.17.0",
        "trimesh>=4.0.5",
        "scikit-image>=0.22.0",
        "ninja>=1.11.1",
    ],
    ext_modules=cuda_extensions,
    cmdclass={'build_ext': build_ext},
    author="NeuroCity Development Team",
    author_email="info@neurocity.dev",
    description="Advanced neural rendering toolkit with Plenoxels support",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/neurocity/neurocity",
    license="MIT",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Visualization",
        "Topic :: Multimedia :: Graphics :: 3D Rendering",
    ],
    keywords="neural rendering, nerf, plenoxels, 3d reconstruction, computer vision",
    project_urls={
        "Bug Reports": "https://github.com/neurocity/neurocity/issues",
        "Source": "https://github.com/neurocity/neurocity",
        "Documentation": "https://neurocity.readthedocs.io/",
    },
    include_package_data=True,
    zip_safe=False,
)
