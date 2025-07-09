#!/usr/bin/env python3
"""
Setup script for SVRaster CUDA extension
"""

from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os


# Get CUDA version
def get_cuda_version():
    import torch

    if torch.cuda.is_available():
        try:
            return torch.version.cuda
        except AttributeError:
            return "11.8"
    return "11.8"  # Default fallback


# Get compute capability
def get_compute_capability():
    import torch

    if torch.cuda.is_available():
        device = torch.cuda.get_device_properties(0)
        return f"sm_{device.major}{device.minor}"
    return "sm_70"  # Default fallback


def main():
    cuda_version = get_cuda_version()
    compute_cap = get_compute_capability()

    print(
        f"Building SVRaster CUDA extension with CUDA {cuda_version}, compute capability {compute_cap}"
    )

    # Get the current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Define the CUDA extensions
    svraster_extension = CUDAExtension(
        name="svraster_cuda",
        sources=[
            os.path.join(current_dir, "trainer", "svraster_cuda.cpp"),
            os.path.join(current_dir, "trainer", "svraster_cuda_kernel.cu"),
        ],
        extra_compile_args={
            "cxx": ["-O3", "-std=c++17", "-Wall", "-Wextra"],
            "nvcc": [
                "-O3",
                "--use_fast_math",
                f"-arch={compute_cap}",
                "--std=c++17",
                "--extended-lambda",
                "--expt-relaxed-constexpr",
                "-Xcompiler",
                "-Wall",
                "-Xcompiler",
                "-Wextra",
            ],
        },
        include_dirs=[os.path.join(current_dir, "include")],
    )

    voxel_rasterizer_extension = CUDAExtension(
        name="voxel_rasterizer_cuda",
        sources=[
            os.path.join(current_dir, "renderer", "voxel_rasterizer_cuda.cpp"),
            os.path.join(current_dir, "renderer", "voxel_rasterizer_cuda_kernel.cu"),
        ],
        extra_compile_args={
            "cxx": ["-O3", "-std=c++17", "-Wall", "-Wextra"],
            "nvcc": [
                "-O3",
                "--use_fast_math",
                f"-arch={compute_cap}",
                "--std=c++17",
                "--extended-lambda",
                "--expt-relaxed-constexpr",
                "-Xcompiler",
                "-Wall",
                "-Xcompiler",
                "-Wextra",
            ],
        },
        include_dirs=[os.path.join(current_dir, "include")],
    )

    # Setup configuration
    setup(
        name="svraster_cuda",
        version="1.0.0",
        description="GPU-optimized SVRaster implementation",
        author="NeuroCity Team",
        packages=find_packages(),
        ext_modules=[svraster_extension, voxel_rasterizer_extension],
        cmdclass={"build_ext": BuildExtension},
        python_requires=">=3.8",
        install_requires=[
            "torch>=1.12.0",
            "numpy>=1.21.0",
        ],
        classifiers=[
            "Development Status :: 4 - Beta",
            "Intended Audience :: Developers",
            "Intended Audience :: Science/Research",
            "License :: OSI Approved :: MIT License",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "Programming Language :: Python :: 3.11",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
            "Topic :: Scientific/Engineering :: Image Processing",
        ],
    )


if __name__ == "__main__":
    main()
