"""
Instant NGP Setup Script

This setup script builds the Instant NGP package with optional CUDA extensions.
"""

import os
import sys
from pathlib import Path
from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext
from pybind11 import get_cmake_dir
import pybind11

# Check if CUDA is available
def cuda_is_available():
    """Check if CUDA is available for compilation."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False

def get_cuda_extensions():
    """Get CUDA extensions if available."""
    if not cuda_is_available():
        print("CUDA not available, skipping CUDA extensions")
        return []
    
    cuda_dir = Path(__file__).parent / "cuda"
    
    # CUDA extension sources
    cuda_sources = [
        str(cuda_dir / "instant_ngp_cuda.cpp"),
        str(cuda_dir / "hash_encoding_kernel.cu"),
    ]
    
    # Check if source files exist
    missing_files = [f for f in cuda_sources if not Path(f).exists()]
    if missing_files:
        print(f"Warning: Missing CUDA source files: {missing_files}")
        return []
    
    # CUDA compilation flags
    cuda_flags = [
        '-DWITH_CUDA',
        '-DTORCH_EXTENSION_NAME=instant_ngp_cuda',
        '--extended-lambda',
        '--expt-relaxed-constexpr',
        '-use_fast_math',
        '-O3',
    ]
    
    # Architecture-specific flags
    cuda_archs = ['60', '61', '70', '75', '80', '86']  # Common architectures
    for arch in cuda_archs:
        cuda_flags.extend(['-gencode', f'arch=compute_{arch},code=sm_{arch}'])
    
    cuda_extension = Pybind11Extension(
        "instant_ngp.cuda.instant_ngp_cuda",
        sources=cuda_sources,
        include_dirs=[
            str(cuda_dir),
            pybind11.get_include(),
        ],
        language='c++',
        cxx_std=17,
        extra_compile_args={
            'cxx': ['-O3', '-std=c++17'],
            'nvcc': cuda_flags
        }
    )
    
    return [cuda_extension]

def get_extensions():
    """Get all extensions (CUDA if available)."""
    extensions = []
    
    # Add CUDA extensions if available
    cuda_extensions = get_cuda_extensions()
    extensions.extend(cuda_extensions)
    
    return extensions

def get_long_description():
    """Get long description from README."""
    readme_path = Path(__file__).parent / "README.md"
    if readme_path.exists():
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "Instant Neural Graphics Primitives with Multiresolution Hash Encoding"

class CustomBuildExt(build_ext):
    """Custom build extension with error handling."""
    
    def build_extension(self, ext):
        try:
            super().build_extension(ext)
        except Exception as e:
            print(f"Failed to build extension {ext.name}: {e}")
            print("Continuing without CUDA acceleration...")

if __name__ == "__main__":
    # Get extensions
    ext_modules = get_extensions()
    
    # Setup configuration
    setup(
        name="instant_ngp",
        version="1.0.0",
        author="NeuroCity Team",
        author_email="team@neurocity.ai",
        description="Instant Neural Graphics Primitives with Multiresolution Hash Encoding",
        long_description=get_long_description(),
        long_description_content_type="text/markdown",
        url="https://github.com/neurocity/instant-ngp",
        ext_modules=ext_modules,
        cmdclass={"build_ext": CustomBuildExt},
        zip_safe=False,
        python_requires=">=3.8",
    )
