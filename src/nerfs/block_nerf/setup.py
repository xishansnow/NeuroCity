"""
Setup script for Block-NeRF package

This script allows Block-NeRF to be installed as a standard Python package
with pip install -e .
"""

from setuptools import setup, find_packages
import os

# Read README
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="block_nerf",
    version="1.0.0",
    author="NeuroCity Team",
    author_email="contact@neurocity.com",
    description="Block-NeRF: Scalable Large Scene Neural View Synthesis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/neurocity/block_nerf",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Multimedia :: Graphics :: 3D Rendering",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "cuda": ["torch>=1.12.0", "torchvision>=0.13.0"],
        "dev": ["pytest>=6.0", "black>=21.0", "flake8>=3.8"],
    },
    entry_points={
        "console_scripts": [
            "block-nerf=block_nerf.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "block_nerf": [
            "config_template.yaml",
            "cuda/*.cu",
            "cuda/*.cpp",
            "cuda/*.py",
            "cuda/*.sh",
        ],
    },
    zip_safe=False,
)
