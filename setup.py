from setuptools import setup, find_packages

setup(
    name="neurocity",
    version="1.0.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.20.0",
        "opencv-python>=4.5.0",
        "imageio>=2.9.0",
        "tqdm>=4.60.0",
        "wandb>=0.12.0",
        "tensorboard>=2.8.0",
    ],
    author="NeuroCity Team",
    description="Neural rendering toolkit for 3D vision",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
