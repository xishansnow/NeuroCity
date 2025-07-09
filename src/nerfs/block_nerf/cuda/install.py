#!/usr/bin/env python3

"""
Block-NeRF CUDA Extension Installation Script

This script installs the Block-NeRF CUDA extension and sets up
the environment for proper usage.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def run_command(cmd, description, check=True):
    """Run a shell command with error handling"""
    print(f"📋 {description}")
    print(f"   Command: {cmd}")
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"   ✓ Success")
        if result.stdout.strip():
            print(f"   Output: {result.stdout.strip()}")
        return True
    else:
        print(f"   ✗ Failed (exit code: {result.returncode})")
        if result.stderr.strip():
            print(f"   Error: {result.stderr.strip()}")
        if check:
            return False
        return True

def check_environment():
    """Check if the environment is ready for building"""
    print("🔍 Checking Environment")
    print("=" * 40)
    
    # Check Python
    print(f"Python version: {sys.version}")
    
    # Check PyTorch
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA device: {torch.cuda.get_device_name()}")
        return True
    except ImportError:
        print("❌ PyTorch not available")
        return False

def clean_build():
    """Clean previous build artifacts"""
    print("\n🧹 Cleaning Previous Builds")
    print("=" * 40)
    
    patterns_to_remove = [
        "*.so",
        "build/",
        "dist/",
        "*.egg-info/",
        "__pycache__/",
    ]
    
    for pattern in patterns_to_remove:
        if pattern.endswith('/'):
            # Directory
            if os.path.exists(pattern):
                shutil.rmtree(pattern)
                print(f"   Removed directory: {pattern}")
        else:
            # Files
            import glob
            files = glob.glob(pattern)
            for file in files:
                os.remove(file)
                print(f"   Removed file: {file}")

def build_extension():
    """Build the CUDA extension"""
    print("\n🔨 Building CUDA Extension")
    print("=" * 40)
    
    # Set environment variables
    os.environ['TORCH_CUDA_ARCH_LIST'] = '6.1'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    
    # Build command
    cmd = "python3 setup.py build_ext --inplace"
    success = run_command(cmd, "Building CUDA extension")
    
    if success:
        # Check if .so file was created
        import glob
        so_files = glob.glob("*.so")
        if so_files:
            print(f"   ✓ Shared library created: {so_files[0]}")
            return True
        else:
            print("   ⚠️  No shared library found")
            return False
    
    return False

def test_installation():
    """Test the installed extension"""
    print("\n🧪 Testing Installation")
    print("=" * 40)
    
    # Set library path
    lib_path = "/home/xishansnow/anaconda3/envs/neurocity/lib/python3.10/site-packages/torch/lib"
    current_ld_path = os.environ.get('LD_LIBRARY_PATH', '')
    if lib_path not in current_ld_path:
        os.environ['LD_LIBRARY_PATH'] = f"{current_ld_path}:{lib_path}"
    
    try:
        import block_nerf_cuda_simple
        print("   ✓ Import successful")
        
        # Quick functionality test
        import torch
        if torch.cuda.is_available():
            test_tensor = torch.randn(100, device='cuda', dtype=torch.float32)
            result = block_nerf_cuda_simple.memory_bandwidth_test(test_tensor)
            print("   ✓ Basic functionality test passed")
        
        functions = [attr for attr in dir(block_nerf_cuda_simple) if not attr.startswith('_')]
        print(f"   Available functions: {functions}")
        return True
        
    except Exception as e:
        print(f"   ✗ Test failed: {e}")
        return False

def create_environment_setup():
    """Create a script to set up environment variables"""
    print("\n📝 Creating Environment Setup Script")
    print("=" * 40)
    
    setup_script = """#!/bin/bash

# Block-NeRF CUDA Extension Environment Setup
# Source this script before using the extension

# Add PyTorch libraries to LD_LIBRARY_PATH
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/home/xishansnow/anaconda3/envs/neurocity/lib/python3.10/site-packages/torch/lib"

# Set CUDA environment
export TORCH_CUDA_ARCH_LIST="6.1"
export CUDA_VISIBLE_DEVICES=0

echo "✓ Block-NeRF CUDA environment configured"
echo "  LD_LIBRARY_PATH updated"
echo "  CUDA architecture set to 6.1 (GTX 1080 Ti)"
"""
    
    with open("env_setup.sh", "w") as f:
        f.write(setup_script)
    
    os.chmod("env_setup.sh", 0o755)
    print("   ✓ Created env_setup.sh")
    print("   Usage: source env_setup.sh")

def main():
    """Main installation function"""
    print("🚀 Block-NeRF CUDA Extension Installation")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not os.path.exists("setup.py"):
        print("❌ setup.py not found. Make sure you're in the CUDA directory.")
        return False
    
    # Check environment
    if not check_environment():
        print("❌ Environment check failed")
        return False
    
    # Clean previous builds
    clean_build()
    
    # Build extension
    if not build_extension():
        print("❌ Build failed")
        return False
    
    # Test installation
    if not test_installation():
        print("❌ Installation test failed")
        return False
    
    # Create environment setup script
    create_environment_setup()
    
    print("\n" + "=" * 50)
    print("🎉 Installation Complete!")
    print("=" * 50)
    print()
    print("To use the Block-NeRF CUDA extension:")
    print("1. Source the environment setup script:")
    print("   source env_setup.sh")
    print()
    print("2. Import and use in Python:")
    print("   import block_nerf_cuda_simple")
    print("   # Use the functions...")
    print()
    print("Available functions:")
    print("  - memory_bandwidth_test(tensor)")
    print("  - simple_add(a, b)")
    print("  - simple_multiply(a, b)")
    print("  - block_visibility(camera_pos, block_centers, block_radii, view_dirs)")
    print("  - block_selection(rays_o, rays_d, block_centers, block_radii)")
    print()
    print("Run test_cuda.py to verify everything works correctly.")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
