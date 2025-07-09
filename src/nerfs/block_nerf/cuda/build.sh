#!/bin/bash

# Simple Build Script for Block-NeRF CUDA Extension
# This script builds the simplified version of the extension

echo "=== Block-NeRF CUDA Extension Simple Build Script ==="
echo "Starting build process..."

# Navigate to the CUDA directory
cd "$(dirname "$0")"
echo "Working directory: $(pwd)"

# Check if we're in the right directory
if [ ! -f "setup.py" ]; then
    echo "ERROR: setup.py not found. Make sure you're in the correct directory."
    exit 1
fi

# Check Python and PyTorch
echo ""
echo "=== Environment Check ==="
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python3 -c "import torch; print(f'CUDA version: {torch.version.cuda}')" 2>/dev/null || echo "CUDA version: Unknown"

# Clean previous builds
echo ""
echo "=== Cleaning Previous Builds ==="
rm -rf build/ dist/ *.egg-info/ __pycache__/
find . -name "*.so" -delete
echo "Cleaned build artifacts"

# Set environment variables
export TORCH_CUDA_ARCH_LIST="6.1"
export CUDA_VISIBLE_DEVICES=0

# Build the extension
echo ""
echo "=== Building CUDA Extension ==="
echo "Building with setup.py..."

python3 setup.py build_ext --inplace --verbose

# Check if build was successful
if [ $? -eq 0 ]; then
    echo ""
    echo "=== Build Success ==="
    echo "CUDA extension built successfully!"
    
    # Check if the shared library was created
    if [ -f "block_nerf_cuda_simple*.so" ] || [ -f "block_nerf_cuda_simple.cpython*.so" ]; then
        echo "Shared library found:"
        ls -la *.so 2>/dev/null || echo "No .so files found in current directory"
        find . -name "*.so" -type f 2>/dev/null
    else
        echo "WARNING: No shared library found. Build may have issues."
    fi
    
    echo ""
    echo "=== Quick Test ==="
    echo "Running basic import test..."
    python3 -c "
try:
    import block_nerf_cuda_simple
    print('✓ Import successful')
    print(f'Available functions: {dir(block_nerf_cuda_simple)}')
except ImportError as e:
    print(f'✗ Import failed: {e}')
except Exception as e:
    print(f'✗ Unexpected error: {e}')
"
    
else
    echo ""
    echo "=== Build Failed ==="
    echo "CUDA extension build failed. Check the error messages above."
    exit 1
fi

echo ""
echo "=== Build Complete ==="
echo "You can now test the extension with:"
echo "  python3 quick_test.py"
echo "  python3 test_unit.py"
