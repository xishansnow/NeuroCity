#!/bin/bash

# SVRaster CUDA Extension Build Script
# This script automates the compilation of the GPU-optimized SVRaster implementation

set -e  # Exit on any error

echo "🚀 Building SVRaster CUDA Extension"
echo "=================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if we're in the right directory
if [ ! -f "svraster_cuda_kernel.h" ]; then
    print_error "Please run this script from the src/nerfs/svraster/ directory"
    exit 1
fi

# Check Python availability
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is required but not installed"
    exit 1
fi

print_status "Python 3 found: $(python3 --version)"

# Check PyTorch installation
print_status "Checking PyTorch installation..."
if ! python3 -c "import torch; print(f'PyTorch {torch.__version__}')" 2>/dev/null; then
    print_error "PyTorch is not installed. Please install PyTorch first."
    exit 1
fi

# Check CUDA availability
print_status "Checking CUDA availability..."
if ! python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')" 2>/dev/null; then
    print_warning "CUDA not available. The extension will be built with CPU fallback."
    CUDA_AVAILABLE=false
else
    CUDA_AVAILABLE=true
    print_success "CUDA is available"
    
    # Get CUDA device info
    python3 -c "
import torch
if torch.cuda.is_available():
    device = torch.cuda.get_device_properties(0)
    print(f'GPU: {device.name}')
    print(f'Compute Capability: {device.major}.{device.minor}')
    print(f'Memory: {device.total_memory / 1e9:.1f} GB')
"
fi

# Check CUDA compiler
if command -v nvcc &> /dev/null; then
    NVCC_VERSION=$(nvcc --version | grep "release" | awk '{print $6}' | cut -d',' -f1)
    print_success "NVCC found: $NVCC_VERSION"
else
    print_warning "NVCC not found. CUDA compilation may fail."
fi

# Check C++ compiler
if command -v g++ &> /dev/null; then
    GCC_VERSION=$(g++ --version | head -n1 | awk '{print $3}')
    print_success "G++ found: $GCC_VERSION"
elif command -v clang++ &> /dev/null; then
    CLANG_VERSION=$(clang++ --version | head -n1 | awk '{print $3}')
    print_success "Clang++ found: $CLANG_VERSION"
else
    print_error "No C++ compiler found. Please install g++ or clang++"
    exit 1
fi

# Create build directory
BUILD_DIR="build"
if [ ! -d "$BUILD_DIR" ]; then
    print_status "Creating build directory..."
    mkdir -p "$BUILD_DIR"
fi

# Clean previous builds
print_status "Cleaning previous builds..."
rm -rf "$BUILD_DIR"/*
rm -f *.so
rm -f *.pyd

# Build the extension
print_status "Building CUDA extension..."
cd "$BUILD_DIR"

# Run setup.py with verbose output
python3 ../setup_cuda.py build_ext --inplace -v

# Check if build was successful
if [ -f "../svraster_cuda*.so" ] || [ -f "../svraster_cuda*.pyd" ]; then
    print_success "CUDA extension built successfully!"
    
    # Get the extension file
    EXTENSION_FILE=$(find .. -name "svraster_cuda*" -type f | head -n1)
    if [ -n "$EXTENSION_FILE" ]; then
        FILE_SIZE=$(du -h "$EXTENSION_FILE" | cut -f1)
        print_success "Extension file: $(basename "$EXTENSION_FILE") ($FILE_SIZE)"
    fi
else
    print_error "Build failed! Extension file not found."
    exit 1
fi

cd ..

# Test the extension
print_status "Testing the extension..."
if python3 -c "
try:
    import svraster_cuda
    print('✅ CUDA extension imported successfully')
    print(f'Available functions: {list(svraster_cuda.__dict__.keys())}')
except ImportError as e:
    print(f'❌ Failed to import extension: {e}')
    exit(1)
" 2>/dev/null; then
    print_success "Extension test passed!"
else
    print_error "Extension test failed!"
    exit 1
fi

# Run comprehensive tests if available
if [ -f "test_svraster_gpu.py" ]; then
    print_status "Running GPU tests..."
    if python3 test_svraster_gpu.py; then
        print_success "All GPU tests passed!"
    else
        print_warning "Some GPU tests failed. Check the output above."
    fi
else
    print_warning "GPU test file not found. Skipping tests."
fi

# Performance benchmark
print_status "Running performance benchmark..."
python3 -c "
import torch
import time
import sys
sys.path.insert(0, '.')

try:
    from svraster_gpu import SVRasterGPU
    from core import SVRasterConfig
    
    config = SVRasterConfig(
        max_octree_levels=8,
        base_resolution=32,
        scene_bounds=(-1.0, -1.0, -1.0, 1.0, 1.0, 1.0)
    )
    
    model = SVRasterGPU(config)
    
    # Benchmark
    ray_origins = torch.randn(1000, 3, device=model.device)
    ray_directions = torch.randn(1000, 3, device=model.device)
    ray_directions = ray_directions / ray_directions.norm(dim=1, keepdim=True)
    
    # Warmup
    for _ in range(3):
        _ = model(ray_origins, ray_directions)
    
    # Benchmark
    torch.cuda.synchronize()
    start_time = time.time()
    
    for _ in range(10):
        outputs = model(ray_origins, ray_directions)
    
    torch.cuda.synchronize()
    end_time = time.time()
    
    avg_time = (end_time - start_time) / 10
    rays_per_second = 1000 / avg_time
    
    print(f'✅ Performance benchmark completed')
    print(f'   Average time: {avg_time:.4f} seconds')
    print(f'   Rays per second: {rays_per_second:.0f}')
    
except Exception as e:
    print(f'❌ Performance benchmark failed: {e}')
" 2>/dev/null

# Installation
print_status "Installing extension..."
if python3 setup_cuda.py install --user; then
    print_success "Extension installed successfully!"
else
    print_warning "Installation failed. You can still use the extension from the current directory."
fi

# Summary
echo ""
echo "🎉 SVRaster CUDA Extension Build Complete!"
echo "=========================================="
echo ""
echo "📁 Files created:"
ls -la *.so *.pyd 2>/dev/null || echo "   (Extension files in current directory)"
echo ""
echo "🚀 Next steps:"
echo "   1. Import the extension: import svraster_cuda"
echo "   2. Use the GPU model: from svraster_gpu import SVRasterGPU"
echo "   3. Run tests: python3 test_svraster_gpu.py"
echo ""
echo "📚 Documentation: README_GPU.md"
echo ""

print_success "Build completed successfully!" 