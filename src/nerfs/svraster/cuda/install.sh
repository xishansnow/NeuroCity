#!/bin/bash

# SVRaster CUDA Extension Installation Script

set -e

echo "🚀 Installing SVRaster CUDA Extension"
echo "====================================="

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "[INFO] Installation directory: $SCRIPT_DIR"

# Check if conda environment is activated
if [[ -z "$CONDA_DEFAULT_ENV" ]]; then
    echo "[WARNING] No conda environment detected. Please activate your conda environment first."
    echo "Example: conda activate neurocity"
    exit 1
fi

echo "[INFO] Conda environment: $CONDA_DEFAULT_ENV"

# Check Python version
PYTHON_VERSION=$(python --version 2>&1 | cut -d' ' -f2)
echo "[INFO] Python version: $PYTHON_VERSION"

# Check PyTorch installation
echo "[INFO] Checking PyTorch installation..."
python -c "import torch; print(f'PyTorch {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

if python -c "import torch; print(torch.cuda.is_available())" | grep -q "True"; then
    echo "[SUCCESS] CUDA is available"
    python -c "import torch; device = torch.cuda.get_device_properties(0); print(f'GPU: {device.name}'); print(f'Compute Capability: {device.major}.{device.minor}'); print(f'Memory: {device.total_memory / 1024**3:.1f} GB')"
else
    echo "[WARNING] CUDA is not available"
fi

# Check CUDA compiler
if command -v nvcc &> /dev/null; then
    NVCC_VERSION=$(nvcc --version | grep "release" | cut -d' ' -f6)
    echo "[SUCCESS] NVCC found: $NVCC_VERSION"
else
    echo "[ERROR] NVCC not found. Please install CUDA toolkit."
    exit 1
fi

# Check C++ compiler
if command -v g++ &> /dev/null; then
    GPP_VERSION=$(g++ --version | head -n1 | cut -d' ' -f4)
    echo "[SUCCESS] G++ found: gcc $GPP_VERSION"
else
    echo "[ERROR] G++ not found. Please install a C++ compiler."
    exit 1
fi

# Set library path
TORCH_LIB_PATH="$CONDA_PREFIX/lib/python3.10/site-packages/torch/lib"
if [[ -d "$TORCH_LIB_PATH" ]]; then
    export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$TORCH_LIB_PATH"
    echo "[INFO] Added PyTorch library path: $TORCH_LIB_PATH"
else
    echo "[WARNING] PyTorch library path not found: $TORCH_LIB_PATH"
fi

# Build CUDA extension
echo "[INFO] Building CUDA extension..."
cd "$SCRIPT_DIR"

# Clean previous builds
rm -rf build/ *.so

# Build extension
python setup.py build_ext --inplace

# Check if build was successful
if [[ -f "voxel_rasterizer_cuda.cpython-310-x86_64-linux-gnu.so" ]]; then
    echo "[SUCCESS] CUDA extension built successfully"
else
    echo "[ERROR] CUDA extension build failed"
    exit 1
fi

# Test the extension
echo "[INFO] Testing CUDA extension..."
python test_cuda_extension.py

# Create environment setup script
ENV_SETUP_SCRIPT="$SCRIPT_DIR/setup_env.sh"
cat > "$ENV_SETUP_SCRIPT" << EOF
#!/bin/bash
# Environment setup script for SVRaster CUDA Extension
# Source this script before running your Python code

export LD_LIBRARY_PATH="\$LD_LIBRARY_PATH:$TORCH_LIB_PATH"
echo "Environment variables set for SVRaster CUDA Extension"
echo "PyTorch library path: $TORCH_LIB_PATH"
EOF

chmod +x "$ENV_SETUP_SCRIPT"

echo "[SUCCESS] Installation completed successfully!"
echo ""
echo "📋 Installation Summary:"
echo "========================"
echo "✓ CUDA extension built and tested"
echo "✓ Environment setup script created: $ENV_SETUP_SCRIPT"
echo ""
echo "🔧 Usage Instructions:"
echo "====================="
echo "1. Before running your code, set the environment:"
echo "   source $ENV_SETUP_SCRIPT"
echo ""
echo "2. Or add this to your ~/.bashrc:"
echo "   export LD_LIBRARY_PATH=\"\$LD_LIBRARY_PATH:$TORCH_LIB_PATH\""
echo ""
echo "3. Test the installation:"
echo "   python test_cuda_extension.py"
echo ""
echo "🎉 SVRaster CUDA Extension is ready to use!" 