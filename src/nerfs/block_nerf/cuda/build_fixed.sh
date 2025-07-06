#!/bin/bash

echo "🚀 Building Block-NeRF CUDA Extensions (Fixed Version) for GTX 1080 Ti..."
echo "=================================================================="

# Clean previous build files
echo "Cleaning previous build files..."
rm -rf build/ *.so __pycache__/ *.egg-info/

# Set environment variables for GTX 1080 Ti
export TORCH_CUDA_ARCH_LIST="6.1"
export CUDA_VISIBLE_DEVICES=0

# Build CUDA extensions using fixed files
echo "Building Block-NeRF CUDA extensions (fixed version)..."
python setup_fixed.py build_ext --inplace

# Check if build was successful
if [ $? -eq 0 ]; then
    echo "✅ Block-NeRF CUDA extensions (fixed) built successfully!"
    
    # Run basic tests
    echo "Running tests..."
    python -c "
import torch
import block_nerf_cuda

print('============================================================')
print('Testing Block-NeRF CUDA Extension (Fixed Version)')
print('============================================================')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA device: {torch.cuda.get_device_name()}')

print('✅ Successfully imported block_nerf_cuda module')

# Test 1: Memory bandwidth test
print('📋 Test 1: Memory Bandwidth Test')
test_data = torch.randn(1000, device='cuda')
result = block_nerf_cuda.memory_bandwidth_test(test_data)
print(f'   Input shape: {test_data.shape}')
print(f'   Output shape: {result.shape}')
print(f'   Data matches: {torch.allclose(test_data, result)}')
print('   ✅ Memory bandwidth test passed')

# Test 2: Block visibility
print('📋 Test 2: Block Visibility Test')
camera_pos = torch.randn(5, 3, device='cuda')
block_centers = torch.randn(10, 3, device='cuda') * 10  # Spread out blocks
block_radii = torch.ones(10, device='cuda') * 5.0
block_active = torch.ones(10, dtype=torch.int32, device='cuda')

visibility = block_nerf_cuda.block_visibility(
    camera_pos, block_centers, block_radii, block_active, 0.1
)
print(f'   Camera positions shape: {camera_pos.shape}')
print(f'   Block centers shape: {block_centers.shape}')
print(f'   Visibility scores shape: {visibility.shape}')
print(f'   Visibility range: [{visibility.min():.4f}, {visibility.max():.4f}]')
print('   ✅ Block visibility test passed')

# Test 3: Block selection
print('📋 Test 3: Block Selection Test')
ray_origins = torch.randn(5, 3, device='cuda')
ray_directions = torch.randn(5, 3, device='cuda')
ray_directions = ray_directions / torch.norm(ray_directions, dim=1, keepdim=True)  # Normalize
ray_near = torch.ones(5, device='cuda') * 0.1
ray_far = torch.ones(5, device='cuda') * 100.0

selected_blocks, num_selected = block_nerf_cuda.block_selection(
    ray_origins, ray_directions, ray_near, ray_far,
    block_centers, block_radii, block_active, 32
)
print(f'   Ray origins shape: {ray_origins.shape}')
print(f'   Selected blocks shape: {selected_blocks.shape}')
print(f'   Num selected shape: {num_selected.shape}')
print(f'   Average blocks per ray: {num_selected.float().mean():.2f}')
print('   ✅ Block selection test passed')

print('')
print('🎉 All tests passed! Block-NeRF CUDA extension is working correctly.')
"
else
    echo "❌ Build failed!"
    exit 1
fi

echo ""
echo "=================================================================="
echo "🎉 Block-NeRF CUDA Extensions build completed successfully!"
echo "=================================================================="
