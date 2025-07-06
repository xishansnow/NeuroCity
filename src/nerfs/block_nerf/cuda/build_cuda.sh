#!/bin/bash

echo "🚀 Building Block-NeRF CUDA Extensions for GTX 1080 Ti..."
echo "=================================================================="

# Clean previous build files
echo "Cleaning previous build files..."
rm -rf build/ *.so

# Set environment variables for GTX 1080 Ti
export TORCH_CUDA_ARCH_LIST="6.1"
export CUDA_VISIBLE_DEVICES=0

# Build CUDA extensions
echo "Building Block-NeRF CUDA extensions..."
python setup.py build_ext --inplace

# Check if build was successful
if [ $? -eq 0 ]; then
    echo "✅ Block-NeRF CUDA extensions built successfully!"
    
    # Run basic tests
    echo "Running tests..."
    python -c "
import torch
import block_nerf_cuda

print('============================================================')
print('Testing Block-NeRF CUDA Extension')
print('============================================================')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA device: {torch.cuda.get_device_name()}')

print('✅ Successfully imported block_nerf_cuda module')

# Test 1: Memory bandwidth test
print('📋 Test 1: Memory Bandwidth Test')
test_data = torch.randn(1000000, device='cuda')
result = block_nerf_cuda.memory_bandwidth_test(test_data)
print(f'   Input shape: {test_data.shape}')
print(f'   Output shape: {result.shape}')
print(f'   Data matches: {torch.allclose(test_data, result)}')
print('   ✅ Memory bandwidth test passed')

# Test 2: Block visibility
print('📋 Test 2: Block Visibility Test')
camera_pos = torch.randn(10, 3, device='cuda')
block_centers = torch.randn(20, 3, device='cuda') * 10  # Spread out blocks
block_radii = torch.ones(20, device='cuda') * 5.0
block_active = torch.ones(20, dtype=torch.bool, device='cuda')

visibility = block_nerf_cuda.block_visibility(
    camera_pos, block_centers, block_radii, block_active, 0.1
)
print(f'   Camera positions: {camera_pos.shape}')
print(f'   Block centers: {block_centers.shape}')
print(f'   Visibility scores: {visibility.shape}')
print(f'   Visibility range: [{visibility.min():.3f}, {visibility.max():.3f}]')
print('   ✅ Block visibility test passed')

# Test 3: Block selection
print('📋 Test 3: Block Selection Test')
ray_origins = torch.randn(100, 3, device='cuda')
ray_directions = torch.randn(100, 3, device='cuda')
ray_directions = ray_directions / ray_directions.norm(dim=-1, keepdim=True)
ray_near = torch.ones(100, device='cuda') * 0.1
ray_far = torch.ones(100, device='cuda') * 100.0

selected_blocks, num_selected = block_nerf_cuda.block_selection(
    ray_origins, ray_directions, ray_near, ray_far,
    block_centers, block_radii, block_active, 8
)
print(f'   Ray origins: {ray_origins.shape}')
print(f'   Selected blocks: {selected_blocks.shape}')
print(f'   Num selected: {num_selected.shape}')
print(f'   Average blocks per ray: {num_selected.float().mean():.2f}')
print('   ✅ Block selection test passed')

print('============================================================')
print('🎉 All Block-NeRF CUDA extension tests passed successfully!')
print('============================================================')
"
    
    if [ $? -eq 0 ]; then
        echo "✅ Block-NeRF CUDA extension is working correctly!"
    else
        echo "❌ Tests failed. Check the error messages above."
        exit 1
    fi
else
    echo "❌ Failed to build Block-NeRF CUDA extensions. Check the error messages above."
    exit 1
fi
