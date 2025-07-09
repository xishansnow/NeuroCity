#!/usr/bin/env python3

"""
Block-NeRF CUDA Extension Test Script

This script tests the simplified CUDA extension functionality
including memory bandwidth, basic operations, and block-specific functions.
"""

import torch
import numpy as np
import time
import os
import sys

# Set library path
lib_path = "/home/xishansnow/anaconda3/envs/neurocity/lib/python3.10/site-packages/torch/lib"
if lib_path not in os.environ.get('LD_LIBRARY_PATH', ''):
    os.environ['LD_LIBRARY_PATH'] = f"{os.environ.get('LD_LIBRARY_PATH', '')}:{lib_path}"

def test_import():
    """Test importing the CUDA extension"""
    print("=== Testing Import ===")
    try:
        import block_nerf_cuda_simple
        print("✓ Import successful")
        functions = [attr for attr in dir(block_nerf_cuda_simple) if not attr.startswith('_')]
        print(f"Available functions: {functions}")
        return block_nerf_cuda_simple
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return None
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return None

def test_memory_bandwidth(cuda_ext):
    """Test memory bandwidth functionality"""
    print("\n=== Testing Memory Bandwidth ===")
    try:
        # Create test tensor
        size = 1024 * 1024  # 1M elements
        input_tensor = torch.randn(size, device='cuda', dtype=torch.float32)
        
        # Test memory bandwidth
        start_time = time.time()
        output = cuda_ext.memory_bandwidth_test(input_tensor)
        end_time = time.time()
        
        # Verify result
        expected = input_tensor * 2.0
        diff = torch.abs(output - expected).max().item()
        
        print(f"✓ Memory bandwidth test passed")
        print(f"  Tensor size: {input_tensor.numel()} elements")
        print(f"  Memory used: {input_tensor.numel() * 4 / 1024 / 1024:.2f} MB")
        print(f"  Execution time: {(end_time - start_time) * 1000:.2f} ms")
        print(f"  Max difference: {diff:.6f}")
        
        return True
    except Exception as e:
        print(f"✗ Memory bandwidth test failed: {e}")
        return False

def test_basic_operations(cuda_ext):
    """Test basic CUDA operations"""
    print("\n=== Testing Basic Operations ===")
    try:
        # Test addition
        a = torch.randn(1000, device='cuda', dtype=torch.float32)
        b = torch.randn(1000, device='cuda', dtype=torch.float32)
        
        result_add = cuda_ext.simple_add(a, b)
        expected_add = a + b
        diff_add = torch.abs(result_add - expected_add).max().item()
        
        print(f"✓ Addition test passed (max diff: {diff_add:.6f})")
        
        # Test multiplication
        result_mul = cuda_ext.simple_multiply(a, b)
        expected_mul = a * b
        diff_mul = torch.abs(result_mul - expected_mul).max().item()
        
        print(f"✓ Multiplication test passed (max diff: {diff_mul:.6f})")
        
        return True
    except Exception as e:
        print(f"✗ Basic operations test failed: {e}")
        return False

def test_block_visibility(cuda_ext):
    """Test block visibility computation"""
    print("\n=== Testing Block Visibility ===")
    try:
        # Create test data
        num_cameras = 10
        num_blocks = 20
        
        camera_positions = torch.randn(num_cameras, 3, device='cuda', dtype=torch.float32)
        block_centers = torch.randn(num_blocks, 3, device='cuda', dtype=torch.float32)
        block_radii = torch.rand(num_blocks, device='cuda', dtype=torch.float32) * 2.0 + 0.5
        view_directions = torch.randn(num_cameras, 3, device='cuda', dtype=torch.float32)
        
        # Normalize view directions
        view_directions = view_directions / torch.norm(view_directions, dim=1, keepdim=True)
        
        # Test block visibility
        start_time = time.time()
        visibility = cuda_ext.block_visibility(
            camera_positions, block_centers, block_radii, view_directions, 0.3
        )
        end_time = time.time()
        
        # Verify output shape
        expected_shape = (num_cameras, num_blocks)
        if visibility.shape != expected_shape:
            raise ValueError(f"Wrong output shape: {visibility.shape} vs {expected_shape}")
        
        # Check visibility values
        vis_min = visibility.min().item()
        vis_max = visibility.max().item()
        vis_mean = visibility.mean().item()
        
        print(f"✓ Block visibility test passed")
        print(f"  Output shape: {visibility.shape}")
        print(f"  Visibility range: [{vis_min:.3f}, {vis_max:.3f}]")
        print(f"  Mean visibility: {vis_mean:.3f}")
        print(f"  Execution time: {(end_time - start_time) * 1000:.2f} ms")
        
        return True
    except Exception as e:
        print(f"✗ Block visibility test failed: {e}")
        return False

def test_block_selection(cuda_ext):
    """Test block selection functionality"""
    print("\n=== Testing Block Selection ===")
    try:
        # Create test data
        num_rays = 100
        num_blocks = 50
        max_blocks = 8
        
        rays_o = torch.randn(num_rays, 3, device='cuda', dtype=torch.float32)
        rays_d = torch.randn(num_rays, 3, device='cuda', dtype=torch.float32)
        rays_d = rays_d / torch.norm(rays_d, dim=1, keepdim=True)  # Normalize
        
        block_centers = torch.randn(num_blocks, 3, device='cuda', dtype=torch.float32)
        block_radii = torch.rand(num_blocks, device='cuda', dtype=torch.float32) * 2.0 + 0.5
        
        # Test block selection
        start_time = time.time()
        selected_blocks, num_selected = cuda_ext.block_selection(
            rays_o, rays_d, block_centers, block_radii, max_blocks
        )
        end_time = time.time()
        
        # Verify output shapes
        if selected_blocks.shape != (num_rays, max_blocks):
            raise ValueError(f"Wrong selected_blocks shape: {selected_blocks.shape}")
        if num_selected.shape != (num_rays,):
            raise ValueError(f"Wrong num_selected shape: {num_selected.shape}")
        
        # Check selection statistics
        avg_selected = num_selected.float().mean().item()
        max_selected = num_selected.max().item()
        
        print(f"✓ Block selection test passed")
        print(f"  Selected blocks shape: {selected_blocks.shape}")
        print(f"  Average blocks selected: {avg_selected:.2f}")
        print(f"  Max blocks selected: {max_selected}")
        print(f"  Execution time: {(end_time - start_time) * 1000:.2f} ms")
        
        return True
    except Exception as e:
        print(f"✗ Block selection test failed: {e}")
        return False

def test_performance_benchmark(cuda_ext):
    """Run performance benchmark"""
    print("\n=== Performance Benchmark ===")
    try:
        # Test different sizes
        sizes = [1024, 10240, 102400, 1024000]
        
        for size in sizes:
            input_tensor = torch.randn(size, device='cuda', dtype=torch.float32)
            
            # Warm up
            for _ in range(5):
                _ = cuda_ext.memory_bandwidth_test(input_tensor)
            
            # Benchmark
            torch.cuda.synchronize()
            start_time = time.time()
            for _ in range(10):
                _ = cuda_ext.memory_bandwidth_test(input_tensor)
            torch.cuda.synchronize()
            end_time = time.time()
            
            avg_time = (end_time - start_time) / 10 * 1000  # ms
            bandwidth = (size * 4 * 2) / (avg_time / 1000) / 1024 / 1024 / 1024  # GB/s
            
            print(f"  Size {size:>7}: {avg_time:>6.2f} ms, {bandwidth:>6.2f} GB/s")
        
        return True
    except Exception as e:
        print(f"✗ Performance benchmark failed: {e}")
        return False

def main():
    """Main test function"""
    print("🧪 Block-NeRF CUDA Extension Test Suite")
    print("=" * 50)
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return False
    
    print(f"✓ CUDA available: {torch.cuda.get_device_name()}")
    
    # Test import
    cuda_ext = test_import()
    if cuda_ext is None:
        print("❌ Cannot import CUDA extension")
        return False
    
    # Run tests
    tests = [
        ("Memory Bandwidth", test_memory_bandwidth),
        ("Basic Operations", test_basic_operations),
        ("Block Visibility", test_block_visibility),
        ("Block Selection", test_block_selection),
        ("Performance Benchmark", test_performance_benchmark),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func(cuda_ext)
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Summary")
    print("=" * 50)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{test_name:<25}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed!")
        return True
    else:
        print("❌ Some tests failed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
