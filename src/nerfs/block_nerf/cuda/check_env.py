#!/usr/bin/env python3
"""
Simple test verification script for Block-NeRF CUDA environment
"""
import torch
import sys
import os

def main():
    print("🔍 Block-NeRF CUDA Test Environment Verification")
    print("=" * 50)
    
    # Check Python version
    print(f"Python version: {sys.version}")
    
    # Check PyTorch
    print(f"PyTorch version: {torch.__version__}")
    
    # Check CUDA
    if torch.cuda.is_available():
        print(f"✅ CUDA available")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Device count: {torch.cuda.device_count()}")
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name(0)}")
        props = torch.cuda.get_device_properties(0)
        print(f"Compute capability: {props.major}.{props.minor}")
        print(f"Total memory: {props.total_memory / 1024**3:.2f} GB")
        print(f"Multi-processor count: {props.multi_processor_count}")
    else:
        print("❌ CUDA not available")
        return False
    
    # Test basic tensor operations
    try:
        print("\n🧪 Testing basic CUDA operations...")
        a = torch.randn(1000, 1000, device='cuda')
        b = torch.randn(1000, 1000, device='cuda')
        c = torch.matmul(a, b)
        result = torch.sum(c).item()
        print(f"✅ Basic CUDA operations successful")
        print(f"   Matrix multiplication result: {result}")
    except Exception as e:
        print(f"❌ Basic CUDA operations failed: {e}")
        return False
    
    # Test memory allocation
    try:
        print("\n💾 Testing memory allocation...")
        large_tensor = torch.randn(5000, 5000, device='cuda')
        memory_used = torch.cuda.memory_allocated() / 1024**2
        print(f"✅ Large tensor allocation successful")
        print(f"   Memory used: {memory_used:.1f} MB")
        del large_tensor
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"❌ Memory allocation test failed: {e}")
        return False
    
    # Try to import extension
    try:
        print("\n📦 Testing Block-NeRF CUDA extension import...")
        import block_nerf_cuda
        print("✅ Block-NeRF CUDA extension imported successfully")
        
        # Test simple function if available
        try:
            test_tensor = torch.randn(100, 100, device='cuda')
            result = block_nerf_cuda.test_memory_bandwidth(test_tensor)
            print(f"✅ Extension function call successful")
            print(f"   Input shape: {test_tensor.shape}")
            print(f"   Output shape: {result.shape}")
            print(f"   Data preserved: {torch.allclose(test_tensor, result)}")
        except Exception as e:
            print(f"⚠️  Extension function call failed: {e}")
            print("   This may be normal if the function is not yet implemented")
        
    except ImportError:
        print("⚠️  Block-NeRF CUDA extension not available")
        print("   To build the extension, run:")
        print("   cd /home/xishansnow/3DVision/NeuroCity/src/nerfs/block_nerf/cuda")
        print("   bash build_cuda.sh")
        return False
    except Exception as e:
        print(f"❌ Extension import failed: {e}")
        return False
    
    # Check build files
    print("\n🔨 Checking build files...")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    build_files = [
        "setup.py",
        "block_nerf_cuda_kernels.cu", 
        "block_nerf_cuda.cpp",
        "block_nerf_cuda.h",
        "build_cuda.sh"
    ]
    
    for file in build_files:
        file_path = os.path.join(current_dir, file)
        if os.path.exists(file_path):
            print(f"   ✅ {file}")
        else:
            print(f"   ❌ {file} (missing)")
    
    print("\n🎉 Environment verification completed!")
    print("📋 Available test files:")
    test_files = [
        "comprehensive_test.py - Complete test suite", 
        "test_unit.py - Unit tests",
        "test_functional.py - Functional tests",
        "test_benchmark.py - Performance benchmarks", 
        "integration_example.py - End-to-end example",
        "test_full_resolution.py - 1920x1200 resolution test",
        "run_tests.py --all - Run all tests"
    ]
    
    for test_file in test_files:
        file_name = test_file.split(' - ')[0]
        file_path = os.path.join(current_dir, file_name)
        status = "✅" if os.path.exists(file_path) else "❌"
        print(f"   {status} {test_file}")
    
    return True

if __name__ == "__main__":
    success = main()
    print(f"\n{'✅ Environment ready!' if success else '❌ Environment setup needed'}")
    exit(0 if success else 1)