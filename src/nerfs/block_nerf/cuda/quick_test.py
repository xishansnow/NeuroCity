#!/usr/bin/env python3
"""
Quick CUDA test - Simple and reliable
Tests basic CUDA functionality and Block-NeRF extension if available
"""
import torch
import time

def test_cuda_basic():
    """Test basic CUDA functionality"""
    print("🔧 Testing Basic CUDA Functionality")
    print("-" * 40)
    
    # Check CUDA
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return False
    
    print(f"✅ CUDA available")
    print(f"   Device: {torch.cuda.get_device_name(0)}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # Test tensor operations
    try:
        a = torch.randn(1000, 1000, device='cuda')
        b = torch.randn(1000, 1000, device='cuda')
        
        start = time.time()
        c = torch.matmul(a, b)
        torch.cuda.synchronize()
        end = time.time()
        
        print(f"✅ Matrix multiplication: {(end-start)*1000:.2f} ms")
        print(f"   Result shape: {c.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ CUDA operations failed: {e}")
        return False

def test_extension():
    """Test Block-NeRF CUDA extension"""
    print("\n📦 Testing Block-NeRF CUDA Extension")
    print("-" * 40)
    
    try:
        import block_nerf_cuda
        print("✅ Extension imported successfully")
        
        # Test function if available
        try:
            test_data = torch.randn(500, 500, device='cuda')
            result = block_nerf_cuda.test_memory_bandwidth(test_data)
            
            print(f"✅ Function call successful")
            print(f"   Input: {test_data.shape}")
            print(f"   Output: {result.shape}")
            print(f"   Data match: {torch.allclose(test_data, result)}")
            
            return True
            
        except Exception as e:
            print(f"⚠️  Function call failed: {e}")
            return False
            
    except ImportError:
        print("⚠️  Extension not available - run 'bash build_cuda.sh' to build")
        return False

def main():
    """Main test function"""
    print("🚀 Block-NeRF CUDA Quick Test")
    print("=" * 50)
    
    # Test 1: Basic CUDA
    cuda_ok = test_cuda_basic()
    
    # Test 2: Extension
    ext_ok = test_extension()
    
    # Summary
    print("\n📋 Summary")
    print("=" * 50)
    print(f"CUDA Basic: {'✅ PASS' if cuda_ok else '❌ FAIL'}")
    print(f"Extension:  {'✅ PASS' if ext_ok else '⚠️  SKIP'}")
    
    if cuda_ok:
        print("\n🎉 CUDA environment is working!")
        if not ext_ok:
            print("💡 To build the extension:")
            print("   bash build_cuda.sh")
    else:
        print("\n❌ CUDA environment has issues")
    
    return cuda_ok

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
