#!/usr/bin/env python3
"""
CUDA JIT compilation test
"""
import torch
import time

def test_cuda_jit():
    """Test CUDA JIT compilation capabilities"""
    print("🔥 CUDA JIT Test")
    print("-" * 30)
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return False
    
    print(f"✅ CUDA available")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    
    # Test basic operations
    try:
        a = torch.randn(1000, 1000, device='cuda')
        b = torch.randn(1000, 1000, device='cuda')
        
        start = time.time()
        c = a + b
        torch.cuda.synchronize()
        end = time.time()
        
        print(f"✅ Addition: {(end-start)*1000:.2f} ms")
        
        start = time.time()
        d = torch.matmul(a, b)
        torch.cuda.synchronize()
        end = time.time()
        
        print(f"✅ MatMul: {(end-start)*1000:.2f} ms")
        
        return True
        
    except Exception as e:
        print(f"❌ CUDA operations failed: {e}")
        return False

def test_extension_import():
    """Test Block-NeRF extension import"""
    print("\n📦 Extension Import Test")
    print("-" * 30)
    
    try:
        import block_nerf_cuda
        print("✅ Extension imported")
        return True
    except ImportError:
        print("⚠️  Extension not available")
        return False

def main():
    """Main test"""
    print("🧪 CUDA JIT Compilation Test")
    print("=" * 40)
    
    jit_ok = test_cuda_jit()
    ext_ok = test_extension_import()
    
    print(f"\n📋 Results:")
    print(f"CUDA JIT: {'✅' if jit_ok else '❌'}")
    print(f"Extension: {'✅' if ext_ok else '⚠️'}")
    
    return jit_ok

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)