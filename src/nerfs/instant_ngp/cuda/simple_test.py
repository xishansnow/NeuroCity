#!/usr/bin/env python3
"""Simple test to check if both CUDA extensions work"""

import torch
import sys
import os

def test_original():
    """Test original CUDA extension"""
    try:
        # Add local path to make sure we import from current directory
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        
        import instant_ngp_cuda
        print("✅ Original extension imported successfully")
        print(f"Available functions: {[f for f in dir(instant_ngp_cuda) if not f.startswith('_')]}")
        
        # Simple test
        positions = torch.randn(100, 3, device='cuda')
        
        # Test spherical harmonics (use actual function name)
        directions = positions / positions.norm(dim=-1, keepdim=True)
        sh_output = instant_ngp_cuda.sh_encode(directions, 4)
        print(f"Spherical harmonics output shape: {sh_output.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Original extension failed: {e}")
        return False

def test_optimized():
    """Test optimized CUDA extension"""
    try:
        # Add local path to make sure we import from current directory
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        
        import instant_ngp_optimized_cuda
        print("✅ Optimized extension imported successfully")
        print(f"Available functions: {[f for f in dir(instant_ngp_optimized_cuda) if not f.startswith('_')]}")
        
        # Simple test
        positions = torch.randn(100, 3, device='cuda')
        
        # Test spherical harmonics
        directions = positions / positions.norm(dim=-1, keepdim=True)
        sh_output = instant_ngp_optimized_cuda.spherical_harmonics_encode_optimized(directions, 4)
        print(f"Spherical harmonics output shape: {sh_output.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Optimized extension failed: {e}")
        return False

def main():
    print("🧪 Testing CUDA Extensions")
    print("=" * 40)
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return False
    
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    print("\n📋 Testing Original Extension:")
    original_ok = test_original()
    
    print("\n📋 Testing Optimized Extension:")
    optimized_ok = test_optimized()
    
    print(f"\n📊 Results:")
    print(f"  Original: {'✅ OK' if original_ok else '❌ Failed'}")
    print(f"  Optimized: {'✅ OK' if optimized_ok else '❌ Failed'}")
    
    return original_ok and optimized_ok

if __name__ == "__main__":
    main()
