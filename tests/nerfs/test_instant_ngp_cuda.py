#!/usr/bin/env python3
"""
Test script for the Instant NGP CUDA extension.
"""

import torch
import numpy as np
import sys
import os
import time

def test_instant_ngp_cuda_extension():
    """Test the Instant NGP CUDA extension specifically."""
    print("=" * 60)
    print("Testing Instant NGP CUDA Extension")
    print("=" * 60)
    
    # Check if CUDA is available
    print(f"CUDA available: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("❌ CUDA is not available, cannot test CUDA extension")
        return False
        
    print(f"CUDA device: {torch.cuda.get_device_name()}")
    
    try:
        # Add the CUDA extension to the path
        cuda_path = '/home/xishansnow/3DVision/NeuroCity/src/nerfs/instant_ngp/cuda'
        if cuda_path not in sys.path:
            sys.path.insert(0, cuda_path)
        
        # Import the CUDA extension
        import instant_ngp_cuda
        print("✅ Successfully imported instant_ngp_cuda module")
        
        # Test 1: Hash Encoding Forward
        print("\n📋 Test 1: Hash Encoding Forward")
        N = 1000
        num_levels = 16
        feature_dim = 2
        base_resolution = 16
        finest_resolution = 512
        log2_hashmap_size = 19
        hashmap_size = 2 ** log2_hashmap_size
        
        # Create test data
        positions = torch.randn(N, 3, device='cuda', dtype=torch.float32)
        positions = torch.clamp(positions, -1.0, 1.0)  # Clamp to valid range
        
        # Calculate total embedding parameters
        total_params = 0
        resolutions = []
        offsets = []
        per_level_scale = 2.0
        
        for level in range(num_levels):
            resolution = int(base_resolution * (per_level_scale ** level))
            resolution = min(resolution, finest_resolution)
            resolutions.append(resolution)
            
            params_in_level = min(resolution ** 3, hashmap_size)
            offsets.append(total_params)
            total_params += params_in_level
        
        embeddings = torch.randn(total_params, feature_dim, device='cuda', dtype=torch.float32) * 0.01
        resolutions_tensor = torch.tensor(resolutions, dtype=torch.int32, device='cuda')
        offsets_tensor = torch.tensor(offsets, dtype=torch.uint32, device='cuda')
        aabb_min = torch.tensor([-1.0, -1.0, -1.0], device='cuda')
        aabb_max = torch.tensor([1.0, 1.0, 1.0], device='cuda')
        
        start_time = time.time()
        encoded = instant_ngp_cuda.hash_encode_forward(
            positions,
            embeddings,
            resolutions_tensor,
            offsets_tensor,
            num_levels,
            feature_dim,
            hashmap_size,
            1.0,  # scale
            aabb_min,
            aabb_max
        )
        end_time = time.time()
        
        print(f"   Input shape: {positions.shape}")
        print(f"   Embeddings shape: {embeddings.shape}")
        print(f"   Output shape: {encoded.shape}")
        print(f"   Expected output shape: ({N}, {num_levels * feature_dim})")
        print(f"   Output range: [{encoded.min():.6f}, {encoded.max():.6f}]")
        print(f"   Time taken: {(end_time - start_time)*1000:.2f} ms")
        
        assert encoded.shape == (N, num_levels * feature_dim), f"Shape mismatch: {encoded.shape} vs {(N, num_levels * feature_dim)}"
        print("   ✅ Hash encoding forward test passed")
        
        # Test 2: Spherical Harmonics Encoding
        print("\n📋 Test 2: Spherical Harmonics Encoding")
        N_dirs = 1000
        degree = 4
        
        # Create normalized direction vectors
        directions = torch.randn(N_dirs, 3, device='cuda', dtype=torch.float32)
        directions = directions / torch.norm(directions, dim=1, keepdim=True)
        
        start_time = time.time()
        sh_encoded = instant_ngp_cuda.sh_encode(directions, degree)
        end_time = time.time()
        
        expected_sh_dim = (degree + 1) ** 2
        print(f"   Input shape: {directions.shape}")
        print(f"   Output shape: {sh_encoded.shape}")
        print(f"   Expected output shape: ({N_dirs}, {expected_sh_dim})")
        print(f"   Degree: {degree}")
        print(f"   Output range: [{sh_encoded.min():.6f}, {sh_encoded.max():.6f}]")
        print(f"   Time taken: {(end_time - start_time)*1000:.2f} ms")
        
        assert sh_encoded.shape == (N_dirs, expected_sh_dim), f"Shape mismatch: {sh_encoded.shape} vs {(N_dirs, expected_sh_dim)}"
        print("   ✅ Spherical harmonics encoding test passed")
        
        # Test 3: Hash Encoding Backward
        print("\n📋 Test 3: Hash Encoding Backward")
        grad_encoded = torch.randn_like(encoded)
        embeddings_shape = torch.tensor(embeddings.shape, dtype=torch.int64)
        
        start_time = time.time()
        grad_embeddings = instant_ngp_cuda.hash_encode_backward(
            positions,
            grad_encoded,
            embeddings_shape,
            resolutions_tensor,
            offsets_tensor,
            num_levels,
            feature_dim,
            hashmap_size,
            1.0,  # scale
            aabb_min,
            aabb_max
        )
        end_time = time.time()
        
        print(f"   Grad encoded shape: {grad_encoded.shape}")
        print(f"   Grad embeddings shape: {grad_embeddings.shape}")
        print(f"   Expected grad embeddings shape: {embeddings.shape}")
        print(f"   Grad range: [{grad_embeddings.min():.6f}, {grad_embeddings.max():.6f}]")
        print(f"   Time taken: {(end_time - start_time)*1000:.2f} ms")
        
        assert grad_embeddings.shape == embeddings.shape, f"Shape mismatch: {grad_embeddings.shape} vs {embeddings.shape}"
        print("   ✅ Hash encoding backward test passed")
        
        # Test 4: Performance Comparison
        print("\n📋 Test 4: Performance Comparison")
        # Load our PyTorch fallback implementation
        sys.path.insert(0, '/home/xishansnow/3DVision/NeuroCity/src/nerfs/instant_ngp')
        from cuda_model import HashEncoder, SHEncoder
        
        # Create PyTorch fallback encoders
        hash_encoder_torch = HashEncoder(
            num_levels=num_levels,
            feature_dim=feature_dim,
            base_resolution=base_resolution,
            finest_resolution=finest_resolution,
            log2_hashmap_size=log2_hashmap_size,
            use_cuda=False
        ).cuda()
        
        sh_encoder_torch = SHEncoder(degree=degree, use_cuda=False).cuda()
        
        # Performance test for hash encoding
        N_perf = 10000
        positions_perf = torch.randn(N_perf, 3, device='cuda') * 0.5
        
        # CUDA version timing
        torch.cuda.synchronize()
        start_time = time.time()
        for _ in range(10):
            encoded_cuda = instant_ngp_cuda.hash_encode_forward(
                positions_perf, embeddings, resolutions_tensor, offsets_tensor,
                num_levels, feature_dim, hashmap_size, 1.0, aabb_min, aabb_max
            )
        torch.cuda.synchronize()
        cuda_time = (time.time() - start_time) / 10
        
        # PyTorch version timing
        torch.cuda.synchronize()
        start_time = time.time()
        for _ in range(10):
            encoded_torch = hash_encoder_torch(positions_perf)
        torch.cuda.synchronize()
        torch_time = (time.time() - start_time) / 10
        
        hash_speedup = torch_time / cuda_time
        
        # Performance test for spherical harmonics
        directions_perf = torch.randn(N_perf, 3, device='cuda')
        directions_perf = directions_perf / torch.norm(directions_perf, dim=1, keepdim=True)
        
        # CUDA version timing
        torch.cuda.synchronize()
        start_time = time.time()
        for _ in range(10):
            sh_cuda = instant_ngp_cuda.sh_encode(directions_perf, degree)
        torch.cuda.synchronize()
        cuda_sh_time = (time.time() - start_time) / 10
        
        # PyTorch version timing
        torch.cuda.synchronize()
        start_time = time.time()
        for _ in range(10):
            sh_torch = sh_encoder_torch(directions_perf)
        torch.cuda.synchronize()
        torch_sh_time = (time.time() - start_time) / 10
        
        sh_speedup = torch_sh_time / cuda_sh_time
        
        print(f"   Hash Encoding:")
        print(f"     CUDA time: {cuda_time*1000:.2f} ms")
        print(f"     PyTorch time: {torch_time*1000:.2f} ms")
        print(f"     Speedup: {hash_speedup:.1f}x")
        
        print(f"   Spherical Harmonics:")
        print(f"     CUDA time: {cuda_sh_time*1000:.2f} ms")
        print(f"     PyTorch time: {torch_sh_time*1000:.2f} ms")
        print(f"     Speedup: {sh_speedup:.1f}x")
        
        print("   ✅ Performance comparison completed")
        
        print("\n" + "=" * 60)
        print("🎉 All Instant NGP CUDA extension tests passed successfully!")
        print("=" * 60)
        
        return True
        
    except ImportError as e:
        print(f"❌ Failed to import instant_ngp_cuda: {e}")
        return False
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_instant_ngp_cuda_extension()
    if success:
        print("\n✅ Instant NGP CUDA extension is working correctly!")
    else:
        print("\n❌ Instant NGP CUDA extension test failed!")
        sys.exit(1)
