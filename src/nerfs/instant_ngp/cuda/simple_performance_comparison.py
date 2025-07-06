#!/usr/bin/env python3
"""
Simplified Performance comparison between original and optimized Instant NGP CUDA extensions
"""

import os
import sys
import torch
import time
import numpy as np

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def benchmark_spherical_harmonics():
    """Benchmark spherical harmonics encoding only (simpler test)"""
    print("\n🧪 Benchmarking Spherical Harmonics Encoding")
    print("=" * 50)
    
    # Test configurations
    test_sizes = [1000, 5000, 10000, 20000, 50000]
    degree = 4
    num_warmup = 10
    num_iterations = 100
    
    results = {'original': {}, 'optimized': {}}
    
    # Test original implementation
    try:
        import instant_ngp_cuda
        print("📋 Testing Original Implementation:")
        
        for N in test_sizes:
            directions = torch.randn(N, 3, device='cuda')
            directions = directions / directions.norm(dim=-1, keepdim=True)
            
            # Warmup
            for _ in range(num_warmup):
                _ = instant_ngp_cuda.sh_encode(directions, degree)
            
            # Benchmark
            torch.cuda.synchronize()
            start_time = time.time()
            
            for _ in range(num_iterations):
                _ = instant_ngp_cuda.sh_encode(directions, degree)
            
            torch.cuda.synchronize()
            end_time = time.time()
            
            latency = (end_time - start_time) / num_iterations * 1000  # ms
            throughput = N / (latency / 1000)  # samples/sec
            
            results['original'][N] = {'latency': latency, 'throughput': throughput}
            print(f"  {N:6,} samples: {latency:.3f} ms ({throughput:,.0f} samples/sec)")
            
    except Exception as e:
        print(f"❌ Original failed: {e}")
    
    # Test optimized implementation
    try:
        import instant_ngp_optimized_cuda
        print("\n📋 Testing Optimized Implementation:")
        
        for N in test_sizes:
            directions = torch.randn(N, 3, device='cuda')
            directions = directions / directions.norm(dim=-1, keepdim=True)
            
            # Warmup
            for _ in range(num_warmup):
                _ = instant_ngp_optimized_cuda.spherical_harmonics_encode_optimized(directions, degree)
            
            # Benchmark
            torch.cuda.synchronize()
            start_time = time.time()
            
            for _ in range(num_iterations):
                _ = instant_ngp_optimized_cuda.spherical_harmonics_encode_optimized(directions, degree)
            
            torch.cuda.synchronize()
            end_time = time.time()
            
            latency = (end_time - start_time) / num_iterations * 1000  # ms
            throughput = N / (latency / 1000)  # samples/sec
            
            results['optimized'][N] = {'latency': latency, 'throughput': throughput}
            print(f"  {N:6,} samples: {latency:.3f} ms ({throughput:,.0f} samples/sec)")
            
    except Exception as e:
        print(f"❌ Optimized failed: {e}")
    
    return results

def benchmark_hash_encoding():
    """Benchmark hash encoding (more complex test)"""
    print("\n🔥 Benchmarking Hash Encoding")
    print("=" * 50)
    
    test_configs = [
        {"N": 1000, "levels": 4, "feature_dim": 2},
        {"N": 5000, "levels": 8, "feature_dim": 2}, 
        {"N": 10000, "levels": 16, "feature_dim": 2},
    ]
    
    num_warmup = 5
    num_iterations = 50
    results = {'original': {}, 'optimized': {}}
    
    # Test original implementation
    try:
        import instant_ngp_cuda
        print("📋 Testing Original Implementation:")
        
        for config in test_configs:
            N = config["N"]
            num_levels = config["levels"]
            feature_dim = config["feature_dim"]
            hashmap_size = 1024
            
            positions = torch.randn(N, 3, device='cuda')
            
            # Original format: [num_levels * hashmap_size, feature_dim]
            total_embeddings = num_levels * hashmap_size
            embeddings = torch.randn(total_embeddings, feature_dim, device='cuda')
            
            # Create level configurations
            base_resolution = 16
            resolutions = torch.tensor([min(base_resolution * (2**i), 512) for i in range(num_levels)], 
                                      device='cuda', dtype=torch.int32)
            offsets = torch.tensor([i * hashmap_size for i in range(num_levels)], 
                                  device='cuda', dtype=torch.uint32)
            
            aabb_min = torch.tensor([-1.0, -1.0, -1.0], device='cuda')
            aabb_max = torch.tensor([1.0, 1.0, 1.0], device='cuda')
            scale = 1.0
            
            # Warmup
            for _ in range(num_warmup):
                _ = instant_ngp_cuda.hash_encode_forward(
                    positions, embeddings, resolutions, offsets,
                    num_levels, feature_dim, hashmap_size, scale,
                    aabb_min, aabb_max
                )
            
            # Benchmark
            torch.cuda.synchronize()
            start_time = time.time()
            
            for _ in range(num_iterations):
                _ = instant_ngp_cuda.hash_encode_forward(
                    positions, embeddings, resolutions, offsets,
                    num_levels, feature_dim, hashmap_size, scale,
                    aabb_min, aabb_max
                )
            
            torch.cuda.synchronize()
            end_time = time.time()
            
            latency = (end_time - start_time) / num_iterations * 1000  # ms
            throughput = N / (latency / 1000)  # samples/sec
            
            config_name = f"{N}_{num_levels}L"
            results['original'][config_name] = {'latency': latency, 'throughput': throughput}
            print(f"  {N:6,} samples, {num_levels:2}L: {latency:.3f} ms ({throughput:,.0f} samples/sec)")
            
    except Exception as e:
        print(f"❌ Original hash encoding failed: {e}")
    
    # Test optimized implementation
    try:
        import instant_ngp_optimized_cuda
        print("\n📋 Testing Optimized Implementation:")
        
        for config in test_configs:
            N = config["N"]
            num_levels = config["levels"]
            feature_dim = config["feature_dim"]
            hashmap_size = 1024
            
            positions = torch.randn(N, 3, device='cuda')
            
            # Optimized format: flat array [total_features]
            total_features = num_levels * hashmap_size * feature_dim
            embeddings = torch.randn(total_features, device='cuda')
            
            # Create level configurations
            base_resolution = 16
            resolutions = torch.tensor([min(base_resolution * (2**i), 512) for i in range(num_levels)], 
                                      device='cuda', dtype=torch.int32)
            offsets = torch.tensor([i * hashmap_size * feature_dim for i in range(num_levels)], 
                                  device='cuda', dtype=torch.uint32)
            
            aabb_min = torch.tensor([-1.0, -1.0, -1.0], device='cuda')
            aabb_max = torch.tensor([1.0, 1.0, 1.0], device='cuda')
            scale = 1.0
            
            # Warmup
            for _ in range(num_warmup):
                _ = instant_ngp_optimized_cuda.hash_encode_forward_optimized(
                    positions, embeddings, resolutions, offsets,
                    num_levels, feature_dim, hashmap_size, scale,
                    aabb_min, aabb_max
                )
            
            # Benchmark
            torch.cuda.synchronize()
            start_time = time.time()
            
            for _ in range(num_iterations):
                _ = instant_ngp_optimized_cuda.hash_encode_forward_optimized(
                    positions, embeddings, resolutions, offsets,
                    num_levels, feature_dim, hashmap_size, scale,
                    aabb_min, aabb_max
                )
            
            torch.cuda.synchronize()
            end_time = time.time()
            
            latency = (end_time - start_time) / num_iterations * 1000  # ms
            throughput = N / (latency / 1000)  # samples/sec
            
            config_name = f"{N}_{num_levels}L"
            results['optimized'][config_name] = {'latency': latency, 'throughput': throughput}
            print(f"  {N:6,} samples, {num_levels:2}L: {latency:.3f} ms ({throughput:,.0f} samples/sec)")
            
    except Exception as e:
        print(f"❌ Optimized hash encoding failed: {e}")
    
    return results

def print_comparison(sh_results, hash_results):
    """Print detailed comparison results"""
    print("\n📊 Performance Comparison Summary")
    print("=" * 80)
    
    # Spherical harmonics comparison
    if sh_results['original'] and sh_results['optimized']:
        print("\n🌐 Spherical Harmonics Encoding:")
        print(f"{'Samples':<10} {'Original (ms)':<15} {'Optimized (ms)':<15} {'Speedup':<10}")
        print("-" * 60)
        
        for N in sorted(sh_results['original'].keys()):
            if N in sh_results['optimized']:
                orig_lat = sh_results['original'][N]['latency']
                opt_lat = sh_results['optimized'][N]['latency']
                speedup = orig_lat / opt_lat
                
                print(f"{N:<10,} {orig_lat:<15.3f} {opt_lat:<15.3f} {speedup:<10.2f}x")
    
    # Hash encoding comparison
    if hash_results['original'] and hash_results['optimized']:
        print("\n🔥 Hash Encoding:")
        print(f"{'Config':<15} {'Original (ms)':<15} {'Optimized (ms)':<15} {'Speedup':<10}")
        print("-" * 60)
        
        for config in sorted(hash_results['original'].keys()):
            if config in hash_results['optimized']:
                orig_lat = hash_results['original'][config]['latency']
                opt_lat = hash_results['optimized'][config]['latency']
                speedup = orig_lat / opt_lat
                
                print(f"{config:<15} {orig_lat:<15.3f} {opt_lat:<15.3f} {speedup:<10.2f}x")

def main():
    print("🚀 Instant NGP CUDA Performance Comparison")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return False
    
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Compute Capability: {torch.cuda.get_device_capability()}")
    print(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Run benchmarks
    sh_results = benchmark_spherical_harmonics()
    hash_results = benchmark_hash_encoding()
    
    # Print comparison
    print_comparison(sh_results, hash_results)
    
    print(f"\n✅ Performance comparison completed!")
    return True

if __name__ == "__main__":
    main()
