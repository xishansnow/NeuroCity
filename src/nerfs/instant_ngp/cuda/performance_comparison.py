#!/usr/bin/env python3
"""
Performance comparison between original and optimized Instant NGP CUDA extensions
"""

import os
import sys
import torch
import time
import numpy as np
import matplotlib.pyplot as plt

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def benchmark_implementation(module_name, test_configs, num_warmup=5, num_iterations=50):
    """Benchmark a specific implementation"""
    try:
        if module_name == "original":
            import instant_ngp_cuda as ngp_module
            hash_func = ngp_module.hash_encode_forward_cuda
            sh_func = ngp_module.spherical_harmonics_encode_cuda
        else:  # optimized
            import instant_ngp_optimized_cuda as ngp_module
            hash_func = ngp_module.hash_encode_forward_optimized
            sh_func = ngp_module.spherical_harmonics_encode_optimized
        
        results = {}
        
        print(f"\n🔥 Benchmarking {module_name.title()} Implementation")
        print("=" * 50)
        
        for config_name, config in test_configs.items():
            print(f"\nConfig: {config_name}")
            print(f"  Samples: {config['N']:,}")
            print(f"  Levels: {config['num_levels']}")
            print(f"  Feature Dim: {config['feature_dim']}")
            
            # Prepare data for hash encoding
            N = config['N']
            num_levels = config['num_levels']
            feature_dim = config['feature_dim']
            hashmap_size = config['hashmap_size']
            
            positions = torch.randn(N, 3, device='cuda')
            
            if module_name == "original":
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
                
                # Test hash encoding
                def hash_test():
                    return hash_func(
                        positions, embeddings, resolutions, offsets,
                        num_levels, feature_dim, hashmap_size, scale,
                        aabb_min, aabb_max
                    )
            else:
                # Optimized format: [total_features]
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
                
                # Test hash encoding
                def hash_test():
                    return hash_func(
                        positions, embeddings, resolutions, offsets,
                        num_levels, feature_dim, hashmap_size, scale,
                        aabb_min, aabb_max
                    )
            
            # Warmup
            for _ in range(num_warmup):
                _ = hash_test()
            
            # Benchmark hash encoding
            torch.cuda.synchronize()
            start_time = time.time()
            
            for _ in range(num_iterations):
                _ = hash_test()
            
            torch.cuda.synchronize()
            end_time = time.time()
            
            hash_time = (end_time - start_time) / num_iterations * 1000  # ms
            hash_throughput = N / (hash_time / 1000)  # samples/sec
            
            # Test spherical harmonics
            directions = torch.randn(N, 3, device='cuda')
            directions = directions / directions.norm(dim=-1, keepdim=True)
            degree = 4
            
            # Warmup
            for _ in range(num_warmup):
                _ = sh_func(directions, degree)
            
            # Benchmark spherical harmonics
            torch.cuda.synchronize()
            start_time = time.time()
            
            for _ in range(num_iterations):
                _ = sh_func(directions, degree)
            
            torch.cuda.synchronize()
            end_time = time.time()
            
            sh_time = (end_time - start_time) / num_iterations * 1000  # ms
            sh_throughput = N / (sh_time / 1000)  # samples/sec
            
            results[config_name] = {
                'hash_time': hash_time,
                'hash_throughput': hash_throughput,
                'sh_time': sh_time,
                'sh_throughput': sh_throughput,
                'N': N,
                'num_levels': num_levels,
                'feature_dim': feature_dim
            }
            
            print(f"  Hash Encoding: {hash_time:.3f} ms ({hash_throughput:,.0f} samples/sec)")
            print(f"  Spherical Harmonics: {sh_time:.3f} ms ({sh_throughput:,.0f} samples/sec)")
        
        return results
        
    except Exception as e:
        print(f"❌ Failed to benchmark {module_name}: {e}")
        return {}

def create_comparison_plots(original_results, optimized_results):
    """Create comparison plots"""
    if not original_results or not optimized_results:
        print("❌ Cannot create plots - missing results")
        return
    
    # Prepare data
    configs = list(original_results.keys())
    
    # Hash encoding comparison
    orig_hash_times = [original_results[c]['hash_time'] for c in configs]
    opt_hash_times = [optimized_results[c]['hash_time'] for c in configs]
    
    orig_hash_throughput = [original_results[c]['hash_throughput'] / 1e6 for c in configs]  # M samples/sec
    opt_hash_throughput = [optimized_results[c]['hash_throughput'] / 1e6 for c in configs]  # M samples/sec
    
    # Spherical harmonics comparison
    orig_sh_times = [original_results[c]['sh_time'] for c in configs]
    opt_sh_times = [optimized_results[c]['sh_time'] for c in configs]
    
    orig_sh_throughput = [original_results[c]['sh_throughput'] / 1e6 for c in configs]  # M samples/sec
    opt_sh_throughput = [optimized_results[c]['sh_throughput'] / 1e6 for c in configs]  # M samples/sec
    
    # Create plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    x = np.arange(len(configs))
    width = 0.35
    
    # Hash encoding latency
    ax1.bar(x - width/2, orig_hash_times, width, label='Original', alpha=0.8, color='skyblue')
    ax1.bar(x + width/2, opt_hash_times, width, label='Optimized', alpha=0.8, color='lightcoral')
    ax1.set_xlabel('Configuration')
    ax1.set_ylabel('Latency (ms)')
    ax1.set_title('Hash Encoding Latency Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(configs, rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Hash encoding throughput
    ax2.bar(x - width/2, orig_hash_throughput, width, label='Original', alpha=0.8, color='skyblue')
    ax2.bar(x + width/2, opt_hash_throughput, width, label='Optimized', alpha=0.8, color='lightcoral')
    ax2.set_xlabel('Configuration')
    ax2.set_ylabel('Throughput (M samples/sec)')
    ax2.set_title('Hash Encoding Throughput Comparison')
    ax2.set_xticks(x)
    ax2.set_xticklabels(configs, rotation=45)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Spherical harmonics latency
    ax3.bar(x - width/2, orig_sh_times, width, label='Original', alpha=0.8, color='lightgreen')
    ax3.bar(x + width/2, opt_sh_times, width, label='Optimized', alpha=0.8, color='orange')
    ax3.set_xlabel('Configuration')
    ax3.set_ylabel('Latency (ms)')
    ax3.set_title('Spherical Harmonics Latency Comparison')
    ax3.set_xticks(x)
    ax3.set_xticklabels(configs, rotation=45)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Spherical harmonics throughput
    ax4.bar(x - width/2, orig_sh_throughput, width, label='Original', alpha=0.8, color='lightgreen')
    ax4.bar(x + width/2, opt_sh_throughput, width, label='Optimized', alpha=0.8, color='orange')
    ax4.set_xlabel('Configuration')
    ax4.set_ylabel('Throughput (M samples/sec)')
    ax4.set_title('Spherical Harmonics Throughput Comparison')
    ax4.set_xticks(x)
    ax4.set_xticklabels(configs, rotation=45)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('performance_comparison.png', dpi=300, bbox_inches='tight')
    print("📊 Performance comparison plots saved as 'performance_comparison.png'")

def print_detailed_comparison(original_results, optimized_results):
    """Print detailed comparison table"""
    if not original_results or not optimized_results:
        print("❌ Cannot create comparison - missing results")
        return
    
    print(f"\n📊 Detailed Performance Comparison")
    print("=" * 80)
    
    # Header
    print(f"{'Config':<15} {'Metric':<20} {'Original':<15} {'Optimized':<15} {'Speedup':<10}")
    print("-" * 80)
    
    for config_name in original_results.keys():
        if config_name not in optimized_results:
            continue
            
        orig = original_results[config_name]
        opt = optimized_results[config_name]
        
        # Hash encoding comparison
        hash_speedup = orig['hash_time'] / opt['hash_time']
        hash_throughput_ratio = opt['hash_throughput'] / orig['hash_throughput']
        
        print(f"{config_name:<15} {'Hash Latency (ms)':<20} {orig['hash_time']:<15.3f} {opt['hash_time']:<15.3f} {hash_speedup:<10.2f}x")
        print(f"{'':15} {'Hash Throughput':<20} {orig['hash_throughput']/1e6:<15.1f}M {opt['hash_throughput']/1e6:<15.1f}M {hash_throughput_ratio:<10.2f}x")
        
        # Spherical harmonics comparison
        sh_speedup = orig['sh_time'] / opt['sh_time']
        sh_throughput_ratio = opt['sh_throughput'] / orig['sh_throughput']
        
        print(f"{'':15} {'SH Latency (ms)':<20} {orig['sh_time']:<15.3f} {opt['sh_time']:<15.3f} {sh_speedup:<10.2f}x")
        print(f"{'':15} {'SH Throughput':<20} {orig['sh_throughput']/1e6:<15.1f}M {opt['sh_throughput']/1e6:<15.1f}M {sh_throughput_ratio:<10.2f}x")
        print("-" * 80)

def main():
    print("🚀 Instant NGP CUDA Performance Comparison")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return False
    
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Compute Capability: {torch.cuda.get_device_capability()}")
    print(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Test configurations
    test_configs = {
        "Small": {"N": 1000, "num_levels": 4, "feature_dim": 2, "hashmap_size": 256},
        "Medium": {"N": 5000, "num_levels": 8, "feature_dim": 2, "hashmap_size": 512},
        "Large": {"N": 10000, "num_levels": 16, "feature_dim": 2, "hashmap_size": 1024},
        "XLarge": {"N": 20000, "num_levels": 16, "feature_dim": 2, "hashmap_size": 2048},
    }
    
    # Benchmark original implementation
    print(f"\n🔥 Testing Original Implementation")
    original_results = benchmark_implementation("original", test_configs)
    
    # Benchmark optimized implementation
    print(f"\n🔥 Testing Optimized Implementation")
    optimized_results = benchmark_implementation("optimized", test_configs)
    
    # Create comparison
    if original_results and optimized_results:
        print_detailed_comparison(original_results, optimized_results)
        
        try:
            create_comparison_plots(original_results, optimized_results)
        except Exception as e:
            print(f"❌ Failed to create plots: {e}")
    
    print(f"\n✅ Performance comparison completed!")
    return True

if __name__ == "__main__":
    main()
