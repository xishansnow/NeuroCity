"""
Benchmark Tests for Block-NeRF CUDA Extension

This module contains performance benchmarks for the CUDA-accelerated
Block-NeRF implementation.
"""

import torch
import time
import numpy as np
import block_nerf_cuda
import pytest


class TestBlockNeRFBenchmarks:
    """Benchmark tests for Block-NeRF CUDA operations"""
    
    def setup_method(self):
        """Setup for each test method"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        # Set device
        self.device = torch.device('cuda')
        
        # Create test data
        self.num_cameras = 50
        self.num_blocks = 100
        self.num_rays = 1000
        self.max_blocks_per_ray = 32
        
        # Camera data
        self.camera_positions = torch.randn(self.num_cameras, 3, device=self.device)
        
        # Block data  
        self.block_centers = torch.randn(self.num_blocks, 3, device=self.device) * 10
        self.block_radii = torch.ones(self.num_blocks, device=self.device) * 5.0
        self.block_active = torch.ones(self.num_blocks, dtype=torch.int32, device=self.device)
        
        # Ray data
        self.ray_origins = torch.randn(self.num_rays, 3, device=self.device)
        self.ray_directions = torch.randn(self.num_rays, 3, device=self.device)
        self.ray_directions = self.ray_directions / torch.norm(self.ray_directions, dim=1, keepdim=True)
        self.ray_near = torch.ones(self.num_rays, device=self.device) * 0.1
        self.ray_far = torch.ones(self.num_rays, device=self.device) * 100.0
    
    def benchmark_function(self, func, num_iterations=100, warmup=10):
        """Benchmark a function with timing"""
        # Warmup
        for _ in range(warmup):
            func()
        
        torch.cuda.synchronize()
        
        # Benchmark
        start_time = time.time()
        for _ in range(num_iterations):
            func()
        torch.cuda.synchronize()
        end_time = time.time()
        
        total_time = end_time - start_time
        avg_time = total_time / num_iterations
        fps = 1.0 / avg_time
        
        return {
            'total_time': total_time,
            'avg_time': avg_time,
            'fps': fps,
            'iterations': num_iterations
        }
    
    def test_memory_bandwidth_benchmark(self):
        """Benchmark memory bandwidth test"""
        # Test different sizes
        sizes = [1000, 10000, 100000, 1000000]
        
        print("\n📊 Memory Bandwidth Benchmark")
        print("=" * 50)
        
        for size in sizes:
            test_data = torch.randn(size, device=self.device)
            
            def benchmark_func():
                return block_nerf_cuda.memory_bandwidth_test(test_data)
            
            results = self.benchmark_function(benchmark_func, num_iterations=100)
            
            # Calculate bandwidth (GB/s)
            data_size_gb = size * 4 * 2 / (1024**3)  # 4 bytes per float, read+write
            bandwidth = data_size_gb / results['avg_time']
            
            print(f"Size: {size:>8} | Time: {results['avg_time']*1000:.3f}ms | "
                  f"Bandwidth: {bandwidth:.2f} GB/s")
            
            assert results['avg_time'] < 0.1  # Should be fast
    
    def test_block_visibility_benchmark(self):
        """Benchmark block visibility computation"""
        print("\n📊 Block Visibility Benchmark")
        print("=" * 50)
        
        def benchmark_func():
            return block_nerf_cuda.block_visibility(
                self.camera_positions,
                self.block_centers,
                self.block_radii,
                self.block_active,
                0.1
            )
        
        results = self.benchmark_function(benchmark_func)
        
        print(f"Cameras: {self.num_cameras} | Blocks: {self.num_blocks}")
        print(f"Avg Time: {results['avg_time']*1000:.3f}ms | FPS: {results['fps']:.1f}")
        
        # Performance assertion
        assert results['avg_time'] < 0.01  # Should be very fast
    
    def test_block_selection_benchmark(self):
        """Benchmark block selection for rays"""
        print("\n📊 Block Selection Benchmark")
        print("=" * 50)
        
        def benchmark_func():
            return block_nerf_cuda.block_selection(
                self.ray_origins,
                self.ray_directions,
                self.ray_near,
                self.ray_far,
                self.block_centers,
                self.block_radii,
                self.block_active,
                self.max_blocks_per_ray
            )
        
        results = self.benchmark_function(benchmark_func)
        
        print(f"Rays: {self.num_rays} | Blocks: {self.num_blocks}")
        print(f"Avg Time: {results['avg_time']*1000:.3f}ms | FPS: {results['fps']:.1f}")
        
        # Performance assertion
        assert results['avg_time'] < 0.05  # Should be reasonably fast
    
    def test_scalability_benchmark(self):
        """Test performance scaling with different input sizes"""
        print("\n📊 Scalability Benchmark")
        print("=" * 50)
        
        ray_counts = [100, 500, 1000, 5000, 10000]
        block_counts = [50, 100, 200, 500]
        
        print(f"{'Rays':<8} | {'Blocks':<8} | {'Time (ms)':<12} | {'FPS':<8}")
        print("-" * 50)
        
        for num_rays in ray_counts:
            for num_blocks in block_counts:
                # Create test data
                ray_origins = torch.randn(num_rays, 3, device=self.device)
                ray_directions = torch.randn(num_rays, 3, device=self.device)
                ray_directions = ray_directions / torch.norm(ray_directions, dim=1, keepdim=True)
                ray_near = torch.ones(num_rays, device=self.device) * 0.1
                ray_far = torch.ones(num_rays, device=self.device) * 100.0
                
                block_centers = torch.randn(num_blocks, 3, device=self.device) * 10
                block_radii = torch.ones(num_blocks, device=self.device) * 5.0
                block_active = torch.ones(num_blocks, dtype=torch.int32, device=self.device)
                
                def benchmark_func():
                    return block_nerf_cuda.block_selection(
                        ray_origins, ray_directions, ray_near, ray_far,
                        block_centers, block_radii, block_active, 32
                    )
                
                results = self.benchmark_function(benchmark_func, num_iterations=20)
                
                print(f"{num_rays:<8} | {num_blocks:<8} | {results['avg_time']*1000:<12.3f} | {results['fps']:<8.1f}")
                
                # Basic performance check
                assert results['avg_time'] < 1.0  # Should complete in reasonable time


def run_benchmarks():
    """Run all benchmarks"""
    print("🚀 Starting Block-NeRF CUDA Benchmarks...")
    print("=" * 60)
    
    benchmarks = TestBlockNeRFBenchmarks()
    benchmarks.setup_method()
    
    try:
        benchmarks.test_memory_bandwidth_benchmark()
        benchmarks.test_block_visibility_benchmark()
        benchmarks.test_block_selection_benchmark() 
        benchmarks.test_scalability_benchmark()
        
        print("\n🎉 All benchmarks completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Benchmark failed: {e}")
        raise


if __name__ == "__main__":
    run_benchmarks()
