#!/usr/bin/env python3
"""
Unit tests for Block-NeRF CUDA kernels
Tests individual CUDA kernel functions in isolation
"""
import unittest
import torch
import numpy as np
import time
import os
import sys

class TestCUDAEnvironment(unittest.TestCase):
    """Test CUDA environment and basic functionality"""
    
    def test_cuda_availability(self):
        """Test if CUDA is available"""
        self.assertTrue(torch.cuda.is_available(), "CUDA should be available")
        self.assertGreater(torch.cuda.device_count(), 0, "At least one CUDA device should be available")
    
    def test_cuda_device_properties(self):
        """Test CUDA device properties"""
        if torch.cuda.is_available():
            device = torch.cuda.get_device_properties(0)
            self.assertGreater(device.total_memory, 0, "Device should have memory")
            self.assertGreater(device.multi_processor_count, 0, "Device should have multiprocessors")
            print(f"Device: {device.name}")
            print(f"Memory: {device.total_memory / 1024**3:.2f} GB")
            print(f"SMs: {device.multi_processor_count}")
    
    def test_basic_tensor_operations(self):
        """Test basic CUDA tensor operations"""
        if torch.cuda.is_available():
            device = torch.device('cuda')
            
            # Test tensor creation
            a = torch.randn(100, 100, device=device)
            b = torch.randn(100, 100, device=device)
            
            # Test operations
            c = a + b
            d = torch.matmul(a, b)
            e = torch.sum(d)
            
            # Test data transfer
            result = e.cpu().item()
            
            self.assertIsInstance(result, float)
            self.assertEqual(a.device.type, 'cuda')
            self.assertEqual(c.device.type, 'cuda')

class TestBlockNeRFCUDAExtension(unittest.TestCase):
    """Test Block-NeRF CUDA extension import and basic functionality"""
    
    def setUp(self):
        """Set up test data"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Skip tests if CUDA not available
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
    
    def test_extension_import(self):
        """Test importing the CUDA extension"""
        try:
            import block_nerf_cuda
            print("✅ Block-NeRF CUDA extension imported successfully")
        except ImportError as e:
            self.skipTest(f"Block-NeRF CUDA extension not built: {e}")
    
    def test_memory_bandwidth_function(self):
        """Test memory bandwidth test function"""
        try:
            import block_nerf_cuda
            
            # Create test data
            input_tensor = torch.randn(1000, 1000, device=self.device)
            
            # Test function
            result = block_nerf_cuda.test_memory_bandwidth(input_tensor)
            
            # Validate output
            self.assertEqual(result.shape, input_tensor.shape)
            self.assertEqual(result.device.type, 'cuda')
            
            # Check if data is preserved (simple copy operation)
            self.assertTrue(torch.allclose(input_tensor, result, atol=1e-6))
            
        except ImportError:
            self.skipTest("Block-NeRF CUDA extension not built")
    
    def test_extension_functions_exist(self):
        """Test that expected functions exist in the extension"""
        try:
            import block_nerf_cuda
            
            # List of expected functions
            expected_functions = [
                'test_memory_bandwidth',
                # Add more function names as they are implemented
            ]
            
            for func_name in expected_functions:
                self.assertTrue(hasattr(block_nerf_cuda, func_name), 
                              f"Function {func_name} not found in extension")
                
        except ImportError:
            self.skipTest("Block-NeRF CUDA extension not built")

class TestPerformance(unittest.TestCase):
    """Test performance characteristics"""
    
    def setUp(self):
        """Set up performance test data"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
    
    def test_memory_bandwidth_performance(self):
        """Test memory bandwidth performance"""
        try:
            import block_nerf_cuda
            
            sizes = [1024, 2048]
            
            for size in sizes:
                input_tensor = torch.randn(size, size, device=self.device)
                
                # Warmup
                for _ in range(3):
                    _ = block_nerf_cuda.test_memory_bandwidth(input_tensor)
                
                # Measure
                torch.cuda.synchronize()
                start_time = time.time()
                result = block_nerf_cuda.test_memory_bandwidth(input_tensor)
                torch.cuda.synchronize()
                end_time = time.time()
                
                # Calculate bandwidth
                bytes_transferred = size * size * 4 * 2  # float32, read + write
                bandwidth = bytes_transferred / (end_time - start_time) / 1024**3
                
                print(f"Size {size}x{size}: {bandwidth:.2f} GB/s, {(end_time - start_time) * 1000:.2f} ms")
                
                # Basic performance check - should be faster than 1 second
                self.assertLess(end_time - start_time, 1.0, f"Performance too slow for size {size}")
                
        except ImportError:
            self.skipTest("Block-NeRF CUDA extension not built")

if __name__ == '__main__':
    # Run with verbose output
    unittest.main(verbosity=2)