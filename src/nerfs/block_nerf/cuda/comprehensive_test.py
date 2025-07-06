#!/usr/bin/env python3
"""
Comprehensive test suite for Block-NeRF CUDA extension
Tests all major CUDA kernels and their integration with PyTorch
"""
import os
import sys
import time
import torch
import numpy as np
import unittest
from typing import Tuple, List

class CUDATestEnvironment:
    """Test environment setup and validation"""
    
    @staticmethod
    def check_cuda_availability():
        """Check if CUDA is available and get device info"""
        if not torch.cuda.is_available():
            return False, "CUDA not available"
        
        device_count = torch.cuda.device_count()
        device_name = torch.cuda.get_device_name(0)
        memory_total = torch.cuda.get_device_properties(0).total_memory
        
        print(f"CUDA devices: {device_count}")
        print(f"Device 0: {device_name}")
        print(f"Memory: {memory_total / 1024**3:.2f} GB")
        
        return True, f"CUDA available with {device_count} device(s)"
    
    @staticmethod
    def test_basic_tensor_operations():
        """Test basic CUDA tensor operations"""
        try:
            # Create tensors on GPU
            a = torch.randn(1000, 1000, device='cuda')
            b = torch.randn(1000, 1000, device='cuda')
            
            # Perform operations
            c = torch.matmul(a, b)
            d = torch.sum(c)
            
            # Move back to CPU
            result = d.cpu().item()
            
            print(f"Basic tensor test result: {result}")
            return True, "Basic tensor operations successful"
        except Exception as e:
            return False, f"Basic tensor operations failed: {str(e)}"

class BlockNeRFCUDATests(unittest.TestCase):
    """Test cases for Block-NeRF CUDA kernels"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = 1024
        self.num_blocks = 64
        self.block_size = 32
        
    def test_cuda_environment(self):
        """Test CUDA environment setup"""
        available, msg = CUDATestEnvironment.check_cuda_availability()
        self.assertTrue(available, msg)
        
        success, msg = CUDATestEnvironment.test_basic_tensor_operations()
        self.assertTrue(success, msg)
    
    def test_extension_import(self):
        """Test importing the CUDA extension"""
        try:
            import block_nerf_cuda
            print("✅ Block-NeRF CUDA extension imported successfully")
        except ImportError:
            self.skipTest("Block-NeRF CUDA extension not available")
    
    def test_memory_bandwidth_kernel(self):
        """Test memory bandwidth kernel"""
        try:
            import block_nerf_cuda
            
            # Create test data
            input_tensor = torch.randn(1000, 1000, device=self.device)
            
            # Run kernel
            start_time = time.time()
            result = block_nerf_cuda.test_memory_bandwidth(input_tensor)
            torch.cuda.synchronize()
            end_time = time.time()
            
            # Validate results
            self.assertEqual(result.shape, input_tensor.shape)
            self.assertTrue(torch.allclose(input_tensor, result, atol=1e-6))
            
            # Calculate bandwidth
            bytes_transferred = input_tensor.numel() * 4 * 2  # float32, read + write
            bandwidth = bytes_transferred / (end_time - start_time) / 1024**3
            
            print(f"Memory bandwidth test time: {(end_time - start_time) * 1000:.2f} ms")
            print(f"Bandwidth: {bandwidth:.2f} GB/s")
            
        except ImportError:
            self.skipTest("Block-NeRF CUDA extension not available")

def run_comprehensive_tests():
    """Run all comprehensive tests"""
    print("=" * 60)
    print("Block-NeRF CUDA Comprehensive Test Suite")
    print("=" * 60)
    
    # Check CUDA availability first
    cuda_available, cuda_msg = CUDATestEnvironment.check_cuda_availability()
    print(f"CUDA Status: {cuda_msg}")
    
    if not cuda_available:
        print("CUDA not available. Tests will be skipped.")
        return
    
    # Run basic environment tests
    print("\n" + "-" * 40)
    print("Testing CUDA Environment")
    print("-" * 40)
    
    success, msg = CUDATestEnvironment.test_basic_tensor_operations()
    print(f"Basic Operations: {msg}")
    
    # Run unit tests
    print("\n" + "-" * 40)
    print("Running Tests")
    print("-" * 40)
    
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(BlockNeRFCUDATests))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_comprehensive_tests()
    exit(0 if success else 1)