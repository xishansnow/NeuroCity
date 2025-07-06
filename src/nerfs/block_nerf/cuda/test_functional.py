"""
Functional Tests for Block-NeRF CUDA Extension

This module contains functional tests that validate the correctness
of the CUDA-accelerated Block-NeRF implementation.
"""

import torch
import numpy as np
import math
import pytest

# Try to import the CUDA extension
try:
    import block_nerf_cuda
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False


class TestBlockNeRFFunctional:
    """Functional tests for Block-NeRF CUDA operations"""
    
    def setup_method(self):
        """Setup for each test method"""
        if not torch.cuda.is_available() or not CUDA_AVAILABLE:
            pytest.skip("CUDA or block_nerf_cuda not available")
        
        # Set device
        self.device = torch.device('cuda')
        
        # Create deterministic test data
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Small test data for precise validation
        self.num_cameras = 3
        self.num_blocks = 5
        self.num_rays = 10
        
        # Camera positions in a simple pattern
        self.camera_positions = torch.tensor([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0]
        ], device=self.device)
        
        # Block centers in a grid pattern
        self.block_centers = torch.tensor([
            [0.0, 0.0, 5.0],
            [2.0, 0.0, 5.0],
            [0.0, 2.0, 5.0],
            [2.0, 2.0, 5.0],
            [1.0, 1.0, 3.0]
        ], device=self.device)
        
        self.block_radii = torch.ones(self.num_blocks, device=self.device) * 1.5
        self.block_active = torch.ones(self.num_blocks, dtype=torch.int32, device=self.device)
        
        # Simple ray setup (looking towards blocks)
        self.ray_origins = torch.zeros(self.num_rays, 3, device=self.device)
        self.ray_directions = torch.tensor([
            [0.0, 0.0, 1.0],  # Forward
            [0.1, 0.0, 1.0],  # Slightly right
            [-0.1, 0.0, 1.0], # Slightly left
            [0.0, 0.1, 1.0],  # Slightly up
            [0.0, -0.1, 1.0], # Slightly down
            [0.2, 0.2, 1.0],  # Diagonal
            [0.5, 0.0, 1.0],  # More right
            [0.0, 0.5, 1.0],  # More up
            [1.0, 0.0, 1.0],  # Far right
            [0.0, 1.0, 1.0],  # Far up
        ], device=self.device)
        
        # Normalize ray directions
        self.ray_directions = self.ray_directions / torch.norm(self.ray_directions, dim=1, keepdim=True)
        
        self.ray_near = torch.ones(self.num_rays, device=self.device) * 0.1
        self.ray_far = torch.ones(self.num_rays, device=self.device) * 10.0
    
    def test_memory_bandwidth_correctness(self):
        """Test memory bandwidth operation correctness"""
        # Test with known data
        test_data = torch.arange(100, dtype=torch.float32, device=self.device)
        result = block_nerf_cuda.memory_bandwidth_test(test_data)
        
        # Should be identical
        assert torch.allclose(test_data, result, atol=1e-6)
        
        # Test with different patterns
        test_patterns = [
            torch.zeros(50, device=self.device),
            torch.ones(50, device=self.device),
            torch.full((50,), 3.14159, device=self.device),
            torch.linspace(0, 10, 50, device=self.device)
        ]
        
        for pattern in test_patterns:
            result = block_nerf_cuda.memory_bandwidth_test(pattern)
            assert torch.allclose(pattern, result, atol=1e-6)
    
    def test_block_visibility_correctness(self):
        """Test block visibility computation correctness"""
        visibility = block_nerf_cuda.block_visibility(
            self.camera_positions,
            self.block_centers,
            self.block_radii,
            self.block_active,
            0.1
        )
        
        # Check output shape
        assert visibility.shape == (self.num_blocks,)
        
        # Check output range (should be between 0 and 1)
        assert torch.all(visibility >= 0.0)
        assert torch.all(visibility <= 1.0)
        
        # Test with inactive blocks
        block_active_mixed = torch.tensor([1, 0, 1, 0, 1], dtype=torch.int32, device=self.device)
        visibility_mixed = block_nerf_cuda.block_visibility(
            self.camera_positions,
            self.block_centers,
            self.block_radii,
            block_active_mixed,
            0.1
        )
        
        # Inactive blocks should have zero visibility
        assert visibility_mixed[1] == 0.0
        assert visibility_mixed[3] == 0.0
        
        # Active blocks should have non-zero visibility
        assert visibility_mixed[0] > 0.0
        assert visibility_mixed[2] > 0.0
        assert visibility_mixed[4] > 0.0
    
    def test_block_selection_correctness(self):
        """Test block selection correctness"""
        selected_blocks, num_selected = block_nerf_cuda.block_selection(
            self.ray_origins,
            self.ray_directions,
            self.ray_near,
            self.ray_far,
            self.block_centers,
            self.block_radii,
            self.block_active,
            32
        )
        
        # Check output shapes
        assert selected_blocks.shape == (self.num_rays, 32)
        assert num_selected.shape == (self.num_rays,)
        
        # Check that number of selected blocks is reasonable
        assert torch.all(num_selected >= 0)
        assert torch.all(num_selected <= self.num_blocks)
        
        # Test ray-sphere intersection logic
        # Ray pointing forward should intersect with blocks in front
        forward_ray_idx = 0
        forward_selected = num_selected[forward_ray_idx]
        
        # Should select at least one block (the closest one)
        assert forward_selected > 0
        
        # Test with inactive blocks
        block_active_mixed = torch.tensor([1, 0, 1, 0, 1], dtype=torch.int32, device=self.device)
        selected_blocks_mixed, num_selected_mixed = block_nerf_cuda.block_selection(
            self.ray_origins,
            self.ray_directions,
            self.ray_near,
            self.ray_far,
            self.block_centers,
            self.block_radii,
            block_active_mixed,
            32
        )
        
        # Should select fewer blocks when some are inactive
        assert torch.all(num_selected_mixed <= num_selected)
    
    def test_edge_cases(self):
        """Test edge cases and boundary conditions"""
        # Test with single camera
        single_camera = self.camera_positions[:1]
        visibility_single = block_nerf_cuda.block_visibility(
            single_camera,
            self.block_centers,
            self.block_radii,
            self.block_active,
            0.1
        )
        
        assert visibility_single.shape == (self.num_blocks,)
        assert torch.all(visibility_single >= 0.0)
        
        # Test with single block
        single_block_center = self.block_centers[:1]
        single_block_radius = self.block_radii[:1]
        single_block_active = self.block_active[:1]
        
        visibility_single_block = block_nerf_cuda.block_visibility(
            self.camera_positions,
            single_block_center,
            single_block_radius,
            single_block_active,
            0.1
        )
        
        assert visibility_single_block.shape == (1,)
        assert visibility_single_block[0] > 0.0
        
        # Test with single ray
        single_ray_origin = self.ray_origins[:1]
        single_ray_direction = self.ray_directions[:1]
        single_ray_near = self.ray_near[:1]
        single_ray_far = self.ray_far[:1]
        
        selected_single_ray, num_selected_single_ray = block_nerf_cuda.block_selection(
            single_ray_origin,
            single_ray_direction,
            single_ray_near,
            single_ray_far,
            self.block_centers,
            self.block_radii,
            self.block_active,
            32
        )
        
        assert selected_single_ray.shape == (1, 32)
        assert num_selected_single_ray.shape == (1,)
    
    def test_different_visibility_thresholds(self):
        """Test different visibility thresholds"""
        thresholds = [0.0, 0.1, 0.5, 0.9]
        
        for threshold in thresholds:
            visibility = block_nerf_cuda.block_visibility(
                self.camera_positions,
                self.block_centers,
                self.block_radii,
                self.block_active,
                threshold
            )
            
            # Higher thresholds should generally result in lower visibility scores
            assert torch.all(visibility >= 0.0)
            assert torch.all(visibility <= 1.0)
    
    def test_deterministic_behavior(self):
        """Test that results are deterministic"""
        # Run the same computation multiple times
        results = []
        for _ in range(5):
            visibility = block_nerf_cuda.block_visibility(
                self.camera_positions,
                self.block_centers,
                self.block_radii,
                self.block_active,
                0.1
            )
            results.append(visibility.clone())
        
        # All results should be identical
        for i in range(1, len(results)):
            assert torch.allclose(results[0], results[i], atol=1e-6)
    
    def test_gpu_memory_efficiency(self):
        """Test GPU memory usage is reasonable"""
        # Record initial memory usage
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated()
        
        # Run operations
        visibility = block_nerf_cuda.block_visibility(
            self.camera_positions,
            self.block_centers,
            self.block_radii,
            self.block_active,
            0.1
        )
        
        selected_blocks, num_selected = block_nerf_cuda.block_selection(
            self.ray_origins,
            self.ray_directions,
            self.ray_near,
            self.ray_far,
            self.block_centers,
            self.block_radii,
            self.block_active,
            32
        )
        
        # Memory usage should be reasonable
        peak_memory = torch.cuda.max_memory_allocated()
        memory_used = peak_memory - initial_memory
        
        # Should use less than 100MB for small test
        assert memory_used < 100 * 1024 * 1024  # 100MB
        
        # Cleanup
        del visibility, selected_blocks, num_selected
        torch.cuda.empty_cache()


def run_functional_tests():
    """Run all functional tests"""
    print("🧪 Starting Block-NeRF CUDA Functional Tests...")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return False
    
    if not CUDA_AVAILABLE:
        print("❌ block_nerf_cuda extension not available")
        return False
    
    tests = TestBlockNeRFFunctional()
    tests.setup_method()
    
    try:
        tests.test_memory_bandwidth_correctness()
        print("✅ Memory bandwidth correctness test passed")
        
        tests.test_block_visibility_correctness()
        print("✅ Block visibility correctness test passed")
        
        tests.test_block_selection_correctness()
        print("✅ Block selection correctness test passed")
        
        tests.test_edge_cases()
        print("✅ Edge cases test passed")
        
        tests.test_different_visibility_thresholds()
        print("✅ Different visibility thresholds test passed")
        
        tests.test_deterministic_behavior()
        print("✅ Deterministic behavior test passed")
        
        tests.test_gpu_memory_efficiency()
        print("✅ GPU memory efficiency test passed")
        
        print("\n🎉 All functional tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Functional test failed: {e}")
        return False


if __name__ == "__main__":
    success = run_functional_tests()
    exit(0 if success else 1)
