"""
Test suite for Mega-NeRF++

This module contains comprehensive tests for all components of Mega-NeRF++
including unit tests, integration tests, and performance benchmarks.
"""

import torch
import torch.nn as nn
import numpy as np
import pytest
import tempfile
import shutil
from pathlib import Path
import time
import json

from .core import MegaNeRFPlus, MegaNeRFPlusConfig, HierarchicalSpatialEncoder
from .spatial_partitioner import AdaptiveOctree, PhotogrammetricPartitioner, PartitionConfig
from .multires_renderer import MultiResolutionRenderer, AdaptiveLODRenderer
from .memory_manager import MemoryManager, CacheManager, MemoryOptimizer
from .dataset import PhotogrammetricDataset, create_meganerf_plus_dataset


class TestMegaNeRFPlusCore:
    """Test core components of Mega-NeRF++"""
    
    def setup_method(self):
        """Setup test environment"""
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.config = MegaNeRFPlusConfig(
            num_levels=4,  # Smaller for testing
            base_resolution=16,
            max_resolution=128,
            netdepth=4,
            netwidth=64,
            batch_size=256,
            num_samples=32,
            num_importance=64
        )
    
    def test_config_creation(self):
        """Test configuration creation"""
        config = MegaNeRFPlusConfig()
        
        assert config.num_levels == 8
        assert config.base_resolution == 32
        assert config.max_resolution == 2048
        assert config.batch_size == 4096
        assert config.use_viewdirs == True
    
    def test_hierarchical_spatial_encoder(self):
        """Test hierarchical spatial encoder"""
        encoder = HierarchicalSpatialEncoder(self.config)
        
        # Test forward pass
        batch_size = 16
        positions = torch.randn(batch_size, 3)
        
        features = encoder(positions)
        
        assert features.shape[0] == batch_size
        assert features.shape[1] == encoder.feature_dim
        assert not torch.isnan(features).any()
    
    def test_meganerf_plus_model(self):
        """Test main Mega-NeRF++ model"""
        model = MegaNeRFPlus(self.config).to(self.device)
        
        # Test parameters
        total_params = sum(p.numel() for p in model.parameters())
        assert total_params > 0
        
        # Test forward pass
        batch_size = 32
        rays_o = torch.randn(batch_size, 3, device=self.device)
        rays_d = torch.randn(batch_size, 3, device=self.device)
        rays_d = torch.nn.functional.normalize(rays_d, dim=-1)
        
        with torch.no_grad():
            results = model.render_rays(rays_o, rays_d, near=0.1, far=10.0)
        
        assert 'coarse' in results
        assert 'rgb' in results['coarse']
        assert 'depth' in results['coarse']
        
        rgb = results['coarse']['rgb']
        assert rgb.shape == (batch_size, 3)
        assert torch.all(rgb >= 0) and torch.all(rgb <= 1)
    
    def test_model_with_fine_network(self):
        """Test model with fine network enabled"""
        config = self.config
        config.num_importance = 32
        
        model = MegaNeRFPlus(config).to(self.device)
        
        batch_size = 16
        rays_o = torch.randn(batch_size, 3, device=self.device)
        rays_d = torch.nn.functional.normalize(
            torch.randn(batch_size, 3, device=self.device), dim=-1
        )
        
        with torch.no_grad():
            results = model.render_rays(rays_o, rays_d, near=0.1, far=10.0)
        
        assert 'fine' in results
        assert results['fine']['rgb'].shape == (batch_size, 3)
    
    def test_different_lod_levels(self):
        """Test rendering at different LOD levels"""
        model = MegaNeRFPlus(self.config).to(self.device)
        
        rays_o = torch.randn(8, 3, device=self.device)
        rays_d = torch.nn.functional.normalize(
            torch.randn(8, 3, device=self.device), dim=-1
        )
        
        results_lod0 = model.render_rays(rays_o, rays_d, near=0.1, far=10.0, lod=0)
        results_lod1 = model.render_rays(rays_o, rays_d, near=0.1, far=10.0, lod=1)
        
        # Both should produce valid results
        assert 'coarse' in results_lod0 and 'coarse' in results_lod1
        assert results_lod0['coarse']['rgb'].shape == results_lod1['coarse']['rgb'].shape


class TestSpatialPartitioning:
    """Test spatial partitioning components"""
    
    def setup_method(self):
        """Setup test environment"""
        self.config = PartitionConfig(
            max_partition_size=64,
            min_partition_size=16,
            max_depth=3
        )
    
    def test_adaptive_octree(self):
        """Test adaptive octree partitioning"""
        partitioner = AdaptiveOctree(self.config)
        
        # Create test scene
        scene_bounds = torch.tensor([[-5, -5, -5], [5, 5, 5]], dtype=torch.float32)
        camera_positions = torch.randn(20, 3) * 3  # Random camera positions
        
        partitions = partitioner.partition_scene(scene_bounds, camera_positions)
        
        assert len(partitions) > 0
        assert all('bounds' in p for p in partitions)
        assert all('cameras' in p for p in partitions)
        
        # Test point-to-partition mapping
        test_point = torch.tensor([1.0, 1.0, 1.0])
        partition_idx = partitioner.get_partition_for_point(test_point)
        assert partition_idx is not None or len(partitions) == 0
    
    def test_photogrammetric_partitioner(self):
        """Test photogrammetric-aware partitioning"""
        partitioner = PhotogrammetricPartitioner(self.config)
        
        # Create test data
        scene_bounds = torch.tensor([[-10, -10, -10], [10, 10, 10]], dtype=torch.float32)
        num_cameras = 16
        
        camera_positions = torch.randn(num_cameras, 3) * 5
        camera_orientations = torch.eye(3).unsqueeze(0).repeat(num_cameras, 1, 1)
        image_resolutions = torch.full((num_cameras, 2), 1024, dtype=torch.float32)
        intrinsics = torch.eye(3).unsqueeze(0).repeat(num_cameras, 1, 1)
        
        # Set focal length
        intrinsics[:, 0, 0] = 800  # fx
        intrinsics[:, 1, 1] = 800  # fy
        intrinsics[:, 0, 2] = 512  # cx
        intrinsics[:, 1, 2] = 512  # cy
        
        partitions = partitioner.partition_scene(
            scene_bounds, camera_positions, camera_orientations,
            image_resolutions, intrinsics
        )
        
        assert len(partitions) > 0
        for partition in partitions:
            assert 'bounds' in partition
            assert 'photogrammetric' in partition
            assert partition['photogrammetric'] == True


class TestMultiResolutionRenderer:
    """Test multi-resolution rendering components"""
    
    def setup_method(self):
        """Setup test environment"""
        self.config = MegaNeRFPlusConfig(
            num_lods=3,
            num_samples=16,
            num_importance=32
        )
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def test_multiresolution_renderer(self):
        """Test multi-resolution renderer"""
        renderer = MultiResolutionRenderer(self.config)
        
        # Test LOD determination
        rays_o = torch.randn(10, 3)
        rays_d = torch.nn.functional.normalize(torch.randn(10, 3), dim=-1)
        scene_bounds = torch.tensor([[-5, -5, -5], [5, 5, 5]], dtype=torch.float32)
        
        lod_levels = renderer.determine_lod(rays_o, rays_d, scene_bounds, 1.0)
        
        assert lod_levels.shape[0] == 10
        assert torch.all(lod_levels >= 0) and torch.all(lod_levels < self.config.num_lods)
    
    def test_adaptive_sampling(self):
        """Test adaptive sampling"""
        renderer = MultiResolutionRenderer(self.config)
        
        rays_o = torch.randn(8, 3)
        rays_d = torch.nn.functional.normalize(torch.randn(8, 3), dim=-1)
        lod_levels = torch.randint(0, self.config.num_lods, (8,))
        
        sampling_results = renderer.adaptive_sampling(
            rays_o, rays_d, near=0.1, far=10.0, lod_levels=lod_levels
        )
        
        assert 't_vals' in sampling_results
        assert 'lod_groups' in sampling_results
        assert sampling_results['t_vals'].shape[0] == 8


class TestMemoryManager:
    """Test memory management components"""
    
    def test_memory_manager_creation(self):
        """Test memory manager creation"""
        manager = MemoryManager(max_memory_gb=8.0)
        
        assert manager.max_memory_gb == 8.0
        assert manager.cleanup_threshold == 0.9
    
    def test_memory_stats(self):
        """Test memory statistics"""
        manager = MemoryManager()
        
        stats = manager.get_memory_stats()
        
        assert 'cpu_percent' in stats
        assert 'cpu_available' in stats
        
        if torch.cuda.is_available():
            assert 'gpu_allocated' in stats
            assert 'gpu_cached' in stats
    
    def test_tensor_caching(self):
        """Test tensor caching functionality"""
        manager = MemoryManager()
        
        # Cache a tensor
        test_tensor = torch.randn(100, 100)
        manager.cache_tensor('test_key', test_tensor)
        
        # Retrieve cached tensor
        cached_tensor = manager.get_cached_tensor('test_key')
        assert cached_tensor is not None
        assert torch.equal(cached_tensor, test_tensor)
        
        # Test non-existent key
        missing_tensor = manager.get_cached_tensor('missing_key')
        assert missing_tensor is None
    
    def test_cache_manager(self):
        """Test cache manager"""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_manager = CacheManager(temp_dir, max_cache_size_gb=1.0)
            
            # Cache some data
            test_data = {'test': 'data'}
            cache_manager.cache_data('test_key', test_data)
            
            # Retrieve cached data
            retrieved_data = cache_manager.get_cached_data('test_key')
            assert retrieved_data == test_data
    
    def test_memory_optimizer(self):
        """Test memory optimizer utilities"""
        # Test tensor memory estimation
        test_tensor = torch.randn(1000, 1000)
        memory_mb = MemoryOptimizer.estimate_tensor_memory(test_tensor)
        
        assert memory_mb > 0
        assert isinstance(memory_mb, float)
        
        # Test model memory analysis
        model = nn.Sequential(
            nn.Linear(100, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )
        
        memory_usage = MemoryOptimizer.get_model_memory_usage(model)
        
        assert 'total_parameters' in memory_usage
        assert 'total_memory_mb' in memory_usage
        assert memory_usage['total_parameters'] > 0


class TestDataset:
    """Test dataset components"""
    
    def setup_method(self):
        """Setup test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        
    def teardown_method(self):
        """Cleanup test environment"""
        shutil.rmtree(self.temp_dir)
    
    def create_mock_dataset(self):
        """Create a mock dataset for testing"""
        # Create directory structure
        images_dir = self.temp_path / 'images'
        images_dir.mkdir()
        
        # Create mock images (just save random data as placeholder)
        for i in range(5):
            img_path = images_dir / f'img_{i:03d}.jpg'
            # Just create empty files for testing
            img_path.touch()
        
        # Create poses file
        poses_file = self.temp_path / 'poses.txt'
        poses_data = np.random.randn(20, 4)  # 5 images * 4 rows per pose
        np.savetxt(poses_file, poses_data)
        
        # Create intrinsics file
        intrinsics_file = self.temp_path / 'intrinsics.txt'
        intrinsics_data = np.eye(3).flatten()
        np.savetxt(intrinsics_file, intrinsics_data)
    
    def test_dataset_creation(self):
        """Test dataset creation"""
        self.create_mock_dataset()
        
        try:
            dataset = create_meganerf_plus_dataset(
                str(self.temp_path),
                dataset_type='photogrammetric',
                split='train',
                use_cached_rays=False
            )
            
            # Basic checks
            assert hasattr(dataset, 'images')
            assert hasattr(dataset, 'poses')
            assert hasattr(dataset, 'intrinsics')
            
        except Exception as e:
            # Dataset creation might fail due to missing actual image files
            # This is expected in the test environment
            assert "Failed to load image" in str(e) or "Images directory not found" in str(e)


class TestIntegration:
    """Integration tests for complete workflows"""
    
    def setup_method(self):
        """Setup test environment"""
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.config = MegaNeRFPlusConfig(
            # Small configuration for testing
            num_levels=2,
            base_resolution=8,
            max_resolution=32,
            netdepth=2,
            netwidth=32,
            batch_size=64,
            num_samples=8,
            num_importance=16,
            lr_decay_steps=100
        )
    
    def test_end_to_end_forward_pass(self):
        """Test complete forward pass through the system"""
        
        # Create model
        model = MegaNeRFPlus(self.config).to(self.device)
        
        # Create synthetic batch
        batch_size = 32
        rays_o = torch.randn(batch_size, 3, device=self.device)
        rays_d = torch.nn.functional.normalize(
            torch.randn(batch_size, 3, device=self.device), dim=-1
        )
        target_rgb = torch.rand(batch_size, 3, device=self.device)
        
        # Forward pass
        model.train()
        results = model.render_rays(rays_o, rays_d, near=0.1, far=10.0)
        
        # Compute loss
        coarse_rgb = results['coarse']['rgb']
        loss = torch.nn.functional.mse_loss(coarse_rgb, target_rgb)
        
        # Backward pass
        loss.backward()
        
        # Check gradients
        has_gradients = any(
            p.grad is not None and p.grad.abs().sum() > 0 
            for p in model.parameters() if p.requires_grad
        )
        assert has_gradients, "Model should have gradients after backward pass"
    
    def test_memory_optimization_integration(self):
        """Test memory optimization with model"""
        
        model = MegaNeRFPlus(self.config)
        
        # Apply memory optimizations
        optimized_model = MemoryOptimizer.optimize_model_memory(
            model,
            use_checkpointing=False,  # Skip checkpointing for simplicity
            use_mixed_precision=False
        )
        
        # Model should still be functional
        rays_o = torch.randn(8, 3)
        rays_d = torch.nn.functional.normalize(torch.randn(8, 3), dim=-1)
        
        with torch.no_grad():
            results = optimized_model.render_rays(rays_o, rays_d, near=0.1, far=10.0)
        
        assert 'coarse' in results
        assert results['coarse']['rgb'].shape[0] == 8


class TestPerformance:
    """Performance benchmarks and tests"""
    
    def setup_method(self):
        """Setup test environment"""
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.config = MegaNeRFPlusConfig(
            num_levels=4,
            base_resolution=32,
            max_resolution=256,
            netdepth=6,
            netwidth=128,
            batch_size=1024,
            num_samples=64,
            num_importance=128
        )
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_rendering_speed(self):
        """Test rendering speed benchmark"""
        
        model = MegaNeRFPlus(self.config).to(self.device)
        model.eval()
        
        # Warm up
        rays_o = torch.randn(100, 3, device=self.device)
        rays_d = torch.nn.functional.normalize(
            torch.randn(100, 3, device=self.device), dim=-1
        )
        
        with torch.no_grad():
            for _ in range(5):
                _ = model.render_rays(rays_o, rays_d, near=0.1, far=10.0)
        
        # Benchmark
        batch_sizes = [256, 512, 1024]
        for batch_size in batch_sizes:
            rays_o = torch.randn(batch_size, 3, device=self.device)
            rays_d = torch.nn.functional.normalize(
                torch.randn(batch_size, 3, device=self.device), dim=-1
            )
            
            torch.cuda.synchronize()
            start_time = time.time()
            
            with torch.no_grad():
                results = model.render_rays(rays_o, rays_d, near=0.1, far=10.0)
            
            torch.cuda.synchronize() 
            end_time = time.time()
            
            rays_per_second = batch_size / (end_time - start_time)
            print(f"Batch size {batch_size}: {rays_per_second:.0f} rays/second")
            
            # Basic performance check - should process at least 1000 rays/second
            assert rays_per_second > 1000, f"Rendering too slow: {rays_per_second} rays/sec"
    
    def test_memory_usage(self):
        """Test memory usage scaling"""
        
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        model = MegaNeRFPlus(self.config).to(self.device)
        
        # Measure baseline memory
        torch.cuda.empty_cache()
        baseline_memory = torch.cuda.memory_allocated()
        
        batch_sizes = [256, 512, 1024]
        memory_usage = []
        
        for batch_size in batch_sizes:
            torch.cuda.empty_cache()
            
            rays_o = torch.randn(batch_size, 3, device=self.device)
            rays_d = torch.nn.functional.normalize(
                torch.randn(batch_size, 3, device=self.device), dim=-1
            )
            
            with torch.no_grad():
                results = model.render_rays(rays_o, rays_d, near=0.1, far=10.0)
            
            current_memory = torch.cuda.memory_allocated()
            memory_usage.append(current_memory - baseline_memory)
            
            print(f"Batch size {batch_size}: {(current_memory - baseline_memory) / 1024**2:.1f} MB")
        
        # Memory usage should scale reasonably with batch size
        assert memory_usage[1] > memory_usage[0], "Memory should increase with batch size"
        assert memory_usage[2] > memory_usage[1], "Memory should increase with batch size"


def run_all_tests():
    """Run all tests with detailed output"""
    
    print("Running Mega-NeRF++ Test Suite")
    print("=" * 50)
    
    # Run tests
    test_classes = [
        TestMegaNeRFPlusCore,
        TestSpatialPartitioning,
        TestMultiResolutionRenderer,
        TestMemoryManager,
        TestDataset,
        TestIntegration,
        TestPerformance
    ]
    
    passed_tests = 0
    failed_tests = 0
    
    for test_class in test_classes:
        print(f"\nRunning {test_class.__name__}...")
        
        test_instance = test_class()
        test_methods = [method for method in dir(test_instance) if method.startswith('test_')]
        
        for test_method in test_methods:
            try:
                # Setup
                if hasattr(test_instance, 'setup_method'):
                    test_instance.setup_method()
                
                # Run test
                getattr(test_instance, test_method)()
                
                # Teardown
                if hasattr(test_instance, 'teardown_method'):
                    test_instance.teardown_method()
                
                print(f"  ✓ {test_method}")
                passed_tests += 1
                
            except Exception as e:
                print(f"  ✗ {test_method}: {str(e)}")
                failed_tests += 1
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed_tests} passed, {failed_tests} failed")
    
    if failed_tests == 0:
        print("🎉 All tests passed!")
    else:
        print(f"⚠️  {failed_tests} tests failed")
    
    return failed_tests == 0


if __name__ == '__main__':
    success = run_all_tests()
    exit(0 if success else 1) 