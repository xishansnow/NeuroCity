from __future__ import annotations

"""
Instant NGP CUDA 性能基准测试

测试 CUDA 扩展的性能优势。
"""

import pytest
import torch
import time
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))

from nerfs.instant_ngp.core import InstantNGPConfig, InstantNGPModel, HashEncoder


class TestInstantNGPCUDAPerformance:
    """CUDA 性能测试类"""

    def setup_method(self):
        """测试设置"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = InstantNGPConfig(
            num_levels=8,
            base_resolution=16,
            finest_resolution=128,
            feature_dim=2,
            log2_hashmap_size=18,
            num_samples=64,
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_vs_cpu_performance(self):
        """测试 CUDA vs CPU 性能比较"""
        print(f"\n🏃 CUDA vs CPU 性能比较")
        print("=" * 50)
        
        # 测试数据
        batch_sizes = [1000, 5000, 10000]
        
        for batch_size in batch_sizes:
            print(f"\n📊 批次大小: {batch_size}")
            
            # 创建测试输入
            positions_cpu = torch.randn(batch_size, 3)
            positions_cuda = positions_cpu.to('cuda')
            
            # CPU 测试
            model_cpu = InstantNGPModel(self.config).to('cpu')
            model_cpu.eval()
            
            # 预热
            with torch.no_grad():
                _ = model_cpu.encoding(positions_cpu[:100])
            
            # CPU 性能测试
            start_time = time.time()
            with torch.no_grad():
                for _ in range(10):
                    _ = model_cpu.encoding(positions_cpu)
            cpu_time = (time.time() - start_time) / 10
            
            # CUDA 测试
            model_cuda = InstantNGPModel(self.config).to('cuda')
            model_cuda.eval()
            
            # 预热
            with torch.no_grad():
                _ = model_cuda.encoding(positions_cuda[:100])
            torch.cuda.synchronize()
            
            # CUDA 性能测试
            start_time = time.time()
            with torch.no_grad():
                for _ in range(10):
                    _ = model_cuda.encoding(positions_cuda)
            torch.cuda.synchronize()
            cuda_time = (time.time() - start_time) / 10
            
            # 打印结果
            speedup = cpu_time / cuda_time if cuda_time > 0 else float('inf')
            print(f"  CPU 时间:   {cpu_time*1000:.2f} ms")
            print(f"  CUDA 时间:  {cuda_time*1000:.2f} ms")
            print(f"  加速比:     {speedup:.2f}x")
            
            # 基本断言
            assert cuda_time > 0
            assert cpu_time > 0
            # 通常 CUDA 应该比 CPU 快，但不强制要求
            if speedup < 1.0:
                print(f"  ⚠️  警告: CUDA 比 CPU 慢 (可能是小批次开销)")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_memory_usage(self):
        """测试内存使用情况"""
        print(f"\n💾 内存使用测试")
        print("=" * 50)
        
        # 清空 CUDA 缓存
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated()
        
        # 创建模型
        model = InstantNGPModel(self.config).to('cuda')
        model_memory = torch.cuda.memory_allocated() - initial_memory
        
        # 测试前向传播内存
        batch_size = 10000
        positions = torch.randn(batch_size, 3, device='cuda')
        
        forward_memory_start = torch.cuda.memory_allocated()
        with torch.no_grad():
            output = model.encoding(positions)
        forward_memory = torch.cuda.memory_allocated() - forward_memory_start
        
        print(f"模型内存:     {model_memory / 1024**2:.2f} MB")
        print(f"前向传播内存: {forward_memory / 1024**2:.2f} MB")
        print(f"输出形状:     {output.shape}")
        
        # 基本检查
        assert model_memory > 0
        assert forward_memory >= 0
        assert output.shape[0] == batch_size

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_batch_scaling(self):
        """测试批次大小缩放性能"""
        print(f"\n📈 批次缩放性能测试")
        print("=" * 50)
        
        model = InstantNGPModel(self.config).to('cuda')
        model.eval()
        
        batch_sizes = [100, 500, 1000, 5000, 10000]
        times = []
        
        for batch_size in batch_sizes:
            positions = torch.randn(batch_size, 3, device='cuda')
            
            # 预热
            with torch.no_grad():
                _ = model.encoding(positions[:min(100, batch_size)])
            torch.cuda.synchronize()
            
            # 性能测试
            start_time = time.time()
            with torch.no_grad():
                for _ in range(5):
                    _ = model.encoding(positions)
            torch.cuda.synchronize()
            avg_time = (time.time() - start_time) / 5
            
            times.append(avg_time)
            throughput = batch_size / avg_time
            
            print(f"批次: {batch_size:5d}, 时间: {avg_time*1000:6.2f} ms, "
                  f"吞吐量: {throughput:8.0f} samples/s")
        
        # 检查性能合理性
        assert all(t > 0 for t in times)
        # 通常更大的批次应该有更高的吞吐量
        # 但不强制要求，因为可能有内存限制

    def test_model_consistency(self):
        """测试模型在 CPU 和 CUDA 上的一致性"""
        print(f"\n🔍 CPU/CUDA 一致性测试")
        print("=" * 50)
        
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        # 创建相同的模型
        torch.manual_seed(42)
        model_cpu = InstantNGPModel(self.config).to('cpu')
        
        torch.manual_seed(42)
        model_cuda = InstantNGPModel(self.config).to('cuda')
        
        # 复制权重确保一致
        model_cuda.load_state_dict(model_cpu.state_dict())
        
        # 测试输入
        positions = torch.randn(100, 3)
        positions_cuda = positions.to('cuda')
        
        # 前向传播
        model_cpu.eval()
        model_cuda.eval()
        
        with torch.no_grad():
            output_cpu = model_cpu.encoding(positions)
            output_cuda = model_cuda.encoding(positions_cuda).cpu()
        
        # 检查一致性
        max_diff = torch.max(torch.abs(output_cpu - output_cuda)).item()
        mean_diff = torch.mean(torch.abs(output_cpu - output_cuda)).item()
        
        print(f"最大差异: {max_diff:.6f}")
        print(f"平均差异: {mean_diff:.6f}")
        
        # 允许小的数值误差
        assert max_diff < 1e-4, f"CPU/CUDA 输出差异过大: {max_diff}"
        print("✅ CPU/CUDA 输出一致")

    def teardown_method(self):
        """清理"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
