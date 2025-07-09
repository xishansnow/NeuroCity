"""
SVRaster One CUDA 测试

测试 CUDA 加速功能，包括混合精度训练、GPU 内存管理等。
"""

import pytest
import torch
import torch.utils.data as data
import tempfile
from pathlib import Path

from ..config import SVRasterOneConfig
from ..core import SVRasterOne
from ..trainer import SVRasterOneTrainer


class MockDataset(data.Dataset):
    """模拟数据集用于 CUDA 测试"""
    
    def __init__(self, num_samples=50, image_size=(32, 32)):
        self.num_samples = num_samples
        self.image_size = image_size
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        batch_size = 1
        height, width = self.image_size
        
        # 创建模拟数据
        camera_matrix = torch.eye(4)
        camera_matrix[2, 3] = 2.0
        
        intrinsics = torch.tensor([
            [500, 0, width/2],
            [0, 500, height/2],
            [0, 0, 1]
        ], dtype=torch.float32)
        
        target_data = {
            "rgb": torch.rand(batch_size, height, width, 3),
            "depth": torch.rand(batch_size, height, width),
        }
        
        return {
            "camera_matrix": camera_matrix,
            "intrinsics": intrinsics,
            "target": target_data,
        }


class TestSVRasterOneCUDA:
    """SVRaster One CUDA 测试"""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_availability(self):
        """测试 CUDA 可用性"""
        assert torch.cuda.is_available()
        assert torch.cuda.device_count() > 0
        
        # 检查 CUDA 版本
        cuda_version = torch.version.cuda
        assert cuda_version is not None

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_model_cuda_init(self):
        """测试 CUDA 模型初始化"""
        config = SVRasterOneConfig()
        config.device = "cuda"
        config.voxel.max_voxels = 500
        config.rendering.image_width = 32
        config.rendering.image_height = 32
        
        model = SVRasterOne(config)
        
        # 检查设备
        assert model.device.type == "cuda"
        assert next(model.parameters()).device.type == "cuda"
        
        # 检查组件设备
        assert model.voxel_grid.device.type == "cuda"
        assert model.rasterizer.background_color.device.type == "cuda"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_trainer_cuda_init(self):
        """测试 CUDA 训练器初始化"""
        config = SVRasterOneConfig()
        config.device = "cuda"
        config.training.use_amp = True
        config.training.batch_size = 4
        
        model = SVRasterOne(config)
        trainer = SVRasterOneTrainer(model, config)
        
        # 检查设备
        assert trainer.device.type == "cuda"
        assert trainer.model.device.type == "cuda"
        
        # 检查混合精度
        assert trainer.use_amp is True
        assert hasattr(trainer, 'scaler')
        assert isinstance(trainer.scaler, torch.cuda.amp.GradScaler)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_forward_pass(self):
        """测试 CUDA 前向传播"""
        config = SVRasterOneConfig()
        config.device = "cuda"
        config.rendering.image_width = 32
        config.rendering.image_height = 32
        
        model = SVRasterOne(config)
        
        # 创建 CUDA 输入数据
        camera_matrix = torch.eye(4, device="cuda")
        camera_matrix[2, 3] = 2.0
        
        intrinsics = torch.eye(3, device="cuda")
        
        # 前向传播
        result = model.forward(camera_matrix, intrinsics)
        
        # 检查输出设备
        assert result["rgb"].device.type == "cuda"
        assert result["depth"].device.type == "cuda"
        assert result["alpha"].device.type == "cuda"
        
        # 检查形状和数值
        assert result["rgb"].shape == (32, 32, 3)
        assert result["depth"].shape == (32, 32)
        assert result["alpha"].shape == (32, 32)
        
        assert torch.all(result["rgb"] >= 0) and torch.all(result["rgb"] <= 1)
        assert torch.all(result["depth"] >= 0)
        assert torch.all(result["alpha"] >= 0) and torch.all(result["alpha"] <= 1)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_training_step(self):
        """测试 CUDA 训练步骤"""
        config = SVRasterOneConfig()
        config.device = "cuda"
        config.training.use_amp = True
        config.rendering.image_width = 16
        config.rendering.image_height = 16
        
        model = SVRasterOne(config)
        trainer = SVRasterOneTrainer(model, config)
        
        # 创建 CUDA 数据集
        dataset = MockDataset(num_samples=4, image_size=(16, 16))
        dataloader = data.DataLoader(dataset, batch_size=2, shuffle=False)
        
        # 训练一个 epoch
        stats = trainer.train_epoch(dataloader, epoch=0)
        
        # 检查统计信息
        assert "epoch" in stats
        assert "avg_loss" in stats
        assert "epoch_time" in stats
        assert "learning_rate" in stats
        
        assert stats["epoch"] == 0
        assert stats["avg_loss"] >= 0
        assert stats["epoch_time"] >= 0
        assert stats["learning_rate"] > 0

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_validation(self):
        """测试 CUDA 验证"""
        config = SVRasterOneConfig()
        config.device = "cuda"
        config.training.use_amp = True
        config.rendering.image_width = 16
        config.rendering.image_height = 16
        
        model = SVRasterOne(config)
        trainer = SVRasterOneTrainer(model, config)
        
        # 创建 CUDA 验证数据集
        val_dataset = MockDataset(num_samples=4, image_size=(16, 16))
        val_dataloader = data.DataLoader(val_dataset, batch_size=2, shuffle=False)
        
        # 执行验证
        val_stats = trainer.validate(val_dataloader, epoch=0)
        
        # 检查验证统计信息
        assert "epoch" in val_stats
        assert "avg_val_loss" in val_stats
        assert "best_loss" in val_stats
        
        assert val_stats["epoch"] == 0
        assert val_stats["avg_val_loss"] >= 0
        assert val_stats["best_loss"] >= 0

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_gradient_flow(self):
        """测试 CUDA 梯度流"""
        config = SVRasterOneConfig()
        config.device = "cuda"
        config.rendering.image_width = 16
        config.rendering.image_height = 16
        
        model = SVRasterOne(config)
        
        # 创建需要梯度的 CUDA 输入
        camera_matrix = torch.eye(4, device="cuda", requires_grad=True)
        camera_matrix[2, 3] = 2.0
        
        intrinsics = torch.eye(3, device="cuda", requires_grad=True)
        
        target_data = {
            "rgb": torch.rand(1, 16, 16, 3, device="cuda"),
            "depth": torch.rand(1, 16, 16, device="cuda"),
        }
        
        # 训练步骤前向传播
        result = model.training_step_forward(camera_matrix, intrinsics, target_data)
        total_loss = result["total_loss"]
        
        # 反向传播
        total_loss.backward()
        
        # 检查梯度
        assert camera_matrix.grad is not None
        assert intrinsics.grad is not None
        
        assert camera_matrix.grad.device.type == "cuda"
        assert intrinsics.grad.device.type == "cuda"
        
        assert not torch.isnan(camera_matrix.grad).any()
        assert not torch.isnan(intrinsics.grad).any()
        
        # 检查模型参数梯度
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None
                assert param.grad.device.type == "cuda"
                assert not torch.isnan(param.grad).any()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_memory_management(self):
        """测试 CUDA 内存管理"""
        config = SVRasterOneConfig()
        config.device = "cuda"
        config.voxel.max_voxels = 1000
        config.rendering.image_width = 32
        config.rendering.image_height = 32
        
        model = SVRasterOne(config)
        
        # 记录初始内存
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated()
        
        # 执行一些操作
        for i in range(5):
            camera_matrix = torch.eye(4, device="cuda")
            camera_matrix[2, 3] = 2.0
            
            intrinsics = torch.eye(3, device="cuda")
            
            result = model.forward(camera_matrix, intrinsics)
            
            # 清理中间变量
            del result
        
        # 检查内存使用
        current_memory = torch.cuda.memory_allocated()
        
        # 内存使用应该合理
        assert current_memory >= 0
        
        # 清理缓存
        torch.cuda.empty_cache()
        final_memory = torch.cuda.memory_allocated()
        
        # 清理后内存应该减少或保持不变
        assert final_memory <= current_memory

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_checkpoint_save_load(self):
        """测试 CUDA 检查点保存和加载"""
        config = SVRasterOneConfig()
        config.device = "cuda"
        config.training.use_amp = True
        
        model = SVRasterOne(config)
        trainer = SVRasterOneTrainer(model, config)
        
        # 模拟训练状态
        trainer.current_epoch = 2
        trainer.global_step = 30
        trainer.best_loss = 0.3
        
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            checkpoint_path = f.name
        
        try:
            # 保存检查点
            trainer.save_checkpoint(checkpoint_path)
            
            # 创建新的模型和训练器
            new_config = SVRasterOneConfig()
            new_config.device = "cuda"
            new_config.training.use_amp = True
            
            new_model = SVRasterOne(new_config)
            new_trainer = SVRasterOneTrainer(new_model, new_config)
            
            # 加载检查点
            new_trainer.load_checkpoint(checkpoint_path)
            
            # 检查状态恢复
            assert new_trainer.current_epoch == 2
            assert new_trainer.global_step == 30
            assert new_trainer.best_loss == 0.3
            assert new_trainer.use_amp is True
            assert hasattr(new_trainer, 'scaler')
            
        finally:
            Path(checkpoint_path).unlink(missing_ok=True)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_device_consistency(self):
        """测试 CUDA 设备一致性"""
        config = SVRasterOneConfig()
        config.device = "cuda"
        config.voxel.max_voxels = 500
        config.rendering.image_width = 32
        config.rendering.image_height = 32
        
        model = SVRasterOne(config)
        trainer = SVRasterOneTrainer(model, config)
        
        # 检查所有组件都在 CUDA 上
        assert model.device.type == "cuda"
        assert trainer.device.type == "cuda"
        assert model.voxel_grid.device.type == "cuda"
        assert model.rasterizer.background_color.device.type == "cuda"
        
        # 检查模型参数
        for param in model.parameters():
            assert param.device.type == "cuda"
        
        # 检查优化器参数
        for param_group in trainer.optimizer.param_groups:
            for param in param_group['params']:
                assert param.device.type == "cuda"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_mixed_precision_training(self):
        """测试 CUDA 混合精度训练"""
        config = SVRasterOneConfig()
        config.device = "cuda"
        config.training.use_amp = True
        config.training.batch_size = 4
        config.rendering.image_width = 16
        config.rendering.image_height = 16
        
        model = SVRasterOne(config)
        trainer = SVRasterOneTrainer(model, config)
        
        # 创建数据集
        dataset = MockDataset(num_samples=8, image_size=(16, 16))
        dataloader = data.DataLoader(dataset, batch_size=4, shuffle=False)
        
        # 训练一个 epoch
        stats = trainer.train_epoch(dataloader, epoch=0)
        
        # 检查混合精度训练成功
        assert trainer.use_amp is True
        assert hasattr(trainer, 'scaler')
        assert stats["avg_loss"] >= 0
        
        # 检查 scaler 状态
        assert trainer.scaler._scale is not None

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_adaptive_optimization(self):
        """测试 CUDA 自适应优化"""
        config = SVRasterOneConfig()
        config.device = "cuda"
        config.voxel.max_voxels = 1000
        config.voxel.adaptive_subdivision = True
        config.rendering.image_width = 16
        config.rendering.image_height = 16
        
        model = SVRasterOne(config)
        
        # 记录初始状态
        initial_stats = model.voxel_grid.get_stats()
        initial_voxel_count = initial_stats["total_voxels"]
        
        # 执行自适应优化
        for i in range(3):
            camera_matrix = torch.eye(4, device="cuda")
            camera_matrix[2, 3] = 2.0
            
            intrinsics = torch.eye(3, device="cuda")
            
            target_data = {
                "rgb": torch.rand(1, 16, 16, 3, device="cuda"),
                "depth": torch.rand(1, 16, 16, device="cuda"),
            }
            
            # 前向传播
            result = model.training_step_forward(camera_matrix, intrinsics, target_data)
            
            # 反向传播
            result["total_loss"].backward()
            
            # 执行自适应优化
            if "density_gradients" in result:
                model.adaptive_optimization(result["density_gradients"])
            
            # 清零梯度
            model.zero_grad()
        
        # 检查体素数量变化
        final_stats = model.voxel_grid.get_stats()
        final_voxel_count = final_stats["total_voxels"]
        
        # 体素数量应该在合理范围内
        assert final_voxel_count >= 0
        assert final_voxel_count <= config.voxel.max_voxels

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_memory_optimization(self):
        """测试 CUDA 内存优化"""
        config = SVRasterOneConfig()
        config.device = "cuda"
        config.voxel.max_voxels = 2000
        config.rendering.image_width = 32
        config.rendering.image_height = 32
        
        model = SVRasterOne(config)
        
        # 记录初始内存使用
        initial_memory = model.get_memory_usage()
        
        # 执行一些操作
        for i in range(3):
            camera_matrix = torch.eye(4, device="cuda")
            camera_matrix[2, 3] = 2.0
            
            intrinsics = torch.eye(3, device="cuda")
            
            result = model.forward(camera_matrix, intrinsics)
        
        # 执行内存优化
        model.optimize_memory(target_memory_mb=100.0)
        
        # 检查内存使用
        final_memory = model.get_memory_usage()
        
        # 内存使用应该合理
        assert final_memory["total_memory_mb"] >= 0
        assert final_memory["active_memory_mb"] >= 0

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_performance_benchmark(self):
        """测试 CUDA 性能基准"""
        config = SVRasterOneConfig()
        config.device = "cuda"
        config.rendering.image_width = 64
        config.rendering.image_height = 64
        config.voxel.max_voxels = 1000
        
        model = SVRasterOne(config)
        
        # 预热
        for i in range(3):
            camera_matrix = torch.eye(4, device="cuda")
            camera_matrix[2, 3] = 2.0
            
            intrinsics = torch.eye(3, device="cuda")
            
            with torch.no_grad():
                result = model.forward(camera_matrix, intrinsics)
        
        # 同步 GPU
        torch.cuda.synchronize()
        
        # 性能测试
        import time
        
        num_iterations = 10
        start_time = time.time()
        
        for i in range(num_iterations):
            camera_matrix = torch.eye(4, device="cuda")
            camera_matrix[2, 3] = 2.0
            
            intrinsics = torch.eye(3, device="cuda")
            
            with torch.no_grad():
                result = model.forward(camera_matrix, intrinsics)
        
        # 同步 GPU
        torch.cuda.synchronize()
        
        end_time = time.time()
        
        # 计算平均时间
        avg_time = (end_time - start_time) / num_iterations
        
        # 检查性能合理
        assert avg_time >= 0
        assert avg_time < 10.0  # 应该很快


def test_cuda_edge_cases():
    """测试 CUDA 边界情况"""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    config = SVRasterOneConfig()
    config.device = "cuda"
    config.rendering.image_width = 1
    config.rendering.image_height = 1
    config.voxel.max_voxels = 1
    
    model = SVRasterOne(config)
    
    # 测试最小配置
    camera_matrix = torch.eye(4, device="cuda")
    intrinsics = torch.eye(3, device="cuda")
    
    result = model.forward(camera_matrix, intrinsics)
    
    assert result["rgb"].shape == (1, 1, 3)
    assert result["depth"].shape == (1, 1)
    assert result["alpha"].shape == (1, 1)
    
    assert result["rgb"].device.type == "cuda"
    assert result["depth"].device.type == "cuda"
    assert result["alpha"].device.type == "cuda"


if __name__ == "__main__":
    pytest.main([__file__]) 