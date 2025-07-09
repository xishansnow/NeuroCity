"""
SVRaster One 集成测试

测试整个系统的端到端功能，包括训练、验证、推理等完整流程。
"""

import pytest
import torch
import torch.utils.data as data
import tempfile
import json
from pathlib import Path
import numpy as np

from ..config import SVRasterOneConfig
from ..core import SVRasterOne
from ..trainer import SVRasterOneTrainer


class MockDataset(data.Dataset):
    """模拟数据集用于集成测试"""
    
    def __init__(self, num_samples=100, image_size=(64, 64)):
        self.num_samples = num_samples
        self.image_size = image_size
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        batch_size = 1
        height, width = self.image_size
        
        # 创建模拟数据
        camera_matrix = torch.eye(4)
        camera_matrix[2, 3] = 2.0  # 相机在 z=2 位置
        
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


class TestSVRasterOneIntegration:
    """SVRaster One 集成测试"""

    def test_full_training_workflow(self):
        """测试完整训练工作流程"""
        config = SVRasterOneConfig()
        config.training.num_epochs = 2
        config.training.batch_size = 4
        config.training.use_amp = False
        config.rendering.image_width = 32
        config.rendering.image_height = 32
        config.voxel.max_voxels = 500
        
        # 创建模型和训练器
        model = SVRasterOne(config)
        trainer = SVRasterOneTrainer(model, config)
        
        # 创建数据集
        train_dataset = MockDataset(num_samples=8, image_size=(32, 32))
        val_dataset = MockDataset(num_samples=4, image_size=(32, 32))
        
        train_loader = data.DataLoader(train_dataset, batch_size=4, shuffle=False)
        val_loader = data.DataLoader(val_dataset, batch_size=4, shuffle=False)
        
        # 执行训练
        trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=2,
            save_dir="test_checkpoints"
        )
        
        # 检查训练状态
        assert trainer.current_epoch >= 0
        assert trainer.global_step > 0
        assert len(trainer.train_losses) > 0
        assert len(trainer.val_losses) > 0
        assert len(trainer.learning_rates) > 0
        
        # 清理
        Path("test_checkpoints").rmdir()

    def test_end_to_end_inference(self):
        """测试端到端推理"""
        config = SVRasterOneConfig()
        config.rendering.image_width = 64
        config.rendering.image_height = 64
        config.voxel.max_voxels = 1000
        
        model = SVRasterOne(config)
        
        # 创建测试相机参数
        num_cameras = 5
        camera_matrices = torch.stack([torch.eye(4) for _ in range(num_cameras)])
        camera_matrices[:, 2, 3] = torch.linspace(1.0, 3.0, num_cameras)
        
        intrinsics = torch.eye(3).unsqueeze(0).repeat(num_cameras, 1, 1)
        
        # 执行推理
        with torch.no_grad():
            result = model.render_sequence(camera_matrices, intrinsics)
        
        # 检查输出
        assert "rgb" in result
        assert "depth" in result
        assert "alpha" in result
        
        assert result["rgb"].shape == (num_cameras, 64, 64, 3)
        assert result["depth"].shape == (num_cameras, 64, 64)
        assert result["alpha"].shape == (num_cameras, 64, 64)
        
        # 检查数值范围
        assert torch.all(result["rgb"] >= 0) and torch.all(result["rgb"] <= 1)
        assert torch.all(result["depth"] >= 0)
        assert torch.all(result["alpha"] >= 0) and torch.all(result["alpha"] <= 1)

    def test_checkpoint_save_load_workflow(self):
        """测试检查点保存和加载工作流程"""
        config = SVRasterOneConfig()
        config.training.use_amp = False
        config.rendering.image_width = 32
        config.rendering.image_height = 32
        
        # 创建模型和训练器
        model = SVRasterOne(config)
        trainer = SVRasterOneTrainer(model, config)
        
        # 模拟训练状态
        trainer.current_epoch = 3
        trainer.global_step = 50
        trainer.best_loss = 0.2
        trainer.train_losses = [0.5, 0.4, 0.3]
        trainer.val_losses = [0.6, 0.5, 0.4]
        trainer.learning_rates = [1e-3, 1e-3, 1e-3]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "checkpoint.pth"
            
            # 保存检查点
            trainer.save_checkpoint(str(checkpoint_path))
            
            # 创建新的模型和训练器
            new_config = SVRasterOneConfig()
            new_model = SVRasterOne(new_config)
            new_trainer = SVRasterOneTrainer(new_model, new_config)
            
            # 加载检查点
            new_trainer.load_checkpoint(str(checkpoint_path))
            
            # 验证状态恢复
            assert new_trainer.current_epoch == 3
            assert new_trainer.global_step == 50
            assert new_trainer.best_loss == 0.2
            assert new_trainer.train_losses == [0.5, 0.4, 0.3]
            assert new_trainer.val_losses == [0.6, 0.5, 0.4]
            assert new_trainer.learning_rates == [1e-3, 1e-3, 1e-3]

    def test_adaptive_optimization_integration(self):
        """测试自适应优化集成"""
        config = SVRasterOneConfig()
        config.voxel.max_voxels = 1000
        config.voxel.adaptive_subdivision = True
        config.rendering.image_width = 32
        config.rendering.image_height = 32
        
        model = SVRasterOne(config)
        
        # 记录初始状态
        initial_stats = model.voxel_grid.get_stats()
        initial_voxel_count = initial_stats["total_voxels"]
        
        # 执行多次自适应优化
        for i in range(5):
            # 模拟训练步骤
            camera_matrix = torch.eye(4)
            camera_matrix[2, 3] = 2.0
            
            intrinsics = torch.eye(3)
            
            target_data = {
                "rgb": torch.rand(1, 32, 32, 3),
                "depth": torch.rand(1, 32, 32),
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

    def test_memory_optimization_integration(self):
        """测试内存优化集成"""
        config = SVRasterOneConfig()
        config.voxel.max_voxels = 2000
        config.rendering.image_width = 64
        config.rendering.image_height = 64
        
        model = SVRasterOne(config)
        
        # 记录初始内存使用
        initial_memory = model.get_memory_usage()
        
        # 执行一些操作
        for i in range(3):
            camera_matrix = torch.eye(4)
            camera_matrix[2, 3] = 2.0
            
            intrinsics = torch.eye(3)
            
            result = model.forward(camera_matrix, intrinsics)
        
        # 执行内存优化
        model.optimize_memory(target_memory_mb=100.0)
        
        # 检查内存使用
        final_memory = model.get_memory_usage()
        
        # 内存使用应该合理
        assert final_memory["total_memory_mb"] >= 0
        assert final_memory["active_memory_mb"] >= 0

    def test_voxel_export_import_integration(self):
        """测试体素导出导入集成"""
        config = SVRasterOneConfig()
        config.voxel.max_voxels = 500
        
        model = SVRasterOne(config)
        
        # 执行一些训练步骤来改变体素状态
        for i in range(3):
            camera_matrix = torch.eye(4)
            camera_matrix[2, 3] = 2.0
            
            intrinsics = torch.eye(3)
            
            target_data = {
                "rgb": torch.rand(1, 32, 32, 3),
                "depth": torch.rand(1, 32, 32),
            }
            
            result = model.training_step_forward(camera_matrix, intrinsics, target_data)
            result["total_loss"].backward()
            model.zero_grad()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            voxel_path = Path(temp_dir) / "voxels.npz"
            
            # 导出体素
            model.export_voxels(str(voxel_path))
            
            # 检查文件存在
            assert voxel_path.exists()
            
            # 创建新模型并导入体素
            new_model = SVRasterOne(config)
            new_model.import_voxels(str(voxel_path))
            
            # 比较体素数据
            original_voxels = model.voxel_grid.get_active_voxels()
            imported_voxels = new_model.voxel_grid.get_active_voxels()
            
            # 检查形状一致性
            assert original_voxels["positions"].shape == imported_voxels["positions"].shape
            assert original_voxels["densities"].shape == imported_voxels["densities"].shape
            assert original_voxels["colors"].shape == imported_voxels["colors"].shape
            assert original_voxels["sizes"].shape == imported_voxels["sizes"].shape

    def test_training_validation_integration(self):
        """测试训练和验证集成"""
        config = SVRasterOneConfig()
        config.training.batch_size = 2
        config.training.use_amp = False
        config.rendering.image_width = 32
        config.rendering.image_height = 32
        config.voxel.max_voxels = 300
        
        model = SVRasterOne(config)
        trainer = SVRasterOneTrainer(model, config)
        
        # 创建数据集
        train_dataset = MockDataset(num_samples=6, image_size=(32, 32))
        val_dataset = MockDataset(num_samples=4, image_size=(32, 32))
        
        train_loader = data.DataLoader(train_dataset, batch_size=2, shuffle=False)
        val_loader = data.DataLoader(val_dataset, batch_size=2, shuffle=False)
        
        # 执行训练和验证
        for epoch in range(2):
            # 训练
            train_stats = trainer.train_epoch(train_loader, epoch)
            
            # 验证
            val_stats = trainer.validate(val_loader, epoch)
            
            # 检查统计信息
            assert train_stats["epoch"] == epoch
            assert val_stats["epoch"] == epoch
            assert train_stats["avg_loss"] >= 0
            assert val_stats["avg_val_loss"] >= 0
            
            # 检查损失记录
            assert len(trainer.train_losses) == epoch + 1
            assert len(trainer.val_losses) == epoch + 1

    def test_model_info_integration(self):
        """测试模型信息集成"""
        config = SVRasterOneConfig()
        config.voxel.max_voxels = 500
        config.rendering.image_width = 64
        config.rendering.image_height = 64
        
        model = SVRasterOne(config)
        
        # 执行一些操作
        for i in range(3):
            camera_matrix = torch.eye(4)
            camera_matrix[2, 3] = 2.0
            
            intrinsics = torch.eye(3)
            
            result = model.forward(camera_matrix, intrinsics)
        
        # 获取模型信息
        model_info = model.get_model_info()
        
        # 检查信息完整性
        assert "model_type" in model_info
        assert "device" in model_info
        assert "voxel_stats" in model_info
        assert "memory_usage" in model_info
        assert "config_summary" in model_info
        
        # 检查体素统计
        voxel_stats = model_info["voxel_stats"]
        assert "total_voxels" in voxel_stats
        assert "active_voxels" in voxel_stats
        assert "subdivision_count" in voxel_stats
        assert "pruning_count" in voxel_stats
        
        # 检查内存使用
        memory_usage = model_info["memory_usage"]
        assert "total_memory_mb" in memory_usage
        assert "active_memory_mb" in memory_usage
        assert "voxel_memory_mb" in memory_usage
        assert "model_memory_mb" in memory_usage

    def test_gradient_flow_integration(self):
        """测试梯度流集成"""
        config = SVRasterOneConfig()
        config.rendering.image_width = 16
        config.rendering.image_height = 16
        config.voxel.max_voxels = 100
        
        model = SVRasterOne(config)
        
        # 创建需要梯度的输入
        camera_matrix = torch.eye(4, requires_grad=True)
        camera_matrix[2, 3] = 2.0
        
        intrinsics = torch.eye(3, requires_grad=True)
        
        target_data = {
            "rgb": torch.rand(1, 16, 16, 3),
            "depth": torch.rand(1, 16, 16),
        }
        
        # 前向传播
        result = model.training_step_forward(camera_matrix, intrinsics, target_data)
        total_loss = result["total_loss"]
        
        # 反向传播
        total_loss.backward()
        
        # 检查梯度流
        assert camera_matrix.grad is not None
        assert intrinsics.grad is not None
        
        # 检查模型参数梯度
        has_gradients = False
        for param in model.parameters():
            if param.requires_grad and param.grad is not None:
                has_gradients = True
                assert not torch.isnan(param.grad).any()
                assert not torch.isinf(param.grad).any()
        
        assert has_gradients, "模型参数应该有梯度"

    def test_device_consistency_integration(self):
        """测试设备一致性集成"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        # CPU 版本
        config_cpu = SVRasterOneConfig()
        config_cpu.device = "cpu"
        config_cpu.rendering.image_width = 32
        config_cpu.rendering.image_height = 32
        
        model_cpu = SVRasterOne(config_cpu)
        trainer_cpu = SVRasterOneTrainer(model_cpu, config_cpu)
        
        # GPU 版本
        config_gpu = SVRasterOneConfig()
        config_gpu.device = "cuda"
        config_gpu.rendering.image_width = 32
        config_gpu.rendering.image_height = 32
        
        model_gpu = SVRasterOne(config_gpu)
        trainer_gpu = SVRasterOneTrainer(model_gpu, config_gpu)
        
        # 检查设备一致性
        assert model_cpu.device.type == "cpu"
        assert model_gpu.device.type == "cuda"
        assert trainer_cpu.device.type == "cpu"
        assert trainer_gpu.device.type == "cuda"
        
        # 检查组件设备一致性
        assert model_cpu.voxel_grid.device.type == "cpu"
        assert model_gpu.voxel_grid.device.type == "cuda"
        assert model_cpu.rasterizer.background_color.device.type == "cpu"
        assert model_gpu.rasterizer.background_color.device.type == "cuda"


def test_integration_edge_cases():
    """测试集成边界情况"""
    config = SVRasterOneConfig()
    config.rendering.image_width = 1
    config.rendering.image_height = 1
    config.voxel.max_voxels = 1
    config.training.batch_size = 1
    
    model = SVRasterOne(config)
    trainer = SVRasterOneTrainer(model, config)
    
    # 测试最小配置
    camera_matrix = torch.eye(4)
    intrinsics = torch.eye(3)
    
    result = model.forward(camera_matrix, intrinsics)
    
    assert result["rgb"].shape == (1, 1, 3)
    assert result["depth"].shape == (1, 1)
    assert result["alpha"].shape == (1, 1)


if __name__ == "__main__":
    pytest.main([__file__]) 