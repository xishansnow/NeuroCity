"""
SVRaster One 训练器测试

测试训练器的初始化、训练步骤、检查点保存等功能。
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
    """模拟数据集用于测试"""
    
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


class TestSVRasterOneTrainer:
    """测试 SVRaster One 训练器"""

    def test_trainer_init(self):
        """测试训练器初始化"""
        config = SVRasterOneConfig()
        config.training.learning_rate = 1e-3
        config.training.use_amp = False  # 测试时禁用混合精度
        
        model = SVRasterOne(config)
        trainer = SVRasterOneTrainer(model, config)
        
        # 检查基本属性
        assert trainer.model == model
        assert trainer.config == config
        assert trainer.device == model.device
        assert trainer.current_epoch == 0
        assert trainer.global_step == 0
        assert trainer.best_loss == float("inf")
        
        # 检查优化器
        assert isinstance(trainer.optimizer, torch.optim.Adam)
        assert trainer.optimizer.param_groups[0]["lr"] == 1e-3
        
        # 检查学习率调度器
        assert isinstance(trainer.scheduler, torch.optim.lr_scheduler.StepLR)
        
        # 检查混合精度
        assert trainer.use_amp == config.training.use_amp
        if config.training.use_amp:
            assert hasattr(trainer, 'scaler')
        
        # 检查日志记录
        assert isinstance(trainer.train_losses, list)
        assert isinstance(trainer.val_losses, list)
        assert isinstance(trainer.learning_rates, list)

    def test_trainer_init_with_amp(self):
        """测试带混合精度的训练器初始化"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available for AMP testing")
        
        config = SVRasterOneConfig()
        config.device = "cuda"
        config.training.use_amp = True
        
        model = SVRasterOne(config)
        trainer = SVRasterOneTrainer(model, config)
        
        assert trainer.use_amp is True
        assert hasattr(trainer, 'scaler')
        assert isinstance(trainer.scaler, torch.cuda.amp.GradScaler)

    def test_train_epoch(self):
        """测试训练一个 epoch"""
        config = SVRasterOneConfig()
        config.training.batch_size = 4
        config.training.use_amp = False
        
        model = SVRasterOne(config)
        trainer = SVRasterOneTrainer(model, config)
        
        # 创建模拟数据集和数据加载器
        dataset = MockDataset(num_samples=8, image_size=(32, 32))
        dataloader = data.DataLoader(dataset, batch_size=4, shuffle=False)
        
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
        
        # 检查训练状态更新
        assert trainer.current_epoch == 0
        assert trainer.global_step > 0
        assert len(trainer.train_losses) > 0
        assert len(trainer.learning_rates) > 0

    def test_validate(self):
        """测试验证"""
        config = SVRasterOneConfig()
        config.training.batch_size = 4
        config.training.use_amp = False
        
        model = SVRasterOne(config)
        trainer = SVRasterOneTrainer(model, config)
        
        # 创建模拟验证数据集
        val_dataset = MockDataset(num_samples=8, image_size=(32, 32))
        val_dataloader = data.DataLoader(val_dataset, batch_size=4, shuffle=False)
        
        # 执行验证
        val_stats = trainer.validate(val_dataloader, epoch=0)
        
        # 检查验证统计信息
        assert "epoch" in val_stats
        assert "avg_val_loss" in val_stats
        assert "best_loss" in val_stats
        
        assert val_stats["epoch"] == 0
        assert val_stats["avg_val_loss"] >= 0
        assert val_stats["best_loss"] >= 0
        
        # 检查验证损失记录
        assert len(trainer.val_losses) > 0

    def test_adaptive_optimization_step(self):
        """测试自适应优化步骤"""
        config = SVRasterOneConfig()
        model = SVRasterOne(config)
        trainer = SVRasterOneTrainer(model, config)
        
        # 模拟训练结果
        result = {
            "density_gradients": torch.rand(10) * 0.1,
            "total_loss": torch.tensor(0.5),
        }
        
        # 执行自适应优化
        trainer._adaptive_optimization_step(result)
        
        # 应该不抛出异常
        assert True

    def test_save_load_checkpoint(self):
        """测试检查点保存和加载"""
        config = SVRasterOneConfig()
        config.training.use_amp = False
        
        model = SVRasterOne(config)
        trainer = SVRasterOneTrainer(model, config)
        
        # 模拟一些训练状态
        trainer.current_epoch = 5
        trainer.global_step = 100
        trainer.best_loss = 0.1
        trainer.train_losses = [0.5, 0.4, 0.3, 0.2, 0.1]
        trainer.val_losses = [0.6, 0.5, 0.4, 0.3, 0.2]
        trainer.learning_rates = [1e-3, 1e-3, 1e-3, 1e-3, 1e-3]
        
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            checkpoint_path = f.name
        
        try:
            # 保存检查点
            trainer.save_checkpoint(checkpoint_path)
            
            # 创建新的训练器
            new_model = SVRasterOne(config)
            new_trainer = SVRasterOneTrainer(new_model, config)
            
            # 加载检查点
            new_trainer.load_checkpoint(checkpoint_path)
            
            # 检查状态恢复
            assert new_trainer.current_epoch == 5
            assert new_trainer.global_step == 100
            assert new_trainer.best_loss == 0.1
            assert new_trainer.train_losses == [0.5, 0.4, 0.3, 0.2, 0.1]
            assert new_trainer.val_losses == [0.6, 0.5, 0.4, 0.3, 0.2]
            assert new_trainer.learning_rates == [1e-3, 1e-3, 1e-3, 1e-3, 1e-3]
            
        finally:
            Path(checkpoint_path).unlink(missing_ok=True)

    def test_save_load_checkpoint_with_amp(self):
        """测试带混合精度的检查点保存和加载"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available for AMP testing")
        
        config = SVRasterOneConfig()
        config.device = "cuda"
        config.training.use_amp = True
        
        model = SVRasterOne(config)
        trainer = SVRasterOneTrainer(model, config)
        
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            checkpoint_path = f.name
        
        try:
            # 保存检查点
            trainer.save_checkpoint(checkpoint_path)
            
            # 创建新的训练器
            new_model = SVRasterOne(config)
            new_trainer = SVRasterOneTrainer(new_model, config)
            
            # 加载检查点
            new_trainer.load_checkpoint(checkpoint_path)
            
            # 检查混合精度状态恢复
            assert new_trainer.use_amp is True
            assert hasattr(new_trainer, 'scaler')
            
        finally:
            Path(checkpoint_path).unlink(missing_ok=True)

    def test_save_training_log(self):
        """测试训练日志保存"""
        config = SVRasterOneConfig()
        model = SVRasterOne(config)
        trainer = SVRasterOneTrainer(model, config)
        
        # 模拟训练状态
        trainer.current_epoch = 3
        trainer.global_step = 50
        trainer.best_loss = 0.2
        trainer.train_losses = [0.5, 0.4, 0.3]
        trainer.val_losses = [0.6, 0.5, 0.4]
        trainer.learning_rates = [1e-3, 1e-3, 1e-3]
        
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            log_path = f.name
        
        try:
            # 保存训练日志
            trainer.save_training_log(log_path)
            
            # 检查文件存在
            assert Path(log_path).exists()
            
            # 读取并验证日志内容
            with open(log_path, 'r') as f:
                log_data = json.load(f)
            
            assert log_data["current_epoch"] == 3
            assert log_data["global_step"] == 50
            assert log_data["best_loss"] == 0.2
            assert log_data["train_losses"] == [0.5, 0.4, 0.3]
            assert log_data["val_losses"] == [0.6, 0.5, 0.4]
            assert log_data["learning_rates"] == [1e-3, 1e-3, 1e-3]
            assert "config" in log_data
            assert "model_info" in log_data
            
        finally:
            Path(log_path).unlink(missing_ok=True)

    def test_get_training_stats(self):
        """测试获取训练统计信息"""
        config = SVRasterOneConfig()
        model = SVRasterOne(config)
        trainer = SVRasterOneTrainer(model, config)
        
        # 模拟训练状态
        trainer.current_epoch = 2
        trainer.global_step = 30
        trainer.best_loss = 0.3
        trainer.train_losses = [0.5, 0.4]
        trainer.val_losses = [0.6, 0.5]
        trainer.learning_rates = [1e-3, 1e-3]
        
        stats = trainer.get_training_stats()
        
        # 检查统计信息
        assert stats["current_epoch"] == 2
        assert stats["global_step"] == 30
        assert stats["best_loss"] == 0.3
        assert stats["train_losses"] == [0.5, 0.4]
        assert stats["val_losses"] == [0.6, 0.5]
        assert stats["learning_rates"] == [1e-3, 1e-3]
        assert "model_info" in stats

    def test_plot_training_curves(self):
        """测试训练曲线绘制"""
        config = SVRasterOneConfig()
        model = SVRasterOne(config)
        trainer = SVRasterOneTrainer(model, config)
        
        # 模拟训练数据
        trainer.train_losses = [0.5, 0.4, 0.3, 0.2]
        trainer.val_losses = [0.6, 0.5, 0.4, 0.3]
        trainer.learning_rates = [1e-3, 1e-3, 1e-3, 1e-3]
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            plot_path = f.name
        
        try:
            # 绘制训练曲线
            trainer.plot_training_curves(plot_path)
            
            # 检查文件是否创建（如果 matplotlib 可用）
            if Path(plot_path).exists():
                assert Path(plot_path).stat().st_size > 0
            
        finally:
            Path(plot_path).unlink(missing_ok=True)

    def test_trainer_with_callbacks(self):
        """测试带回调函数的训练"""
        config = SVRasterOneConfig()
        config.training.batch_size = 4
        config.training.use_amp = False
        
        model = SVRasterOne(config)
        trainer = SVRasterOneTrainer(model, config)
        
        # 创建回调函数
        callback_called = False
        callback_batch_idx = None
        callback_result = None
        
        def test_callback(trainer_instance, batch_idx, result):
            nonlocal callback_called, callback_batch_idx, callback_result
            callback_called = True
            callback_batch_idx = batch_idx
            callback_result = result
        
        # 创建数据集
        dataset = MockDataset(num_samples=8, image_size=(32, 32))
        dataloader = data.DataLoader(dataset, batch_size=4, shuffle=False)
        
        # 训练一个 epoch
        trainer.train_epoch(dataloader, epoch=0, callbacks=[test_callback])
        
        # 检查回调是否被调用
        assert callback_called is True
        assert callback_batch_idx is not None
        assert callback_result is not None
        assert "total_loss" in callback_result

    def test_trainer_gradient_clipping(self):
        """测试梯度裁剪"""
        config = SVRasterOneConfig()
        config.training.grad_clip = 1.0
        config.training.use_amp = False
        
        model = SVRasterOne(config)
        trainer = SVRasterOneTrainer(model, config)
        
        # 创建数据集
        dataset = MockDataset(num_samples=4, image_size=(16, 16))
        dataloader = data.DataLoader(dataset, batch_size=2, shuffle=False)
        
        # 训练一个 epoch（应该应用梯度裁剪）
        stats = trainer.train_epoch(dataloader, epoch=0)
        
        # 应该不抛出异常
        assert stats["avg_loss"] >= 0

    def test_trainer_device_consistency(self):
        """测试训练器设备一致性"""
        config = SVRasterOneConfig()
        config.device = "cpu"  # 强制使用 CPU
        
        model = SVRasterOne(config)
        trainer = SVRasterOneTrainer(model, config)
        
        # 检查所有组件都在同一设备上
        assert trainer.device == model.device
        assert trainer.device.type == "cpu"
        
        # 检查模型参数在正确设备上
        for param in trainer.model.parameters():
            assert param.device == trainer.device


def test_trainer_edge_cases():
    """测试训练器边界情况"""
    config = SVRasterOneConfig()
    config.training.batch_size = 1
    config.training.use_amp = False
    
    model = SVRasterOne(config)
    trainer = SVRasterOneTrainer(model, config)
    
    # 测试空数据集
    empty_dataset = MockDataset(num_samples=0, image_size=(16, 16))
    empty_dataloader = data.DataLoader(empty_dataset, batch_size=1, shuffle=False)
    
    # 应该不抛出异常
    stats = trainer.train_epoch(empty_dataloader, epoch=0)
    assert stats["avg_loss"] == 0.0 or stats["avg_loss"] == float("inf")


if __name__ == "__main__":
    pytest.main([__file__]) 