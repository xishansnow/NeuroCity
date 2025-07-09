"""
InfNeRF 训练器测试

测试 InfNeRFTrainer 的各种功能，包括：
- 训练循环
- 损失计算
- 优化器管理
- 检查点保存和加载
- 训练状态管理
"""

import pytest
import torch
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch

from src.nerfs.inf_nerf import InfNeRF, InfNeRFConfig, InfNeRFTrainer, InfNeRFTrainerConfig


class TestInfNeRFTrainerConfig:
    """测试训练器配置"""

    def test_default_config(self):
        """测试默认配置"""
        config = InfNeRFTrainerConfig()

        assert config.num_epochs == 1000
        assert config.lr_init == 1e-3
        assert config.rays_batch_size == 4096
        assert config.mixed_precision is True
        assert config.save_freq == 1000
        assert config.eval_freq == 500

    def test_custom_config(self):
        """测试自定义配置"""
        config = InfNeRFTrainerConfig(
            num_epochs=100,
            lr_init=5e-4,
            rays_batch_size=2048,
            mixed_precision=False,
            save_freq=500,
            eval_freq=100,
        )

        assert config.num_epochs == 100
        assert config.lr_init == 5e-4
        assert config.rays_batch_size == 2048
        assert config.mixed_precision is False
        assert config.save_freq == 500
        assert config.eval_freq == 100

    def test_config_validation(self):
        """测试配置验证"""
        with pytest.raises(ValueError):
            InfNeRFTrainerConfig(num_epochs=-1)

        with pytest.raises(ValueError):
            InfNeRFTrainerConfig(lr_init=0)

        with pytest.raises(ValueError):
            InfNeRFTrainerConfig(rays_batch_size=0)


class TestInfNeRFTrainer:
    """测试 InfNeRF 训练器"""

    def test_trainer_creation(self, small_config, trainer_config):
        """测试训练器创建"""
        model = InfNeRF(small_config)
        trainer = InfNeRFTrainer(model, trainer_config)

        assert trainer.model is model
        assert trainer.config is trainer_config
        assert trainer.optimizer is not None
        assert trainer.scheduler is not None
        assert trainer.scaler is not None

    def test_trainer_device(self, small_config, trainer_config, device):
        """测试训练器设备管理"""
        model = InfNeRF(small_config).to(device)
        trainer = InfNeRFTrainer(model, trainer_config)

        assert next(trainer.model.parameters()).device == device
        assert trainer.device == device

    def test_loss_computation(self, small_config, trainer_config, device):
        """测试损失计算"""
        model = InfNeRF(small_config).to(device)
        trainer = InfNeRFTrainer(model, trainer_config)

        batch_size = 4
        rays_o = torch.randn(batch_size, 3, device=device)
        rays_d = torch.randn(batch_size, 3, device=device)
        rays_d = torch.nn.functional.normalize(rays_d, dim=-1)
        target_rgb = torch.rand(batch_size, 3, device=device)

        # 前向传播
        output = model(
            rays_o=rays_o,
            rays_d=rays_d,
            near=0.1,
            far=10.0,
            focal_length=800.0,
            pixel_width=1.0 / 800,
        )

        # 计算损失
        loss = trainer.compute_loss(output, target_rgb)

        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad
        assert loss.item() >= 0

    def test_training_step(self, small_config, trainer_config, device, synthetic_dataset):
        """测试训练步骤"""
        model = InfNeRF(small_config).to(device)
        trainer = InfNeRFTrainer(model, trainer_config)

        # 获取一个批次的数据
        batch = synthetic_dataset[0]

        # 执行训练步骤
        loss = trainer.training_step(batch)

        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad
        assert loss.item() >= 0

    def test_validation_step(self, small_config, trainer_config, device, synthetic_dataset):
        """测试验证步骤"""
        model = InfNeRF(small_config).to(device)
        trainer = InfNeRFTrainer(model, trainer_config)

        # 获取一个批次的数据
        batch = synthetic_dataset[0]

        # 执行验证步骤
        metrics = trainer.validation_step(batch)

        assert isinstance(metrics, dict)
        assert "loss" in metrics
        assert "psnr" in metrics
        assert metrics["loss"] >= 0
        assert metrics["psnr"] >= 0

    def test_epoch_training(self, small_config, trainer_config, device, synthetic_dataset):
        """测试完整轮次训练"""
        model = InfNeRF(small_config).to(device)
        trainer = InfNeRFTrainer(model, trainer_config)

        # 模拟数据集
        train_loader = synthetic_dataset[:5]  # 使用前5个样本

        # 执行一个训练轮次
        avg_loss = trainer.train_epoch(train_loader)

        assert isinstance(avg_loss, float)
        assert avg_loss >= 0

    def test_epoch_validation(self, small_config, trainer_config, device, synthetic_dataset):
        """测试完整轮次验证"""
        model = InfNeRF(small_config).to(device)
        trainer = InfNeRFTrainer(model, trainer_config)

        # 模拟数据集
        val_loader = synthetic_dataset[:3]  # 使用前3个样本

        # 执行一个验证轮次
        metrics = trainer.validate_epoch(val_loader)

        assert isinstance(metrics, dict)
        assert "loss" in metrics
        assert "psnr" in metrics
        assert "ssim" in metrics

    def test_checkpoint_save_load(self, small_config, trainer_config, temp_dir):
        """测试检查点保存和加载"""
        model = InfNeRF(small_config)
        trainer = InfNeRFTrainer(model, trainer_config)

        # 保存检查点
        checkpoint_path = temp_dir / "test_checkpoint.pth"
        trainer.save_checkpoint(checkpoint_path)

        assert checkpoint_path.exists()

        # 创建新的训练器并加载检查点
        new_model = InfNeRF(small_config)
        new_trainer = InfNeRFTrainer(new_model, trainer_config)

        new_trainer.load_checkpoint(checkpoint_path)

        # 检查模型参数是否相同
        for p1, p2 in zip(model.parameters(), new_model.parameters()):
            assert torch.allclose(p1, p2)

    def test_learning_rate_scheduling(self, small_config, trainer_config, device):
        """测试学习率调度"""
        model = InfNeRF(small_config).to(device)
        trainer = InfNeRFTrainer(model, trainer_config)

        initial_lr = trainer.optimizer.param_groups[0]["lr"]

        # 模拟一些训练步骤
        for _ in range(10):
            trainer.scheduler.step()

        current_lr = trainer.optimizer.param_groups[0]["lr"]

        # 学习率应该发生变化
        assert current_lr != initial_lr

    def test_mixed_precision_training(self, small_config, device):
        """测试混合精度训练"""
        config = InfNeRFTrainerConfig(mixed_precision=True)
        model = InfNeRF(small_config).to(device)
        trainer = InfNeRFTrainer(model, config)

        assert trainer.scaler is not None

        # 测试混合精度训练步骤
        batch_size = 4
        rays_o = torch.randn(batch_size, 3, device=device)
        rays_d = torch.randn(batch_size, 3, device=device)
        rays_d = torch.nn.functional.normalize(rays_d, dim=-1)
        target_rgb = torch.rand(batch_size, 3, device=device)

        batch = {
            "rays_o": rays_o,
            "rays_d": rays_d,
            "target_rgb": target_rgb,
            "near": torch.tensor(0.1),
            "far": torch.tensor(10.0),
            "focal_length": torch.tensor(800.0),
            "pixel_width": torch.tensor(1.0 / 800),
        }

        loss = trainer.training_step(batch)
        assert isinstance(loss, torch.Tensor)

    def test_gradient_clipping(self, small_config, trainer_config, device):
        """测试梯度裁剪"""
        config = InfNeRFTrainerConfig(gradient_clip_val=1.0)
        model = InfNeRF(small_config).to(device)
        trainer = InfNeRFTrainer(model, config)

        # 创建一些大的梯度
        for param in model.parameters():
            if param.grad is not None:
                param.grad.data.fill_(10.0)

        # 执行梯度裁剪
        trainer.clip_gradients()

        # 检查梯度是否被裁剪
        for param in model.parameters():
            if param.grad is not None:
                assert torch.norm(param.grad) <= 1.0 + 1e-6

    def test_early_stopping(self, small_config, trainer_config, device):
        """测试早停机制"""
        config = InfNeRFTrainerConfig(patience=3, min_delta=1e-4)
        model = InfNeRF(small_config).to(device)
        trainer = InfNeRFTrainer(model, config)

        # 模拟验证损失
        losses = [1.0, 0.9, 0.8, 0.85, 0.86, 0.87]

        for loss in losses:
            should_stop = trainer.early_stopping(loss)
            if should_stop:
                break

        # 应该在第6个损失后触发早停
        assert should_stop

    def test_model_evaluation_mode(self, small_config, trainer_config, device):
        """测试模型评估模式"""
        model = InfNeRF(small_config).to(device)
        trainer = InfNeRFTrainer(model, trainer_config)

        # 确保模型在训练模式
        assert model.training

        # 切换到评估模式
        trainer.eval()
        assert not model.training

        # 切换回训练模式
        trainer.train()
        assert model.training

    def test_training_state_management(self, small_config, trainer_config, device):
        """测试训练状态管理"""
        model = InfNeRF(small_config).to(device)
        trainer = InfNeRFTrainer(model, trainer_config)

        # 检查初始状态
        assert trainer.current_epoch == 0
        assert trainer.global_step == 0
        assert trainer.best_loss == float("inf")

        # 更新状态
        trainer.current_epoch = 5
        trainer.global_step = 100
        trainer.best_loss = 0.5

        assert trainer.current_epoch == 5
        assert trainer.global_step == 100
        assert trainer.best_loss == 0.5

    @pytest.mark.slow
    def test_full_training_cycle(
        self, small_config, trainer_config, device, synthetic_dataset, temp_dir
    ):
        """测试完整训练周期"""
        model = InfNeRF(small_config).to(device)
        trainer = InfNeRFTrainer(model, trainer_config)

        # 使用少量数据进行快速训练
        train_loader = synthetic_dataset[:3]
        val_loader = synthetic_dataset[:2]

        # 训练几个轮次
        for epoch in range(2):
            train_loss = trainer.train_epoch(train_loader)
            val_metrics = trainer.validate_epoch(val_loader)

            assert isinstance(train_loss, float)
            assert isinstance(val_metrics, dict)
            assert train_loss >= 0

        # 保存最终检查点
        checkpoint_path = temp_dir / "final_checkpoint.pth"
        trainer.save_checkpoint(checkpoint_path)
        assert checkpoint_path.exists()
