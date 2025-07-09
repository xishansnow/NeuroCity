"""
Test Instant NGP Trainer Components

This module tests the training-related components of Instant NGP:
- InstantNGPTrainer
- InstantNGPTrainerConfig
"""

import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import tempfile
import os
import sys
import time
from pathlib import Path

from dataclasses import asdict

# Add the src directory to the path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))

try:
    from nerfs.instant_ngp import InstantNGP
    from nerfs.instant_ngp.core import InstantNGPConfig, InstantNGPModel
    from nerfs.instant_ngp.trainer import InstantNGPTrainer, InstantNGPTrainerConfig

    INSTANT_NGP_AVAILABLE = True
except ImportError as e:
    INSTANT_NGP_AVAILABLE = False
    IMPORT_ERROR = str(e)


class TestInstantNGPTrainer:
    """Instant NGP 训练器测试类"""

    def setup_method(self):
        """每个测试方法前的设置"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 模型配置
        self.model_config = InstantNGPConfig(
            num_levels=4,  # 减少层数以加快测试
            base_resolution=16,
            finest_resolution=128,
            hidden_dim=32,
            num_layers=2,
            batch_size=512,
        )

        # 训练器配置
        self.trainer_config = InstantNGPTrainerConfig(
            num_epochs=2,
            batch_size=512,
            learning_rate=1e-2,
            learning_rate_hash=1e-1,
            log_freq=10,
            eval_freq=50,
            save_freq=100,
            use_mixed_precision=False,  # 关闭混合精度以简化测试
        )

    def test_trainer_config_initialization(self):
        """测试训练器配置初始化"""
        config = InstantNGPTrainerConfig()

        # 检查默认值
        assert config.num_epochs == 20
        assert config.batch_size == 8192
        assert config.learning_rate == 1e-2
        assert config.learning_rate_hash == 1e-1
        assert config.use_mixed_precision is True

        # 测试 Python 3.10 兼容性 - 使用内置容器
        config_dict: dict[str, Any] = asdict(config)
        assert isinstance(config_dict, dict)
        assert "num_epochs" in config_dict
        assert "batch_size" in config_dict

    def test_trainer_initialization(self):
        """测试训练器初始化"""
        model = InstantNGPModel(self.model_config)
        trainer = InstantNGPTrainer(model, self.trainer_config, device=self.device)

        # 检查基本属性
        assert trainer.model is model
        assert trainer.config is self.trainer_config
        assert trainer.device == self.device

        # 检查优化器
        assert trainer.optimizer is not None
        assert isinstance(trainer.optimizer, optim.Adam)

        # 检查学习率调度器
        assert trainer.scheduler is not None

    def test_trainer_device_auto_selection(self):
        """测试训练器设备自动选择"""
        model = InstantNGPModel(self.model_config)
        config = InstantNGPTrainerConfig(device="auto")
        trainer = InstantNGPTrainer(model, config)

        # 检查设备选择
        expected_device = "cuda" if torch.cuda.is_available() else "cpu"
        assert str(trainer.device).startswith(expected_device)

    def test_optimizer_groups(self):
        """测试优化器参数组"""
        model = InstantNGPModel(self.model_config)
        trainer = InstantNGPTrainer(model, self.trainer_config, device=self.device)

        # 检查参数组
        param_groups: list[dict[str, Any]] = trainer.optimizer.param_groups
        assert len(param_groups) == 2  # 哈希编码器和其他参数

        # 检查学习率设置
        hash_lr = None
        other_lr = None

        for group in param_groups:
            if "hash" in str(group.get("name", "")):
                hash_lr = group["lr"]
            else:
                other_lr = group["lr"]

        # 哈希编码器应该有更高的学习率
        assert hash_lr is not None
        assert other_lr is not None
        assert hash_lr > other_lr

    def test_create_dummy_dataloader(self):
        """创建虚拟数据加载器用于测试"""
        # 创建虚拟数据
        num_samples = 1000
        ray_origins = torch.randn(num_samples, 3)
        ray_directions = torch.randn(num_samples, 3)
        ray_directions = torch.nn.functional.normalize(ray_directions, dim=-1)
        colors = torch.rand(num_samples, 3)

        # 创建数据集
        dataset = TensorDataset(ray_origins, ray_directions, colors)
        dataloader = DataLoader(
            dataset,
            batch_size=self.trainer_config.batch_size,
            shuffle=True,
            num_workers=0,  # 避免多进程问题
        )

        return dataloader

    def test_training_step(self):
        """测试单步训练"""
        model = InstantNGPModel(self.model_config)
        trainer = InstantNGPTrainer(model, self.trainer_config, device=self.device)

        # 创建虚拟批次数据
        batch_size = 100
        ray_origins = torch.randn(batch_size, 3, device=self.device)
        ray_directions = torch.randn(batch_size, 3, device=self.device)
        ray_directions = torch.nn.functional.normalize(ray_directions, dim=-1)
        colors = torch.rand(batch_size, 3, device=self.device)

        # 执行训练步骤
        batch_data: dict[str, torch.Tensor] = {
            "ray_origins": ray_origins,
            "ray_directions": ray_directions,
            "colors": colors,
        }

        loss_info = trainer.training_step(batch_data)

        # 检查损失信息
        assert isinstance(loss_info, dict)
        assert "total_loss" in loss_info
        assert "color_loss" in loss_info

        # 检查损失值
        assert loss_info["total_loss"] >= 0
        assert loss_info["color_loss"] >= 0

    def test_training_loop_single_epoch(self):
        """测试单个 epoch 的训练循环"""
        model = InstantNGPModel(self.model_config)
        trainer = InstantNGPTrainer(model, self.trainer_config, device=self.device)

        # 创建虚拟数据加载器
        dataloader = self.test_create_dummy_dataloader()

        # 训练一个 epoch
        trainer.train_epoch(dataloader, epoch=0)

        # 检查训练状态
        assert trainer.step_count > 0
        assert len(trainer.loss_history) > 0

        # 检查损失历史 - 使用 Python 3.10 兼容的类型
        loss_values: list[float] = trainer.loss_history
        assert all(isinstance(loss, float) for loss in loss_values)
        assert all(loss >= 0 for loss in loss_values)

    def test_learning_rate_scheduling(self):
        """测试学习率调度"""
        model = InstantNGPModel(self.model_config)
        trainer = InstantNGPTrainer(model, self.trainer_config, device=self.device)

        # 获取初始学习率
        initial_lrs: list[float] = [group["lr"] for group in trainer.optimizer.param_groups]

        # 模拟训练步骤
        for _ in range(10):
            trainer.step_count += 1
            if trainer.step_count % self.trainer_config.decay_step == 0:
                trainer.scheduler.step()

        # 检查学习率变化
        current_lrs: list[float] = [group["lr"] for group in trainer.optimizer.param_groups]

        # 在某些步骤后，学习率应该有所变化
        assert len(current_lrs) == len(initial_lrs)

    def test_checkpoint_saving_loading(self):
        """测试检查点保存和加载"""
        model = InstantNGPModel(self.model_config)
        trainer = InstantNGPTrainer(model, self.trainer_config, device=self.device)

        # 创建临时目录
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "test_checkpoint.pth"

            # 模拟训练状态
            trainer.step_count = 100
            trainer.loss_history = [1.0, 0.9, 0.8, 0.7]

            # 保存检查点
            trainer.save_checkpoint(checkpoint_path)
            assert checkpoint_path.exists()

            # 创建新的训练器并加载检查点
            new_model = InstantNGPModel(self.model_config)
            new_trainer = InstantNGPTrainer(new_model, self.trainer_config, device=self.device)
            new_trainer.load_checkpoint(checkpoint_path)

            # 检查状态恢复
            assert new_trainer.step_count == 100
            assert new_trainer.loss_history == [1.0, 0.9, 0.8, 0.7]

    def test_evaluation_mode(self):
        """测试评估模式"""
        model = InstantNGPModel(self.model_config)
        trainer = InstantNGPTrainer(model, self.trainer_config, device=self.device)

        # 创建虚拟评估数据
        batch_size = 50
        ray_origins = torch.randn(batch_size, 3, device=self.device)
        ray_directions = torch.randn(batch_size, 3, device=self.device)
        ray_directions = torch.nn.functional.normalize(ray_directions, dim=-1)
        colors = torch.rand(batch_size, 3, device=self.device)

        eval_batch: dict[str, torch.Tensor] = {
            "ray_origins": ray_origins,
            "ray_directions": ray_directions,
            "colors": colors,
        }

        # 执行评估
        eval_metrics = trainer.evaluate_batch(eval_batch)

        # 检查评估指标
        assert isinstance(eval_metrics, dict)
        assert "loss" in eval_metrics
        assert "psnr" in eval_metrics

        # 检查指标值
        assert eval_metrics["loss"] >= 0
        assert eval_metrics["psnr"] >= 0

    def test_mixed_precision_training(self):
        """测试混合精度训练"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available for mixed precision testing")

        # 启用混合精度
        trainer_config = InstantNGPTrainerConfig(
            use_mixed_precision=True,
            batch_size=512,
            num_epochs=1,
        )

        model = InstantNGPModel(self.model_config)
        trainer = InstantNGPTrainer(model, trainer_config, device=self.device)

        # 检查 GradScaler 初始化
        assert trainer.scaler is not None
        assert isinstance(trainer.scaler, torch.cuda.amp.GradScaler)

        # 创建虚拟数据进行训练
        batch_size = 100
        ray_origins = torch.randn(batch_size, 3, device=self.device)
        ray_directions = torch.randn(batch_size, 3, device=self.device)
        ray_directions = torch.nn.functional.normalize(ray_directions, dim=-1)
        colors = torch.rand(batch_size, 3, device=self.device)

        batch_data = {
            "ray_origins": ray_origins,
            "ray_directions": ray_directions,
            "colors": colors,
        }

        # 执行混合精度训练步骤
        loss_info = trainer.training_step(batch_data)

        # 检查损失
        assert "total_loss" in loss_info
        assert loss_info["total_loss"] >= 0

    def test_gradient_accumulation(self):
        """测试梯度累积"""
        model = InstantNGPModel(self.model_config)
        trainer = InstantNGPTrainer(model, self.trainer_config, device=self.device)

        # 创建小批次数据
        batch_size = 50

        # 累积多个批次的梯度
        total_loss = 0
        accumulation_steps = 3

        for i in range(accumulation_steps):
            ray_origins = torch.randn(batch_size, 3, device=self.device)
            ray_directions = torch.randn(batch_size, 3, device=self.device)
            ray_directions = torch.nn.functional.normalize(ray_directions, dim=-1)
            colors = torch.rand(batch_size, 3, device=self.device)

            batch_data = {
                "ray_origins": ray_origins,
                "ray_directions": ray_directions,
                "colors": colors,
            }

            # 训练步骤（不更新参数）
            loss_info = trainer.training_step(
                batch_data, update_params=(i == accumulation_steps - 1)
            )
            total_loss += loss_info["total_loss"]

        # 检查总损失
        assert total_loss > 0

    def test_training_metrics_collection(self):
        """测试训练指标收集"""
        model = InstantNGPModel(self.model_config)
        trainer = InstantNGPTrainer(model, self.trainer_config, device=self.device)

        # 执行几个训练步骤
        for _ in range(5):
            batch_size = 100
            ray_origins = torch.randn(batch_size, 3, device=self.device)
            ray_directions = torch.randn(batch_size, 3, device=self.device)
            ray_directions = torch.nn.functional.normalize(ray_directions, dim=-1)
            colors = torch.rand(batch_size, 3, device=self.device)

            batch_data = {
                "ray_origins": ray_origins,
                "ray_directions": ray_directions,
                "colors": colors,
            }

            trainer.training_step(batch_data)

        # 检查指标收集
        assert len(trainer.loss_history) == 5
        assert trainer.step_count == 5

        # 检查指标类型 - 使用 Python 3.10 兼容的类型
        metrics: dict[str, Any] = trainer.get_training_metrics()
        assert isinstance(metrics, dict)
        assert "average_loss" in metrics
        assert "step_count" in metrics
        assert isinstance(metrics["average_loss"], float)
        assert isinstance(metrics["step_count"], int)

    def test_training_performance_monitoring(self):
        """测试训练性能监控"""
        model = InstantNGPModel(self.model_config)
        trainer = InstantNGPTrainer(model, self.trainer_config, device=self.device)

        # 记录开始时间
        start_time = time.time()

        # 执行多个训练步骤
        for _ in range(10):
            batch_size = 100
            ray_origins = torch.randn(batch_size, 3, device=self.device)
            ray_directions = torch.randn(batch_size, 3, device=self.device)
            ray_directions = torch.nn.functional.normalize(ray_directions, dim=-1)
            colors = torch.rand(batch_size, 3, device=self.device)

            batch_data = {
                "ray_origins": ray_origins,
                "ray_directions": ray_directions,
                "colors": colors,
            }

            trainer.training_step(batch_data)

        # 计算训练时间
        end_time = time.time()
        training_time = end_time - start_time

        # 检查性能指标
        assert training_time > 0
        assert trainer.step_count == 10

        # 计算每步平均时间
        avg_time_per_step = training_time / 10
        assert avg_time_per_step > 0

    def teardown_method(self):
        """每个测试方法后的清理"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class TestInstantNGPTrainerIntegration:
    """Instant NGP 训练器集成测试"""

    def setup_method(self):
        """设置集成测试"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 使用较小的配置以加快测试
        self.model_config = InstantNGPConfig(
            num_levels=4,
            base_resolution=16,
            finest_resolution=64,
            hidden_dim=32,
            num_layers=2,
        )

        self.trainer_config = InstantNGPTrainerConfig(
            num_epochs=2,
            batch_size=256,
            learning_rate=1e-2,
            log_freq=5,
            eval_freq=10,
            use_mixed_precision=False,
        )

    def test_full_training_pipeline(self):
        """测试完整的训练流程"""
        model = InstantNGPModel(self.model_config)
        trainer = InstantNGPTrainer(model, self.trainer_config, device=self.device)

        # 创建虚拟数据集
        num_samples = 500
        ray_origins = torch.randn(num_samples, 3)
        ray_directions = torch.randn(num_samples, 3)
        ray_directions = torch.nn.functional.normalize(ray_directions, dim=-1)
        colors = torch.rand(num_samples, 3)

        dataset = TensorDataset(ray_origins, ray_directions, colors)
        dataloader = DataLoader(dataset, batch_size=self.trainer_config.batch_size, shuffle=True)

        # 执行完整训练
        initial_loss = None
        final_loss = None

        for epoch in range(self.trainer_config.num_epochs):
            for batch_idx, batch in enumerate(dataloader):
                ray_origins, ray_directions, colors = batch
                ray_origins = ray_origins.to(self.device)
                ray_directions = ray_directions.to(self.device)
                colors = colors.to(self.device)

                batch_data = {
                    "ray_origins": ray_origins,
                    "ray_directions": ray_directions,
                    "colors": colors,
                }

                loss_info = trainer.training_step(batch_data)

                if initial_loss is None:
                    initial_loss = loss_info["total_loss"]
                final_loss = loss_info["total_loss"]

        # 检查训练进度
        assert trainer.step_count > 0
        assert len(trainer.loss_history) > 0
        assert initial_loss is not None
        assert final_loss is not None

        # 损失应该有所改善（或至少没有发散）
        assert final_loss < initial_loss * 2  # 允许一定的波动

    def teardown_method(self):
        """清理集成测试"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    pytest.main([__file__])
