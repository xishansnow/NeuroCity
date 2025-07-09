"""
InfNeRF 集成测试

测试 InfNeRF 各个组件之间的集成，包括：
- 端到端训练流程
- 端到端渲染流程
- 模型保存和加载
- 性能基准测试
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import time

from src.nerfs.inf_nerf import (
    InfNeRF,
    InfNeRFConfig,
    InfNeRFTrainer,
    InfNeRFTrainerConfig,
    InfNeRFRenderer,
    InfNeRFRendererConfig,
    InfNeRFDataset,
    InfNeRFDatasetConfig,
)


class TestEndToEndTraining:
    """测试端到端训练流程"""

    @pytest.mark.slow
    def test_complete_training_cycle(self, small_config, trainer_config, device, temp_dir):
        """测试完整训练周期"""
        # 创建模型
        model = InfNeRF(small_config).to(device)

        # 创建训练器
        trainer = InfNeRFTrainer(model, trainer_config)

        # 创建合成数据集
        dataset = self._create_synthetic_dataset(device)

        # 训练几个轮次
        for epoch in range(2):
            train_loss = trainer.train_epoch(dataset[:3])
            val_metrics = trainer.validate_epoch(dataset[:2])

            assert isinstance(train_loss, float)
            assert isinstance(val_metrics, dict)
            assert train_loss >= 0
            assert "loss" in val_metrics
            assert "psnr" in val_metrics

        # 保存检查点
        checkpoint_path = temp_dir / "integration_checkpoint.pth"
        trainer.save_checkpoint(checkpoint_path)
        assert checkpoint_path.exists()

        # 加载检查点
        new_model = InfNeRF(small_config).to(device)
        new_trainer = InfNeRFTrainer(new_model, trainer_config)
        new_trainer.load_checkpoint(checkpoint_path)

        # 验证模型参数相同
        for p1, p2 in zip(model.parameters(), new_model.parameters()):
            assert torch.allclose(p1, p2)

    def _create_synthetic_dataset(self, device):
        """创建合成数据集"""
        dataset = []
        num_samples = 5

        for i in range(num_samples):
            batch_size = 4
            rays_o = torch.randn(batch_size, 3, device=device)
            rays_d = torch.randn(batch_size, 3, device=device)
            rays_d = torch.nn.functional.normalize(rays_d, dim=-1)
            target_rgb = torch.rand(batch_size, 3, device=device)

            sample = {
                "rays_o": rays_o,
                "rays_d": rays_d,
                "target_rgb": target_rgb,
                "near": torch.tensor(0.1, device=device),
                "far": torch.tensor(10.0, device=device),
                "focal_length": torch.tensor(800.0, device=device),
                "pixel_width": torch.tensor(1.0 / 800, device=device),
            }
            dataset.append(sample)

        return dataset


class TestEndToEndRendering:
    """测试端到端渲染流程"""

    def test_complete_rendering_pipeline(self, small_config, renderer_config, device, temp_dir):
        """测试完整渲染流程"""
        # 创建模型
        model = InfNeRF(small_config).to(device)

        # 创建渲染器
        renderer = InfNeRFRenderer(model, renderer_config)

        # 创建相机参数
        camera_pose = torch.eye(4, device=device)
        intrinsics = torch.tensor([[800, 0, 400], [0, 800, 300], [0, 0, 1]], device=device)
        width, height = 32, 32

        # 渲染单张图像
        result = renderer.render_image(camera_pose, intrinsics, width, height)

        assert isinstance(result, dict)
        assert "rgb" in result
        assert "depth" in result
        assert "acc" in result
        assert result["rgb"].shape == (height, width, 3)

        # 批量渲染
        num_views = 3
        camera_poses = torch.stack([torch.eye(4, device=device) for _ in range(num_views)])
        results = renderer.render_batch(camera_poses, intrinsics, width, height)

        assert len(results) == num_views
        for result in results:
            assert "rgb" in result
            assert result["rgb"].shape == (height, width, 3)

        # 保存渲染结果
        output_dir = str(temp_dir / "renders")
        renderer.save_renders(results, output_dir, "integration")

        # 检查输出目录
        output_path = Path(output_dir)
        assert output_path.exists()

    def test_model_checkpoint_rendering(self, small_config, renderer_config, device, temp_dir):
        """测试从检查点渲染"""
        # 创建模型和检查点
        model = InfNeRF(small_config).to(device)
        checkpoint_path = temp_dir / "render_checkpoint.pth"

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "model_config": small_config,
            "epoch": 0,
            "global_step": 0,
        }
        torch.save(checkpoint, checkpoint_path)

        # 从检查点创建渲染器
        renderer = InfNeRFRenderer.from_checkpoint(str(checkpoint_path), renderer_config)

        # 测试渲染
        camera_pose = torch.eye(4, device=device)
        intrinsics = torch.tensor([[800, 0, 400], [0, 800, 300], [0, 0, 1]], device=device)

        result = renderer.render_image(camera_pose, intrinsics, 32, 32)

        assert isinstance(result, dict)
        assert "rgb" in result
        assert result["rgb"].shape == (32, 32, 3)


class TestModelSerialization:
    """测试模型序列化"""

    def test_model_save_load(self, small_config, device, temp_dir):
        """测试模型保存和加载"""
        # 创建模型
        model = InfNeRF(small_config).to(device)

        # 保存模型
        model_path = temp_dir / "test_model.pth"
        torch.save(model.state_dict(), model_path)

        # 加载模型
        new_model = InfNeRF(small_config).to(device)
        new_model.load_state_dict(torch.load(model_path))

        # 验证参数相同
        for p1, p2 in zip(model.parameters(), new_model.parameters()):
            assert torch.allclose(p1, p2)

    def test_config_serialization(self, small_config, temp_dir):
        """测试配置序列化"""
        import json

        # 保存配置
        config_path = temp_dir / "config.json"
        config_dict = {
            "max_depth": small_config.max_depth,
            "hidden_dim": small_config.hidden_dim,
            "num_samples": small_config.num_samples,
            "scene_bound": small_config.scene_bound,
        }

        with open(config_path, "w") as f:
            json.dump(config_dict, f)

        # 加载配置
        with open(config_path, "r") as f:
            loaded_config = json.load(f)

        assert loaded_config["max_depth"] == small_config.max_depth
        assert loaded_config["hidden_dim"] == small_config.hidden_dim
        assert loaded_config["num_samples"] == small_config.num_samples
        assert loaded_config["scene_bound"] == small_config.scene_bound


class TestPerformanceBenchmarks:
    """测试性能基准"""

    @pytest.mark.performance
    def test_training_performance(self, small_config, trainer_config, device):
        """测试训练性能"""
        model = InfNeRF(small_config).to(device)
        trainer = InfNeRFTrainer(model, trainer_config)

        # 创建测试数据
        batch_size = 8
        rays_o = torch.randn(batch_size, 3, device=device)
        rays_d = torch.randn(batch_size, 3, device=device)
        rays_d = torch.nn.functional.normalize(rays_d, dim=-1)
        target_rgb = torch.rand(batch_size, 3, device=device)

        batch = {
            "rays_o": rays_o,
            "rays_d": rays_d,
            "target_rgb": target_rgb,
            "near": torch.tensor(0.1, device=device),
            "far": torch.tensor(10.0, device=device),
            "focal_length": torch.tensor(800.0, device=device),
            "pixel_width": torch.tensor(1.0 / 800, device=device),
        }

        # 预热
        for _ in range(3):
            trainer.training_step(batch)

        # 测量性能
        num_iterations = 10
        start_time = time.time()

        for _ in range(num_iterations):
            trainer.training_step(batch)

        end_time = time.time()
        total_time = end_time - start_time
        avg_time_per_iteration = total_time / num_iterations

        # 性能基准：每次迭代应该小于1秒
        assert avg_time_per_iteration < 1.0

        print(f"Average training iteration time: {avg_time_per_iteration:.4f}s")

    @pytest.mark.performance
    def test_rendering_performance(self, small_config, renderer_config, device):
        """测试渲染性能"""
        model = InfNeRF(small_config).to(device)
        renderer = InfNeRFRenderer(model, renderer_config)

        # 创建相机参数
        camera_pose = torch.eye(4, device=device)
        intrinsics = torch.tensor([[800, 0, 400], [0, 800, 300], [0, 0, 1]], device=device)

        # 测试不同分辨率
        resolutions = [(32, 32), (64, 64), (128, 128)]

        for width, height in resolutions:
            start_time = time.time()
            result = renderer.render_image(camera_pose, intrinsics, width, height)
            end_time = time.time()

            render_time = end_time - start_time

            # 性能基准：渲染时间应该与像素数量成比例
            expected_time = (width * height) / 1000  # 每1000像素1秒
            assert render_time < expected_time

            print(f"Rendering {width}x{height}: {render_time:.4f}s")

    @pytest.mark.performance
    def test_memory_usage(self, small_config, device):
        """测试内存使用"""
        model = InfNeRF(small_config).to(device)

        # 获取内存使用情况
        memory_info = model.get_memory_usage()

        assert "total_mb" in memory_info
        assert memory_info["total_mb"] >= 0

        # 内存使用应该在合理范围内（小于10GB）
        assert memory_info["total_mb"] < 10000

        print(f"Model memory usage: {memory_info['total_mb']:.2f} MB")


class TestErrorHandling:
    """测试错误处理"""

    def test_invalid_config_handling(self):
        """测试无效配置处理"""
        # 测试无效参数
        with pytest.raises(ValueError):
            InfNeRFConfig(max_depth=-1)

        with pytest.raises(ValueError):
            InfNeRFConfig(hidden_dim=0)

        with pytest.raises(ValueError):
            InfNeRFConfig(scene_bound=0)

    def test_invalid_data_handling(self, small_config, device):
        """测试无效数据处理"""
        model = InfNeRF(small_config).to(device)

        # 测试无效的光线数据
        invalid_rays_o = torch.randn(4, 2, device=device)  # 错误的维度
        invalid_rays_d = torch.randn(4, 3, device=device)

        with pytest.raises((ValueError, RuntimeError)):
            model(
                rays_o=invalid_rays_o,
                rays_d=invalid_rays_d,
                near=0.1,
                far=10.0,
                focal_length=800.0,
                pixel_width=1.0 / 800,
            )

    def test_device_mismatch_handling(self, small_config):
        """测试设备不匹配处理"""
        model = InfNeRF(small_config)

        # 创建在不同设备上的数据
        if torch.cuda.is_available():
            rays_o = torch.randn(4, 3, device="cuda")
            rays_d = torch.randn(4, 3, device="cuda")

            # 应该抛出设备不匹配错误
            with pytest.raises((RuntimeError, ValueError)):
                model(
                    rays_o=rays_o,
                    rays_d=rays_d,
                    near=0.1,
                    far=10.0,
                    focal_length=800.0,
                    pixel_width=1.0 / 800,
                )


class TestIntegrationScenarios:
    """测试集成场景"""

    def test_training_to_rendering_workflow(
        self, small_config, trainer_config, renderer_config, device, temp_dir
    ):
        """测试从训练到渲染的完整工作流"""
        # 1. 创建模型
        model = InfNeRF(small_config).to(device)

        # 2. 训练模型
        trainer = InfNeRFTrainer(model, trainer_config)
        dataset = self._create_simple_dataset(device)

        # 训练一个轮次
        trainer.train_epoch(dataset[:2])

        # 3. 保存检查点
        checkpoint_path = temp_dir / "workflow_checkpoint.pth"
        trainer.save_checkpoint(checkpoint_path)

        # 4. 创建渲染器
        renderer = InfNeRFRenderer.from_checkpoint(str(checkpoint_path), renderer_config)

        # 5. 渲染图像
        camera_pose = torch.eye(4, device=device)
        intrinsics = torch.tensor([[800, 0, 400], [0, 800, 300], [0, 0, 1]], device=device)

        result = renderer.render_image(camera_pose, intrinsics, 32, 32)

        # 6. 验证结果
        assert isinstance(result, dict)
        assert "rgb" in result
        assert result["rgb"].shape == (32, 32, 3)

        print("Training to rendering workflow completed successfully")

    def _create_simple_dataset(self, device):
        """创建简单数据集"""
        dataset = []
        for i in range(3):
            batch_size = 4
            rays_o = torch.randn(batch_size, 3, device=device)
            rays_d = torch.randn(batch_size, 3, device=device)
            rays_d = torch.nn.functional.normalize(rays_d, dim=-1)
            target_rgb = torch.rand(batch_size, 3, device=device)

            sample = {
                "rays_o": rays_o,
                "rays_d": rays_d,
                "target_rgb": target_rgb,
                "near": torch.tensor(0.1, device=device),
                "far": torch.tensor(10.0, device=device),
                "focal_length": torch.tensor(800.0, device=device),
                "pixel_width": torch.tensor(1.0 / 800, device=device),
            }
            dataset.append(sample)

        return dataset

    def test_multi_gpu_integration(self, small_config, device):
        """测试多GPU集成"""
        if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
            pytest.skip("需要至少2个GPU")

        # 测试模型在多GPU上的行为
        model = InfNeRF(small_config)

        # 移动到第一个GPU
        model = model.to("cuda:0")

        # 创建数据在第二个GPU上
        rays_o = torch.randn(4, 3, device="cuda:1")
        rays_d = torch.randn(4, 3, device="cuda:1")
        rays_d = torch.nn.functional.normalize(rays_d, dim=-1)

        # 应该处理设备不匹配
        with pytest.raises((RuntimeError, ValueError)):
            model(
                rays_o=rays_o,
                rays_d=rays_d,
                near=0.1,
                far=10.0,
                focal_length=800.0,
                pixel_width=1.0 / 800,
            )


def test_integration_smoke_test(small_config, device):
    """集成冒烟测试"""
    # 创建模型
    model = InfNeRF(small_config).to(device)

    # 创建简单输入
    rays_o = torch.randn(4, 3, device=device)
    rays_d = torch.randn(4, 3, device=device)
    rays_d = torch.nn.functional.normalize(rays_d, dim=-1)

    # 前向传播
    output = model(
        rays_o=rays_o, rays_d=rays_d, near=0.1, far=10.0, focal_length=800.0, pixel_width=1.0 / 800
    )

    # 验证输出
    assert isinstance(output, dict)
    assert "rgb" in output
    assert "depth" in output
    assert "acc" in output
    assert output["rgb"].shape == (4, 3)
    assert output["depth"].shape == (4,)
    assert output["acc"].shape == (4,)

    print("Integration smoke test passed")
