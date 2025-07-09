"""
SVRaster One 核心模型测试

测试 SVRasterOne 模型的初始化、前向传播、训练等功能。
"""

import pytest
import torch
import tempfile
from pathlib import Path

from ..config import SVRasterOneConfig
from ..core import SVRasterOne


class TestSVRasterOne:
    """测试 SVRaster One 核心模型"""

    def test_model_init(self):
        """测试模型初始化"""
        config = SVRasterOneConfig()
        config.voxel.grid_resolution = 64
        config.voxel.max_voxels = 1000
        
        model = SVRasterOne(config)
        
        # 检查基本属性
        assert model.config == config
        assert model.device == torch.device(config.device)
        assert model.training_step == 0
        assert model.best_loss == float("inf")
        
        # 检查组件
        assert hasattr(model, 'voxel_grid')
        assert hasattr(model, 'rasterizer')
        assert hasattr(model, 'loss_fn')
        
        # 检查设备一致性
        assert model.voxel_grid.device == model.device
        assert next(model.voxel_grid.parameters()).device == model.device

    def test_model_forward(self):
        """测试模型前向传播"""
        config = SVRasterOneConfig()
        config.rendering.image_width = 64
        config.rendering.image_height = 64
        
        model = SVRasterOne(config)
        
        # 创建输入数据
        camera_matrix = torch.eye(4)
        camera_matrix[2, 3] = 2.0  # 相机在 z=2 位置
        
        intrinsics = torch.tensor([
            [500, 0, 32],
            [0, 500, 32],
            [0, 0, 1]
        ], dtype=torch.float32)
        
        # 前向传播
        result = model.forward(camera_matrix, intrinsics)
        
        # 检查输出
        assert "rgb" in result
        assert "depth" in result
        assert "alpha" in result
        assert "voxel_stats" in result
        
        assert result["rgb"].shape == (64, 64, 3)
        assert result["depth"].shape == (64, 64)
        assert result["alpha"].shape == (64, 64)
        
        # 检查数值范围
        assert torch.all(result["rgb"] >= 0) and torch.all(result["rgb"] <= 1)
        assert torch.all(result["depth"] >= 0)
        assert torch.all(result["alpha"] >= 0) and torch.all(result["alpha"] <= 1)

    def test_model_forward_training_mode(self):
        """测试训练模式的前向传播"""
        config = SVRasterOneConfig()
        config.rendering.image_width = 32
        config.rendering.image_height = 32
        
        model = SVRasterOne(config)
        
        camera_matrix = torch.eye(4)
        camera_matrix[2, 3] = 2.0
        
        intrinsics = torch.eye(3)
        
        # 训练模式
        result = model.forward(camera_matrix, intrinsics, mode="training")
        
        # 检查软光栅化设置
        assert config.rendering.soft_rasterization is True
        assert config.rendering.use_soft_sorting is True
        
        # 检查输出
        assert "rgb" in result
        assert "depth" in result
        assert "alpha" in result

    def test_model_forward_inference_mode(self):
        """测试推理模式的前向传播"""
        config = SVRasterOneConfig()
        config.rendering.image_width = 32
        config.rendering.image_height = 32
        
        model = SVRasterOne(config)
        
        camera_matrix = torch.eye(4)
        camera_matrix[2, 3] = 2.0
        
        intrinsics = torch.eye(3)
        
        # 推理模式
        result = model.forward(camera_matrix, intrinsics, mode="inference")
        
        # 检查硬光栅化设置
        assert config.rendering.soft_rasterization is False
        assert config.rendering.use_soft_sorting is False
        
        # 检查输出
        assert "rgb" in result
        assert "depth" in result
        assert "alpha" in result

    def test_compute_loss(self):
        """测试损失计算"""
        config = SVRasterOneConfig()
        model = SVRasterOne(config)
        
        # 创建渲染输出
        batch_size = 2
        height, width = 32, 32
        
        rendered_output = {
            "rgb": torch.rand(batch_size, height, width, 3),
            "depth": torch.rand(batch_size, height, width),
            "alpha": torch.rand(batch_size, height, width),
            "voxel_stats": {"total_voxels": 100, "active_voxels": 50},
        }
        
        # 创建目标数据
        target_data = {
            "rgb": torch.rand(batch_size, height, width, 3),
            "depth": torch.rand(batch_size, height, width),
        }
        
        # 计算损失
        losses = model.compute_loss(rendered_output, target_data)
        
        # 检查损失字典
        assert "total_loss" in losses
        assert "rgb_loss" in losses
        assert "depth_loss" in losses
        assert "density_reg" in losses
        assert "sparsity_loss" in losses
        
        # 检查损失值
        for loss_name, loss_value in losses.items():
            assert isinstance(loss_value, torch.Tensor)
            assert loss_value.shape == ()
            assert loss_value.item() >= 0
            assert not torch.isnan(loss_value)
            assert not torch.isinf(loss_value)

    def test_training_step_forward(self):
        """测试训练步骤前向传播"""
        config = SVRasterOneConfig()
        config.rendering.image_width = 32
        config.rendering.image_height = 32
        
        model = SVRasterOne(config)
        
        # 创建输入数据
        camera_matrix = torch.eye(4)
        camera_matrix[2, 3] = 2.0
        
        intrinsics = torch.eye(3)
        
        target_data = {
            "rgb": torch.rand(1, 32, 32, 3),
            "depth": torch.rand(1, 32, 32),
        }
        
        # 训练步骤前向传播
        result = model.training_step_forward(camera_matrix, intrinsics, target_data)
        
        # 检查输出包含渲染结果和损失
        assert "rgb" in result
        assert "depth" in result
        assert "alpha" in result
        assert "total_loss" in result
        assert "rgb_loss" in result
        assert "depth_loss" in result
        assert "density_reg" in result
        assert "sparsity_loss" in result
        
        # 检查损失值
        assert result["total_loss"].item() >= 0
        assert not torch.isnan(result["total_loss"])

    def test_adaptive_optimization(self):
        """测试自适应优化"""
        config = SVRasterOneConfig()
        config.voxel.max_voxels = 200
        
        model = SVRasterOne(config)
        
        # 记录初始体素数量
        initial_stats = model.voxel_grid.get_stats()
        initial_voxel_count = initial_stats["total_voxels"]
        
        # 模拟梯度幅度
        gradient_magnitudes = torch.rand(initial_voxel_count) * 0.5
        
        # 执行自适应优化
        model.adaptive_optimization(gradient_magnitudes)
        
        # 检查体素数量可能变化
        new_stats = model.voxel_grid.get_stats()
        new_voxel_count = new_stats["total_voxels"]
        
        # 体素数量应该保持合理范围
        assert new_voxel_count >= 0
        assert new_voxel_count <= config.voxel.max_voxels

    def test_get_trainable_parameters(self):
        """测试获取可训练参数"""
        config = SVRasterOneConfig()
        model = SVRasterOne(config)
        
        trainable_params = model.get_trainable_parameters()
        
        # 检查参数列表
        assert isinstance(trainable_params, list)
        assert len(trainable_params) > 0
        
        # 检查所有参数都需要梯度
        for param in trainable_params:
            assert param.requires_grad

    def test_save_load_checkpoint(self):
        """测试检查点保存和加载"""
        config = SVRasterOneConfig()
        model = SVRasterOne(config)
        
        # 模拟训练状态
        model.training_step = 10
        model.best_loss = 0.1
        
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            checkpoint_path = f.name
        
        try:
            # 保存检查点
            model.save_checkpoint(checkpoint_path)
            
            # 创建新模型
            new_model = SVRasterOne(config)
            
            # 加载检查点
            new_model.load_checkpoint(checkpoint_path)
            
            # 检查状态恢复
            assert new_model.training_step == 10
            assert new_model.best_loss == 0.1
            
        finally:
            Path(checkpoint_path).unlink(missing_ok=True)

    def test_render_sequence(self):
        """测试序列渲染"""
        config = SVRasterOneConfig()
        config.rendering.image_width = 32
        config.rendering.image_height = 32
        
        model = SVRasterOne(config)
        
        # 创建相机序列
        num_frames = 5
        camera_matrices = torch.stack([torch.eye(4) for _ in range(num_frames)])
        camera_matrices[:, 2, 3] = torch.linspace(1.0, 3.0, num_frames)  # 相机移动
        
        intrinsics = torch.eye(3).unsqueeze(0).repeat(num_frames, 1, 1)
        
        # 渲染序列
        result = model.render_sequence(camera_matrices, intrinsics)
        
        # 检查输出
        assert "rgb" in result
        assert "depth" in result
        assert "alpha" in result
        
        # 检查形状
        assert result["rgb"].shape == (num_frames, 32, 32, 3)
        assert result["depth"].shape == (num_frames, 32, 32)
        assert result["alpha"].shape == (num_frames, 32, 32)

    def test_get_memory_usage(self):
        """测试内存使用统计"""
        config = SVRasterOneConfig()
        model = SVRasterOne(config)
        
        memory_usage = model.get_memory_usage()
        
        # 检查内存统计
        assert "total_memory_mb" in memory_usage
        assert "active_memory_mb" in memory_usage
        assert "voxel_memory_mb" in memory_usage
        assert "model_memory_mb" in memory_usage
        
        # 检查数值
        for key, value in memory_usage.items():
            assert isinstance(value, float)
            assert value >= 0

    def test_optimize_memory(self):
        """测试内存优化"""
        config = SVRasterOneConfig()
        config.voxel.max_voxels = 1000
        
        model = SVRasterOne(config)
        
        # 记录初始内存使用
        initial_memory = model.get_memory_usage()
        
        # 执行内存优化
        model.optimize_memory(target_memory_mb=100.0)
        
        # 检查内存使用可能减少
        new_memory = model.get_memory_usage()
        
        # 内存使用应该合理
        assert new_memory["total_memory_mb"] >= 0

    def test_export_import_voxels(self):
        """测试体素导出和导入"""
        config = SVRasterOneConfig()
        model = SVRasterOne(config)
        
        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
            voxel_path = f.name
        
        try:
            # 导出体素
            model.export_voxels(voxel_path)
            
            # 检查文件存在
            assert Path(voxel_path).exists()
            
            # 导入体素
            new_model = SVRasterOne(config)
            new_model.import_voxels(voxel_path)
            
            # 检查体素数据一致性
            original_voxels = model.voxel_grid.get_active_voxels()
            imported_voxels = new_model.voxel_grid.get_active_voxels()
            
            assert original_voxels["positions"].shape == imported_voxels["positions"].shape
            assert original_voxels["densities"].shape == imported_voxels["densities"].shape
            assert original_voxels["colors"].shape == imported_voxels["colors"].shape
            
        finally:
            Path(voxel_path).unlink(missing_ok=True)

    def test_get_model_info(self):
        """测试获取模型信息"""
        config = SVRasterOneConfig()
        model = SVRasterOne(config)
        
        model_info = model.get_model_info()
        
        # 检查模型信息
        assert "model_type" in model_info
        assert "device" in model_info
        assert "voxel_stats" in model_info
        assert "memory_usage" in model_info
        assert "config_summary" in model_info
        
        assert model_info["model_type"] == "SVRasterOne"
        assert model_info["device"] == str(model.device)
        
        # 检查体素统计
        voxel_stats = model_info["voxel_stats"]
        assert "total_voxels" in voxel_stats
        assert "active_voxels" in voxel_stats
        
        # 检查内存使用
        memory_usage = model_info["memory_usage"]
        assert "total_memory_mb" in memory_usage
        assert "active_memory_mb" in memory_usage

    def test_model_gradients(self):
        """测试模型梯度传播"""
        config = SVRasterOneConfig()
        config.rendering.image_width = 16
        config.rendering.image_height = 16
        
        model = SVRasterOne(config)
        
        # 创建需要梯度的输入
        camera_matrix = torch.eye(4, requires_grad=True)
        camera_matrix[2, 3] = 2.0
        
        intrinsics = torch.eye(3, requires_grad=True)
        
        target_data = {
            "rgb": torch.rand(1, 16, 16, 3),
            "depth": torch.rand(1, 16, 16),
        }
        
        # 训练步骤前向传播
        result = model.training_step_forward(camera_matrix, intrinsics, target_data)
        total_loss = result["total_loss"]
        
        # 反向传播
        total_loss.backward()
        
        # 检查梯度
        assert camera_matrix.grad is not None
        assert intrinsics.grad is not None
        
        assert not torch.isnan(camera_matrix.grad).any()
        assert not torch.isnan(intrinsics.grad).any()
        
        # 检查模型参数梯度
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None
                assert not torch.isnan(param.grad).any()

    def test_model_device_consistency(self):
        """测试模型设备一致性"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        # CPU 版本
        config_cpu = SVRasterOneConfig()
        config_cpu.device = "cpu"
        model_cpu = SVRasterOne(config_cpu)
        
        # GPU 版本
        config_gpu = SVRasterOneConfig()
        config_gpu.device = "cuda"
        model_gpu = SVRasterOne(config_gpu)
        
        # 检查设备一致性
        assert model_cpu.device.type == "cpu"
        assert model_gpu.device.type == "cuda"
        
        # 检查组件设备一致性
        assert model_cpu.voxel_grid.device.type == "cpu"
        assert model_gpu.voxel_grid.device.type == "cuda"


def test_model_edge_cases():
    """测试模型边界情况"""
    config = SVRasterOneConfig()
    config.rendering.image_width = 1
    config.rendering.image_height = 1
    config.voxel.max_voxels = 1
    
    model = SVRasterOne(config)
    
    # 测试最小视口
    camera_matrix = torch.eye(4)
    intrinsics = torch.eye(3)
    
    result = model.forward(camera_matrix, intrinsics)
    
    assert result["rgb"].shape == (1, 1, 3)
    assert result["depth"].shape == (1, 1)
    assert result["alpha"].shape == (1, 1)


if __name__ == "__main__":
    pytest.main([__file__]) 