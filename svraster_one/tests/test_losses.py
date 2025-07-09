"""
SVRaster One 损失函数测试

测试各种损失函数的计算和梯度传播。
"""

import pytest
import torch
import torch.nn.functional as F
import numpy as np

from ..config import SVRasterOneConfig
from ..losses import SVRasterOneLoss


class TestSVRasterOneLoss:
    """测试 SVRaster One 损失函数"""

    def test_loss_init(self):
        """测试损失函数初始化"""
        config = SVRasterOneConfig()
        
        loss_fn = SVRasterOneLoss(config)
        
        # 检查权重
        assert loss_fn.rgb_loss_weight == config.training.rgb_loss_weight
        assert loss_fn.depth_loss_weight == config.training.depth_loss_weight
        assert loss_fn.density_reg_weight == config.training.density_reg_weight
        assert loss_fn.sparsity_weight == config.training.sparsity_weight

    def test_rgb_loss(self):
        """测试 RGB 损失计算"""
        config = SVRasterOneConfig()
        loss_fn = SVRasterOneLoss(config)
        
        # 创建测试数据
        batch_size = 10
        height, width = 64, 64
        
        rendered_rgb = torch.rand(batch_size, height, width, 3)
        target_rgb = torch.rand(batch_size, height, width, 3)
        
        # 计算 RGB 损失
        rgb_loss = loss_fn._compute_rgb_loss(rendered_rgb, target_rgb)
        
        # 检查损失值
        assert isinstance(rgb_loss, torch.Tensor)
        assert rgb_loss.shape == ()
        assert rgb_loss.item() >= 0
        assert not torch.isnan(rgb_loss)
        assert not torch.isinf(rgb_loss)
        
        # 相同输入应该产生零损失
        zero_loss = loss_fn._compute_rgb_loss(rendered_rgb, rendered_rgb)
        assert zero_loss.item() < 1e-6

    def test_depth_loss(self):
        """测试深度损失计算"""
        config = SVRasterOneConfig()
        loss_fn = SVRasterOneLoss(config)
        
        # 创建测试数据
        batch_size = 10
        height, width = 64, 64
        
        rendered_depth = torch.rand(batch_size, height, width)
        target_depth = torch.rand(batch_size, height, width)
        
        # 计算深度损失
        depth_loss = loss_fn._compute_depth_loss(rendered_depth, target_depth)
        
        # 检查损失值
        assert isinstance(depth_loss, torch.Tensor)
        assert depth_loss.shape == ()
        assert depth_loss.item() >= 0
        assert not torch.isnan(depth_loss)
        assert not torch.isinf(depth_loss)
        
        # 相同输入应该产生零损失
        zero_loss = loss_fn._compute_depth_loss(rendered_depth, rendered_depth)
        assert zero_loss.item() < 1e-6

    def test_density_regularization(self):
        """测试密度正则化"""
        config = SVRasterOneConfig()
        loss_fn = SVRasterOneLoss(config)
        
        # 创建体素数据
        voxel_data = {
            "densities": torch.rand(100),
            "positions": torch.randn(100, 3),
            "sizes": torch.ones(100) * 0.1,
            "colors": torch.rand(100, 3),
        }
        
        # 计算密度正则化
        density_reg = loss_fn._compute_density_regularization(voxel_data)
        
        # 检查损失值
        assert isinstance(density_reg, torch.Tensor)
        assert density_reg.shape == ()
        assert density_reg.item() >= 0
        assert not torch.isnan(density_reg)
        assert not torch.isinf(density_reg)

    def test_sparsity_loss(self):
        """测试稀疏性损失"""
        config = SVRasterOneConfig()
        loss_fn = SVRasterOneLoss(config)
        
        # 创建体素数据
        voxel_data = {
            "densities": torch.rand(100),
            "positions": torch.randn(100, 3),
            "sizes": torch.ones(100) * 0.1,
            "colors": torch.rand(100, 3),
        }
        
        # 计算稀疏性损失
        sparsity_loss = loss_fn._compute_sparsity_loss(voxel_data)
        
        # 检查损失值
        assert isinstance(sparsity_loss, torch.Tensor)
        assert sparsity_loss.shape == ()
        assert sparsity_loss.item() >= 0
        assert not torch.isnan(sparsity_loss)
        assert not torch.isinf(sparsity_loss)

    def test_total_loss(self):
        """测试总损失计算"""
        config = SVRasterOneConfig()
        loss_fn = SVRasterOneLoss(config)
        
        # 创建渲染输出
        batch_size = 5
        height, width = 32, 32
        
        rendered_output = {
            "rgb": torch.rand(batch_size, height, width, 3),
            "depth": torch.rand(batch_size, height, width),
            "alpha": torch.rand(batch_size, height, width),
        }
        
        # 创建目标数据
        target_data = {
            "rgb": torch.rand(batch_size, height, width, 3),
            "depth": torch.rand(batch_size, height, width),
        }
        
        # 创建体素数据
        voxel_data = {
            "densities": torch.rand(50),
            "positions": torch.randn(50, 3),
            "sizes": torch.ones(50) * 0.1,
            "colors": torch.rand(50, 3),
        }
        
        # 计算总损失
        losses = loss_fn(rendered_output, target_data, voxel_data)
        
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
        
        # 检查总损失是各分量的加权和
        expected_total = (
            config.training.rgb_loss_weight * losses["rgb_loss"] +
            config.training.depth_loss_weight * losses["depth_loss"] +
            config.training.density_reg_weight * losses["density_reg"] +
            config.training.sparsity_weight * losses["sparsity_loss"]
        )
        
        assert torch.allclose(losses["total_loss"], expected_total, atol=1e-6)

    def test_loss_gradients(self):
        """测试损失函数的梯度传播"""
        config = SVRasterOneConfig()
        loss_fn = SVRasterOneLoss(config)
        
        # 创建需要梯度的数据
        batch_size = 3
        height, width = 16, 16
        
        rendered_output = {
            "rgb": torch.rand(batch_size, height, width, 3, requires_grad=True),
            "depth": torch.rand(batch_size, height, width, requires_grad=True),
            "alpha": torch.rand(batch_size, height, width),
        }
        
        target_data = {
            "rgb": torch.rand(batch_size, height, width, 3),
            "depth": torch.rand(batch_size, height, width),
        }
        
        voxel_data = {
            "densities": torch.rand(20, requires_grad=True),
            "positions": torch.randn(20, 3),
            "sizes": torch.ones(20) * 0.1,
            "colors": torch.rand(20, 3, requires_grad=True),
        }
        
        # 计算损失
        losses = loss_fn(rendered_output, target_data, voxel_data)
        total_loss = losses["total_loss"]
        
        # 反向传播
        total_loss.backward()
        
        # 检查梯度
        assert rendered_output["rgb"].grad is not None
        assert rendered_output["depth"].grad is not None
        assert voxel_data["densities"].grad is not None
        assert voxel_data["colors"].grad is not None
        
        assert not torch.isnan(rendered_output["rgb"].grad).any()
        assert not torch.isnan(rendered_output["depth"].grad).any()
        assert not torch.isnan(voxel_data["densities"].grad).any()
        assert not torch.isnan(voxel_data["colors"].grad).any()

    def test_loss_weights(self):
        """测试损失权重的影响"""
        config = SVRasterOneConfig()
        
        # 测试不同权重
        test_cases = [
            {"rgb_loss_weight": 1.0, "depth_loss_weight": 0.0, "density_reg_weight": 0.0, "sparsity_weight": 0.0},
            {"rgb_loss_weight": 0.0, "depth_loss_weight": 1.0, "density_reg_weight": 0.0, "sparsity_weight": 0.0},
            {"rgb_loss_weight": 0.0, "depth_loss_weight": 0.0, "density_reg_weight": 1.0, "sparsity_weight": 0.0},
            {"rgb_loss_weight": 0.0, "depth_loss_weight": 0.0, "density_reg_weight": 0.0, "sparsity_weight": 1.0},
        ]
        
        for weights in test_cases:
            config.training.rgb_loss_weight = weights["rgb_loss_weight"]
            config.training.depth_loss_weight = weights["depth_loss_weight"]
            config.training.density_reg_weight = weights["density_reg_weight"]
            config.training.sparsity_weight = weights["sparsity_weight"]
            
            loss_fn = SVRasterOneLoss(config)
            
            # 创建测试数据
            batch_size = 2
            height, width = 8, 8
            
            rendered_output = {
                "rgb": torch.rand(batch_size, height, width, 3),
                "depth": torch.rand(batch_size, height, width),
                "alpha": torch.rand(batch_size, height, width),
            }
            
            target_data = {
                "rgb": torch.rand(batch_size, height, width, 3),
                "depth": torch.rand(batch_size, height, width),
            }
            
            voxel_data = {
                "densities": torch.rand(10),
                "positions": torch.randn(10, 3),
                "sizes": torch.ones(10) * 0.1,
                "colors": torch.rand(10, 3),
            }
            
            # 计算损失
            losses = loss_fn(rendered_output, target_data, voxel_data)
            
            # 检查总损失等于对应分量的加权和
            expected_total = (
                weights["rgb_loss_weight"] * losses["rgb_loss"] +
                weights["depth_loss_weight"] * losses["depth_loss"] +
                weights["density_reg_weight"] * losses["density_reg"] +
                weights["sparsity_weight"] * losses["sparsity_loss"]
            )
            
            assert torch.allclose(losses["total_loss"], expected_total, atol=1e-6)

    def test_loss_edge_cases(self):
        """测试损失函数的边界情况"""
        config = SVRasterOneConfig()
        loss_fn = SVRasterOneLoss(config)
        
        # 测试空数据
        batch_size = 0
        height, width = 0, 0
        
        rendered_output = {
            "rgb": torch.empty(batch_size, height, width, 3),
            "depth": torch.empty(batch_size, height, width),
            "alpha": torch.empty(batch_size, height, width),
        }
        
        target_data = {
            "rgb": torch.empty(batch_size, height, width, 3),
            "depth": torch.empty(batch_size, height, width),
        }
        
        voxel_data = {
            "densities": torch.empty(0),
            "positions": torch.empty(0, 3),
            "sizes": torch.empty(0),
            "colors": torch.empty(0, 3),
        }
        
        # 应该不抛出异常
        losses = loss_fn(rendered_output, target_data, voxel_data)
        
        assert "total_loss" in losses
        assert isinstance(losses["total_loss"], torch.Tensor)
        assert losses["total_loss"].shape == ()

    def test_loss_numerical_stability(self):
        """测试损失函数的数值稳定性"""
        config = SVRasterOneConfig()
        loss_fn = SVRasterOneLoss(config)
        
        # 测试极值
        batch_size = 2
        height, width = 4, 4
        
        # 极小值
        rendered_output = {
            "rgb": torch.full((batch_size, height, width, 3), 1e-10),
            "depth": torch.full((batch_size, height, width), 1e-10),
            "alpha": torch.full((batch_size, height, width), 1e-10),
        }
        
        target_data = {
            "rgb": torch.full((batch_size, height, width, 3), 1e-10),
            "depth": torch.full((batch_size, height, width), 1e-10),
        }
        
        voxel_data = {
            "densities": torch.full((10,), 1e-10),
            "positions": torch.randn(10, 3),
            "sizes": torch.ones(10) * 1e-10,
            "colors": torch.full((10, 3), 1e-10),
        }
        
        losses = loss_fn(rendered_output, target_data, voxel_data)
        
        # 检查没有 NaN 或 Inf
        for loss_name, loss_value in losses.items():
            assert not torch.isnan(loss_value)
            assert not torch.isinf(loss_value)
        
        # 极大值
        rendered_output = {
            "rgb": torch.full((batch_size, height, width, 3), 1e10),
            "depth": torch.full((batch_size, height, width), 1e10),
            "alpha": torch.full((batch_size, height, width), 1e10),
        }
        
        target_data = {
            "rgb": torch.full((batch_size, height, width, 3), 1e10),
            "depth": torch.full((batch_size, height, width), 1e10),
        }
        
        voxel_data = {
            "densities": torch.full((10,), 1e10),
            "positions": torch.randn(10, 3),
            "sizes": torch.ones(10) * 1e10,
            "colors": torch.full((10, 3), 1e10),
        }
        
        losses = loss_fn(rendered_output, target_data, voxel_data)
        
        # 检查没有 NaN 或 Inf
        for loss_name, loss_value in losses.items():
            assert not torch.isnan(loss_value)
            assert not torch.isinf(loss_value)


if __name__ == "__main__":
    pytest.main([__file__]) 