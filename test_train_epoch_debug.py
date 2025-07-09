#!/usr/bin/env python3
"""
测试 train_epoch 函数是否能正常运行到第296行
"""

import torch
import torch.utils.data as data
import numpy as np
import logging

# 设置日志级别
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 添加 src 到路径
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

try:
    import nerfs.svraster as svraster

    SVRASTER_AVAILABLE = True
except ImportError as e:
    SVRASTER_AVAILABLE = False
    logger.error(f"SVRaster not available: {e}")
    sys.exit(1)


class DummyDataset(data.Dataset):
    """简单的测试数据集"""

    def __init__(self, num_samples=10):
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 创建简单的测试数据
        batch_size = 4
        H, W = 32, 32

        rays_o = torch.randn(batch_size, H, W, 3)
        rays_d = torch.randn(batch_size, H, W, 3)
        colors = torch.rand(batch_size, H, W, 3)

        return {
            "rays_o": rays_o,
            "rays_d": rays_d,
            "colors": colors,
        }


def test_train_epoch():
    """测试 train_epoch 函数"""
    logger.info("开始测试 train_epoch 函数")

    try:
        # 1. 创建模型和配置
        logger.info("创建模型和配置...")
        config = svraster.SVRasterConfig(
            max_octree_levels=2,
            base_resolution=8,
            sh_degree=1,
        )
        model = svraster.SVRasterModel(config)

        # 2. 创建体积渲染器
        logger.info("创建体积渲染器...")
        volume_renderer = svraster.VolumeRenderer(config)

        # 3. 创建训练器配置
        trainer_config = svraster.SVRasterTrainerConfig(
            num_epochs=1,
            batch_size=1,
            learning_rate=1e-3,
        )

        # 4. 创建数据集和数据加载器
        logger.info("创建数据集...")
        dataset = DummyDataset(num_samples=3)
        dataloader = data.DataLoader(dataset, batch_size=1, shuffle=False)

        # 5. 创建训练器
        logger.info("创建训练器...")
        trainer = svraster.SVRasterTrainer(
            model=model,
            volume_renderer=volume_renderer,
            config=trainer_config,
            train_dataset=dataset,
        )

        # 6. 测试 train_epoch
        logger.info("开始测试 train_epoch...")
        result = trainer.train_epoch(dataloader)

        logger.info(f"train_epoch 成功完成！返回结果: {result}")
        return True

    except Exception as e:
        logger.error(f"测试失败: {str(e)}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_train_epoch()
    if success:
        print("✅ train_epoch 测试成功！")
    else:
        print("❌ train_epoch 测试失败！")
