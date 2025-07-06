#!/usr/bin/env python3
"""
SVRaster 训练演示

这个演示展示如何使用 SVRaster 进行神经辐射场训练。
包含完整的训练流程：数据加载、模型初始化、训练循环、损失计算等。

特点：
- 使用 VolumeRenderer 进行体积渲染训练
- 支持自适应稀疏体素
- 球谐函数视角相关颜色
- 现代 PyTorch 训练循环
- 实时训练监控
"""

from __future__ import annotations

import sys
import torch
import torch.nn.functional as F
import numpy as np
import time
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional
import json
from tqdm import tqdm

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

# SVRaster 导入
from src.nerfs.svraster import (
    SVRasterModel, SVRasterConfig, 
    SVRasterTrainer, SVRasterTrainerConfig,
    SVRasterDataset, SVRasterDatasetConfig,
    VolumeRenderer, SVRasterLoss
)

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SVRasterTrainingDemo:
    """SVRaster 训练演示类"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"使用设备: {self.device}")
        
        # 训练配置
        self.config = self._create_config()
        self.model = None
        self.trainer = None
        self.dataset = None
        
    def _create_config(self) -> SVRasterConfig:
        """创建 SVRaster 配置"""
        config = SVRasterConfig(
            # 场景设置
            image_width=400,
            image_height=300,
            scene_bounds=(-2.0, -2.0, -2.0, 2.0, 2.0, 2.0),
            
            # 体素网格设置
            base_resolution=64,
            max_octree_levels=8,
            
            # 渲染设置
            ray_samples_per_voxel=8,
            depth_peeling_layers=4,
            morton_ordering=True,
            
            # 外观设置
            sh_degree=2,
            color_activation="sigmoid",
            density_activation="exp",
            
            # 训练设置
            background_color=(0.0, 0.0, 0.0),
            near_plane=0.1,
            far_plane=10.0,
        )
        
        logger.info("SVRaster 配置创建完成")
        logger.info(f"  - 分辨率: {config.image_width}x{config.image_height}")
        logger.info(f"  - 基础网格: {config.base_resolution}^3")
        logger.info(f"  - 球谐阶数: {config.sh_degree}")
        
        return config
        
    def _create_synthetic_dataset(self) -> SVRasterDataset:
        """创建合成训练数据集"""
        logger.info("创建合成数据集...")
        
        # 数据集配置
        dataset_config = SVRasterDatasetConfig(
            data_dir="demo_data",
            image_width=self.config.image_width,
            image_height=self.config.image_height,
            train_split=0.8,
            val_split=0.2,
            test_split=0.0,
        )
        
        # 生成合成场景数据
        self._generate_synthetic_scene_data(dataset_config)
        
        # 创建数据集
        dataset = SVRasterDataset(dataset_config)
        
        logger.info(f"数据集创建完成，训练样本: {len(dataset)}")
        return dataset
        
    def _generate_synthetic_scene_data(self, dataset_config: SVRasterDatasetConfig):
        """生成合成场景数据"""
        import os
        import imageio
        
        # 创建数据目录
        data_dir = Path(dataset_config.data_dir)
        data_dir.mkdir(exist_ok=True)
        
        images_dir = data_dir / "images"
        images_dir.mkdir(exist_ok=True)
        
        poses_dir = data_dir / "poses"
        poses_dir.mkdir(exist_ok=True)
        
        # 生成相机位置
        n_views = 60  # 固定使用60个视角进行演示
        
        # 球面相机分布
        phi = np.random.uniform(0, 2 * np.pi, n_views)
        theta = np.random.uniform(np.pi/6, np.pi/3, n_views)
        radius = 3.0
        
        camera_positions = np.stack([
            radius * np.sin(theta) * np.cos(phi),
            radius * np.sin(theta) * np.sin(phi),
            radius * np.cos(theta)
        ], axis=1)
        
        # 生成图像和相机参数
        transforms = {
            "camera_angle_x": 0.8,
            "frames": []
        }
        
        for i in range(n_views):
            # 相机朝向场景中心
            camera_pos = camera_positions[i]
            target = np.array([0.0, 0.0, 0.0])
            up = np.array([0.0, 0.0, 1.0])
            
            # 计算变换矩阵
            forward = target - camera_pos
            forward = forward / np.linalg.norm(forward)
            right = np.cross(forward, up)
            right = right / np.linalg.norm(right)
            up = np.cross(right, forward)
            
            transform_matrix = np.eye(4)
            transform_matrix[:3, 0] = right
            transform_matrix[:3, 1] = up
            transform_matrix[:3, 2] = -forward
            transform_matrix[:3, 3] = camera_pos
            
            # 生成简单的合成图像（彩色球体）
            image = self._generate_synthetic_image(
                camera_pos, forward, 
                dataset_config.image_width, 
                dataset_config.image_height
            )
            
            # 保存图像
            image_path = images_dir / f"image_{i:03d}.png"
            imageio.imwrite(image_path, (image * 255).astype(np.uint8))
            
            # 保存pose矩阵
            pose_path = poses_dir / f"pose_{i:03d}.txt"
            np.savetxt(pose_path, transform_matrix)
            
            # 添加到变换数据（用于兼容性）
            transforms["frames"].append({
                "file_path": f"images/image_{i:03d}.png",
                "transform_matrix": transform_matrix.tolist()
            })
        
        # 保存变换数据
        with open(data_dir / "transforms.json", "w") as f:
            json.dump(transforms, f, indent=2)
            
        logger.info(f"生成了 {n_views} 个合成视图")
        
    def _generate_synthetic_image(
        self, camera_pos: np.ndarray, forward: np.ndarray, 
        width: int, height: int
    ) -> np.ndarray:
        """生成简单的合成图像"""
        # 创建光线
        i, j = np.meshgrid(
            np.linspace(-1, 1, width),
            np.linspace(-1, 1, height),
            indexing='xy'
        )
        
        # 计算右和上向量
        up = np.array([0.0, 0.0, 1.0])
        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)
        up = np.cross(right, forward)
        
        # 光线方向
        dirs = forward[None, None, :] + i[:, :, None] * right[None, None, :] + j[:, :, None] * up[None, None, :]
        dirs = dirs / np.linalg.norm(dirs, axis=2, keepdims=True)
        
        # 简单的球体渲染
        sphere_center = np.array([0.0, 0.0, 0.0])
        sphere_radius = 1.0
        
        # 计算与球体的交点
        oc = camera_pos - sphere_center
        a = np.sum(dirs * dirs, axis=2)
        b = 2.0 * np.sum(oc * dirs, axis=2)
        c = np.sum(oc * oc) - sphere_radius * sphere_radius
        
        discriminant = b * b - 4 * a * c
        
        # 计算颜色
        image = np.zeros((height, width, 3))
        
        # 球体内部
        hit_mask = discriminant >= 0
        if np.any(hit_mask):
            t = (-b - np.sqrt(discriminant)) / (2 * a)
            hit_points = camera_pos + t[:, :, None] * dirs
            
            # 根据位置计算颜色
            colors = (hit_points + sphere_radius) / (2 * sphere_radius)
            colors = np.clip(colors, 0, 1)
            
            image[hit_mask] = colors[hit_mask]
        
        return image
        
    def setup_training(self):
        """设置训练组件"""
        logger.info("设置训练组件...")
        
        # 创建数据集
        self.dataset = self._create_synthetic_dataset()
        
        # 创建模型
        self.model = SVRasterModel(self.config).to(self.device)
        logger.info(f"模型参数数量: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # 创建训练器配置
        trainer_config = SVRasterTrainerConfig(
            learning_rate=1e-3,
            batch_size=1,  # SVRaster训练一般使用batch_size=1
            num_epochs=50,  # 演示用较少的epochs
            log_every=10,
            save_every=200,
            validate_every=100,
            checkpoint_dir="demos/checkpoints/svraster_training",
            use_amp=True,
            grad_clip_norm=1.0,
        )
        
        # 创建体积渲染器
        volume_renderer = VolumeRenderer(self.config)
        
        # 创建训练器
        self.trainer = SVRasterTrainer(
            model=self.model,
            volume_renderer=volume_renderer,
            config=trainer_config,
            train_dataset=self.dataset,
            val_dataset=self.dataset  # 演示中使用相同数据集
        )
        
        logger.info("训练组件设置完成")
        
    def run_training_epoch(self, epoch: int) -> Dict[str, float]:
        """运行一个训练 epoch"""
        self.model.train()
        
        epoch_losses = {
            'total_loss': 0.0,
            'rgb_loss': 0.0,
            'depth_loss': 0.0,
            'regularization_loss': 0.0
        }
        
        num_batches = len(self.dataset) // self.trainer.config.batch_size
        
        with tqdm(range(num_batches), desc=f"Epoch {epoch}") as pbar:
            for batch_idx in pbar:
                # 获取批次数据
                batch_data = self._get_training_batch(batch_idx)
                
                # 训练步骤
                losses = self.trainer.train_step(batch_data)
                
                # 累积损失
                for key in epoch_losses:
                    if key in losses:
                        epoch_losses[key] += losses[key]
                
                # 更新进度条
                current_loss = losses.get('total_loss', 0.0)
                pbar.set_postfix({'loss': f'{current_loss:.6f}'})
        
        # 平均损失
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
            
        return epoch_losses
        
    def _get_training_batch(self, batch_idx: int) -> Dict[str, torch.Tensor]:
        """获取训练批次数据"""
        batch_size = self.trainer.config.batch_size
        
        # 随机采样光线
        H, W = self.config.image_height, self.config.image_width
        
        # 生成随机像素坐标
        pixels_y = torch.randint(0, H, (batch_size,), device=self.device)
        pixels_x = torch.randint(0, W, (batch_size,), device=self.device)
        
        # 生成光线（简化版本）
        camera_pos = torch.tensor([0.0, 0.0, 3.0], device=self.device)
        
        # 归一化像素坐标到 [-1, 1]
        x_norm = (pixels_x.float() / W - 0.5) * 2
        y_norm = (pixels_y.float() / H - 0.5) * 2
        
        # 光线方向
        ray_dirs = torch.stack([
            x_norm * 0.5,
            y_norm * 0.5,
            -torch.ones_like(x_norm)
        ], dim=1)
        ray_dirs = F.normalize(ray_dirs, dim=1)
        
        # 光线起点
        ray_origins = camera_pos.unsqueeze(0).expand(batch_size, -1)
        
        # 目标颜色（简化：基于光线方向的颜色）
        target_colors = (ray_dirs + 1) / 2  # 归一化到 [0, 1]
        
        return {
            'ray_origins': ray_origins,
            'ray_directions': ray_dirs,
            'target_rgb': target_colors,
            'pixels': torch.stack([pixels_x, pixels_y], dim=1)
        }
        
    def validate(self, epoch: int) -> Dict[str, float]:
        """验证模型"""
        self.model.eval()
        
        val_losses = {
            'val_rgb_loss': 0.0,
            'val_psnr': 0.0
        }
        
        num_val_batches = 10
        
        with torch.no_grad():
            for _ in range(num_val_batches):
                batch_data = self._get_training_batch(0)  # 简化：使用相同的批次生成
                
                # 渲染
                outputs = self.model(
                    batch_data['ray_origins'],
                    batch_data['ray_directions'],
                    mode="training"  # 使用体积渲染
                )
                
                # 计算损失
                rgb_loss = F.mse_loss(outputs['rgb'], batch_data['target_rgb'])
                val_losses['val_rgb_loss'] += rgb_loss.item()
                
                # 计算 PSNR
                mse = rgb_loss.item()
                psnr = -10 * np.log10(mse + 1e-8)
                val_losses['val_psnr'] += psnr
        
        # 平均验证指标
        for key in val_losses:
            val_losses[key] /= num_val_batches
            
        return val_losses
        
    def run_full_training(self):
        """运行完整训练"""
        logger.info("开始 SVRaster 训练...")
        
        best_psnr = 0.0
        training_history = []
        
        for epoch in range(self.trainer.config.num_epochs):
            start_time = time.time()
            
            # 训练
            train_losses = self.run_training_epoch(epoch)
            
            # 验证
            val_metrics = {}
            if epoch % self.trainer.config.validate_every == 0:
                val_metrics = self.validate(epoch)
                
                # 检查是否是最佳模型
                current_psnr = val_metrics.get('val_psnr', 0.0)
                if current_psnr > best_psnr:
                    best_psnr = current_psnr
                    self._save_checkpoint(epoch, "best_model.pth")
            
            epoch_time = time.time() - start_time
            
            # 记录训练历史
            epoch_info = {
                'epoch': epoch,
                'epoch_time': epoch_time,
                **train_losses,
                **val_metrics
            }
            training_history.append(epoch_info)
            
            # 打印训练信息
            logger.info(
                f"Epoch {epoch:3d} | "
                f"Loss: {train_losses['total_loss']:.6f} | "
                f"RGB: {train_losses['rgb_loss']:.6f} | "
                f"Time: {epoch_time:.1f}s"
            )
            
            if val_metrics:
                logger.info(
                    f"         | "
                    f"Val RGB: {val_metrics['val_rgb_loss']:.6f} | "
                    f"PSNR: {val_metrics['val_psnr']:.2f}dB"
                )
            
            # 保存检查点
            if epoch % self.trainer.config.save_every == 0:
                self._save_checkpoint(epoch, f"checkpoint_epoch_{epoch}.pth")
        
        logger.info(f"训练完成！最佳 PSNR: {best_psnr:.2f}dB")
        
        # 保存训练历史
        self._save_training_history(training_history)
        
    def _save_checkpoint(self, epoch: int, filename: str):
        """保存检查点"""
        checkpoint_dir = Path(self.trainer.config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
        }
        
        torch.save(checkpoint, checkpoint_dir / filename)
        logger.info(f"保存检查点: {filename}")
        
    def _save_training_history(self, history: list):
        """保存训练历史"""
        checkpoint_dir = Path(self.trainer.config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        with open(checkpoint_dir / "training_history.json", "w") as f:
            json.dump(history, f, indent=2)
        
        logger.info("训练历史已保存")


def main():
    """主函数"""
    print("=" * 70)
    print("SVRaster 训练演示")
    print("=" * 70)
    
    try:
        # 创建训练演示
        demo = SVRasterTrainingDemo()
        
        # 设置训练
        demo.setup_training()
        
        # 运行训练
        demo.run_full_training()
        
        print("\n🎉 SVRaster 训练演示完成！")
        print("\n训练特点:")
        print("✅ 使用 VolumeRenderer 进行体积渲染训练")
        print("✅ 自适应稀疏体素表示")
        print("✅ 球谐函数视角相关颜色")
        print("✅ 现代 PyTorch 训练循环")
        print("✅ 实时损失监控和验证")
        
    except KeyboardInterrupt:
        print("\n⏹️  训练被用户中断")
    except Exception as e:
        print(f"\n❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
