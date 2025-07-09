"""
SVRaster One 核心模型

实现可微分光栅化渲染的完整模型，支持端到端训练。
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
import logging

from .config import SVRasterOneConfig
from .voxels import SparseVoxelGrid
from .renderer import DifferentiableVoxelRasterizer
from .losses import SVRasterOneLoss

logger = logging.getLogger(__name__)


class SVRasterOne(nn.Module):
    """
    SVRaster One 核心模型

    整合稀疏体素网格和可微分光栅化渲染器，
    实现端到端的可微分光栅化渲染。
    """

    def __init__(self, config: SVRasterOneConfig):
        super().__init__()
        self.config = config

        # 移动到指定设备
        self.device = torch.device(config.device)

        # 初始化组件
        self.voxel_grid = SparseVoxelGrid(config).to(self.device)
        self.rasterizer = DifferentiableVoxelRasterizer(config).to(self.device)
        self.loss_fn = SVRasterOneLoss(config)

        # 训练状态
        self.training_step = 0
        self.best_loss = float("inf")

        logger.info(f"Initialized SVRaster One model on device: {self.device}")

    def forward(
        self,
        camera_matrix: torch.Tensor,
        intrinsics: torch.Tensor,
        viewport_size: Optional[Tuple[int, int]] = None,
        mode: str = "training",
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播

        Args:
            camera_matrix: 相机变换矩阵 [4, 4]
            intrinsics: 相机内参矩阵 [3, 3]
            viewport_size: 视口尺寸 (width, height)
            mode: 模式 ("training" 或 "inference")

        Returns:
            渲染结果字典
        """
        # 获取活跃体素数据
        voxel_data = self.voxel_grid.get_active_voxels()

        if len(voxel_data["positions"]) == 0:
            # 没有活跃体素，返回背景
            if viewport_size is None:
                viewport_size = (
                    self.config.rendering.image_width,
                    self.config.rendering.image_height,
                )
            return self.rasterizer._create_background_image(viewport_size, self.device)

        # 渲染
        if mode == "training":
            # 训练模式：使用软光栅化
            self.config.rendering.soft_rasterization = True
            self.config.rendering.use_soft_sorting = True
        else:
            # 推理模式：使用硬光栅化
            self.config.rendering.soft_rasterization = False
            self.config.rendering.use_soft_sorting = False

        rendered_output = self.rasterizer(voxel_data, camera_matrix, intrinsics, viewport_size)

        # 添加体素统计信息
        rendered_output["voxel_stats"] = self.voxel_grid.get_stats()

        return rendered_output

    def compute_loss(
        self,
        rendered_output: Dict[str, torch.Tensor],
        target_data: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        计算损失

        Args:
            rendered_output: 渲染输出
            target_data: 目标数据

        Returns:
            损失字典
        """
        # 获取体素数据用于正则化
        voxel_data = self.voxel_grid.get_active_voxels()

        # 计算损失
        losses = self.loss_fn(rendered_output, target_data, voxel_data)

        return losses

    def training_step_forward(
        self,
        camera_matrix: torch.Tensor,
        intrinsics: torch.Tensor,
        target_data: Dict[str, torch.Tensor],
        viewport_size: Optional[Tuple[int, int]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        训练步骤前向传播

        Args:
            camera_matrix: 相机变换矩阵
            intrinsics: 相机内参矩阵
            target_data: 目标数据
            viewport_size: 视口尺寸

        Returns:
            包含损失的结果字典
        """
        # 前向传播
        rendered_output = self.forward(camera_matrix, intrinsics, viewport_size, mode="training")

        # 计算损失
        losses = self.compute_loss(rendered_output, target_data)

        # 合并结果
        result = rendered_output.copy()
        result.update(losses)

        return result

    def adaptive_optimization(self, gradient_magnitudes: torch.Tensor):
        """
        自适应优化

        基于梯度幅度进行体素细分和剪枝
        """
        # 自适应体素细分
        self.voxel_grid.adaptive_subdivision(gradient_magnitudes)

        # 自适应体素剪枝
        self.voxel_grid.adaptive_pruning()

        # Morton 排序
        self.voxel_grid.sort_by_morton()

    def get_trainable_parameters(self) -> list:
        """获取可训练参数"""
        return list(self.voxel_grid.parameters())

    def save_checkpoint(self, filepath: str):
        """保存检查点"""
        checkpoint = {
            "model_state_dict": self.state_dict(),
            "config": self.config.to_dict(),
            "training_step": self.training_step,
            "best_loss": self.best_loss,
            "voxel_stats": self.voxel_grid.get_stats(),
        }
        torch.save(checkpoint, filepath)
        logger.info(f"Saved checkpoint to {filepath}")

    def load_checkpoint(self, filepath: str):
        """加载检查点"""
        checkpoint = torch.load(filepath, map_location=self.device)

        # 加载模型状态
        self.load_state_dict(checkpoint["model_state_dict"])

        # 加载训练状态
        self.training_step = checkpoint.get("training_step", 0)
        self.best_loss = checkpoint.get("best_loss", float("inf"))

        logger.info(f"Loaded checkpoint from {filepath}")
        logger.info(f"Training step: {self.training_step}, Best loss: {self.best_loss}")

    def render_sequence(
        self,
        camera_matrices: torch.Tensor,
        intrinsics: torch.Tensor,
        viewport_size: Optional[Tuple[int, int]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        渲染序列

        Args:
            camera_matrices: 相机变换矩阵序列 [N, 4, 4]
            intrinsics: 相机内参矩阵 [3, 3]
            viewport_size: 视口尺寸

        Returns:
            渲染序列结果
        """
        num_frames = camera_matrices.shape[0]
        rendered_frames = []

        for i in range(num_frames):
            camera_matrix = camera_matrices[i]
            rendered_output = self.forward(
                camera_matrix, intrinsics, viewport_size, mode="inference"
            )
            rendered_frames.append(rendered_output["rgb"])

        # 堆叠所有帧
        rendered_sequence = torch.stack(rendered_frames, dim=0)

        return {
            "rgb_sequence": rendered_sequence,
            "num_frames": num_frames,
        }

    def get_memory_usage(self) -> Dict[str, float]:
        """获取内存使用情况"""
        voxel_stats = self.voxel_grid.get_stats()

        # 估算内存使用
        total_voxels = voxel_stats["total_voxels"]
        active_voxels = voxel_stats["active_voxels"]

        # 每个体素的参数：位置(3) + 特征(4) + 大小(1) = 8个float32
        bytes_per_voxel = 8 * 4  # 8个float32，每个4字节
        total_memory_mb = total_voxels * bytes_per_voxel / (1024 * 1024)
        active_memory_mb = active_voxels * bytes_per_voxel / (1024 * 1024)

        return {
            "total_memory_mb": total_memory_mb,
            "active_memory_mb": active_memory_mb,
            "memory_efficiency": active_voxels / total_voxels if total_voxels > 0 else 0.0,
        }

    def optimize_memory(self, target_memory_mb: float = 1000.0):
        """
        优化内存使用

        Args:
            target_memory_mb: 目标内存使用量（MB）
        """
        current_memory = self.get_memory_usage()

        if current_memory["total_memory_mb"] > target_memory_mb:
            # 计算需要移除的体素数量
            excess_memory_mb = current_memory["total_memory_mb"] - target_memory_mb
            bytes_per_voxel = 8 * 4
            voxels_to_remove = int(excess_memory_mb * 1024 * 1024 / bytes_per_voxel)

            # 移除低密度体素
            voxel_data = self.voxel_grid.get_active_voxels()
            if len(voxel_data["densities"]) > 0:
                # 按密度排序，移除最低密度的体素
                sorted_indices = torch.argsort(voxel_data["densities"])
                indices_to_remove = sorted_indices[:voxels_to_remove]

                # 标记体素为非活跃
                active_indices = torch.where(self.voxel_grid.active_mask)[0]
                remove_indices = active_indices[indices_to_remove]
                self.voxel_grid.active_mask[remove_indices] = False

                logger.info(f"Removed {len(remove_indices)} low-density voxels to optimize memory")

    def export_voxels(self, filepath: str):
        """导出体素数据"""
        voxel_data = self.voxel_grid.get_active_voxels()

        export_data = {
            "positions": voxel_data["positions"].cpu().numpy(),
            "densities": voxel_data["densities"].cpu().numpy(),
            "colors": voxel_data["colors"].cpu().numpy(),
            "sizes": voxel_data["sizes"].cpu().numpy(),
            "config": self.config.to_dict(),
        }

        if "morton_codes" in voxel_data:
            export_data["morton_codes"] = voxel_data["morton_codes"].cpu().numpy()

        torch.save(export_data, filepath)
        logger.info(f"Exported voxel data to {filepath}")

    def import_voxels(self, filepath: str):
        """导入体素数据"""
        import_data = torch.load(filepath, map_location=self.device)

        # 更新体素数据
        positions = torch.tensor(import_data["positions"], device=self.device, dtype=torch.float32)
        densities = torch.tensor(import_data["densities"], device=self.device, dtype=torch.float32)
        colors = torch.tensor(import_data["colors"], device=self.device, dtype=torch.float32)
        sizes = torch.tensor(import_data["sizes"], device=self.device, dtype=torch.float32)

        # 重建体素特征
        features = torch.cat([densities.unsqueeze(1), colors], dim=1)

        # 更新体素网格
        self.voxel_grid.voxel_coords = nn.Parameter(positions, requires_grad=False)
        self.voxel_grid.voxel_features = nn.Parameter(features, requires_grad=True)
        self.voxel_grid.voxel_sizes = nn.Parameter(sizes, requires_grad=True)

        # 更新活跃掩码
        num_voxels = positions.shape[0]
        self.voxel_grid.active_mask = torch.ones(num_voxels, dtype=torch.bool, device=self.device)

        # 更新 Morton 编码
        if "morton_codes" in import_data:
            morton_codes = torch.tensor(
                import_data["morton_codes"], device=self.device, dtype=torch.int64
            )
            self.voxel_grid.morton_codes = morton_codes

        logger.info(f"Imported {num_voxels} voxels from {filepath}")

    def get_model_info(self) -> Dict[str, any]:
        """获取模型信息"""
        voxel_stats = self.voxel_grid.get_stats()
        memory_usage = self.get_memory_usage()

        return {
            "model_name": "SVRaster One",
            "version": "1.0.0",
            "device": str(self.device),
            "voxel_stats": voxel_stats,
            "memory_usage": memory_usage,
            "config": self.config.to_dict(),
        }
