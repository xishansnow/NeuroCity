from __future__ import annotations

"""
Mega-NeRF Utilities

This module contains utility functions and classes for Mega-NeRF implementation.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Union, Any
from pathlib import Path
import logging
import json
import os

logger = logging.getLogger(__name__)

# 尝试导入可选依赖
try:
    import cv2

    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    logger.warning("cv2 not available for image processing")

try:
    import imageio

    IMAGEIO_AVAILABLE = True
except ImportError:
    IMAGEIO_AVAILABLE = False
    logger.warning("imageio not available for image/video saving")


# ============================================================================
# 相机工具函数
# ============================================================================


def generate_rays(
    camera_pose: torch.Tensor,
    intrinsics: torch.Tensor,
    width: int,
    height: int,
    device: str = "cuda",
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    生成相机光线

    Args:
        camera_pose: [4, 4] 相机姿态矩阵
        intrinsics: [3, 3] 相机内参矩阵
        width: 图像宽度
        height: 图像高度
        device: 设备

    Returns:
        rays_o: [N, 3] 光线原点
        rays_d: [N, 3] 光线方向
    """
    # 创建像素坐标网格
    i, j = torch.meshgrid(
        torch.arange(height, device=device),
        torch.arange(width, device=device),
        indexing="ij",
    )

    # 转换为相机坐标
    directions = torch.stack(
        [
            (j - intrinsics[0, 2]) / intrinsics[0, 0],
            (i - intrinsics[1, 2]) / intrinsics[1, 1],
            torch.ones_like(i),
        ],
        dim=-1,
    )

    # 转换到世界坐标
    directions = directions @ camera_pose[:3, :3].T
    origins = camera_pose[:3, -1].expand_as(directions)

    # 扁平化
    rays_o = origins.reshape(-1, 3)
    rays_d = F.normalize(directions.reshape(-1, 3), dim=-1)

    return rays_o, rays_d


def create_spiral_path(
    center: torch.Tensor,
    radius: float,
    num_frames: int,
    height_offset: float = 0.0,
    device: str = "cuda",
) -> torch.Tensor:
    """
    创建螺旋相机路径

    Args:
        center: 螺旋中心点
        radius: 螺旋半径
        num_frames: 帧数
        height_offset: 高度偏移
        device: 设备

    Returns:
        trajectory: [num_frames, 4, 4] 相机轨迹
    """
    t = torch.linspace(0, 2 * np.pi, num_frames, device=device)

    # 螺旋参数
    x = center[0] + radius * torch.cos(t)
    y = center[1] + radius * torch.sin(t)
    z = center[2] + height_offset

    # 创建相机姿态
    poses = []
    for i in range(num_frames):
        # 简单的相机姿态，始终看向中心
        pos = torch.tensor([x[i], y[i], z], device=device)
        forward = F.normalize(center - pos, dim=0)
        right = F.normalize(torch.cross(forward, torch.tensor([0, 0, 1], device=device)), dim=0)
        up = F.normalize(torch.cross(right, forward), dim=0)

        pose = torch.eye(4, device=device)
        pose[:3, 0] = right
        pose[:3, 1] = up
        pose[:3, 2] = forward
        pose[:3, 3] = pos

        poses.append(pose)

    return torch.stack(poses)


def interpolate_poses(
    pose1: torch.Tensor, pose2: torch.Tensor, num_interpolations: int
) -> torch.Tensor:
    """
    在两个相机姿态之间插值

    Args:
        pose1: [4, 4] 第一个相机姿态
        pose2: [4, 4] 第二个相机姿态
        num_interpolations: 插值数量

    Returns:
        interpolated_poses: [num_interpolations, 4, 4] 插值后的姿态
    """
    # 提取位置和旋转
    pos1 = pose1[:3, 3]
    pos2 = pose2[:3, 3]
    rot1 = pose1[:3, :3]
    rot2 = pose2[:3, :3]

    # 位置线性插值
    t = torch.linspace(0, 1, num_interpolations, device=pose1.device)
    positions = pos1.unsqueeze(0) + t.unsqueeze(1) * (pos2 - pos1).unsqueeze(0)

    # 旋转球面线性插值 (SLERP)
    rotations = []
    for i in range(num_interpolations):
        alpha = t[i]
        # 简单的线性插值，实际应用中可能需要更复杂的四元数插值
        rot_interp = rot1 + alpha * (rot2 - rot1)
        # 重新正交化
        u, _, v = torch.svd(rot_interp)
        rot_interp = u @ v.T
        rotations.append(rot_interp)

    rotations = torch.stack(rotations)

    # 构建完整的变换矩阵
    poses = torch.eye(4, device=pose1.device).unsqueeze(0).expand(num_interpolations, -1, -1)
    poses[:, :3, :3] = rotations
    poses[:, :3, 3] = positions

    return poses


def generate_random_poses(
    center: torch.Tensor, radius: float, num_poses: int, device: str = "cuda"
) -> torch.Tensor:
    """
    生成随机相机姿态

    Args:
        center: 中心点
        radius: 半径
        num_poses: 姿态数量
        device: 设备

    Returns:
        poses: [num_poses, 4, 4] 随机相机姿态
    """
    poses = []

    for _ in range(num_poses):
        # 随机球面坐标
        theta = torch.rand(1, device=device) * 2 * np.pi
        phi = torch.acos(2 * torch.rand(1, device=device) - 1)

        # 转换为笛卡尔坐标
        x = center[0] + radius * torch.sin(phi) * torch.cos(theta)
        y = center[1] + radius * torch.sin(phi) * torch.sin(theta)
        z = center[2] + radius * torch.cos(phi)

        pos = torch.tensor([x, y, z], device=device)

        # 朝向中心
        forward = F.normalize(center - pos, dim=0)
        right = F.normalize(torch.cross(forward, torch.tensor([0, 0, 1], device=device)), dim=0)
        up = F.normalize(torch.cross(right, forward), dim=0)

        pose = torch.eye(4, device=device)
        pose[:3, 0] = right
        pose[:3, 1] = up
        pose[:3, 2] = forward
        pose[:3, 3] = pos

        poses.append(pose)

    return torch.stack(poses)


# ============================================================================
# 渲染工具函数
# ============================================================================


def save_image(image: np.ndarray, path: str, normalize: bool = True) -> None:
    """
    保存图像

    Args:
        image: 图像数组
        path: 保存路径
        normalize: 是否归一化到 [0, 255]
    """
    if not IMAGEIO_AVAILABLE:
        logger.warning("imageio not available, skipping image save")
        return

    if normalize:
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)

    imageio.imwrite(path, image)


def save_video(images: list[np.ndarray], path: str, fps: int = 30) -> None:
    """
    保存视频

    Args:
        images: 图像列表
        path: 保存路径
        fps: 帧率
    """
    if not IMAGEIO_AVAILABLE:
        logger.warning("imageio not available, skipping video save")
        return

    # 确保图像格式正确
    processed_images = []
    for img in images:
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)
        processed_images.append(img)

    imageio.mimsave(path, processed_images, fps=fps)


def create_depth_visualization(depth: np.ndarray, colormap: str = "plasma") -> np.ndarray:
    """
    创建深度可视化

    Args:
        depth: 深度图
        colormap: 颜色映射

    Returns:
        colored_depth: 彩色深度图
    """
    if not CV2_AVAILABLE:
        logger.warning("cv2 not available, returning grayscale depth")
        return depth

    # 归一化深度
    depth_normalized = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
    depth_uint8 = (depth_normalized * 255).astype(np.uint8)

    # 应用颜色映射
    if colormap == "plasma":
        colored_depth = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_PLASMA)
    elif colormap == "viridis":
        colored_depth = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_VIRIDIS)
    elif colormap == "magma":
        colored_depth = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_MAGMA)
    else:
        colored_depth = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_PLASMA)

    return colored_depth


def compute_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    计算峰值信噪比 (PSNR)

    Args:
        pred: 预测图像
        target: 目标图像

    Returns:
        psnr: PSNR 值
    """
    mse = torch.mean((pred - target) ** 2)
    if mse == 0:
        return float("inf")
    return -10 * torch.log10(mse).item()


def compute_ssim(pred: torch.Tensor, target: torch.Tensor, window_size: int = 11) -> float:
    """
    计算结构相似性指数 (SSIM)

    Args:
        pred: 预测图像
        target: 目标图像
        window_size: 窗口大小

    Returns:
        ssim: SSIM 值
    """
    # 简化的 SSIM 实现
    # 实际应用中可能需要更复杂的实现
    mu1 = torch.mean(pred)
    mu2 = torch.mean(target)

    sigma1 = torch.var(pred)
    sigma2 = torch.var(target)
    sigma12 = torch.mean((pred - mu1) * (target - mu2))

    c1 = 0.01**2
    c2 = 0.03**2

    ssim = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1**2 + mu2**2 + c1) * (sigma1 + sigma2 + c2)
    )

    return ssim.item()


def compute_lpips(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    计算 LPIPS (Learned Perceptual Image Patch Similarity)

    Args:
        pred: 预测图像
        target: 目标图像

    Returns:
        lpips: LPIPS 值
    """
    # 这里需要安装 lpips 库
    try:
        import lpips

        loss_fn = lpips.LPIPS(net="alex")
        return loss_fn(pred.unsqueeze(0), target.unsqueeze(0)).item()
    except ImportError:
        logger.warning("lpips not available, returning 0")
        return 0.0


# ============================================================================
# I/O 工具函数
# ============================================================================


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    path: str,
    **kwargs,
) -> None:
    """
    保存检查点

    Args:
        model: 模型
        optimizer: 优化器
        epoch: 当前轮次
        loss: 当前损失
        path: 保存路径
        **kwargs: 其他要保存的数据
    """
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "loss": loss,
        **kwargs,
    }

    torch.save(checkpoint, path)
    logger.info(f"Checkpoint saved to {path}")


def load_checkpoint(
    model: torch.nn.Module, optimizer: torch.optim.Optimizer, path: str
) -> tuple[int, float]:
    """
    加载检查点

    Args:
        model: 模型
        optimizer: 优化器
        path: 检查点路径

    Returns:
        epoch: 轮次
        loss: 损失
    """
    checkpoint = torch.load(path, map_location="cpu")

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    epoch = checkpoint.get("epoch", 0)
    loss = checkpoint.get("loss", float("inf"))

    logger.info(f"Checkpoint loaded from {path}")
    return epoch, loss


def save_config(config: Any, path: str) -> None:
    """
    保存配置

    Args:
        config: 配置对象
        path: 保存路径
    """
    config_dict = {}

    if hasattr(config, "__dict__"):
        config_dict = config.__dict__
    elif hasattr(config, "__dataclass_fields__"):
        config_dict = {field: getattr(config, field) for field in config.__dataclass_fields__}

    with open(path, "w") as f:
        json.dump(config_dict, f, indent=2, default=str)

    logger.info(f"Config saved to {path}")


def load_config(config_class: type, path: str) -> Any:
    """
    加载配置

    Args:
        config_class: 配置类
        path: 配置文件路径

    Returns:
        config: 配置对象
    """
    with open(path, "r") as f:
        config_dict = json.load(f)

    # 尝试转换数据类型
    for key, value in config_dict.items():
        if isinstance(value, str):
            # 尝试转换为数字
            try:
                if "." in value:
                    config_dict[key] = float(value)
                else:
                    config_dict[key] = int(value)
            except ValueError:
                pass

    config = config_class(**config_dict)
    logger.info(f"Config loaded from {path}")
    return config


# ============================================================================
# 空间分区工具函数
# ============================================================================


def compute_scene_bounds(
    camera_positions: np.ndarray, margin: float = 0.1
) -> tuple[np.ndarray, np.ndarray]:
    """
    计算场景边界

    Args:
        camera_positions: [N, 3] 相机位置
        margin: 边界边距

    Returns:
        min_bounds: 最小边界
        max_bounds: 最大边界
    """
    min_bounds = np.min(camera_positions, axis=0)
    max_bounds = np.max(camera_positions, axis=0)

    # 添加边距
    extent = max_bounds - min_bounds
    min_bounds -= extent * margin
    max_bounds += extent * margin

    return min_bounds, max_bounds


def create_spatial_grid(
    min_bounds: np.ndarray, max_bounds: np.ndarray, grid_size: tuple[int, int, int]
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    创建空间网格

    Args:
        min_bounds: 最小边界
        max_bounds: 最大边界
        grid_size: 网格大小 (nx, ny, nz)

    Returns:
        grid_cells: 网格单元列表，每个元素为 (min_bounds, max_bounds)
    """
    grid_cells = []

    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            for k in range(grid_size[2]):
                cell_min = np.array(
                    [
                        min_bounds[0] + i * (max_bounds[0] - min_bounds[0]) / grid_size[0],
                        min_bounds[1] + j * (max_bounds[1] - min_bounds[1]) / grid_size[1],
                        min_bounds[2] + k * (max_bounds[2] - min_bounds[2]) / grid_size[2],
                    ]
                )

                cell_max = np.array(
                    [
                        min_bounds[0] + (i + 1) * (max_bounds[0] - min_bounds[0]) / grid_size[0],
                        min_bounds[1] + (j + 1) * (max_bounds[1] - min_bounds[1]) / grid_size[1],
                        min_bounds[2] + (k + 1) * (max_bounds[2] - min_bounds[2]) / grid_size[2],
                    ]
                )

                grid_cells.append((cell_min, cell_max))

    return grid_cells


def assign_points_to_cells(
    points: np.ndarray, grid_cells: list[tuple[np.ndarray, np.ndarray]]
) -> list[list[int]]:
    """
    将点分配给网格单元

    Args:
        points: [N, 3] 点坐标
        grid_cells: 网格单元列表

    Returns:
        assignments: 每个单元的点的索引列表
    """
    assignments = [[] for _ in range(len(grid_cells))]

    for point_idx, point in enumerate(points):
        for cell_idx, (cell_min, cell_max) in enumerate(grid_cells):
            if np.all((point >= cell_min) & (point <= cell_max)):
                assignments[cell_idx].append(point_idx)
                break

    return assignments


# ============================================================================
# 导出所有函数
# ============================================================================

__all__ = [
    # 相机工具函数
    "generate_rays",
    "create_spiral_path",
    "interpolate_poses",
    "generate_random_poses",
    # 渲染工具函数
    "save_image",
    "save_video",
    "create_depth_visualization",
    "compute_psnr",
    "compute_ssim",
    "compute_lpips",
    # I/O 工具函数
    "save_checkpoint",
    "load_checkpoint",
    "save_config",
    "load_config",
    # 空间分区工具函数
    "compute_scene_bounds",
    "create_spatial_grid",
    "assign_points_to_cells",
]
