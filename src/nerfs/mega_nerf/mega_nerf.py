from __future__ import annotations

#!/usr/bin/env python3
"""
Mega-NeRF: Scalable Construction of Large-Scale NeRFs for Virtual Fly-Throughs
基于CVPR 2022论文的完整实现
"""

from typing import Optional

import matplotlib.pyplot as plt

核心特性:
- 空间分解 (Spatial Partitioning)
- 几何感知的数据分割 (Geometry-aware Data Partitioning)
- 并行训练 (Parallel Training)
- 时间一致性渲染 (Temporal Coherence Rendering)
- 大规模场景支持
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import json
import cv2
from tqdm import tqdm
import logging
from dataclasses import dataclass
from collections import defaultdict
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import time
import pickle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MegaNeRFConfig:
    """Mega-NeRF配置参数"""
    # 场景分解参数
    num_submodules: int = 8
    grid_size: tuple[int, int] = (4, 2)  # 2D网格分解
    overlap_factor: float = 0.15
    
    # 网络参数
    hidden_dim: int = 256
    num_layers: int = 8
    skip_connections: list[int] = None
    use_viewdirs: bool = True
    
    # 训练参数
    batch_size: int = 1024
    learning_rate: float = 5e-4
    lr_decay: float = 0.1
    max_iterations: int = 500000
    
    # 采样参数
    num_coarse: int = 256
    num_fine: int = 512
    near: float = 0.1
    far: float = 1000.0
    
    # 外观嵌入
    use_appearance_embedding: bool = True
    appearance_dim: int = 48
    
    # 边界参数
    scene_bounds: tuple[float, float, float, float, float, float] = (-100, -100, -10, 100, 100, 50)
    foreground_ratio: float = 0.8
    
    def __post_init__(self):
        if self.skip_connections is None:
            self.skip_connections = [4]

class PositionalEncoding(nn.Module):
    """位置编码模块"""
    def __init__(self, input_dim: int, max_freq_log2: int = 10, num_freqs: int = 10):
        super().__init__()
        self.input_dim = input_dim
        self.max_freq_log2 = max_freq_log2
        self.num_freqs = num_freqs
        
        self.freq_bands = 2.0 ** torch.linspace(0.0, max_freq_log2, num_freqs)
        self.output_dim = input_dim * (1 + 2 * num_freqs)
    
    def forward(self, x):
        """
        Args:
            x: [..., input_dim]
        Returns:
            encoded: [..., output_dim]
        """
        encoded = [x]
        for freq in self.freq_bands:
            encoded.append(torch.sin(freq * x))
            encoded.append(torch.cos(freq * x))
        return torch.cat(encoded, dim=-1)

class MegaNeRFSubmodule(nn.Module):
    """Mega-NeRF子模块，处理场景的一个空间区域"""
    def __init__(self, config: MegaNeRFConfig, centroid: np.ndarray):
        super().__init__()
        self.config = config
        self.centroid = torch.tensor(centroid, dtype=torch.float32)
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(3, max_freq_log2=10, num_freqs=10)
        self.dir_encoder = PositionalEncoding(
            3,
            max_freq_log2=4,
            num_freqs=4,
        )
        
        # 计算输入维度
        pos_dim = self.pos_encoder.output_dim
        input_dim = pos_dim
        if config.use_viewdirs:
            dir_dim = self.dir_encoder.output_dim
        else:
            dir_dim = 0
        
        # 外观嵌入
        if config.use_appearance_embedding:
            self.appearance_embedding = nn.Embedding(1000, config.appearance_dim)  # 假设最多1000张图片
            input_dim += config.appearance_dim
        
        # 主网络 - 预测密度和特征
        layers = []
        current_dim = input_dim
        
        for i in range(config.num_layers):
            if i in config.skip_connections:
                layers.append(nn.Linear(current_dim + input_dim, config.hidden_dim))
            else:
                layers.append(nn.Linear(current_dim, config.hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            current_dim = config.hidden_dim
        
        self.main_layers = nn.ModuleList(layers)
        
        # 密度预测头
        self.density_head = nn.Linear(config.hidden_dim, 1)
        
        # 颜色预测网络
        if config.use_viewdirs:
            self.feature_head = nn.Linear(config.hidden_dim, config.hidden_dim)
            color_input_dim = config.hidden_dim + dir_dim
            self.color_layers = nn.Sequential(
                nn.Linear(
                    color_input_dim,
                    config.hidden_dim // 2,
                )
            )
        else:
            self.color_head = nn.Sequential(
                nn.Linear(config.hidden_dim, 3), nn.Sigmoid()
            )
    
    def forward(self, points, viewdirs=None, appearance_idx=None):
        """
        Args:
            points: [N, 3] 3D坐标
            viewdirs: [N, 3] 视角方向 (可选)
            appearance_idx: [N] 外观嵌入索引 (可选)
        Returns:
            density: [N, 1] 密度值
            color: [N, 3] RGB颜色
        """
        # 位置编码
        pos_encoded = self.pos_encoder(points)
        
        # 外观嵌入
        features = [pos_encoded]
        if self.config.use_appearance_embedding and appearance_idx is not None:
            app_embed = self.appearance_embedding(appearance_idx)
            features.append(app_embed)
        
        x = torch.cat(features, dim=-1)
        input_x = x
        
        # 主网络前向传播
        for i, layer in enumerate(self.main_layers):
            if i // 2 in self.config.skip_connections and i % 2 == 0:
                x = torch.cat([x, input_x], dim=-1)
            x = layer(x)
        
        # 预测密度
        density = self.density_head(x)
        density = F.relu(density)
        
        # 预测颜色
        if self.config.use_viewdirs and viewdirs is not None:
            features = self.feature_head(x)
            dir_encoded = self.dir_encoder(viewdirs)
            color_input = torch.cat([features, dir_encoded], dim=-1)
            color = self.color_layers(color_input)
        else:
            color = self.color_head(x)
        
        return density, color

class MegaNeRF(nn.Module):
    """Mega-NeRF主模型"""
    def __init__(self, config: MegaNeRFConfig):
        super().__init__()
        self.config = config
        
        # 创建空间网格中心点
        self.centroids = self._create_spatial_grid()
        
        # 创建子模块
        self.submodules = nn.ModuleList([
            MegaNeRFSubmodule(config, centroid) 
            for centroid in self.centroids
        ])
        
        # 前景/背景分解
        self.foreground_bounds = self._compute_foreground_bounds()
        
        # 背景NeRF (简化版本)
        self.background_nerf = MegaNeRFSubmodule(config, np.array([0, 0, 0]))
        
        logger.info(f"创建了 {len(self.submodules)} 个子模块")
        logger.info(f"前景边界: {self.foreground_bounds}")
    
    def _create_spatial_grid(self):
        """创建2D空间网格的中心点"""
        bounds = self.config.scene_bounds
        x_min, y_min, z_min, x_max, y_max, z_max = bounds
        
        grid_x, grid_y = self.config.grid_size
        
        x_centers = np.linspace(x_min, x_max, grid_x + 1)[:-1] + (x_max - x_min) / (2 * grid_x)
        y_centers = np.linspace(y_min, y_max, grid_y + 1)[:-1] + (y_max - y_min) / (2 * grid_y)
        z_center = (z_min + z_max) / 2
        
        centroids = []
        for x in x_centers:
            for y in y_centers:
                centroids.append([x, y, z_center])
        
        return np.array(centroids)
    
    def _compute_foreground_bounds(self):
        """计算前景边界（椭球体）"""
        bounds = self.config.scene_bounds
        x_min, y_min, z_min, x_max, y_max, z_max = bounds
        
        center = np.array([(x_min + x_max) / 2, (y_min + y_max) / 2, (z_min + z_max) / 2])
        radius = np.array(
            [,
        )
        
        return {'center': center, 'radius': radius}
    
    def _assign_points_to_submodules(self, points):
        """将3D点分配给最近的子模块"""
        points_np = points.detach().cpu().numpy()
        distances = np.linalg.norm(points_np[:, None, :] - self.centroids[None, :, :], axis=2)
        assignments = np.argmin(distances, axis=1)
        return torch.tensor(assignments, device=points.device)
    
    def _is_in_foreground(self, points):
        """判断点是否在前景区域"""
        center = torch.tensor(self.foreground_bounds['center'], device=points.device)
        radius = torch.tensor(self.foreground_bounds['radius'], device=points.device)
        
        normalized = (points - center) / radius
        distances = torch.norm(normalized, dim=-1)
        return distances <= 1.0
    
    def forward(self, points, viewdirs=None, appearance_idx=None):
        """
        Args:
            points: [N, 3] 3D坐标
            viewdirs: [N, 3] 视角方向
            appearance_idx: [N] 外观嵌入索引
        Returns:
            density: [N, 1] 密度值
            color: [N, 3] RGB颜色
        """
        # 判断前景/背景
        in_foreground = self._is_in_foreground(points)
        
        density = torch.zeros(points.shape[0], 1, device=points.device)
        color = torch.zeros(points.shape[0], 3, device=points.device)
        
        # 处理前景点
        if in_foreground.any():
            fg_points = points[in_foreground]
            fg_viewdirs = viewdirs[in_foreground] if viewdirs is not None else None
            fg_appearance = appearance_idx[in_foreground] if appearance_idx is not None else None
            
            # 分配给子模块
            assignments = self._assign_points_to_submodules(fg_points)
            
            for i, submodule in enumerate(self.submodules):
                mask = assignments == i
                if mask.any():
                    sub_points = fg_points[mask]
                    sub_viewdirs = fg_viewdirs[mask] if fg_viewdirs is not None else None
                    sub_appearance = fg_appearance[mask] if fg_appearance is not None else None
                    
                    sub_density, sub_color = submodule(sub_points, sub_viewdirs, sub_appearance)
                    
                    # 将结果放回对应位置
                    fg_indices = torch.where(in_foreground)[0][mask]
                    density[fg_indices] = sub_density
                    color[fg_indices] = sub_color
        
        # 处理背景点
        if (~in_foreground).any():
            bg_points = points[~in_foreground]
            bg_viewdirs = viewdirs[~in_foreground] if viewdirs is not None else None
            bg_appearance = appearance_idx[~in_foreground] if appearance_idx is not None else None
            
            bg_density, bg_color = self.background_nerf(bg_points, bg_viewdirs, bg_appearance)
            
            bg_indices = torch.where(~in_foreground)[0]
            density[bg_indices] = bg_density
            color[bg_indices] = bg_color
        
        return density, color

class MegaNeRFDataset:
    """Mega-NeRF数据集"""
    def __init__(self, data_dir: str, config: MegaNeRFConfig):
        self.data_dir = data_dir
        self.config = config
        
        # 加载相机参数和图像
        self.cameras, self.images = self._load_data()
        
        # 创建数据分割
        self.data_partitions = self._create_data_partitions()
        
        logger.info(f"加载了 {len(self.images)} 张图像")
        logger.info(f"创建了 {len(self.data_partitions)} 个数据分割")
    
    def _load_data(self):
        """加载相机参数和图像数据"""
        # 这里应该根据实际数据格式实现
        # 示例实现
        cameras = []
        images = []
        
        # 假设有transforms.json文件
        transforms_path = os.path.join(self.data_dir, "transforms.json")
        if os.path.exists(transforms_path):
            with open(transforms_path, 'r') as f:
                data = json.load(f)
            
            for frame in data['frames']:
                # 相机参数
                camera = {
                    'transform_matrix': np.array(
                        frame['transform_matrix'],
                    )
                }
                cameras.append(camera)
                
                # 图像路径
                image_path = os.path.join(self.data_dir, frame['file_path'])
                images.append(image_path)
        
        return cameras, images
    
    def _create_data_partitions(self):
        """创建基于几何的数据分割"""
        partitions = [[] for _ in range(len(self.config.grid_size[0] * self.config.grid_size[1]))]
        
        for img_idx, camera in enumerate(self.cameras):
            # 获取相机位置
            camera_pos = camera['transform_matrix'][:3, 3]
            
            # 分配给最近的子模块
            distances = np.linalg.norm(
                camera_pos[None, :] - np.array(
                    [centroid for centroid in self._get_centroids,
                )
            )
            assignment = np.argmin(distances)
            
            # 添加重叠
            overlap_assignments = []
            for i, dist in enumerate(distances):
                if dist <= distances[assignment] * (1 + self.config.overlap_factor):
                    overlap_assignments.append(i)
            
            for assign_idx in overlap_assignments:
                partitions[assign_idx].append(img_idx)
        
        return partitions
    
    def _get_centroids(self):
        """获取网格中心点"""
        bounds = self.config.scene_bounds
        x_min, y_min, z_min, x_max, y_max, z_max = bounds
        
        grid_x, grid_y = self.config.grid_size
        
        x_centers = np.linspace(x_min, x_max, grid_x + 1)[:-1] + (x_max - x_min) / (2 * grid_x)
        y_centers = np.linspace(y_min, y_max, grid_y + 1)[:-1] + (y_max - y_min) / (2 * grid_y)
        z_center = (z_min + z_max) / 2
        
        centroids = []
        for x in x_centers:
            for y in y_centers:
                centroids.append([x, y, z_center])
        
        return centroids
    
    def get_partition_data(self, partition_idx: int):
        """获取特定分割的数据"""
        image_indices = self.data_partitions[partition_idx]
        
        partition_cameras = [self.cameras[i] for i in image_indices]
        partition_images = [self.images[i] for i in image_indices]
        
        return partition_cameras, partition_images, image_indices

class VolumetricRenderer:
    """体积渲染器"""
    def __init__(self, config: MegaNeRFConfig):
        self.config = config
    
    def render_ray(self, model, ray_o, ray_d, near, far, appearance_idx=None):
        """渲染单条光线"""
        # 分层采样
        t_vals = torch.linspace(near, far, self.config.num_coarse, device=ray_o.device)
        
        # 粗采样点
        points = ray_o[..., None, :] + ray_d[..., None, :] * t_vals[..., :, None]
        points = points.reshape(-1, 3)
        
        # 视角方向
        viewdirs = ray_d / torch.norm(ray_d, dim=-1, keepdim=True)
        viewdirs = viewdirs.expand_as(points)
        
        # 外观索引
        if appearance_idx is not None:
            app_idx = torch.full((points.shape[0], ), appearance_idx, device=points.device)
        else:
            app_idx = None
        
        # 模型前向传播
        with torch.no_grad():
            density, color = model(points, viewdirs, app_idx)
        
        # 重塑为光线形状
        density = density.reshape(*ray_o.shape[:-1], self.config.num_coarse, 1)
        color = color.reshape(*ray_o.shape[:-1], self.config.num_coarse, 3)
        
        # 体积渲染
        dists = t_vals[..., 1:] - t_vals[..., :-1]
        dists = torch.cat([dists, torch.full_like(dists[..., :1], 1e10)], dim=-1)
        
        alpha = 1.0 - torch.exp(-density[..., 0] * dists)
        transmittance = torch.cumprod(1.0 - alpha + 1e-10, dim=-1)
        transmittance = torch.cat(
            [torch.ones_like,
        )
        
        weights = alpha * transmittance
        rgb = torch.sum(weights[..., None] * color, dim=-2)
        
        return rgb, weights, t_vals
    
    def render_image(self, model, camera, height, width, appearance_idx=None):
        """渲染完整图像"""
        # 生成光线
        rays_o, rays_d = self._generate_rays(camera, height, width)
        
        # 批量渲染
        rgb_map = []
        batch_size = 1024
        
        for i in range(0, rays_o.shape[0], batch_size):
            batch_rays_o = rays_o[i:i+batch_size]
            batch_rays_d = rays_d[i:i+batch_size]
            
            rgb, _, _ = self.render_ray(
                model, batch_rays_o, batch_rays_d, self.config.near, self.config.far, appearance_idx
            )
            rgb_map.append(rgb)
        
        rgb_map = torch.cat(rgb_map, dim=0)
        rgb_map = rgb_map.reshape(height, width, 3)
        
        return rgb_map
    
    def _generate_rays(self, camera, height, width):
        """生成相机光线"""
        # 获取相机参数
        transform_matrix = torch.tensor(camera['transform_matrix'], dtype=torch.float32)
        focal = camera['focal_length']
        
        # 生成像素坐标
        i, j = torch.meshgrid(
            torch.arange(
                width,
                dtype=torch.float32,
            )
        )
        
        # 转换为相机坐标
        dirs = torch.stack([
            (i - width * 0.5) / focal, -(j - height * 0.5) / focal, -torch.ones_like(i)
        ], dim=-1)
        
        # 转换为世界坐标
        rays_d = torch.sum(dirs[..., None, :] * transform_matrix[:3, :3], dim=-1)
        rays_o = transform_matrix[:3, 3].expand(rays_d.shape)
        
        return rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)

class MegaNeRFTrainer:
    """Mega-NeRF训练器"""
    def __init__(self, config: MegaNeRFConfig, data_dir: str, output_dir: str):
        self.config = config
        self.data_dir = data_dir
        self.output_dir = output_dir
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 初始化模型和数据
        self.model = MegaNeRF(config)
        self.dataset = MegaNeRFDataset(data_dir, config)
        self.renderer = VolumetricRenderer(config)
        
        # 优化器
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.95)
        
        # 设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        logger.info(f"使用设备: {self.device}")
    
    def train_submodule(self, submodule_idx: int, max_iterations: int = None):
        """训练单个子模块"""
        if max_iterations is None:
            max_iterations = self.config.max_iterations
        
        # 获取分割数据
        cameras, images, image_indices = self.dataset.get_partition_data(submodule_idx)
        
        if not cameras:
            logger.warning(f"子模块 {submodule_idx} 没有训练数据")
            return
        
        logger.info(f"训练子模块 {submodule_idx}，包含 {len(cameras)} 张图像")
        
        # 只训练对应的子模块
        submodule = self.model.submodules[submodule_idx]
        optimizer = torch.optim.Adam(submodule.parameters(), lr=self.config.learning_rate)
        
        for iteration in range(max_iterations):
            # 随机选择图像
            img_idx = np.random.randint(len(cameras))
            camera = cameras[img_idx]
            
            # 生成训练光线
            rays_o, rays_d = self._generate_training_rays(camera)
            
            # 随机采样光线
            ray_indices = torch.randperm(rays_o.shape[0])[:self.config.batch_size]
            batch_rays_o = rays_o[ray_indices].to(self.device)
            batch_rays_d = rays_d[ray_indices].to(self.device)
            
            # 渲染
            rgb_pred, _, _ = self.renderer.render_ray(
                self.model, batch_rays_o, batch_rays_d, self.config.near, self.config.far, appearance_idx=image_indices[img_idx]
            )
            
            # 获取目标颜色（这里需要根据实际情况实现）
            rgb_target = self._get_target_colors(camera, ray_indices)
            
            # 计算损失
            loss = F.mse_loss(rgb_pred, rgb_target)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if iteration % 1000 == 0:
                logger.info(f"子模块 {submodule_idx}, 迭代 {iteration}, 损失: {loss.item():.6f}")
    
    def train_parallel(self):
        """并行训练所有子模块"""
        logger.info("开始并行训练")
        
        # 使用多进程并行训练
        with ProcessPoolExecutor(
            max_workers=min,
        )
            futures = []
            for i in range(len(self.model.submodules)):
                future = executor.submit(self.train_submodule, i)
                futures.append(future)
            
            # 等待所有训练完成
            for i, future in enumerate(futures):
                try:
                    future.result()
                    logger.info(f"子模块 {i} 训练完成")
                except Exception as e:
                    logger.error(f"子模块 {i} 训练失败: {e}")
        
        logger.info("并行训练完成")
    
    def train_sequential(self):
        """顺序训练所有子模块"""
        logger.info("开始顺序训练")
        
        for i in range(len(self.model.submodules)):
            logger.info(f"训练子模块 {i}/{len(self.model.submodules)}")
            self.train_submodule(i)
        
        logger.info("顺序训练完成")
    
    def _generate_training_rays(self, camera):
        """生成训练光线"""
        height, width = camera['height'], camera['width']
        return self.renderer._generate_rays(camera, height, width)
    
    def _get_target_colors(self, camera, ray_indices):
        """获取目标颜色（需要根据实际图像数据实现）"""
        # 这里是示例实现，实际需要从图像中采样
        return torch.rand(len(ray_indices), 3, device=self.device)
    
    def save_model(self, path: str):
        """保存模型"""
        torch.save({
            'model_state_dict': self.model.state_dict(
            )
        }, path)
        logger.info(f"模型已保存到: {path}")
    
    def load_model(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"模型已从 {path} 加载")

class InteractiveRenderer:
    """交互式渲染器，支持实时飞行浏览"""
    def __init__(self, model: MegaNeRF, config: MegaNeRFConfig):
        self.model = model
        self.config = config
        self.renderer = VolumetricRenderer(config)
        
        # 缓存结构
        self.octree_cache = {}
        self.cache_size = 1000000  # 最大缓存元素数
        
        # 时间一致性参数
        self.temporal_coherence = True
        self.cache_reuse_threshold = 0.8
    
    def render_view(self, camera_pose, width=800, height=600, use_cache=True):
        """渲染视图"""
        start_time = time.time()
        
        # 生成相机参数
        camera = {
            'transform_matrix': camera_pose, 'focal_length': 800, 'width': width, 'height': height
        }
        
        # 渲染图像
        if use_cache and self.temporal_coherence:
            rgb = self._render_with_cache(camera, height, width)
        else:
            rgb = self.renderer.render_image(self.model, camera, height, width)
        
        render_time = time.time() - start_time
        logger.info(f"渲染时间: {render_time:.3f}s")
        
        return rgb.detach().cpu().numpy()
    
    def _render_with_cache(self, camera, height, width):
        """使用缓存的渲染"""
        # 简化的缓存实现
        # 实际应该实现基于八叉树的动态缓存
        return self.renderer.render_image(self.model, camera, height, width)
    
    def create_flythrough(self, camera_path, output_path, fps=30):
        """创建飞行浏览视频"""
        logger.info(f"创建飞行浏览视频: {output_path}")
        
        # 设置视频编写器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (800, 600))
        
        for i, camera_pose in enumerate(tqdm(camera_path, desc="渲染帧")):
            # 渲染帧
            rgb = self.render_view(camera_pose)
            
            # 转换为OpenCV格式
            frame = (rgb * 255).astype(np.uint8)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # 写入视频
            video_writer.write(frame)
        
        video_writer.release()
        logger.info(f"视频已保存到: {output_path}")

def create_sample_camera_path(num_frames=100):
    """创建示例相机路径"""
    # 创建圆形飞行路径
    angles = np.linspace(0, 2 * np.pi, num_frames)
    radius = 50
    height = 20
    
    camera_path = []
    for angle in angles:
        # 相机位置
        pos = np.array([
            radius * np.cos(angle), radius * np.sin(angle), height
        ])
        
        # 朝向中心
        target = np.array([0, 0, 0])
        up = np.array([0, 0, 1])
        
        # 构建变换矩阵
        forward = target - pos
        forward = forward / np.linalg.norm(forward)
        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)
        up = np.cross(right, forward)
        
        transform = np.eye(4)
        transform[:3, 0] = right
        transform[:3, 1] = up
        transform[:3, 2] = -forward
        transform[:3, 3] = pos
        
        camera_path.append(transform)
    
    return camera_path

def main():
    """主函数示例"""
    # 配置参数
    config = MegaNeRFConfig(
        num_submodules=8, grid_size=(4, 2), max_iterations=100000, batch_size=1024
    )
    
    logger.info("Mega-NeRF实现创建完成！")
    logger.info(f"配置: {config}")
    
    # 创建模型
    model = MegaNeRF(config)
    logger.info(f"模型参数数量: {sum(p.numel() for p in model.parameters())}")

if __name__ == "__main__":
    main() 