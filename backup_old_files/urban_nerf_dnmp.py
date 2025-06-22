#!/usr/bin/env python3
"""
Urban Radiance Field Representation with Deformable Neural Mesh Primitives
基于论文 "Urban Radiance Field Representation with Deformable Neural Mesh Primitives" (ICCV 2023)

核心特性:
- 可变形神经网格原语 (DNMP)
- 基于光栅化的高效渲染
- 城市级场景表示
- 视角相关的辐射度预测
- 几何和外观解耦
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import json
import cv2
from typing import List, Tuple, Dict, Optional, Union
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
from dataclasses import dataclass
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class UrbanRFConfig:
    """Urban Radiance Field配置参数"""
    # 场景参数
    scene_bounds: Tuple[float, float, float, float, float, float] = (-50, -50, -10, 50, 50, 30)
    voxel_size: float = 2.0                    # 体素大小
    
    # DNMP参数
    primitive_resolution: int = 8              # 原语分辨率 (8x8x8 vertices)
    latent_dim: int = 32                      # 潜在空间维度
    vertex_feature_dim: int = 64              # 顶点特征维度
    
    # 网络参数
    deformation_hidden_dims: List[int] = None  # 变形网络隐藏层
    radiance_hidden_dims: List[int] = None     # 辐射度网络隐藏层
    view_embed_dim: int = 27                   # 视角嵌入维度
    
    # 训练参数
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 4096
    num_epochs: int = 200
    
    # 渲染参数
    near_plane: float = 0.1
    far_plane: float = 100.0
    n_samples: int = 64                       # 每条射线采样点数
    
    def __post_init__(self):
        if self.deformation_hidden_dims is None:
            self.deformation_hidden_dims = [128, 128, 128]
        if self.radiance_hidden_dims is None:
            self.radiance_hidden_dims = [256, 256, 128]

class DeformationNetwork(nn.Module):
    """变形网络 - 从潜在码解码网格顶点位置"""
    
    def __init__(self, latent_dim: int, primitive_resolution: int, hidden_dims: List[int]):
        super().__init__()
        self.latent_dim = latent_dim
        self.primitive_resolution = primitive_resolution
        self.n_vertices = primitive_resolution ** 3
        
        # 构建MLP
        layers = []
        prev_dim = latent_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
            ])
            prev_dim = hidden_dim
        
        # 输出层: 每个顶点的3D偏移
        layers.append(nn.Linear(prev_dim, self.n_vertices * 3))
        
        self.network = nn.Sequential(*layers)
        
        # 初始化基础网格
        self.register_buffer('base_vertices', self._create_base_grid())
    
    def _create_base_grid(self) -> torch.Tensor:
        """创建基础规则网格"""
        res = self.primitive_resolution
        x = torch.linspace(-1, 1, res)
        y = torch.linspace(-1, 1, res)
        z = torch.linspace(-1, 1, res)
        
        xx, yy, zz = torch.meshgrid(x, y, z, indexing='ij')
        vertices = torch.stack([xx.flatten(), yy.flatten(), zz.flatten()], dim=1)
        
        return vertices  # (n_vertices, 3)
    
    def forward(self, latent_codes: torch.Tensor) -> torch.Tensor:
        """
        Args:
            latent_codes: (batch_size, latent_dim)
        Returns:
            vertices: (batch_size, n_vertices, 3)
        """
        batch_size = latent_codes.shape[0]
        
        # 预测偏移
        offsets = self.network(latent_codes)  # (batch_size, n_vertices * 3)
        offsets = offsets.view(batch_size, self.n_vertices, 3)
        
        # 应用偏移到基础网格
        base_vertices = self.base_vertices.unsqueeze(0).expand(batch_size, -1, -1)
        deformed_vertices = base_vertices + offsets
        
        return deformed_vertices

class RadianceNetwork(nn.Module):
    """辐射度网络 - 从顶点特征和视角预测颜色和密度"""
    
    def __init__(self, vertex_feature_dim: int, view_embed_dim: int, hidden_dims: List[int]):
        super().__init__()
        self.vertex_feature_dim = vertex_feature_dim
        self.view_embed_dim = view_embed_dim
        
        # 构建MLP
        layers = []
        prev_dim = vertex_feature_dim + view_embed_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
            ])
            prev_dim = hidden_dim
        
        # 输出层: RGB + density
        self.rgb_head = nn.Sequential(
            nn.Linear(prev_dim, 3),
            nn.Sigmoid()
        )
        self.density_head = nn.Sequential(
            nn.Linear(prev_dim, 1),
            nn.ReLU()
        )
        
        self.feature_network = nn.Sequential(*layers)
    
    def forward(self, vertex_features: torch.Tensor, view_dirs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            vertex_features: (batch_size, vertex_feature_dim)
            view_dirs: (batch_size, view_embed_dim)
        Returns:
            rgb: (batch_size, 3)
            density: (batch_size, 1)
        """
        # 连接特征和视角
        combined = torch.cat([vertex_features, view_dirs], dim=-1)
        
        # 通过网络
        features = self.feature_network(combined)
        
        # 预测RGB和密度
        rgb = self.rgb_head(features)
        density = self.density_head(features)
        
        return rgb, density

class DNMP(nn.Module):
    """可变形神经网格原语"""
    
    def __init__(self, config: UrbanRFConfig):
        super().__init__()
        self.config = config
        
        # 潜在码
        self.latent_code = nn.Parameter(torch.randn(config.latent_dim) * 0.1)
        
        # 顶点特征
        n_vertices = config.primitive_resolution ** 3
        self.vertex_features = nn.Parameter(torch.randn(n_vertices, config.vertex_feature_dim) * 0.1)
        
        # 变形网络
        self.deformation_net = DeformationNetwork(
            config.latent_dim,
            config.primitive_resolution,
            config.deformation_hidden_dims
        )
    
    def get_vertices(self) -> torch.Tensor:
        """获取变形后的顶点位置"""
        latent = self.latent_code.unsqueeze(0)  # (1, latent_dim)
        vertices = self.deformation_net(latent).squeeze(0)  # (n_vertices, 3)
        return vertices
    
    def get_vertex_features(self) -> torch.Tensor:
        """获取顶点特征"""
        return self.vertex_features

class UrbanRadianceField(nn.Module):
    """城市辐射场主类"""
    
    def __init__(self, config: UrbanRFConfig):
        super().__init__()
        self.config = config
        
        # 创建体素网格
        self.voxel_grid = self._create_voxel_grid()
        
        # 为每个体素创建DNMP
        self.dnmps = nn.ModuleDict()
        for i, voxel_center in enumerate(self.voxel_grid):
            self.dnmps[f'dnmp_{i}'] = DNMP(config)
        
        # 辐射度网络
        self.radiance_net = RadianceNetwork(
            config.vertex_feature_dim,
            config.view_embed_dim,
            config.radiance_hidden_dims
        )
        
        logger.info(f"创建了 {len(self.dnmps)} 个DNMP")
    
    def _create_voxel_grid(self) -> torch.Tensor:
        """创建体素网格中心点"""
        x_min, y_min, z_min, x_max, y_max, z_max = self.config.scene_bounds
        voxel_size = self.config.voxel_size
        
        x_centers = torch.arange(x_min + voxel_size/2, x_max, voxel_size)
        y_centers = torch.arange(y_min + voxel_size/2, y_max, voxel_size)
        z_centers = torch.arange(z_min + voxel_size/2, z_max, voxel_size)
        
        xx, yy, zz = torch.meshgrid(x_centers, y_centers, z_centers, indexing='ij')
        voxel_centers = torch.stack([xx.flatten(), yy.flatten(), zz.flatten()], dim=1)
        
        return voxel_centers
    
    def _positional_encoding(self, x: torch.Tensor, num_freqs: int = 10) -> torch.Tensor:
        """位置编码"""
        freq_bands = 2.0 ** torch.linspace(0, num_freqs-1, num_freqs, device=x.device)
        
        encoded = []
        for freq in freq_bands:
            encoded.append(torch.sin(freq * x))
            encoded.append(torch.cos(freq * x))
        
        return torch.cat(encoded, dim=-1)
    
    def render_rays(self, ray_origins: torch.Tensor, ray_directions: torch.Tensor) -> Dict[str, torch.Tensor]:
        """渲染射线"""
        batch_size = ray_origins.shape[0]
        device = ray_origins.device
        
        # 在射线上采样点
        t_vals = torch.linspace(self.config.near_plane, self.config.far_plane, 
                              self.config.n_samples, device=device)
        points = ray_origins.unsqueeze(-2) + ray_directions.unsqueeze(-2) * t_vals.unsqueeze(0).unsqueeze(-1)
        
        # 视角编码
        view_dirs = self._positional_encoding(ray_directions)
        view_dirs = view_dirs.unsqueeze(1).expand(-1, self.config.n_samples, -1)
        
        # 收集所有采样点的RGB和密度
        all_rgbs = []
        all_densities = []
        
        for i, dnmp in enumerate(self.dnmps.values()):
            # 获取DNMP的顶点和特征
            vertices = dnmp.get_vertices()  # (n_vertices, 3)
            vertex_features = dnmp.get_vertex_features()  # (n_vertices, vertex_feature_dim)
            
            # 对于每个采样点，找到最近的顶点并插值特征
            points_flat = points.view(-1, 3)  # (batch_size * n_samples, 3)
            
            # 简化实现：使用最近邻插值
            distances = torch.cdist(points_flat, vertices)  # (batch_size * n_samples, n_vertices)
            nearest_indices = torch.argmin(distances, dim=-1)  # (batch_size * n_samples,)
            
            # 获取插值特征
            interpolated_features = vertex_features[nearest_indices]  # (batch_size * n_samples, vertex_feature_dim)
            view_dirs_flat = view_dirs.view(-1, view_dirs.shape[-1])
            
            # 预测RGB和密度
            rgb, density = self.radiance_net(interpolated_features, view_dirs_flat)
            
            rgb = rgb.view(batch_size, self.config.n_samples, 3)
            density = density.view(batch_size, self.config.n_samples, 1).squeeze(-1)
            
            all_rgbs.append(rgb)
            all_densities.append(density)
        
        # 平均所有DNMP的预测
        if all_rgbs:
            rgb_samples = torch.stack(all_rgbs, dim=0).mean(dim=0)  # (batch_size, n_samples, 3)
            density_samples = torch.stack(all_densities, dim=0).mean(dim=0)  # (batch_size, n_samples)
        else:
            rgb_samples = torch.zeros(batch_size, self.config.n_samples, 3, device=device)
            density_samples = torch.zeros(batch_size, self.config.n_samples, device=device)
        
        # 体渲染
        delta = t_vals[1:] - t_vals[:-1]
        delta = torch.cat([delta, torch.tensor([1e10], device=device)])
        
        alpha = 1.0 - torch.exp(-density_samples * delta.unsqueeze(0))
        
        # 累积透明度
        T = torch.cumprod(1.0 - alpha + 1e-10, dim=-1)
        T = torch.cat([torch.ones(batch_size, 1, device=device), T[:, :-1]], dim=-1)
        
        # 权重
        weights = alpha * T
        
        # 最终颜色
        rgb_final = torch.sum(weights.unsqueeze(-1) * rgb_samples, dim=1)
        depth_final = torch.sum(weights * t_vals.unsqueeze(0), dim=1)
        
        return {
            'rgb': rgb_final,
            'depth': depth_final,
            'weights': weights
        }
    
    def forward(self, ray_origins: torch.Tensor, ray_directions: torch.Tensor) -> Dict[str, torch.Tensor]:
        """前向传播"""
        return self.render_rays(ray_origins, ray_directions)

class UrbanRFDataset:
    """城市辐射场数据集"""
    
    def __init__(self, data_dir: str, config: UrbanRFConfig):
        self.data_dir = data_dir
        self.config = config
        
        # 加载相机参数和图像
        self.cameras, self.images = self._load_data()
        
        logger.info(f"加载了 {len(self.images)} 张图像")
    
    def _load_data(self) -> Tuple[List[Dict], List[np.ndarray]]:
        """加载数据"""
        cameras = []
        images = []
        
        # 假设数据格式类似KITTI-360
        transforms_file = os.path.join(self.data_dir, 'transforms.json')
        
        if os.path.exists(transforms_file):
            with open(transforms_file, 'r') as f:
                data = json.load(f)
            
            for frame in data['frames']:
                # 加载图像
                img_path = os.path.join(self.data_dir, frame['file_path'])
                if os.path.exists(img_path):
                    img = cv2.imread(img_path)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    images.append(img)
                    
                    # 相机参数
                    camera = {
                        'transform_matrix': np.array(frame['transform_matrix']),
                        'camera_angle_x': data.get('camera_angle_x', 1.0),
                        'width': img.shape[1],
                        'height': img.shape[0]
                    }
                    cameras.append(camera)
        
        return cameras, images
    
    def get_rays(self, camera_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """生成射线"""
        camera = self.cameras[camera_idx]
        H, W = camera['height'], camera['width']
        
        # 相机内参
        focal = 0.5 * W / np.tan(0.5 * camera['camera_angle_x'])
        
        # 像素坐标
        i, j = np.meshgrid(np.arange(W), np.arange(H), indexing='xy')
        
        # 归一化设备坐标
        dirs = np.stack([
            (i - W * 0.5) / focal,
            -(j - H * 0.5) / focal,
            -np.ones_like(i)
        ], axis=-1)
        
        # 相机到世界坐标变换
        c2w = camera['transform_matrix']
        
        # 射线方向
        ray_directions = np.sum(dirs[..., None, :] * c2w[:3, :3], axis=-1)
        ray_directions = ray_directions / np.linalg.norm(ray_directions, axis=-1, keepdims=True)
        
        # 射线起点
        ray_origins = np.broadcast_to(c2w[:3, 3], ray_directions.shape)
        
        return torch.FloatTensor(ray_origins), torch.FloatTensor(ray_directions)

class UrbanRFTrainer:
    """城市辐射场训练器"""
    
    def __init__(self, model: UrbanRadianceField, dataset: UrbanRFDataset, config: UrbanRFConfig):
        self.model = model
        self.dataset = dataset
        self.config = config
        
        # 优化器
        self.optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        
        # 损失函数
        self.criterion = nn.MSELoss()
        
        # 训练历史
        self.train_losses = []
    
    def train_epoch(self) -> float:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        for camera_idx in range(len(self.dataset.cameras)):
            # 获取射线和目标图像
            ray_origins, ray_directions = self.dataset.get_rays(camera_idx)
            target_image = torch.FloatTensor(self.dataset.images[camera_idx]) / 255.0
            
            H, W = target_image.shape[:2]
            
            # 随机采样像素
            num_rays = min(self.config.batch_size, H * W)
            indices = torch.randperm(H * W)[:num_rays]
            
            ray_origins_batch = ray_origins.view(-1, 3)[indices]
            ray_directions_batch = ray_directions.view(-1, 3)[indices]
            target_rgb_batch = target_image.view(-1, 3)[indices]
            
            # 前向传播
            outputs = self.model(ray_origins_batch, ray_directions_batch)
            predicted_rgb = outputs['rgb']
            
            # 计算损失
            loss = self.criterion(predicted_rgb, target_rgb_batch)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0
    
    def train(self) -> None:
        """训练模型"""
        logger.info("开始训练Urban Radiance Field...")
        
        for epoch in range(self.config.num_epochs):
            avg_loss = self.train_epoch()
            self.train_losses.append(avg_loss)
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{self.config.num_epochs}, Loss: {avg_loss:.6f}")
        
        logger.info("训练完成")
    
    def render_novel_view(self, camera_params: Dict) -> np.ndarray:
        """渲染新视角"""
        self.model.eval()
        
        with torch.no_grad():
            # 生成射线
            H, W = camera_params['height'], camera_params['width']
            focal = 0.5 * W / np.tan(0.5 * camera_params['camera_angle_x'])
            
            i, j = np.meshgrid(np.arange(W), np.arange(H), indexing='xy')
            dirs = np.stack([
                (i - W * 0.5) / focal,
                -(j - H * 0.5) / focal,
                -np.ones_like(i)
            ], axis=-1)
            
            c2w = camera_params['transform_matrix']
            ray_directions = np.sum(dirs[..., None, :] * c2w[:3, :3], axis=-1)
            ray_directions = ray_directions / np.linalg.norm(ray_directions, axis=-1, keepdims=True)
            ray_origins = np.broadcast_to(c2w[:3, 3], ray_directions.shape)
            
            # 分块渲染
            chunk_size = 1024
            rgb_image = np.zeros((H, W, 3))
            
            ray_origins_flat = ray_origins.reshape(-1, 3)
            ray_directions_flat = ray_directions.reshape(-1, 3)
            
            for i in range(0, len(ray_origins_flat), chunk_size):
                end_i = min(i + chunk_size, len(ray_origins_flat))
                
                origins_chunk = torch.FloatTensor(ray_origins_flat[i:end_i])
                directions_chunk = torch.FloatTensor(ray_directions_flat[i:end_i])
                
                outputs = self.model(origins_chunk, directions_chunk)
                rgb_chunk = outputs['rgb'].cpu().numpy()
                
                rgb_image.reshape(-1, 3)[i:end_i] = rgb_chunk
            
            return np.clip(rgb_image, 0, 1)

def create_sample_data(data_dir: str) -> None:
    """创建示例数据"""
    os.makedirs(data_dir, exist_ok=True)
    
    # 创建简单的transforms.json
    transforms = {
        "camera_angle_x": 0.8575560450553894,
        "frames": []
    }
    
    # 生成一些示例相机位置
    for i in range(10):
        angle = i * 2 * np.pi / 10
        radius = 10
        
        # 相机位置
        cam_pos = np.array([radius * np.cos(angle), radius * np.sin(angle), 2])
        
        # 看向原点
        forward = -cam_pos / np.linalg.norm(cam_pos)
        up = np.array([0, 0, 1])
        right = np.cross(forward, up)
        up = np.cross(right, forward)
        
        # 变换矩阵
        transform_matrix = np.eye(4)
        transform_matrix[:3, 0] = right
        transform_matrix[:3, 1] = up
        transform_matrix[:3, 2] = forward
        transform_matrix[:3, 3] = cam_pos
        
        frame = {
            "file_path": f"image_{i:03d}.png",
            "transform_matrix": transform_matrix.tolist()
        }
        transforms["frames"].append(frame)
        
        # 创建假图像
        img = np.random.randint(0, 255, (400, 600, 3), dtype=np.uint8)
        cv2.imwrite(os.path.join(data_dir, f"image_{i:03d}.png"), img)
    
    # 保存transforms.json
    with open(os.path.join(data_dir, 'transforms.json'), 'w') as f:
        json.dump(transforms, f, indent=2)
    
    logger.info(f"示例数据已创建在: {data_dir}")

def main():
    """主函数 - 演示Urban Radiance Field的使用"""
    logger.info("Urban Radiance Field演示开始...")
    
    # 创建配置
    config = UrbanRFConfig(
        scene_bounds=(-20, -20, -5, 20, 20, 15),
        voxel_size=4.0,
        primitive_resolution=4,  # 减小以加快演示
        num_epochs=50
    )
    
    # 创建示例数据
    data_dir = "urban_rf_demo_data"
    create_sample_data(data_dir)
    
    # 创建数据集
    dataset = UrbanRFDataset(data_dir, config)
    
    # 创建模型
    model = UrbanRadianceField(config)
    
    # 创建训练器
    trainer = UrbanRFTrainer(model, dataset, config)
    
    # 训练模型
    trainer.train()
    
    # 渲染新视角
    novel_camera = {
        'height': 400,
        'width': 600,
        'camera_angle_x': 0.8575560450553894,
        'transform_matrix': np.array([
            [1, 0, 0, 15],
            [0, 0, -1, 0],
            [0, 1, 0, 5],
            [0, 0, 0, 1]
        ])
    }
    
    rendered_image = trainer.render_novel_view(novel_camera)
    
    # 保存结果
    plt.figure(figsize=(12, 8))
    plt.imshow(rendered_image)
    plt.title('Rendered Novel View')
    plt.axis('off')
    plt.savefig('urban_rf_novel_view.png', dpi=300, bbox_inches='tight')
    
    # 绘制训练曲线
    plt.figure(figsize=(10, 6))
    plt.plot(trainer.train_losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.grid(True)
    plt.savefig('urban_rf_training_loss.png', dpi=300, bbox_inches='tight')
    
    logger.info("Urban Radiance Field演示完成")
    logger.info(f"新视角渲染结果保存为: urban_rf_novel_view.png")
    logger.info(f"训练曲线保存为: urban_rf_training_loss.png")

if __name__ == "__main__":
    main() 