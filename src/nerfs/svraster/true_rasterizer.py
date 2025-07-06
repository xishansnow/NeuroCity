"""
真正的体素光栅化渲染器

按照 SVRaster 论文设计，实现基于投影的光栅化渲染方法，
用于高效的推理渲染，与训练时的体积渲染形成对比。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
import math


class TrueVoxelRasterizer:
    """
    真正的体素光栅化渲染器
    
    实现基于投影的光栅化渲染，按照 SVRaster 论文的设计：
    1. 体素投影到屏幕空间
    2. 深度排序和视锥剔除  
    3. 逐像素光栅化
    4. Alpha blending 合成
    
    与体积渲染的区别：
    - 不使用沿光线积分
    - 直接投影体素到屏幕
    - 使用传统图形学管线
    """
    
    def __init__(self, config):
        self.config = config
        self.background_color = torch.tensor(config.background_color)
        
    def __call__(
        self,
        voxels: Dict[str, torch.Tensor],
        camera_matrix: torch.Tensor,
        intrinsics: torch.Tensor,
        viewport_size: Tuple[int, int],
    ) -> Dict[str, torch.Tensor]:
        """
        光栅化渲染主入口
        
        Args:
            voxels: 体素数据字典
            camera_matrix: 相机变换矩阵 [4, 4] 
            intrinsics: 相机内参矩阵 [3, 3]
            viewport_size: 视口尺寸 (width, height)
            
        Returns:
            渲染结果字典
        """
        width, height = viewport_size
        device = voxels['positions'].device
        
        # 1. 投影变换
        screen_voxels = self._project_voxels_to_screen(
            voxels, camera_matrix, intrinsics, viewport_size
        )
        
        # 2. 视锥剔除
        visible_voxels = self._frustum_culling(screen_voxels, viewport_size)
        
        if len(visible_voxels) == 0:
            # 没有可见体素，返回背景
            rgb = self.background_color.expand(height, width, 3).to(device)
            depth = torch.full((height, width), self.config.far_plane, device=device)
            return {'rgb': rgb, 'depth': depth}
        
        # 3. 深度排序（后向前）
        sorted_voxels = self._depth_sort(visible_voxels)
        
        # 4. 光栅化渲染
        framebuffer = self._rasterize_voxels(sorted_voxels, viewport_size)
        
        return {
            'rgb': framebuffer['color'],
            'depth': framebuffer['depth']
        }
    
    def _project_voxels_to_screen(
        self,
        voxels: Dict[str, torch.Tensor], 
        camera_matrix: torch.Tensor,
        intrinsics: torch.Tensor,
        viewport_size: Tuple[int, int]
    ) -> List[Dict]:
        """
        将体素投影到屏幕空间
        
        Args:
            voxels: 体素数据
            camera_matrix: 相机变换矩阵
            intrinsics: 相机内参
            viewport_size: 视口尺寸
            
        Returns:
            投影后的体素列表
        """
        positions = voxels['positions']  # [N, 3]
        sizes = voxels['sizes']  # [N]
        densities = voxels['densities']  # [N]
        colors = voxels['colors']  # [N, color_dim]
        device = positions.device
        
        # Ensure camera matrices are on the correct device
        camera_matrix = camera_matrix.to(device)
        intrinsics = intrinsics.to(device)
        
        # 转换到相机坐标系
        positions_hom = torch.cat([
            positions, torch.ones(positions.shape[0], 1, device=device)
        ], dim=1)  # [N, 4]
        
        camera_positions = torch.matmul(positions_hom, camera_matrix.T)  # [N, 4]
        camera_positions = camera_positions[:, :3] / camera_positions[:, 3:4]  # [N, 3]
        
        # 投影到屏幕空间
        screen_positions = torch.matmul(camera_positions, intrinsics.T)  # [N, 3]
        screen_positions = screen_positions[:, :2] / screen_positions[:, 2:3]  # [N, 2]
        
        # 计算屏幕上的体素尺寸（简化的透视投影）
        depths = camera_positions[:, 2]
        # If sizes is [N, 3], take the first dimension; if [N], use as-is
        if sizes.dim() == 2:
            sizes_scalar = sizes[:, 0]  # Use x-dimension as representative size
        else:
            sizes_scalar = sizes
        screen_sizes = sizes_scalar * intrinsics[0, 0] / torch.clamp(depths, min=0.1)
        
        # 组装投影后的体素数据
        screen_voxels = []
        for i in range(positions.shape[0]):
            screen_voxels.append({
                'screen_pos': screen_positions[i],
                'depth': depths[i],
                'screen_size': screen_sizes[i],
                'density': densities[i],
                'color': colors[i],
                'world_pos': positions[i],
                'world_size': sizes[i],
                'voxel_idx': i
            })
        
        return screen_voxels
    
    def _frustum_culling(
        self, 
        screen_voxels: List[Dict], 
        viewport_size: Tuple[int, int]
    ) -> List[Dict]:
        """
        视锥剔除，移除屏幕外的体素
        
        Args:
            screen_voxels: 屏幕空间体素列表
            viewport_size: 视口尺寸
            
        Returns:
            可见体素列表
        """
        width, height = viewport_size
        visible_voxels = []
        
        for voxel in screen_voxels:
            x, y = voxel['screen_pos']
            size = voxel['screen_size']
            depth = voxel['depth']
            
            # 深度剔除
            if depth <= self.config.near_plane or depth >= self.config.far_plane:
                continue
            
            # 屏幕边界剔除（考虑体素尺寸）
            if (x + size >= 0 and x - size < width and 
                y + size >= 0 and y - size < height):
                visible_voxels.append(voxel)
        
        return visible_voxels
    
    def _depth_sort(self, visible_voxels: List[Dict]) -> List[Dict]:
        """
        按深度排序体素（后向前）
        
        Args:
            visible_voxels: 可见体素列表
            
        Returns:
            按深度排序的体素列表
        """
        return sorted(visible_voxels, key=lambda v: v['depth'], reverse=True)
    
    def _rasterize_voxels(
        self, 
        sorted_voxels: List[Dict], 
        viewport_size: Tuple[int, int]
    ) -> Dict[str, torch.Tensor]:
        """
        光栅化体素到像素
        
        Args:
            sorted_voxels: 按深度排序的体素列表
            viewport_size: 视口尺寸
            
        Returns:
            渲染结果帧缓冲
        """
        width, height = viewport_size
        device = sorted_voxels[0]['screen_pos'].device if sorted_voxels else torch.device('cpu')
        
        # 初始化帧缓冲
        color_buffer = self.background_color.expand(height, width, 3).to(device).clone()
        depth_buffer = torch.full((height, width), self.config.far_plane, device=device)
        alpha_buffer = torch.zeros(height, width, device=device)
        
        # 光栅化每个体素
        for voxel in sorted_voxels:
            self._rasterize_single_voxel(
                voxel, color_buffer, depth_buffer, alpha_buffer, viewport_size
            )
        
        return {
            'color': color_buffer,
            'depth': depth_buffer
        }
    
    def _rasterize_single_voxel(
        self,
        voxel: Dict,
        color_buffer: torch.Tensor,
        depth_buffer: torch.Tensor, 
        alpha_buffer: torch.Tensor,
        viewport_size: Tuple[int, int]
    ) -> None:
        """
        光栅化单个体素
        
        Args:
            voxel: 体素数据
            color_buffer: 颜色缓冲 [H, W, 3]
            depth_buffer: 深度缓冲 [H, W]
            alpha_buffer: Alpha 缓冲 [H, W]
            viewport_size: 视口尺寸
        """
        width, height = viewport_size
        
        # 体素屏幕位置和尺寸
        screen_x, screen_y = voxel['screen_pos']
        screen_size = voxel['screen_size'] 
        depth = voxel['depth']
        
        # 计算体素在屏幕上的像素范围
        half_size = screen_size * 0.5
        min_x = max(0, int(screen_x - half_size))
        max_x = min(width, int(screen_x + half_size) + 1)
        min_y = max(0, int(screen_y - half_size)) 
        max_y = min(height, int(screen_y + half_size) + 1)
        
        if min_x >= max_x or min_y >= max_y:
            return
        
        # 计算体素颜色（使用球谐函数）
        voxel_color = self._compute_voxel_color(voxel)
        
        # 计算体素 alpha
        density = voxel['density']
        if self.config.density_activation == "exp":
            sigma = torch.exp(density)
        else:
            sigma = F.relu(density)
        
        # 简化的 alpha 计算（基于体素尺寸）
        voxel_alpha = 1.0 - torch.exp(-sigma * voxel['world_size'])
        voxel_alpha = torch.clamp(voxel_alpha, 0.0, 1.0)
        
        # 对覆盖的像素进行着色
        for y in range(min_y, max_y):
            for x in range(min_x, max_x):
                # 简单的距离衰减
                dx = x - screen_x
                dy = y - screen_y
                distance = torch.sqrt(dx*dx + dy*dy)
                
                if distance <= half_size:
                    # 深度测试（对于透明体素，我们使用 alpha blending）
                    if depth < depth_buffer[y, x]:
                        # 计算像素 alpha（距离衰减）
                        pixel_alpha = voxel_alpha * (1.0 - distance / half_size)
                        pixel_alpha = torch.clamp(pixel_alpha, 0.0, 1.0)
                        
                        # Alpha blending
                        current_alpha = alpha_buffer[y, x]
                        blend_factor = pixel_alpha * (1.0 - current_alpha)
                        
                        color_buffer[y, x] = (
                            color_buffer[y, x] * (1.0 - blend_factor) + 
                            voxel_color * blend_factor
                        )
                        
                        alpha_buffer[y, x] = current_alpha + blend_factor
                        
                        # 更新深度（使用加权平均）
                        if blend_factor > 0:
                            depth_buffer[y, x] = (
                                depth_buffer[y, x] * (1.0 - blend_factor) +
                                depth * blend_factor
                            )
    
    def _compute_voxel_color(self, voxel: Dict) -> torch.Tensor:
        """
        计算体素颜色（使用球谐函数）
        
        Args:
            voxel: 体素数据
            
        Returns:
            体素颜色 [3]
        """
        # 简化实现：假设观察方向为 z 轴负方向
        view_dir = torch.tensor([0.0, 0.0, -1.0], device=voxel['color'].device)
        
        # 获取球谐系数
        sh_degree = getattr(self.config, 'sh_degree', 2)
        num_sh_coeffs = (sh_degree + 1) ** 2
        
        color_coeffs = voxel['color']  # [3 * num_sh_coeffs]
        if color_coeffs.numel() >= 3 * num_sh_coeffs:
            color_coeffs = color_coeffs[:3 * num_sh_coeffs].view(3, num_sh_coeffs)
            
            # 计算球谐基函数
            from .spherical_harmonics import eval_sh_basis
            sh_basis = eval_sh_basis(sh_degree, view_dir.unsqueeze(0))  # [1, num_sh_coeffs]
            
            # 计算颜色
            rgb = torch.matmul(sh_basis, color_coeffs.t()).squeeze(0)  # [3]
        else:
            # 退化为简单颜色
            rgb = color_coeffs[:3] if color_coeffs.numel() >= 3 else torch.ones(3, device=color_coeffs.device)
        
        # 应用激活函数
        if self.config.color_activation == "sigmoid":
            rgb = torch.sigmoid(rgb)
        elif self.config.color_activation == "tanh":
            rgb = (torch.tanh(rgb) + 1) / 2
        elif self.config.color_activation == "clamp":
            rgb = torch.clamp(rgb, 0, 1)
        
        return rgb


def create_camera_matrix(camera_pose: torch.Tensor) -> torch.Tensor:
    """
    从相机位姿创建变换矩阵
    
    Args:
        camera_pose: 相机位姿矩阵 [4, 4] (world to camera)
        
    Returns:
        相机变换矩阵
    """
    return camera_pose


def rays_to_camera_matrix(
    ray_origins: torch.Tensor,
    ray_directions: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    从光线信息估算相机矩阵（简化实现）
    
    Args:
        ray_origins: 光线起点 [N, 3]
        ray_directions: 光线方向 [N, 3]
        
    Returns:
        estimated camera matrix and intrinsics
    """
    # 简化实现：假设所有光线起点相同
    camera_center = ray_origins.mean(dim=0)
    
    # 估算相机朝向
    mean_direction = ray_directions.mean(dim=0)
    mean_direction = mean_direction / torch.norm(mean_direction)
    
    # 构建相机坐标系
    forward = -mean_direction  # 相机朝向
    up_vec = torch.tensor([0., 1., 0.], device=forward.device, dtype=forward.dtype)
    right = torch.cross(forward, up_vec, dim=0)
    right = right / torch.norm(right)
    up = torch.cross(right, forward, dim=0)
    
    # 构建相机变换矩阵
    rotation = torch.stack([right, up, forward], dim=1)  # [3, 3]
    translation = -torch.matmul(rotation.T, camera_center)  # [3]
    
    camera_matrix = torch.zeros(4, 4, device=ray_origins.device, dtype=ray_origins.dtype)
    camera_matrix[:3, :3] = rotation.T
    camera_matrix[:3, 3] = translation
    camera_matrix[3, 3] = 1.0
    
    # 简化的内参矩阵
    intrinsics = torch.tensor([
        [800, 0, 400],
        [0, 800, 300], 
        [0, 0, 1]
    ], dtype=ray_origins.dtype, device=ray_origins.device)
    
    return camera_matrix, intrinsics
