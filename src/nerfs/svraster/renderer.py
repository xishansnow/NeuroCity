"""
SVRaster Renderer - 与 TrueVoxelRasterizer 紧密耦合

这个渲染器专门用于推理阶段，与 TrueVoxelRasterizer 紧密耦合，
使用光栅化进行快速渲染，符合 SVRaster 论文的设计理念。

渲染器负责：
1. 加载训练好的模型进行推理
2. 与 TrueVoxelRasterizer 配合进行光栅化渲染
3. 生成新视点的高质量图像
4. 支持批量渲染和实时渲染
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
from pathlib import Path
import logging
import json
from tqdm import tqdm
import os

logger = logging.getLogger(__name__)

# 尝试导入 imageio
try:
    import imageio
    IMAGEIO_AVAILABLE = True
except ImportError:
    IMAGEIO_AVAILABLE = False
    logger.warning("imageio not available for video rendering")

from .core import SVRasterModel, SVRasterConfig
from .true_rasterizer import TrueVoxelRasterizer

# 创建 TrueVoxelRasterizer 的简单配置类
from dataclasses import dataclass

@dataclass
class TrueVoxelRasterizerConfig:
    """TrueVoxelRasterizer 配置"""
    background_color: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    near_plane: float = 0.1
    far_plane: float = 100.0
    density_activation: str = "exp"
    color_activation: str = "sigmoid"
    sh_degree: int = 2


@dataclass  
class SVRasterRendererConfig:
    """SVRaster 渲染器配置 - 专门为光栅化推理设计"""
    
    # 渲染质量设置
    image_width: int = 800
    image_height: int = 600
    render_batch_size: int = 4096
    render_chunk_size: int = 1024
    
    # 光栅化参数（与 TrueVoxelRasterizer 紧密相关）
    background_color: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    use_alpha_blending: bool = True
    depth_threshold: float = 1e-6
    
    # 渲染质量控制
    max_rays_per_batch: int = 8192
    use_hierarchical_sampling: bool = True
    
    # 输出设置
    output_format: str = "png"
    save_depth: bool = False
    save_alpha: bool = False
    
    # 优化设置
    use_cached_features: bool = True
    enable_gradient_checkpointing: bool = False


class SVRasterRenderer:
    """
    SVRaster 渲染器 - 与 TrueVoxelRasterizer 紧密耦合
    
    专门负责：
    - 加载训练好的模型进行推理
    - 使用光栅化进行快速渲染
    - 生成新视点的图像
    - 支持批量渲染和实时渲染
    """
    
    def __init__(
        self,
        model: SVRasterModel,
        rasterizer: TrueVoxelRasterizer,
        config: SVRasterRendererConfig
    ):
        self.model = model
        self.rasterizer = rasterizer  # 紧密耦合的体素光栅化器
        self.config = config
        
        # 确保模型处于评估模式
        self.model.eval()
        
        # 设置设备
        self.device = next(self.model.parameters()).device
        
        logger.info(f"SVRasterRenderer initialized with TrueVoxelRasterizer coupling")
        logger.info(f"Model device: {self.device}")
        logger.info(f"Render config: {self.config}")
    
    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        rasterizer_config: Optional[TrueVoxelRasterizerConfig] = None,
        renderer_config: Optional[SVRasterRendererConfig] = None
    ) -> SVRasterRenderer:
        """
        从检查点加载渲染器
        """
        # 加载检查点
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # 创建模型配置
        if 'model_config' in checkpoint:
            model_config = checkpoint['model_config']
        else:
            model_config = SVRasterConfig()
            logger.warning("No model config found in checkpoint, using default")
        
        # 创建模型
        model = SVRasterModel(model_config)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # 移到正确的设备
        if torch.cuda.is_available():
            model = model.cuda()
        
        # 创建光栅化器
        if rasterizer_config is None:
            rasterizer_config = TrueVoxelRasterizerConfig()
        
        rasterizer = TrueVoxelRasterizer(rasterizer_config)
        
        # 创建渲染器配置
        if renderer_config is None:
            renderer_config = SVRasterRendererConfig()
        
        logger.info(f"Loading renderer from checkpoint: {checkpoint_path}")
        
        return cls(model, rasterizer, renderer_config)
    
    def render_image(
        self,
        camera_pose: torch.Tensor,
        intrinsics: torch.Tensor,
        width: Optional[int] = None,
        height: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """
        渲染单张图像 - 使用光栅化
        """
        width = width or self.config.image_width
        height = height or self.config.image_height
        
        with torch.no_grad():
            # 生成光线
            rays_o, rays_d = self._generate_rays(camera_pose, intrinsics, width, height)
            
            # 首先使用模型获取体素数据
            # 这里需要先从模型中提取体素
            voxels = self._extract_voxels_from_model()
            
            # 使用光栅化器进行渲染
            # 这里光栅化器与渲染器紧密耦合
            render_result = self.rasterizer(
                voxels=voxels,
                camera_matrix=self._pose_to_camera_matrix(camera_pose),
                intrinsics=intrinsics,
                viewport_size=(width, height)
            )
            
            # 重整形状为图像格式
            rgb = render_result['rgb']
            depth = render_result['depth']
            
            result = {
                'rgb': rgb,
                'depth': depth,
                'raw_output': render_result
            }
            
            # 添加额外信息
            if 'alpha' in render_result:
                result['alpha'] = render_result['alpha']
            
            if 'weights' in render_result:
                result['weights'] = render_result['weights']
            
            return result
    
    def render_batch(
        self,
        camera_poses: torch.Tensor,
        intrinsics: torch.Tensor,
        width: Optional[int] = None,
        height: Optional[int] = None
    ) -> List[Dict[str, torch.Tensor]]:
        """
        批量渲染多张图像
        """
        results = []
        
        logger.info(f"Rendering batch of {camera_poses.shape[0]} images")
        
        for i in tqdm(range(camera_poses.shape[0]), desc="Rendering images"):
            # 选择相机参数
            pose = camera_poses[i]
            intrinsic = intrinsics[i] if intrinsics.ndim > 2 else intrinsics
            
            # 渲染单张图像
            result = self.render_image(pose, intrinsic, width, height)
            results.append(result)
        
        return results
    
    def render_video(
        self,
        camera_trajectory: torch.Tensor,
        intrinsics: torch.Tensor,
        output_path: str,
        fps: int = 30,
        width: Optional[int] = None,
        height: Optional[int] = None
    ) -> None:
        """
        渲染视频序列
        """
        if not IMAGEIO_AVAILABLE:
            raise ImportError("imageio is required for video rendering. Install with: pip install imageio")
        
        width = width or self.config.image_width
        height = height or self.config.image_height
        
        frames = []
        
        logger.info(f"Rendering {len(camera_trajectory)} frames for video...")
        
        for i, pose in enumerate(tqdm(camera_trajectory, desc="Rendering video frames")):
            result = self.render_image(pose, intrinsics, width, height)
            
            # 转换为 numpy 格式 (0-255)
            rgb_np = (result['rgb'].cpu().numpy() * 255).astype(np.uint8)
            frames.append(rgb_np)
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 保存视频
        imageio.mimsave(output_path, frames, fps=fps)
        logger.info(f"Video saved to {output_path}")
    
    def render_spiral_video(
        self,
        center: torch.Tensor,
        radius: float,
        num_frames: int,
        intrinsics: torch.Tensor,
        output_path: str,
        fps: int = 30,
        height_offset: float = 0.0,
        width: Optional[int] = None,
        height: Optional[int] = None
    ) -> None:
        """
        渲染螺旋轨迹视频
        """
        # 生成螺旋轨迹
        trajectory = self._generate_spiral_trajectory(
            center, radius, num_frames, height_offset
        )
        
        # 渲染视频
        self.render_video(trajectory, intrinsics, output_path, fps, width, height)
    
    def save_renders(
        self,
        renders: List[Dict[str, torch.Tensor]],
        output_dir: str,
        prefix: str = "render"
    ) -> None:
        """
        保存渲染结果到文件
        """
        os.makedirs(output_dir, exist_ok=True)
        
        for i, render in enumerate(renders):
            # 保存 RGB 图像
            rgb_path = os.path.join(output_dir, f"{prefix}_{i:04d}.png")
            self._save_image(render['rgb'], rgb_path)
            
            # 保存深度图（如果启用）
            if self.config.save_depth and 'depth' in render:
                depth_path = os.path.join(output_dir, f"{prefix}_{i:04d}_depth.png")
                self._save_depth_image(render['depth'], depth_path)
            
            # 保存 alpha 通道（如果启用）
            if self.config.save_alpha and 'alpha' in render:
                alpha_path = os.path.join(output_dir, f"{prefix}_{i:04d}_alpha.png")
                self._save_image(render['alpha'], alpha_path)
        
        logger.info(f"Saved {len(renders)} renders to {output_dir}")
    
    def _extract_voxels_from_model(self) -> Dict[str, torch.Tensor]:
        """
        从模型中提取体素数据
        
        Returns:
            体素数据字典
        """
        # 这里需要从 SVRasterModel 中提取体素
        # 实际实现需要根据模型的具体结构调整
        with torch.no_grad():
            # 从 SVRasterModel 中提取体素
            if hasattr(self.model, 'voxels'):
                voxels_obj = self.model.voxels
                
                # 提取所有层级的体素数据
                all_positions = []
                all_sizes = []
                all_densities = []
                all_colors = []
                all_morton_codes = []
                
                for level_idx in range(len(voxels_obj.voxel_positions)):
                    all_positions.append(voxels_obj.voxel_positions[level_idx])
                    all_sizes.append(voxels_obj.voxel_sizes[level_idx])
                    all_densities.append(voxels_obj.voxel_densities[level_idx])
                    all_colors.append(voxels_obj.voxel_colors[level_idx])
                    all_morton_codes.append(voxels_obj.voxel_morton_codes[level_idx])
                
                # 合并所有层级
                if all_positions:
                    positions = torch.cat(all_positions, dim=0)
                    sizes = torch.cat(all_sizes, dim=0)
                    densities = torch.cat(all_densities, dim=0)
                    colors = torch.cat(all_colors, dim=0)
                    morton_codes = torch.cat(all_morton_codes, dim=0)
                    
                    return {
                        'positions': positions.float(),
                        'sizes': sizes.float(),
                        'densities': densities.float(),
                        'colors': colors.float(),
                        'morton_codes': morton_codes.long()
                    }
                else:
                    return self._create_dummy_voxels()
            else:
                # 如果没有找到体素数据，创建一个简单的测试网格
                return self._create_dummy_voxels()
    
    def _create_dummy_voxels(self) -> Dict[str, torch.Tensor]:
        """
        创建测试用的体素数据
        """
        # 创建一个简单的体素网格用于测试
        n_voxels = 100
        
        positions = torch.randn(n_voxels, 3, device=self.device) * 2.0
        sizes = torch.ones(n_voxels, device=self.device) * 0.1
        densities = torch.randn(n_voxels, device=self.device)
        colors = torch.randn(n_voxels, 3, device=self.device)
        morton_codes = torch.randint(0, 1000000, (n_voxels,), device=self.device)
        
        return {
            'positions': positions,
            'sizes': sizes,
            'densities': densities,
            'colors': colors,
            'morton_codes': morton_codes
        }
    
    def _pose_to_camera_matrix(self, pose: torch.Tensor) -> torch.Tensor:
        """
        将相机位姿转换为相机矩阵
        
        Args:
            pose: 相机位姿矩阵 [4, 4]
            
        Returns:
            相机变换矩阵 [4, 4]
        """
        # 假设输入的 pose 是 world-to-camera 变换
        return pose
    
    def _generate_rays(
        self,
        camera_pose: torch.Tensor,
        intrinsics: torch.Tensor,
        width: int,
        height: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        根据相机参数生成光线
        """
        device = camera_pose.device
        
        # 生成像素坐标
        i, j = torch.meshgrid(
            torch.linspace(0, width - 1, width, device=device),
            torch.linspace(0, height - 1, height, device=device),
            indexing='ij'
        )
        i = i.t()
        j = j.t()
        
        # 提取内参
        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]
        
        # 计算归一化设备坐标
        dirs = torch.stack([
            (i - cx) / fx,
            -(j - cy) / fy,
            -torch.ones_like(i)
        ], -1)
        
        # 转换到世界坐标系
        rotation = camera_pose[:3, :3]
        translation = camera_pose[:3, 3]
        
        rays_d = torch.sum(dirs[..., None, :] * rotation, -1)
        rays_o = translation.expand(rays_d.shape)
        
        return rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)
    
    def _generate_spiral_trajectory(
        self,
        center: torch.Tensor,
        radius: float,
        num_frames: int,
        height_offset: float = 0.0
    ) -> torch.Tensor:
        """
        生成螺旋相机轨迹
        """
        device = center.device
        
        # 生成角度
        angles = torch.linspace(0, 2 * np.pi, num_frames, device=device)
        
        # 生成位置
        positions = []
        for angle in angles:
            x = center[0] + radius * torch.cos(angle)
            y = center[1] + radius * torch.sin(angle)
            z = center[2] + height_offset
            positions.append(torch.tensor([x, y, z], device=device))
        
        # 生成相机姿态
        poses = []
        for pos in positions:
            # 计算朝向中心的旋转矩阵
            forward = F.normalize(center - pos, dim=0)
            up = torch.tensor([0, 0, 1], device=device, dtype=torch.float32)
            right = F.normalize(torch.cross(forward, up), dim=0)
            up = torch.cross(right, forward)
            
            # 构建变换矩阵
            pose = torch.eye(4, device=device, dtype=torch.float32)
            pose[:3, 0] = right
            pose[:3, 1] = up
            pose[:3, 2] = -forward
            pose[:3, 3] = pos
            
            poses.append(pose)
        
        return torch.stack(poses)
    
    def _save_image(self, image: torch.Tensor, path: str) -> None:
        """
        保存图像到文件
        """
        if not IMAGEIO_AVAILABLE:
            logger.warning("imageio not available, cannot save image")
            return
        
        # 转换为 numpy 格式
        if image.dim() == 3 and image.shape[2] == 1:
            image = image.squeeze(2)
        
        image_np = (image.cpu().numpy() * 255).astype(np.uint8)
        imageio.imwrite(path, image_np)
    
    def _save_depth_image(self, depth: torch.Tensor, path: str) -> None:
        """
        保存深度图到文件
        """
        if not IMAGEIO_AVAILABLE:
            logger.warning("imageio not available, cannot save depth image")
            return
        
        # 归一化深度值
        depth_np = depth.cpu().numpy().squeeze()
        depth_np = (depth_np - depth_np.min()) / (depth_np.max() - depth_np.min())
        depth_np = (depth_np * 255).astype(np.uint8)
        
        imageio.imwrite(path, depth_np)
    
    def get_memory_usage(self) -> Dict[str, float]:
        """
        获取内存使用情况
        """
        if not torch.cuda.is_available():
            return {}
        
        return {
            'allocated_gb': torch.cuda.memory_allocated() / 1024**3,
            'reserved_gb': torch.cuda.memory_reserved() / 1024**3,
            'max_allocated_gb': torch.cuda.max_memory_allocated() / 1024**3,
        }
    
    def clear_cache(self) -> None:
        """
        清理缓存
        """
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 注意：TrueVoxelRasterizer 当前没有 clear_cache 方法
        # 如果需要，可以在 TrueVoxelRasterizer 中添加该方法


# 使用示例函数
def create_svraster_renderer(
    checkpoint_path: str,
    rasterizer_config: Optional[TrueVoxelRasterizerConfig] = None,
    renderer_config: Optional[SVRasterRendererConfig] = None
) -> SVRasterRenderer:
    """
    创建 SVRaster 渲染器的便捷函数
    """
    return SVRasterRenderer.from_checkpoint(
        checkpoint_path=checkpoint_path,
        rasterizer_config=rasterizer_config,
        renderer_config=renderer_config
    )


def render_demo_images(
    renderer: SVRasterRenderer,
    num_views: int = 8,
    output_dir: str = "demo_renders"
) -> None:
    """
    渲染演示图像
    """
    # 创建演示相机轨迹
    center = torch.tensor([0.0, 0.0, 0.0], device=renderer.device)
    radius = 3.0
    
    # 生成相机姿态
    poses = renderer._generate_spiral_trajectory(center, radius, num_views)
    
    # 创建内参矩阵
    intrinsics = torch.tensor([
        [800, 0, 400],
        [0, 800, 300],
        [0, 0, 1]
    ], dtype=torch.float32, device=renderer.device)
    
    # 批量渲染
    renders = renderer.render_batch(poses, intrinsics)
    
    # 保存结果
    renderer.save_renders(renders, output_dir, "demo")
    
    logger.info(f"Demo rendering completed. Results saved to {output_dir}")


if __name__ == "__main__":
    # 使用示例
    print("SVRaster Renderer - 与 TrueVoxelRasterizer 紧密耦合")
    print("主要功能：")
    print("1. 从检查点加载模型")
    print("2. 使用光栅化进行快速渲染")
    print("3. 支持图像、批量和视频渲染")
    print("4. 与 TrueVoxelRasterizer 紧密耦合优化")
