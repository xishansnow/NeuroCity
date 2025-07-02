"""
Core components for SVRaster: Adaptive sparse voxels and rasterization.

This module implements the main components of the SVRaster method including
adaptive sparse voxel representation, custom rasterizer, and the main model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
import logging
import os
from typing import Optional
from .utils.rendering_utils import ray_direction_dependent_ordering
import math

logger = logging.getLogger(__name__)

@dataclass
class SVRasterConfig:
    """
    SVRaster 模型的配置类

    这个类定义了 SVRaster 模型的所有超参数和配置选项，包括场景表示、体素属性、
    自适应分配、光栅化设置、渲染参数和优化选项。

    主要配置项包括：
    - 场景表示：八叉树的最大层级、基础分辨率、场景边界
    - 体素属性：密度激活函数、颜色激活函数、球谐函数度数
    - 自适应分配：细分阈值、剪枝阈值、每层最大体素数
    - 光栅化：每体素光线采样数、深度剥离层数、Morton 排序
    - 渲染：背景颜色、近远平面
    - 优化：视角相关颜色、不透明度正则化

    使用方法：
        config = SVRasterConfig(
            max_octree_levels=16,
            base_resolution=64,
            scene_bounds=(-1.0, -1.0, -1.0, 1.0, 1.0, 1.0),
            density_activation="exp",
            color_activation="sigmoid",
            sh_degree=2
        )
    """

    # Scene representation
    max_octree_levels: int = 16  # Maximum octree levels (65536^3 resolution)
    base_resolution: int = 64    # Base grid resolution
    scene_bounds: tuple[float, float, float, float, float, float] = (
        -1.0,
        -1.0,
        -1.0,
        1.0,
        1.0,
        1.0,
    )

    # Voxel properties
    density_activation: str = "exp"     # Density activation function
    color_activation: str = "sigmoid"   # Color activation function
    sh_degree: int = 2                  # Spherical harmonics degree

    # Adaptive allocation
    subdivision_threshold: float = 0.01  # Threshold for voxel subdivision
    pruning_threshold: float = 0.001     # Threshold for voxel pruning
    max_voxels_per_level: int = 1000000  # Maximum voxels per octree level

    # Rasterization
    ray_samples_per_voxel: int = 8       # Number of samples per voxel along ray
    depth_peeling_layers: int = 4        # Number of depth peeling layers
    morton_ordering: bool = True         # Use Morton ordering for depth sorting

    # Rendering
    background_color: tuple[float, float, float] = (0.0, 0.0, 0.0)
    near_plane: float = 0.1
    far_plane: float = 100.0

    # Optimization
    use_view_dependent_color: bool = True
    use_opacity_regularization: bool = True
    opacity_reg_weight: float = 0.01
    use_ssim_loss: bool = True
    ssim_loss_weight: float = 0.1
    use_distortion_loss: bool = True
    distortion_loss_weight: float = 0.01
    use_pointwise_rgb_loss: bool = False
    pointwise_rgb_loss_weight: float = 0.1

def eval_sh_basis(degree: int, dirs: torch.Tensor) -> torch.Tensor:
    """
    计算球谐函数基（支持 0~3 阶），输入方向 shape [..., 3]，输出 shape [..., num_sh_coeffs]
    """
    # dirs: [..., 3], 必须归一化
    x, y, z = dirs.unbind(-1)
    sh_list = []
    sh_list.append(torch.ones_like(x))  # l=0, m=0
    if degree >= 1:
        sh_list += [y, z, x]  # l=1, m=-1,0,1
    if degree >= 2:
        sh_list += [
            x * y,  # l=2, m=-2
            y * z,  # l=2, m=-1
            3 * z ** 2 - 1,  # l=2, m=0
            x * z,  # l=2, m=1
            x ** 2 - y ** 2  # l=2, m=2
        ]
    if degree >= 3:
        sh_list += [
            y * (3 * x ** 2 - y ** 2),  # l=3, m=-3
            x * y * z,  # l=3, m=-2
            y * (5 * z ** 2 - 1),  # l=3, m=-1
            z * (5 * z ** 2 - 3),  # l=3, m=0
            x * (5 * z ** 2 - 1),  # l=3, m=1
            (x ** 2 - y ** 2) * z,  # l=3, m=2
            x * (x ** 2 - 3 * y ** 2)  # l=3, m=3
        ]
    return torch.stack(sh_list, dim=-1)

class AdaptiveSparseVoxels(nn.Module):
    """
    自适应稀疏体素表示类
    
    这个类实现了基于八叉树的多分辨率自适应稀疏体素表示，是 SVRaster 的核心组件。
    它管理不同八叉树层级的稀疏体素，只存储叶子节点，不存储完整的八叉树结构。
    
    主要功能：
    1. 初始化基础体素网格
    2. 根据梯度或密度进行自适应细分
    3. 基于密度阈值进行体素剪枝
    4. 计算 Morton 码用于深度排序
    5. 管理多层级体素的参量值
    
    核心特性：
    - 多分辨率表示：支持 16 层八叉树，最高分辨率可达 65536³
    - 自适应细分：根据重建误差或梯度自动细分体素
    - 稀疏存储：只存储有意义的体素，节省内存
    - Morton 排序：使用 Morton 码进行深度排序，避免伪影
    
    Morton 码容量说明：
    - 原始实现：每个坐标分量 10 位，总共支持 1024³ ≈ 10 亿个位置
    - 改进实现：每个坐标分量 21 位，总共支持 2097151³ ≈ 9.2×10¹⁸ 个位置
    - 容量提升：约 9.2×10⁹ 倍，完全满足 SVRaster 的 65536³ 需求
    - 性能优化：使用向量化操作，批量计算 Morton 码，提高性能
    
    使用方法：
        config = SVRasterConfig()
        sparse_voxels = AdaptiveSparseVoxels(config)
        
        # 获取所有体素
        voxels = sparse_voxels.get_all_voxels()
        
        # 自适应细分
        subdivision_mask = torch.rand(1000) > 0.5
        sparse_voxels.subdivide_voxels(subdivision_mask, level_idx=0)
        
        # 体素剪枝
        sparse_voxels.prune_voxels(threshold=0.001)
        
        # 获取统计信息
        total_count = sparse_voxels.get_total_voxel_count()
    """

    def __init__(self, config: SVRasterConfig):
        super().__init__()
        self.config = config

        # Initialize voxel storage
        self.voxel_positions = nn.ParameterList()  # Voxel center positions
        self.voxel_sizes = nn.ParameterList()      # Voxel sizes (level-dependent)
        self.voxel_densities = nn.ParameterList()  # Density values
        self.voxel_colors = nn.ParameterList()     # Color/SH coefficients
        self.voxel_levels: list[torch.Tensor] = []                     # Octree levels
        self.voxel_morton_codes: list[torch.Tensor] = []               # Morton codes for sorting

        # Scene bounds
        self.register_buffer('scene_min', torch.tensor(config.scene_bounds[:3]))
        self.register_buffer('scene_max', torch.tensor(config.scene_bounds[3:]))
        # Calculate scene size after buffers are registered
        self.scene_size: torch.Tensor = torch.tensor([0.0, 0.0, 0.0])  # Placeholder

        # SH coefficients count
        self.num_sh_coeffs = (config.sh_degree + 1) ** 2

        # Initialize with base level voxels
        self._initialize_base_voxels()

    def _initialize_base_voxels(self):
        """
        初始化基础体素网格

        在场景边界内创建规则的基础体素网格，覆盖整个场景空间。
        每个体素包含位置、大小、密度和颜色/球谐函数系数等参量。
        """
        base_res = self.config.base_resolution

        # Calculate scene size now that buffers are properly initialized
        scene_min_tensor = torch.as_tensor(self.scene_min)
        scene_max_tensor = torch.as_tensor(self.scene_max)
        self.scene_size = scene_max_tensor - scene_min_tensor

        # Create regular grid at base level
        x = torch.linspace(0, 1, base_res + 1)[:-1] + 0.5 / base_res
        y = torch.linspace(0, 1, base_res + 1)[:-1] + 0.5 / base_res
        z = torch.linspace(0, 1, base_res + 1)[:-1] + 0.5 / base_res

        xx, yy, zz = torch.meshgrid(x, y, z, indexing='ij') # 创建网格
        positions = torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3)

        # Convert to world coordinates
        positions = positions * self.scene_size + scene_min_tensor

        # Initialize parameters
        num_voxels = positions.shape[0]
        voxel_size = float(self.scene_size.max()) / base_res

        self.voxel_positions.append(nn.Parameter(positions))
        self.voxel_sizes.append(nn.Parameter(torch.full((num_voxels, ), voxel_size)))
        self.voxel_densities.append(nn.Parameter(torch.randn(num_voxels) * 0.1))

        # Initialize SH coefficients (RGB + SH)
        color_dim = 3 * self.num_sh_coeffs
        self.voxel_colors.append(nn.Parameter(torch.randn(num_voxels, color_dim) * 0.1))

        # set levels and Morton codes
        self.voxel_levels.append(torch.zeros(num_voxels, dtype=torch.int))
        self.voxel_morton_codes.append(self._compute_morton_codes(positions, 0))

    def _compute_morton_codes(self, positions: torch.Tensor, level: int) -> torch.Tensor:
        """
        计算体素位置的 Morton 码

        Morton 码是一种空间填充曲线，用于将 3D 坐标编码为 1D 索引，
        便于进行深度排序和空间局部性优化。

        Args:
            positions: 体素中心位置 [N, 3]
            level: 八叉树层级

        Returns:
            Morton 码 [N]
        """
        # Normalize positions to [0, 1]
        scene_min_tensor = torch.as_tensor(self.scene_min)
        normalized_pos = (positions - scene_min_tensor) / self.scene_size
        
        # Discretize to grid coordinates
        grid_res = self.config.base_resolution * (2 ** level)
        grid_coords = (normalized_pos * grid_res).long()
        grid_coords = torch.clamp(grid_coords, 0, grid_res - 1)
        
        # Compute Morton codes using vectorized operations for better performance
        morton_codes = self._morton_encode_3d_vectorized(grid_coords)
        
        return morton_codes
    
    def _morton_encode_3d_vectorized(self, coords: torch.Tensor) -> torch.Tensor:
        """
        向量化的 3D Morton 码编码
        
        使用 PyTorch 张量操作进行批量 Morton 码计算，提高性能。
        
        Args:
            coords: 坐标张量 [N, 3]
            
        Returns:
            Morton 码张量 [N]
        """
        x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
        
        # 确保坐标在合理范围内
        max_coord = 0x1fffff  # 21 位最大值
        x = torch.clamp(x, 0, max_coord)
        y = torch.clamp(y, 0, max_coord)
        z = torch.clamp(z, 0, max_coord)
        
        # 位交错函数
        def part1by2_vectorized(n):
            n = n & 0x1fffff
            n = (n ^ (n << 32)) & 0x1f00000000ffff
            n = (n ^ (n << 16)) & 0x1f0000ff0000ff
            n = (n ^ (n << 8)) & 0x100f00f00f00f00f
            n = (n ^ (n << 4)) & 0x10c30c30c30c30c3
            n = (n ^ (n << 2)) & 0x1249249249249249
            return n
        
        # 计算 Morton 码
        morton_x = part1by2_vectorized(x)
        morton_y = part1by2_vectorized(y)
        morton_z = part1by2_vectorized(z)
        
        return (morton_z << 2) + (morton_y << 1) + morton_x

    def _morton_encode_3d(self, x: int, y: int, z: int) -> int:
        """
        将单个 3D 坐标编码为 Morton 码（兼容性方法）
        
        使用位交错技术将 3 个坐标分量编码为单个整数，
        保持空间局部性。支持更高的分辨率。

        Args:
            x, y, z: 3D 坐标分量

        Returns:
            Morton 码
        """
        def part1by2(n):
            # 支持更大的坐标值，使用更高效的位交错
            # 将 21 位输入扩展为 63 位输出（3 * 21 = 63）
            n &= 0x1fffff  # 21 位掩码，支持 0-2097151
            n = (n ^ (n << 32)) & 0x1f00000000ffff
            n = (n ^ (n << 16)) & 0x1f0000ff0000ff
            n = (n ^ (n << 8)) & 0x100f00f00f00f00f
            n = (n ^ (n << 4)) & 0x10c30c30c30c30c3
            n = (n ^ (n << 2)) & 0x1249249249249249
            return n
        
        # 确保输入在合理范围内
        x = max(0, min(x, 0x1fffff))
        y = max(0, min(y, 0x1fffff))
        z = max(0, min(z, 0x1fffff))
        
        return (part1by2(z) << 2) + (part1by2(y) << 1) + part1by2(x)

    def subdivide_voxels(self, subdivision_mask: torch.Tensor, level_idx: int):
        """
        根据细分掩码细分体素

        将选中的体素细分为 8 个子体素，每个子体素的大小是父体素的一半。
        细分后的子体素被添加到下一层级，父体素被移除。

        Args:
            subdivision_mask: 细分掩码，True 表示需要细分的体素
            level_idx: 当前层级索引
        """
        if not subdivision_mask.any():
            return

        # Get voxels to subdivide
        parent_positions = self.voxel_positions[level_idx][subdivision_mask]
        parent_sizes = self.voxel_sizes[level_idx][subdivision_mask]
        parent_densities = self.voxel_densities[level_idx][subdivision_mask]
        parent_colors = self.voxel_colors[level_idx][subdivision_mask]

        # Create 8 child voxels for each parent
        num_parents = parent_positions.shape[0]
        child_size = parent_sizes / 2

        # Child offsets (8 corners of cube)
        offsets = torch.tensor([
            [-1, -1, -1], [1, -1, -1], [-1, 1, -1], [1, 1, -1], [-1, -1, 1], [1, -1, 1], [-1, 1, 1], [1, 1, 1]
        ], dtype=torch.float32, device=parent_positions.device) * 0.25

        # Generate child positions
        child_positions = []
        child_sizes_list = []
        child_densities_list = []
        child_colors_list = []

        for i in range(num_parents):
            parent_pos = parent_positions[i]
            size = child_size[i]

            for offset in offsets:
                child_pos = parent_pos + offset * parent_sizes[i]
                child_positions.append(child_pos)
                child_sizes_list.append(size)
                child_densities_list.append(parent_densities[i])
                child_colors_list.append(parent_colors[i])

        # Convert to tensors
        child_positions = torch.stack(child_positions)
        child_sizes_tensor = torch.stack(child_sizes_list)
        child_densities_tensor = torch.stack(child_densities_list)
        child_colors_tensor = torch.stack(child_colors_list)

        # Add to next level
        new_level = level_idx + 1
        if new_level >= len(self.voxel_positions):
            # Create new level
            self.voxel_positions.append(nn.Parameter(child_positions))
            self.voxel_sizes.append(nn.Parameter(child_sizes_tensor))
            self.voxel_densities.append(nn.Parameter(child_densities_tensor))
            self.voxel_colors.append(nn.Parameter(child_colors_tensor))
            self.voxel_levels.append(torch.full((child_positions.shape[0], ), new_level))
            self.voxel_morton_codes.append(self._compute_morton_codes(child_positions, new_level))
        else:
            # Append to existing level
            old_positions = self.voxel_positions[new_level].data
            old_sizes = self.voxel_sizes[new_level].data
            old_densities = self.voxel_densities[new_level].data
            old_colors = self.voxel_colors[new_level].data

            self.voxel_positions[new_level].data = torch.cat([old_positions, child_positions])
            self.voxel_sizes[new_level].data = torch.cat([old_sizes, child_sizes_tensor])
            self.voxel_densities[new_level].data = torch.cat(
                [old_densities,
                child_densities_tensor],
            )
            self.voxel_colors[new_level].data = torch.cat([old_colors, child_colors_tensor])

            # Update levels and Morton codes
            new_levels = torch.full((child_positions.shape[0], ), new_level)
            self.voxel_levels[new_level] = torch.cat([self.voxel_levels[new_level], new_levels])
            new_morton = self._compute_morton_codes(child_positions, new_level)
            self.voxel_morton_codes[new_level] = torch.cat(
                [self.voxel_morton_codes[new_level],
                new_morton],
            )

        # Remove subdivided parent voxels
        keep_mask = ~subdivision_mask
        self.voxel_positions[level_idx].data = self.voxel_positions[level_idx].data[keep_mask]
        self.voxel_sizes[level_idx].data = self.voxel_sizes[level_idx].data[keep_mask]
        self.voxel_densities[level_idx].data = self.voxel_densities[level_idx].data[keep_mask]
        self.voxel_colors[level_idx].data = self.voxel_colors[level_idx].data[keep_mask]
        self.voxel_levels[level_idx] = self.voxel_levels[level_idx][keep_mask]
        self.voxel_morton_codes[level_idx] = self.voxel_morton_codes[level_idx][keep_mask]

    def prune_voxels(self, threshold: Optional[float] = None):
        """
        移除低密度体素

        根据密度阈值移除不重要的体素，减少内存使用并提高渲染效率。
        密度低于阈值的体素被认为是透明的，对最终渲染结果贡献很小。

        Args:
            threshold: 密度阈值，None 时使用配置中的默认值
        """
        if threshold is None:
            threshold = self.config.pruning_threshold

        for level_idx in range(len(self.voxel_densities)):
            if self.config.density_activation == "exp":
                densities = torch.exp(self.voxel_densities[level_idx])
            else:
                densities = torch.relu(self.voxel_densities[level_idx])

            keep_mask = densities > threshold

            if keep_mask.any() and not keep_mask.all():
                self.voxel_positions[level_idx].data = self.voxel_positions[level_idx].data[keep_mask]
                self.voxel_sizes[level_idx].data = self.voxel_sizes[level_idx].data[keep_mask]
                self.voxel_densities[level_idx].data = self.voxel_densities[level_idx].data[keep_mask]
                self.voxel_colors[level_idx].data = self.voxel_colors[level_idx].data[keep_mask]
                self.voxel_levels[level_idx] = self.voxel_levels[level_idx][keep_mask]
                self.voxel_morton_codes[level_idx] = self.voxel_morton_codes[level_idx][keep_mask]

    def get_all_voxels(self) -> dict[str, torch.Tensor]:
        """
        获取所有层级的体素数据

        将所有层级的体素数据合并为一个字典，包含位置、大小、密度、
        颜色、层级和 Morton 码。

        Returns:
            包含所有体素数据的字典
        """
        all_positions = []
        all_sizes = []
        all_densities = []
        all_colors = []
        all_levels = []
        all_morton_codes = []

        for level_idx in range(len(self.voxel_positions)):
            all_positions.append(self.voxel_positions[level_idx].data)
            all_sizes.append(self.voxel_sizes[level_idx].data)
            all_densities.append(self.voxel_densities[level_idx].data)
            all_colors.append(self.voxel_colors[level_idx].data)
            all_levels.append(self.voxel_levels[level_idx])
            all_morton_codes.append(self.voxel_morton_codes[level_idx])

        return {
            'positions': torch.cat(all_positions),
            'sizes': torch.cat(all_sizes),
            'densities': torch.cat(all_densities),
            'colors': torch.cat(all_colors),
            'levels': torch.cat(all_levels),
            'morton_codes': torch.cat(all_morton_codes)
        }

    def get_total_voxel_count(self) -> int:
        """
        获取所有层级的体素总数

        Returns:
            体素总数
        """
        return sum(pos.shape[0] for pos in self.voxel_positions)

    def add_voxels(self, positions, sizes, densities, colors, level=0):
        """
        向指定层级添加体素。
        Args:
            positions: [N, 3] 体素中心
            sizes: [N] 体素尺寸
            densities: [N] 体素密度
            colors: [N, 3 * (sh_degree+1)^2] 体素颜色/SH系数
            level: int, 层级索引
        """
        self.voxel_positions[level].data = torch.cat([self.voxel_positions[level].data, positions])
        self.voxel_sizes[level].data = torch.cat([self.voxel_sizes[level].data, sizes])
        self.voxel_densities[level].data = torch.cat([self.voxel_densities[level].data, densities])
        self.voxel_colors[level].data = torch.cat([self.voxel_colors[level].data, colors])
        self.voxel_levels[level] = torch.cat([self.voxel_levels[level], torch.full((positions.shape[0], ), level)])
        self.voxel_morton_codes[level] = self._compute_morton_codes(positions, level)

class VoxelRasterizer(nn.Module):
    """
    体素光栅化器

    这个类实现了高效的稀疏体素渲染器，使用基于光线方向的自适应 Morton 排序
    进行正确的深度排序，避免了高斯溅射中的弹出伪影。

    主要功能：
    1. 光线-体素相交检测
    2. 基于 Morton 码的深度排序
    3. 体素内光线采样和积分
    4. 体积渲染方程求解

    核心特性：
    - 自适应深度排序：根据平均光线方向进行 Morton 排序
    - 高效相交检测：使用 AABB 进行快速光线-体素相交测试
    - 体积渲染：支持透明度和颜色累积
    - 背景处理：正确处理背景颜色贡献

    使用方法：
        config = SVRasterConfig()
        rasterizer = VoxelRasterizer(config)

        # 渲染图像
        outputs = rasterizer(
            voxels=voxel_data,
            ray_origins=ray_origins,
            ray_directions=ray_directions
        )

        rgb = outputs['rgb']  # 渲染的 RGB 颜色
        depth = outputs['depth']  # 深度值
        weights = outputs['weights']  # 权重
    """

    def __init__(self, config: SVRasterConfig):
        super().__init__()
        self.config = config

    def forward(
        self,
        voxels: dict[str, torch.Tensor],
        ray_origins: torch.Tensor,
        ray_directions: torch.Tensor,
        camera_params: Optional[dict[str, torch.Tensor]] = None,
    ) -> dict[str, torch.Tensor]:
        """
        前向传播：体素光栅化

        Args:
            voxels: 体素数据字典，包含位置、大小、密度、颜色等
            ray_origins: 光线起点 [B, 3]
            ray_directions: 光线方向 [B, 3]
            camera_params: 相机参数（可选）

        Returns:
            渲染结果字典，包含 RGB、深度和权重
        """
        batch_size = ray_origins.shape[0]
        device = ray_origins.device

        # Sort voxels by mean ray direction for better depth ordering
        mean_ray_direction = ray_directions.mean(dim=0)
        sorted_voxels = self._sort_voxels_by_ray_direction(voxels, mean_ray_direction)

        # Process each ray
        rgb_list = []
        depth_list = []
        weights_list = []

        for i in range(batch_size):
            ray_o = ray_origins[i]
            ray_d = ray_directions[i]

            # Find ray-voxel intersections
            intersections = self._ray_voxel_intersections(ray_o, ray_d, sorted_voxels)

            # Render ray through intersected voxels
            rgb, depth, weights = self._render_ray(ray_o, ray_d, intersections, sorted_voxels)

            rgb_list.append(rgb)
            depth_list.append(depth)
            weights_list.append(weights)

        # Stack results
        rgb = torch.stack(rgb_list)
        depth = torch.stack(depth_list)
        weights = torch.stack(weights_list)

        return {
            'rgb': rgb,
            'depth': depth,
            'weights': weights
        }

    def _sort_voxels_by_ray_direction(
        self,
        voxels: dict[str, torch.Tensor],
        mean_ray_direction: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        根据平均光线方向对体素进行排序

        将体素中心投影到平均光线方向上，按投影距离排序，
        确保正确的深度顺序。

        Args:
            voxels: 体素数据
            mean_ray_direction: 平均光线方向

        Returns:
            排序后的体素数据
        """
        if not self.config.morton_ordering:
            return voxels

        # 使用真正的视角相关 Morton 排序
        sort_indices = ray_direction_dependent_ordering(
            voxels['positions'],
            voxels['morton_codes'], 
            mean_ray_direction
        )

        # Sort all voxel attributes using the returned indices
        sorted_voxels = {}
        for key, value in voxels.items():
            sorted_voxels[key] = value[sort_indices]

        return sorted_voxels

    def _ray_voxel_intersections(
        self,
        ray_o: torch.Tensor,
        ray_d: torch.Tensor,
        voxels: dict[str, torch.Tensor],
    ) -> list[tuple[int, float, float]]:
        """
        计算光线与体素的相交

        使用向量化的 AABB（轴对齐包围盒）进行快速光线-体素相交测试，
        返回相交的体素索引和相交参数。

        Args:
            ray_o: 光线起点
            ray_d: 光线方向
            voxels: 体素数据

        Returns:
            相交列表，每个元素为（体素索引，t_near, t_far）
        """
        positions = voxels['positions']
        sizes = voxels['sizes']

        # 计算所有体素的 AABB
        half_sizes = (sizes / 2).unsqueeze(-1)
        box_mins = positions - half_sizes
        box_maxs = positions + half_sizes

        # 向量化的光线-AABB 相交检测
        t_mins = (box_mins - ray_o) / ray_d
        t_maxs = (box_maxs - ray_o) / ray_d

        t1 = torch.min(t_mins, t_maxs)
        t2 = torch.max(t_mins, t_maxs)

        t_near = torch.max(t1, dim=-1)[0]
        t_far = torch.min(t2, dim=-1)[0]

        # 找到有效的相交
        valid_mask = (t_far > t_near) & (t_far > 0)
        valid_indices = torch.where(valid_mask)[0]

        # 构建相交列表
        intersections = []
        for idx in valid_indices:
            intersections.append((idx.item(), t_near[idx].item(), t_far[idx].item()))

        # Sort by t_near
        intersections.sort(key=lambda x: x[1])
        return intersections

    def _render_ray(
        self,
        ray_o: torch.Tensor,
        ray_d: torch.Tensor,
        intersections: list[tuple[int, float, float]],
        voxels: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        渲染单条光线（支持体积积分、球谐函数、视角相关颜色）
        """
        device = ray_o.device
        sh_degree = getattr(self.config, 'sh_degree', 2)
        num_sh_coeffs = (sh_degree + 1) ** 2

        if not intersections:
            # 确保返回的张量有梯度
            bg_color = torch.tensor(self.config.background_color, device=device, requires_grad=True)
            far_depth = torch.tensor([self.config.far_plane], device=device, requires_grad=True)
            zero_weight = torch.tensor([0.0], device=device, requires_grad=True)
            return bg_color, far_depth, zero_weight

        rgb_acc = torch.zeros(3, device=device, requires_grad=True)
        depth_acc = torch.zeros(1, device=device, requires_grad=True)
        weight_acc = torch.zeros(1, device=device, requires_grad=True)
        transmittance = 1.0

        for voxel_idx, t_near, t_far in intersections:
            # 多点采样
            n_samples = self.config.ray_samples_per_voxel
            t_samples = torch.linspace(t_near, t_far, n_samples, device=device)
            sample_points = ray_o + ray_d * t_samples.unsqueeze(-1)  # [n_samples, 3]
            # 所有采样点方向都用 ray_d
            dirs = ray_d.expand(n_samples, 3)

            # 获取体素 SH 系数
            color_coeffs = voxels['colors'][voxel_idx]  # [3 * num_sh_coeffs]
            color_coeffs = color_coeffs.view(3, num_sh_coeffs)  # [3, num_sh_coeffs]
            sh_basis = eval_sh_basis(sh_degree, dirs)  # [n_samples, num_sh_coeffs]
            rgb_samples = torch.matmul(sh_basis, color_coeffs.t())  # [n_samples, 3]

            # 激活函数
            if self.config.color_activation == "sigmoid":
                rgb_samples = torch.sigmoid(rgb_samples)
            elif self.config.color_activation == "tanh":
                rgb_samples = (torch.tanh(rgb_samples) + 1) / 2
            elif self.config.color_activation == "clamp":
                rgb_samples = torch.clamp(rgb_samples, 0, 1)

            # 密度
            density = voxels['densities'][voxel_idx]
            if self.config.density_activation == "exp":
                sigma = torch.exp(density)
            else:
                sigma = F.relu(density)
            # 假设体素内密度均匀
            sigmas = sigma.expand(n_samples)

            # 体积渲染积分
            delta_t = (t_far - t_near) / n_samples
            alphas = 1.0 - torch.exp(-sigmas * delta_t)  # [n_samples]
            trans = torch.cumprod(torch.cat([torch.ones(1, device=device), 1 - alphas + 1e-8]), dim=0)[:-1]
            weights = alphas * trans  # [n_samples]

            rgb = torch.sum(weights.unsqueeze(-1) * rgb_samples, dim=0)  # [3]
            depth = torch.sum(weights * t_samples)
            weight_sum = torch.sum(weights)

            rgb_acc = rgb_acc + transmittance * rgb
            depth_acc = depth_acc + transmittance * depth
            weight_acc = weight_acc + transmittance * weight_sum

            transmittance = transmittance * (1 - weight_sum)
            if transmittance < 0.01:
                break

        # 背景
        if transmittance > 0:
            bg_color = torch.tensor(self.config.background_color, device=device, requires_grad=True)
            rgb_acc = rgb_acc + transmittance * bg_color

        # 输出 shape 兼容
        return rgb_acc, depth_acc, weight_acc

class SVRasterModel(nn.Module):
    """
    SVRaster 主模型

    这个类将自适应稀疏体素表示和体素光栅化器组合成完整的 SVRaster 模型。
    它是整个系统的核心，负责管理体素表示、执行渲染和提供训练接口。

    主要功能：
    1. 管理自适应稀疏体素表示
    2. 执行体素光栅化渲染
    3. 提供自适应细分接口
    4. 统计体素分布信息
    5. 可视化八叉树结构

    核心特性：
    - 端到端训练：支持梯度反向传播和参数更新
    - 自适应优化：支持基于梯度的体素细分
    - 内存高效：稀疏表示减少内存占用
    - 实时渲染：优化的光栅化算法支持实时渲染

    使用方法：
        config = SVRasterConfig()
        model = SVRasterModel(config)

        # 前向传播
        outputs = model(ray_origins, ray_directions)
        rgb = outputs['rgb']

        # 自适应细分
        subdivision_criteria = torch.rand(1000)
        model.adaptive_subdivision(subdivision_criteria)

        # 获取统计信息
        stats = model.get_voxel_statistics()

        # 可视化结构
        model.visualize_structure('output_dir')
    """

    def __init__(self, config: SVRasterConfig):
        super().__init__()
        self.config = config

        # Core components
        self.sparse_voxels = AdaptiveSparseVoxels(config)
        self.rasterizer = VoxelRasterizer(config)

        # Background color
        self.register_buffer('background_color', torch.tensor(config.background_color))

    def forward(
        self,
        ray_origins: torch.Tensor,
        ray_directions: torch.Tensor,
        camera_params: Optional[dict[str, torch.Tensor]] = None,
    ) -> dict[str, torch.Tensor]:
        """
        前向传播：渲染图像

        Args:
            ray_origins: 光线起点 [B, 3]
            ray_directions: 光线方向 [B, 3]
            camera_params: 相机参数（可选）

        Returns:
            渲染结果字典
        """
        # Get current voxel representation
        voxels = self.sparse_voxels.get_all_voxels()

        # Rasterize voxels
        render_output = self.rasterizer(
            voxels,
            ray_origins,
            ray_directions,
            camera_params
        )

        return render_output

    def adaptive_subdivision(self, subdivision_criteria: torch.Tensor) -> None:
        """
        根据细分标准对体素进行自适应细分，提高重建质量。
        细分标准通常基于梯度幅度或重建误差。
        """
        for level_idx in range(len(self.sparse_voxels.voxel_positions)):
            if level_idx >= self.config.max_octree_levels - 1:
                break
            # 应用细分
            self.sparse_voxels.subdivide_voxels(subdivision_criteria[level_idx], level_idx)

    def get_voxel_statistics(self) -> dict[str, int]:
        """
        获取体素分布统计信息

        Returns:
            包含体素统计信息的字典
        """
        stats = {
            'total_voxels': self.sparse_voxels.get_total_voxel_count(),
            'num_levels': len(self.sparse_voxels.voxel_positions)
        }

        for level_idx in range(len(self.sparse_voxels.voxel_positions)):
            stats[f'level_{level_idx}_voxels'] = self.sparse_voxels.voxel_positions[level_idx].shape[0]

        return stats

    def visualize_structure(self, output_dir: str) -> None:
        """
        可视化八叉树结构

        将各层级的体素位置和大小保存为 numpy 数组，用于后续可视化。

        Args:
            output_dir: 输出目录
        """
        os.makedirs(output_dir, exist_ok=True)

        # Save voxel positions and sizes for each level
        for level_idx in range(len(self.sparse_voxels.voxel_positions)):
            positions = self.sparse_voxels.voxel_positions[level_idx].data.cpu().numpy()
            sizes = self.sparse_voxels.voxel_sizes[level_idx].data.cpu().numpy()

            np.save(
                os.path.join(output_dir, f'level_{level_idx}_positions.npy'),
                positions
            )
            np.save(
                os.path.join(output_dir, f'level_{level_idx}_sizes.npy'),
                sizes
            )

    def add_voxels(self, positions, sizes, densities, colors, level=0):
        """
        向指定层级添加体素。
        Args:
            positions: [N, 3] 体素中心
            sizes: [N] 体素尺寸
            densities: [N] 体素密度
            colors: [N, 3 * (sh_degree+1)^2] 体素颜色/SH系数
            level: int, 层级索引
        """
        self.sparse_voxels.add_voxels(positions, sizes, densities, colors, level)
    
    def render_image(
        self,
        camera_pose: torch.Tensor,
        camera_intrinsics: torch.Tensor,
        image_size: tuple[int, int],
        device: Optional[torch.device] = None,
    ) -> dict[str, torch.Tensor]:
        """
        渲染完整图像
        
        Args:
            camera_pose: 相机姿态矩阵 [4, 4] (世界坐标系到相机坐标系)
            camera_intrinsics: 相机内参矩阵 [3, 3]
            image_size: 图像尺寸 (height, width)
            device: 计算设备
            
        Returns:
            渲染结果字典，包含 RGB 图像、深度图等
        """
        if device is None:
            device = next(self.parameters()).device
        
        height, width = image_size
        
        # 生成像素坐标网格
        u = torch.arange(width, device=device, dtype=torch.float32)
        v = torch.arange(height, device=device, dtype=torch.float32)
        u, v = torch.meshgrid(u, v, indexing='xy')  # [H, W]
        
        # 转换为归一化坐标
        u_norm = (u - camera_intrinsics[0, 2]) / camera_intrinsics[0, 0]  # (u - cx) / fx
        v_norm = (v - camera_intrinsics[1, 2]) / camera_intrinsics[1, 1]  # (v - cy) / fy
        
        # 创建光线方向 (相机坐标系)
        ray_directions_cam = torch.stack([u_norm, v_norm, torch.ones_like(u_norm)], dim=-1)  # [H, W, 3]
        
        # 转换到世界坐标系
        rotation = camera_pose[:3, :3]  # [3, 3]
        translation = camera_pose[:3, 3]  # [3]
        
        # 重塑并转换光线方向
        ray_directions_cam_flat = ray_directions_cam.reshape(-1, 3)  # [H*W, 3]
        ray_directions_world = torch.matmul(ray_directions_cam_flat, rotation.t())  # [H*W, 3]
        ray_directions_world = ray_directions_world / torch.norm(ray_directions_world, dim=-1, keepdim=True)
        
        # 光线起点 (相机位置)
        ray_origins = translation.unsqueeze(0).expand(height * width, 3)  # [H*W, 3]
        
        # 渲染
        outputs = self(ray_origins, ray_directions_world)
        
        # 重塑为图像格式
        rgb_image = outputs['rgb'].reshape(height, width, 3)  # [H, W, 3]
        depth_image = outputs['depth'].reshape(height, width, 1)  # [H, W, 1]
        weights_image = outputs['weights'].reshape(height, width, 1)  # [H, W, 1]
        
        return {
            'rgb': rgb_image,
            'depth': depth_image,
            'weights': weights_image,
            'ray_origins': ray_origins.reshape(height, width, 3),
            'ray_directions': ray_directions_world.reshape(height, width, 3)
        }
    
    def render_novel_view(
        self,
        camera_pose: torch.Tensor,
        camera_intrinsics: torch.Tensor,
        image_size: tuple[int, int],
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        渲染新视角图像（简化接口）
        
        Args:
            camera_pose: 相机姿态矩阵 [4, 4]
            camera_intrinsics: 相机内参矩阵 [3, 3]
            image_size: 图像尺寸 (height, width)
            device: 计算设备
            
        Returns:
            RGB 图像 [H, W, 3]
        """
        with torch.no_grad():
            result = self.render_image(camera_pose, camera_intrinsics, image_size, device)
            return result['rgb']
    
    def create_camera_pose(
        self,
        position: torch.Tensor,
        look_at: torch.Tensor,
        up: torch.Tensor = torch.tensor([0.0, 1.0, 0.0]),
    ) -> torch.Tensor:
        """
        创建相机姿态矩阵
        
        Args:
            position: 相机位置 [3]
            look_at: 看向的点 [3]
            up: 上方向向量 [3]
            
        Returns:
            相机姿态矩阵 [4, 4]
        """
        # 计算相机坐标系
        forward = look_at - position
        forward = forward / torch.norm(forward)
        
        right = torch.cross(forward, up)
        right = right / torch.norm(right)
        
        up_cam = torch.cross(right, forward)
        
        # 构建旋转矩阵
        rotation = torch.stack([right, up_cam, -forward], dim=1)  # [3, 3]
        
        # 构建变换矩阵
        pose = torch.eye(4, device=position.device)
        pose[:3, :3] = rotation
        pose[:3, 3] = position
        
        return pose
    
    def create_camera_intrinsics(
        self,
        focal_length: float,
        image_size: tuple[int, int],
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        创建相机内参矩阵
        
        Args:
            focal_length: 焦距
            image_size: 图像尺寸 (height, width)
            device: 计算设备
            
        Returns:
            相机内参矩阵 [3, 3]
        """
        if device is None:
            device = next(self.parameters()).device
        
        height, width = image_size
        cx, cy = width / 2, height / 2
        
        intrinsics = torch.tensor([
            [focal_length, 0, cx],
            [0, focal_length, cy],
            [0, 0, 1]
        ], device=device, dtype=torch.float32)
        
        return intrinsics
    
    def render_circular_path(
        self,
        center: torch.Tensor,
        radius: float,
        num_views: int,
        image_size: tuple[int, int],
        focal_length: float,
        elevation: float = 0.0,
        device: Optional[torch.device] = None,
    ) -> list[torch.Tensor]:
        """
        渲染圆形路径上的多个视角
        
        Args:
            center: 场景中心点 [3]
            radius: 相机距离中心的半径
            num_views: 视角数量
            image_size: 图像尺寸 (height, width)
            focal_length: 焦距
            elevation: 仰角（弧度）
            device: 计算设备
            
        Returns:
            RGB 图像列表
        """
        if device is None:
            device = next(self.parameters()).device
        
        images = []
        intrinsics = self.create_camera_intrinsics(focal_length, image_size, device)
        
        for i in range(num_views):
            # 计算相机位置
            angle = 2 * torch.pi * i / num_views
            x = center[0] + radius * torch.cos(torch.tensor(angle, device=device))
            z = center[2] + radius * torch.sin(torch.tensor(angle, device=device))
            y = center[1] + radius * torch.sin(torch.tensor(elevation, device=device))
            
            position = torch.tensor([x, y, z], device=device)
            
            # 创建相机姿态
            pose = self.create_camera_pose(position, center)
            
            # 渲染图像
            image = self.render_novel_view(pose, intrinsics, image_size, device)
            images.append(image)
        
        return images

class SVRasterLoss(nn.Module):
    """
    SVRaster 损失函数

    这个类定义了 SVRaster 模型的损失函数，包括 RGB 重建损失、
    深度损失和可选的不透明度正则化。

    主要损失项：
    1. RGB 重建损失：预测颜色与目标颜色的 MSE 损失
    2. 深度损失：预测深度与目标深度的损失（可选）
    3. 不透明度正则化：防止过度透明或过度不透明

    使用方法：
        config = SVRasterConfig()
        loss_fn = SVRasterLoss(config)

        # 计算损失
        losses = loss_fn(outputs, targets, model)
        total_loss = losses['total_loss']
    """

    def __init__(self, config: SVRasterConfig):
        super().__init__()
        self.config = config

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
        model: SVRasterModel,
    ) -> dict[str, torch.Tensor]:
        """
        前向传播：计算损失

        Args:
            outputs: 模型输出
            targets: 目标数据
            model: SVRaster 模型实例

        Returns:
            损失字典
        """
        loss_dict = {}

        # RGB reconstruction loss
        rgb_loss = F.mse_loss(outputs['rgb'], targets['rgb'])
        loss_dict['rgb_loss'] = rgb_loss

        # Initialize all loss components to zero
        loss_dict['ssim_loss'] = torch.tensor(0.0, device=outputs['rgb'].device)
        loss_dict['opacity_reg'] = torch.tensor(0.0, device=outputs['rgb'].device)
        loss_dict['distortion_loss'] = torch.tensor(0.0, device=outputs['rgb'].device)
        loss_dict['pointwise_rgb_loss'] = torch.tensor(0.0, device=outputs['rgb'].device)

        # SSIM loss for better perceptual quality
        if self.config.use_ssim_loss:
            # Reshape to image format for SSIM computation
            pred_rgb = outputs['rgb'].view(-1, 3)  # [N, 3]
            target_rgb = targets['rgb'].view(-1, 3)  # [N, 3]
            
            # Use a simple SSIM approximation for efficiency
            # Convert to grayscale for SSIM
            pred_gray = 0.299 * pred_rgb[:, 0] + 0.587 * pred_rgb[:, 1] + 0.114 * pred_rgb[:, 2]
            target_gray = 0.299 * target_rgb[:, 0] + 0.587 * target_rgb[:, 1] + 0.114 * target_rgb[:, 2]
            
            # Simple SSIM approximation
            mu1 = torch.mean(pred_gray)
            mu2 = torch.mean(target_gray)
            
            sigma1_sq = torch.var(pred_gray)
            sigma2_sq = torch.var(target_gray)
            sigma12 = torch.mean((pred_gray - mu1) * (target_gray - mu2))
            
            c1 = 0.01 ** 2
            c2 = 0.03 ** 2
            
            ssim = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / \
                   ((mu1**2 + mu2**2 + c1) * (sigma1_sq + sigma2_sq + c2))
            
            # SSIM loss (1 - SSIM since we want to minimize loss)
            ssim_loss = 1.0 - ssim
            loss_dict['ssim_loss'] = ssim_loss * self.config.ssim_loss_weight

        # Optional opacity regularization
        if self.config.use_opacity_regularization:
            opacity_reg = 0
            for level_idx in range(len(model.sparse_voxels.voxel_densities)):
                densities = model.sparse_voxels.voxel_densities[level_idx]
                if self.config.density_activation == "exp":
                    opacity = torch.exp(densities)
                else:
                    opacity = F.relu(densities)
                opacity_reg += torch.mean(opacity)

            loss_dict['opacity_reg'] = opacity_reg * self.config.opacity_reg_weight

        # Distortion loss for better geometry
        if self.config.use_distortion_loss and 'weights' in outputs and 'depth' in outputs:
            distortion_loss = self._compute_distortion_loss(outputs['weights'], outputs['depth'])
            loss_dict['distortion_loss'] = distortion_loss * self.config.distortion_loss_weight

        # Pointwise RGB loss (per-pixel L1 loss)
        if self.config.use_pointwise_rgb_loss:
            pointwise_loss = F.l1_loss(outputs['rgb'], targets['rgb'])
            loss_dict['pointwise_rgb_loss'] = pointwise_loss * self.config.pointwise_rgb_loss_weight

        # Update total loss to include all components
        loss_dict['total_loss'] = (loss_dict['rgb_loss'] + 
                                  loss_dict['ssim_loss'] + 
                                  loss_dict['opacity_reg'] + 
                                  loss_dict['distortion_loss'] + 
                                  loss_dict['pointwise_rgb_loss'])

        return loss_dict

    def _compute_distortion_loss(self, weights: torch.Tensor, depths: torch.Tensor) -> torch.Tensor:
        """
        计算畸变损失，鼓励权重分布更加集中，提升几何质量
        
        Args:
            weights: [N, 1] 光线权重
            depths: [N, 1] 深度值
            
        Returns:
            畸变损失值
        """
        # 归一化权重
        weights_norm = weights / (weights.sum(dim=0, keepdim=True) + 1e-8)
        
        # 计算质心
        center_of_mass = (weights_norm * depths).sum(dim=0, keepdim=True)
        
        # 计算方差（畸变度量）
        variance = (weights_norm * (depths - center_of_mass) ** 2).sum(dim=0)
        
        return variance.mean()