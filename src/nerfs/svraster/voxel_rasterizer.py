"""
真正的体素光栅化渲染器

按照 SVRaster 论文设计，实现基于投影的光栅化渲染方法，
用于高效的推理渲染，与训练时的体积渲染形成对比。

支持 CPU 和 GPU (CUDA) 两种实现方式，自动选择最优方案。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional
import math
import logging

logger = logging.getLogger(__name__)

# 尝试导入 CUDA 扩展
try:
    import nerfs.svraster.cuda.renderer.voxel_rasterizer_gpu as voxel_rasterizer_gpu_module

    # 检查关键函数是否可用
    if voxel_rasterizer_gpu_module.CUDA_AVAILABLE:
        CUDA_AVAILABLE = True
        voxel_rasterizer_cuda = voxel_rasterizer_gpu_module.get_voxel_rasterization_function()
        logger.info("VoxelRasterizer CUDA 扩展加载成功")
    else:
        raise AttributeError("CUDA extension not available")
except (ImportError, AttributeError):
    CUDA_AVAILABLE = False
    voxel_rasterizer_cuda = None
    logger.warning("VoxelRasterizer CUDA 扩展不可用，将使用 CPU 实现")


class VoxelRasterizer:
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

    def __init__(self, config, use_cuda: Optional[bool] = None):
        """
        初始化体素光栅化渲染器

        Args:
            config: 配置对象
            use_cuda: 是否强制使用 CUDA，None 表示自动选择
        """
        self.config = config

        # 确保 background_color 是 3 元素的 RGB 张量
        bg_color = torch.tensor(config.background_color, dtype=torch.float32)

        if bg_color.numel() >= 3:
            self.background_color = bg_color.flatten()[:3]  # 只取前3个元素 (RGB)
        else:
            # 如果少于3个元素，用0填充
            self.background_color = torch.zeros(3, dtype=torch.float32)
            self.background_color[: bg_color.numel()] = bg_color.flatten()

        # 确定是否使用 CUDA
        if use_cuda is None:
            self.use_cuda = CUDA_AVAILABLE and torch.cuda.is_available()
        else:
            self.use_cuda = use_cuda and CUDA_AVAILABLE and torch.cuda.is_available()

        if self.use_cuda:
            logger.info("使用 CUDA 加速的体素光栅化渲染器")
        else:
            logger.info("使用 CPU 体素光栅化渲染器")

    def __call__(
        self,
        voxels: dict[str, torch.Tensor],
        camera_matrix: torch.Tensor,
        intrinsics: torch.Tensor,
        viewport_size: tuple[int, int],
    ) -> dict[str, torch.Tensor]:
        """
        光栅化渲染主入口

        Args:
            voxels: 体素数据字典，包含 positions, sizes, densities, colors
            camera_matrix: 相机变换矩阵 [4, 4]
            intrinsics: 相机内参矩阵 [3, 3]
            viewport_size: 视口尺寸 (width, height)

        Returns:
            渲染结果字典
        """
        if self.use_cuda:
            return self._render_cuda(voxels, camera_matrix, intrinsics, viewport_size)
        else:
            return self._render_cpu(voxels, camera_matrix, intrinsics, viewport_size)

    def _render_cuda(
        self,
        voxels: dict[str, torch.Tensor],
        camera_matrix: torch.Tensor,
        intrinsics: torch.Tensor,
        viewport_size: tuple[int, int],
    ) -> dict[str, torch.Tensor]:
        """
        CUDA 加速的渲染实现
        """
        device = voxels["positions"].device

        # 确保数据在 GPU 上
        voxels_gpu = {k: v.to(device) for k, v in voxels.items()}
        camera_matrix = camera_matrix.to(device)
        intrinsics = intrinsics.to(device)
        viewport_tensor = torch.tensor(viewport_size, dtype=torch.int32, device=device)
        background_color = self.background_color.to(device, dtype=torch.float32)

        # 重塑 colors 为正确的形状
        if voxels_gpu["colors"].dim() == 3:  # [N, 3, num_sh_coeffs]
            N, C, num_sh_coeffs = voxels_gpu["colors"].shape
            voxels_gpu["colors"] = voxels_gpu["colors"].reshape(
                N, C * num_sh_coeffs
            )  # [N, 3 * num_sh_coeffs]

        # 调用 CUDA 扩展
        if voxel_rasterizer_cuda is not None:
            # 确保 background_color 是 [3] 的 1D float32 张量
            bg_color_tensor = background_color.flatten().to(torch.float32)[:3]
            if bg_color_tensor.shape[0] != 3:
                raise ValueError(
                    f"background_color must have at least 3 elements (got {bg_color_tensor.shape})"
                )

            rgb, depth = voxel_rasterizer_cuda(
                voxels_gpu["positions"],
                voxels_gpu["sizes"],
                voxels_gpu["densities"],
                voxels_gpu["colors"],
                camera_matrix,
                intrinsics,
                viewport_tensor,
                self.config.near_plane,
                self.config.far_plane,
                bg_color_tensor,
                self.config.density_activation,
                self.config.color_activation,
                getattr(self.config, "sh_degree", 2),
            )
        else:
            raise RuntimeError("CUDA extension not available")

        return {"rgb": rgb, "depth": depth}

    def _render_cpu(
        self,
        voxels: dict[str, torch.Tensor],
        camera_matrix: torch.Tensor,
        intrinsics: torch.Tensor,
        viewport_size: tuple[int, int],
    ) -> dict[str, torch.Tensor]:
        """
        CPU 渲染实现（原有逻辑）
        """
        width, height = viewport_size
        device = voxels["positions"].device

        # 重塑 colors 为正确的形状（与 CUDA 路径保持一致）
        voxels_cpu = {k: v.clone() for k, v in voxels.items()}
        if voxels_cpu["colors"].dim() == 3:  # [N, 3, num_sh_coeffs]
            N, C, num_sh_coeffs = voxels_cpu["colors"].shape
            voxels_cpu["colors"] = voxels_cpu["colors"].reshape(
                N, C * num_sh_coeffs
            )  # [N, 3 * num_sh_coeffs]

        # 1. 投影变换
        screen_voxels = self._project_voxels_to_screen(
            voxels_cpu, camera_matrix, intrinsics, viewport_size
        )

        # 2. 视锥剔除
        visible_voxels = self._frustum_culling(screen_voxels, viewport_size)

        if len(visible_voxels) == 0:
            # 没有可见体素，返回背景
            rgb = self.background_color.expand(height, width, -1).to(device)
            depth = torch.full((height, width), self.config.far_plane, device=device)
            return {"rgb": rgb, "depth": depth}

        # 3. 深度排序（后向前）
        sorted_voxels = self._depth_sort(visible_voxels)

        # 4. 光栅化渲染
        framebuffer = self._rasterize_voxels(sorted_voxels, viewport_size)

        return {"rgb": framebuffer["color"], "depth": framebuffer["depth"]}

    def _project_voxels_to_screen(
        self,
        voxels: dict[str, torch.Tensor],
        camera_matrix: torch.Tensor,
        intrinsics: torch.Tensor,
        viewport_size: tuple[int, int],
    ) -> list[dict[str, torch.Tensor]]:
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
        positions = voxels["positions"]  # [N, 3]
        sizes = voxels["sizes"]  # [N]
        densities = voxels["densities"]  # [N]
        colors = voxels["colors"]  # [N, color_dim]
        device = positions.device

        # Ensure camera matrices are on the correct device and dtype
        camera_matrix = camera_matrix.to(device, dtype=torch.float32)
        intrinsics = intrinsics.to(device, dtype=torch.float32)

        # 转换到相机坐标系
        positions_hom = torch.cat(
            [positions, torch.ones(positions.shape[0], 1, device=device)], dim=1
        )  # [N, 4]

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
            screen_voxels.append(
                {
                    "screen_pos": screen_positions[i],
                    "depth": depths[i],
                    "screen_size": screen_sizes[i],
                    "density": densities[i],
                    "color": colors[i],
                    "world_pos": positions[i],
                    "world_size": sizes[i],
                    "voxel_idx": i,
                }
            )

        return screen_voxels

    def _frustum_culling(
        self, screen_voxels: list[dict[str, torch.Tensor]], viewport_size: tuple[int, int]
    ) -> list[dict[str, torch.Tensor]]:
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
            x, y = voxel["screen_pos"]
            size = voxel["screen_size"]
            depth = voxel["depth"]

            # 深度剔除
            if depth <= self.config.near_plane or depth >= self.config.far_plane:
                continue

            # 屏幕边界剔除（考虑体素尺寸）
            if x + size >= 0 and x - size < width and y + size >= 0 and y - size < height:
                visible_voxels.append(voxel)

        return visible_voxels

    def _depth_sort(
        self, visible_voxels: list[dict[str, torch.Tensor]]
    ) -> list[dict[str, torch.Tensor]]:
        """
        按深度排序体素（后向前）

        Args:
            visible_voxels: 可见体素列表

        Returns:
            按深度排序的体素列表
        """
        return sorted(visible_voxels, key=lambda v: v["depth"].item(), reverse=True)

    def _rasterize_voxels(
        self, sorted_voxels: list[dict[str, torch.Tensor]], viewport_size: tuple[int, int]
    ) -> dict[str, torch.Tensor]:
        """
        光栅化体素到像素

        Args:
            sorted_voxels: 按深度排序的体素列表
            viewport_size: 视口尺寸

        Returns:
            渲染结果帧缓冲
        """
        width, height = viewport_size
        device = sorted_voxels[0]["screen_pos"].device if sorted_voxels else torch.device("cpu")

        # 初始化帧缓冲
        color_buffer = self.background_color.expand(height, width, -1).to(device).clone()
        depth_buffer = torch.full((height, width), self.config.far_plane, device=device)
        alpha_buffer = torch.zeros(height, width, device=device)

        # 光栅化每个体素
        for voxel in sorted_voxels:
            self._rasterize_single_voxel(
                voxel, color_buffer, depth_buffer, alpha_buffer, viewport_size
            )

        return {"color": color_buffer, "depth": depth_buffer}

    def _rasterize_single_voxel(
        self,
        voxel: dict[str, torch.Tensor],
        color_buffer: torch.Tensor,
        depth_buffer: torch.Tensor,
        alpha_buffer: torch.Tensor,
        viewport_size: tuple[int, int],
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
        screen_x, screen_y = voxel["screen_pos"]
        screen_size = voxel["screen_size"]
        depth = voxel["depth"]

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
        density = voxel["density"]
        if self.config.density_activation == "exp":
            sigma = torch.exp(density)
        else:
            sigma = F.relu(density)

        # 简化的 alpha 计算（基于体素尺寸）
        voxel_alpha = 1.0 - torch.exp(-sigma * voxel["world_size"])
        voxel_alpha = torch.clamp(voxel_alpha, 0.0, 1.0)

        # 对覆盖的像素进行着色
        for y in range(min_y, max_y):
            for x in range(min_x, max_x):
                # 简单的距离衰减
                dx = x - screen_x
                dy = y - screen_y
                distance = torch.sqrt(dx * dx + dy * dy)

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
                            color_buffer[y, x] * (1.0 - blend_factor) + voxel_color * blend_factor
                        )

                        alpha_buffer[y, x] = current_alpha + blend_factor

                        # 更新深度（使用加权平均）
                        if blend_factor > 0:
                            depth_buffer[y, x] = (
                                depth_buffer[y, x] * (1.0 - blend_factor) + depth * blend_factor
                            )

    def _compute_voxel_color(self, voxel: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        计算体素颜色（使用球谐函数）

        Args:
            voxel: 体素数据

        Returns:
            体素颜色 [3]
        """
        # 简化实现：假设观察方向为 z 轴负方向
        view_dir = torch.tensor([0.0, 0.0, -1.0], device=voxel["color"].device)

        # 获取球谐系数
        sh_degree = getattr(self.config, "sh_degree", 2)
        num_sh_coeffs = (sh_degree + 1) ** 2

        color_coeffs = voxel["color"]  # [3 * num_sh_coeffs]

        if color_coeffs.numel() >= 3 * num_sh_coeffs:
            color_coeffs = color_coeffs[: 3 * num_sh_coeffs].view(3, num_sh_coeffs)

            # 计算球谐基函数
            from .spherical_harmonics import eval_sh_basis

            sh_basis = eval_sh_basis(sh_degree, view_dir.unsqueeze(0))  # [1, num_sh_coeffs]

            # 计算颜色
            rgb = torch.matmul(sh_basis, color_coeffs.t()).squeeze(0)  # [3]
        else:
            # 退化为简单颜色
            rgb = (
                color_coeffs[:3]
                if color_coeffs.numel() >= 3
                else torch.ones(3, device=color_coeffs.device)
            )

        # 应用激活函数
        if self.config.color_activation == "sigmoid":
            rgb = torch.sigmoid(rgb)
        elif self.config.color_activation == "tanh":
            rgb = (torch.tanh(rgb) + 1) / 2
        elif self.config.color_activation == "clamp":
            rgb = torch.clamp(rgb, 0, 1)

        rgb = rgb[:3]  # 保证 shape 始终为 [3]
        return rgb


def create_camera_matrix(camera_pose: torch.Tensor) -> torch.Tensor:
    """
    从相机位姿创建变换矩阵

    Args:
        camera_pose: 相机位姿矩阵 [4, 4] (world to camera)

    Returns:
        相机变换矩阵
    """
    if CUDA_AVAILABLE:
        try:
            import nerfs.svraster.cuda.renderer.voxel_rasterizer_gpu as vrg

            create_camera_matrix_func = vrg.get_create_camera_matrix_function()
            if create_camera_matrix_func is not None:
                return create_camera_matrix_func(camera_pose)
        except (ImportError, AttributeError):
            pass
    return camera_pose


def rays_to_camera_matrix(
    ray_origins: torch.Tensor, ray_directions: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    从光线信息估算相机矩阵（简化实现）

    Args:
        ray_origins: 光线起点 [N, 3]
        ray_directions: 光线方向 [N, 3]

    Returns:
        estimated camera matrix and intrinsics
    """
    if CUDA_AVAILABLE:
        try:
            import nerfs.svraster.cuda.renderer.voxel_rasterizer_gpu as vrg

            rays_to_camera_matrix_func = vrg.get_rays_to_camera_matrix_function()
            if rays_to_camera_matrix_func is not None:
                return rays_to_camera_matrix_func(ray_origins, ray_directions)
        except (ImportError, AttributeError):
            pass

    # CPU 实现
    # 简化实现：假设所有光线起点相同
    camera_center = ray_origins.mean(dim=0)

    # 估算相机朝向
    mean_direction = ray_directions.mean(dim=0)
    mean_direction = mean_direction / torch.norm(mean_direction)

    # 构建相机坐标系
    forward = -mean_direction  # 相机朝向
    up_vec = torch.tensor([0.0, 1.0, 0.0], device=forward.device, dtype=forward.dtype)
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
    intrinsics = torch.tensor(
        [[800, 0, 400], [0, 800, 300], [0, 0, 1]],
        dtype=ray_origins.dtype,
        device=ray_origins.device,
    )

    return camera_matrix, intrinsics


def benchmark_voxel_rasterizer(
    voxels: dict[str, torch.Tensor],
    camera_matrix: torch.Tensor,
    intrinsics: torch.Tensor,
    viewport_size: tuple[int, int],
    num_iterations: int = 100,
    use_cuda: Optional[bool] = None,
) -> dict[str, float]:
    """
    基准测试体素光栅化渲染器性能

    Args:
        voxels: 体素数据
        camera_matrix: 相机矩阵
        intrinsics: 相机内参
        viewport_size: 视口尺寸
        num_iterations: 迭代次数
        use_cuda: 是否使用 CUDA

    Returns:
        性能统计结果
    """
    if use_cuda is None:
        use_cuda = CUDA_AVAILABLE and torch.cuda.is_available()

    if use_cuda and CUDA_AVAILABLE:
        try:
            import nerfs.svraster.cuda.renderer.voxel_rasterizer_gpu as vrg

            benchmark_func = vrg.get_benchmark_function()
            if benchmark_func is not None:
                # CUDA 基准测试
                device = voxels["positions"].device
                voxels_gpu = {k: v.to(device) for k, v in voxels.items()}
                camera_matrix = camera_matrix.to(device)
                intrinsics = intrinsics.to(device)
                viewport_tensor = torch.tensor(viewport_size, dtype=torch.int32, device=device)

                return benchmark_func(
                    voxels_gpu["positions"],
                    voxels_gpu["sizes"],
                    voxels_gpu["densities"],
                    voxels_gpu["colors"],
                    camera_matrix,
                    intrinsics,
                    viewport_tensor,
                    num_iterations,
                )
        except (ImportError, AttributeError):
            pass

        # CUDA 可用但 benchmark 函数不可用，回退到 CPU
        use_cuda = False

    # CPU 基准测试
    import time

    config = type(
        "Config",
        (),
        {
            "near_plane": 0.1,
            "far_plane": 100.0,
            "background_color": [0.0, 0.0, 0.0],
            "density_activation": "exp",
            "color_activation": "sigmoid",
        },
    )()

    rasterizer = VoxelRasterizer(config, use_cuda=False)

    # 预热
    for _ in range(3):
        _ = rasterizer(voxels, camera_matrix, intrinsics, viewport_size)

    # 基准测试
    start_time = time.time()
    for _ in range(num_iterations):
        _ = rasterizer(voxels, camera_matrix, intrinsics, viewport_size)
    end_time = time.time()

    total_time = (end_time - start_time) * 1000  # 转换为毫秒
    avg_time = total_time / num_iterations
    fps = 1000.0 / avg_time

    return {"total_time_ms": total_time, "avg_time_ms": avg_time, "fps": fps}


def is_cuda_available() -> bool:
    """检查 CUDA 是否可用"""
    return CUDA_AVAILABLE and torch.cuda.is_available()


def get_recommended_device() -> str:
    """获取推荐的设备类型"""
    if is_cuda_available():
        return "cuda"
    else:
        return "cpu"
