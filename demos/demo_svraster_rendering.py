#!/usr/bin/env python3
"""
SVRaster 高效渲染演示

这个演示展示如何使用 SVRaster 进行实时高效渲染。
重点展示推理阶段的光栅化渲染性能和质量。

特点：
- 使用 VoxelRasterizer 进行快速光栅化
- 实时渲染性能优化
- 多种渲染模式对比
- GPU 加速渲染
- 渲染质量评估
"""

from __future__ import annotations

import sys
import torch
import numpy as np
import time
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import json
import imageio
from tqdm import tqdm

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

# SVRaster 导入
from src.nerfs.svraster import (
    SVRasterConfig, SVRasterModel,
    SVRasterRenderer, SVRasterRendererConfig,
    VoxelRasterizer, VolumeRenderer
)

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SVRasterRenderingDemo:
    """SVRaster 高效渲染演示类"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"使用设备: {self.device}")
        
        # 渲染配置
        self.model_config = self._create_model_config()
        self.render_config = self._create_render_config()
        
        self.model = None
        self.volume_renderer = None
        self.true_rasterizer = None
        self.svraster_renderer = None
        
    def _create_model_config(self) -> SVRasterConfig:
        """创建模型配置"""
        config = SVRasterConfig(
            # 场景设置
            image_width=800,
            image_height=600,
            scene_bounds=(-2.0, -2.0, -2.0, 2.0, 2.0, 2.0),
            
            # 体素网格设置
            base_resolution=128,  # 高分辨率用于高质量渲染
            max_octree_levels=10,
            
            # 渲染设置
            ray_samples_per_voxel=4,  # 推理时可以减少采样
            depth_peeling_layers=2,
            morton_ordering=True,
            
            # 外观设置
            sh_degree=2,
            color_activation="sigmoid",
            density_activation="exp",
            
            # 渲染设置
            background_color=(1.0, 1.0, 1.0),  # 白色背景
            near_plane=0.1,
            far_plane=10.0,
        )
        
        logger.info("模型配置创建完成")
        logger.info(f"  - 渲染分辨率: {config.image_width}x{config.image_height}")
        logger.info(f"  - 体素分辨率: {config.base_resolution}^3")
        
        return config
        
    def _create_render_config(self) -> SVRasterRendererConfig:
        """创建渲染器配置"""
        config = SVRasterRendererConfig(
            image_width=self.model_config.image_width,
            image_height=self.model_config.image_height,
            render_batch_size=8192,  # 大批次提高效率
            render_chunk_size=2048,
            background_color=(1.0, 1.0, 1.0),
            use_alpha_blending=True,
            depth_threshold=1e-6,
            max_rays_per_batch=16384,
            use_hierarchical_sampling=True,
            output_format='png',
            save_depth=True,
            save_alpha=True,
            use_cached_features=True,
            enable_gradient_checkpointing=False,  # 推理时关闭
        )
        
        logger.info("渲染器配置创建完成")
        logger.info(f"  - 批次大小: {config.render_batch_size}")
        logger.info(f"  - 最大光线数: {config.max_rays_per_batch}")
        
        return config
        
    def setup_model_and_renderers(self):
        """设置模型和渲染器"""
        logger.info("设置模型和渲染器...")
        
        # 创建模型
        self.model = SVRasterModel(self.model_config).to(self.device)
        
        # 初始化模型为演示场景
        self._initialize_demo_scene()
        
        # 创建体积渲染器（训练用）
        self.volume_renderer = VolumeRenderer(self.model_config)
        
        # 创建真正的光栅化器（推理用）
        self.true_rasterizer = VoxelRasterizer(self.model_config)
        
        # 创建 SVRaster 渲染器（高级接口）
        self.svraster_renderer = SVRasterRenderer(
            model=self.model,
            rasterizer=self.true_rasterizer,
            config=self.render_config
        )
        
        logger.info("模型和渲染器设置完成")
        logger.info(f"模型参数数量: {sum(p.numel() for p in self.model.parameters()):,}")
        
    def _initialize_demo_scene(self):
        """初始化演示场景"""
        logger.info("初始化演示场景...")
        
        with torch.no_grad():
            # 创建多个球体的复杂场景
            res = self.model_config.base_resolution
            
            # 生成网格坐标
            coords = torch.stack(torch.meshgrid(
                torch.linspace(-2, 2, res),
                torch.linspace(-2, 2, res),
                torch.linspace(-2, 2, res),
                indexing='ij'
            ), dim=-1).to(self.device)
            
            # 创建多个球体
            densities = torch.zeros(res, res, res, device=self.device)
            
            # 球体 1: 中心球（红色）
            center1 = torch.tensor([0.0, 0.0, 0.0], device=self.device)
            dist1 = torch.norm(coords - center1, dim=-1)
            sphere1 = torch.exp(-(dist1 - 0.8) ** 2 / 0.1)
            densities += sphere1
            
            # 球体 2: 左侧球（绿色）
            center2 = torch.tensor([-1.2, 0.0, 0.0], device=self.device)
            dist2 = torch.norm(coords - center2, dim=-1)
            sphere2 = torch.exp(-(dist2 - 0.5) ** 2 / 0.08)
            densities += sphere2 * 0.8
            
            # 球体 3: 右侧球（蓝色）
            center3 = torch.tensor([1.2, 0.0, 0.0], device=self.device)
            dist3 = torch.norm(coords - center3, dim=-1)
            sphere3 = torch.exp(-(dist3 - 0.5) ** 2 / 0.08)
            densities += sphere3 * 0.6
            
            # 设置密度
            self.model.voxels.densities = densities.unsqueeze(-1)
            
            # 创建彩色特征（球谐系数）
            num_sh_coeffs = (self.model_config.sh_degree + 1) ** 2
            features = torch.zeros(res, res, res, 3 * num_sh_coeffs, device=self.device)
            
            # 为不同区域设置不同颜色，使用正确的索引方法
            # 中心球：红色
            mask1 = sphere1 > 0.1
            i1, j1, k1 = torch.where(mask1)
            features[i1, j1, k1, 0] = 0.8  # R 通道的 0 阶系数
            features[i1, j1, k1, num_sh_coeffs] = 0.2  # G 通道的 0 阶系数
            features[i1, j1, k1, 2 * num_sh_coeffs] = 0.2  # B 通道的 0 阶系数
            
            # 左侧球：绿色
            mask2 = sphere2 > 0.1
            i2, j2, k2 = torch.where(mask2)
            features[i2, j2, k2, 0] = 0.2
            features[i2, j2, k2, num_sh_coeffs] = 0.8
            features[i2, j2, k2, 2 * num_sh_coeffs] = 0.2
            
            # 右侧球：蓝色
            mask3 = sphere3 > 0.1
            i3, j3, k3 = torch.where(mask3)
            features[i3, j3, k3, 0] = 0.2
            features[i3, j3, k3, num_sh_coeffs] = 0.2
            features[i3, j3, k3, 2 * num_sh_coeffs] = 0.8
            
            # 添加一些视角相关效果（高阶球谐）
            if num_sh_coeffs > 1:
                # 为每个球体添加轻微的视角依赖
                features[i1, j1, k1, 1:4] = 0.1  # 1阶系数
                features[i2, j2, k2, num_sh_coeffs+1:num_sh_coeffs+4] = 0.1
                features[i3, j3, k3, 2*num_sh_coeffs+1:2*num_sh_coeffs+4] = 0.1
            
            self.model.voxels.colors = features
            
        logger.info("演示场景初始化完成")
        logger.info(f"  - 密度范围: [{densities.min():.3f}, {densities.max():.3f}]")
        logger.info(f"  - 特征维度: {features.shape[-1]}")
        
    def generate_camera_path(self, num_frames: int = 60) -> List[Tuple[np.ndarray, np.ndarray]]:
        """生成相机路径"""
        camera_path = []
        
        # 圆形轨道
        radius = 4.0
        for i in range(num_frames):
            angle = 2 * np.pi * i / num_frames
            
            # 相机位置
            camera_pos = np.array([
                radius * np.cos(angle),
                radius * np.sin(angle),
                1.0  # 稍微从上往下看
            ])
            
            # 朝向场景中心
            target = np.array([0.0, 0.0, 0.0])
            forward = target - camera_pos
            forward = forward / np.linalg.norm(forward)
            
            camera_path.append((camera_pos, forward))
            
        return camera_path
        
    def generate_ray_batch(
        self, camera_pos: np.ndarray, camera_forward: np.ndarray, 
        subset_ratio: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """为给定相机位置生成光线"""
        H, W = self.model_config.image_height, self.model_config.image_width
        
        # 生成像素网格
        if subset_ratio < 1.0:
            # 子集渲染（用于性能测试）
            subset_H = int(H * subset_ratio)
            subset_W = int(W * subset_ratio)
            i, j = torch.meshgrid(
                torch.linspace(0, W-1, subset_W),
                torch.linspace(0, H-1, subset_H),
                indexing='xy'
            )
            pixels = torch.stack([i, j], dim=-1).reshape(-1, 2)
        else:
            # 完整渲染
            i, j = torch.meshgrid(
                torch.arange(W),
                torch.arange(H),
                indexing='xy'
            )
            pixels = torch.stack([i, j], dim=-1).reshape(-1, 2)
        
        # 归一化到 [-1, 1]
        x_norm = (pixels[:, 0] / W - 0.5) * 2
        y_norm = (pixels[:, 1] / H - 0.5) * 2
        
        # 计算相机坐标系
        up = np.array([0.0, 0.0, 1.0])
        right = np.cross(camera_forward, up)
        right = right / np.linalg.norm(right)
        up = np.cross(right, camera_forward)
        
        # 转换为tensor
        camera_forward_t = torch.tensor(camera_forward, dtype=torch.float32)
        right_t = torch.tensor(right, dtype=torch.float32)
        up_t = torch.tensor(up, dtype=torch.float32)
        
        # 焦距（field of view）
        focal_length = 1.0
        
        # 光线方向
        ray_dirs = (
            camera_forward_t[None, :] * focal_length +
            right_t[None, :] * x_norm[:, None] * 0.5 +
            up_t[None, :] * y_norm[:, None] * 0.5
        )
        ray_dirs = ray_dirs / torch.norm(ray_dirs, dim=1, keepdim=True)
        
        # 转换为 tensor
        ray_origins = torch.tensor(camera_pos, dtype=torch.float32, device=self.device)
        ray_origins = ray_origins.unsqueeze(0).expand(len(pixels), -1)
        ray_directions = torch.tensor(ray_dirs, dtype=torch.float32, device=self.device)
        
        return ray_origins, ray_directions
        
    def benchmark_rendering_modes(self):
        """对比不同渲染模式的性能"""
        logger.info("开始渲染性能基准测试...")
        
        # 生成测试相机位置
        camera_pos = np.array([3.0, 0.0, 1.0])
        target = np.array([0.0, 0.0, 0.0])
        camera_forward = target - camera_pos
        camera_forward = camera_forward / np.linalg.norm(camera_forward)
        
        # 测试不同的渲染分辨率
        test_ratios = [0.25, 0.5, 1.0]  # 1/4, 1/2, 全分辨率
        
        results = {}
        
        for ratio in test_ratios:
            logger.info(f"\n测试分辨率比例: {ratio}")
            
            # 生成光线
            ray_origins, ray_directions = self.generate_ray_batch(
                camera_pos, camera_forward, ratio
            )
            
            num_rays = len(ray_origins)
            actual_res = f"{int(self.model_config.image_width * ratio)}x{int(self.model_config.image_height * ratio)}"
            logger.info(f"  光线数量: {num_rays:,} ({actual_res})")
            
            # 1. 体积渲染（训练模式）
            logger.info("  测试体积渲染...")
            with torch.no_grad():
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                start_time = time.time()
                
                volume_outputs = self.model(
                    ray_origins, ray_directions, mode="training"
                )
                
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                volume_time = time.time() - start_time
            
            # 2. 光栅化渲染（推理模式）
            logger.info("  测试光栅化渲染...")
            with torch.no_grad():
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                start_time = time.time()
                
                raster_outputs = self.model(
                    ray_origins, ray_directions, mode="inference"
                )
                
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                raster_time = time.time() - start_time
            
            # 计算性能指标
            volume_fps = 1.0 / volume_time if volume_time > 0 else float('inf')
            raster_fps = 1.0 / raster_time if raster_time > 0 else float('inf')
            speedup = volume_time / raster_time if raster_time > 0 else float('inf')
            
            results[ratio] = {
                'resolution': actual_res,
                'num_rays': num_rays,
                'volume_time': volume_time,
                'raster_time': raster_time,
                'volume_fps': volume_fps,
                'raster_fps': raster_fps,
                'speedup': speedup,
                'volume_outputs': volume_outputs,
                'raster_outputs': raster_outputs
            }
            
            logger.info(f"  体积渲染: {volume_time:.4f}s ({volume_fps:.1f} FPS)")
            logger.info(f"  光栅化渲染: {raster_time:.4f}s ({raster_fps:.1f} FPS)")
            logger.info(f"  加速比: {speedup:.2f}x")
            
        return results
        
    def render_quality_comparison(self, camera_pos: np.ndarray, camera_forward: np.ndarray):
        """渲染质量对比"""
        logger.info("生成质量对比图像...")
        
        # 生成光线
        ray_origins, ray_directions = self.generate_ray_batch(camera_pos, camera_forward)
        
        with torch.no_grad():
            # 体积渲染
            volume_outputs = self.model(ray_origins, ray_directions, mode="training")
            
            # 光栅化渲染
            raster_outputs = self.model(ray_origins, ray_directions, mode="inference")
        
        # 重新整理为图像格式
        H, W = self.model_config.image_height, self.model_config.image_width
        
        volume_image = volume_outputs['rgb'].reshape(H, W, 3).cpu().numpy()
        raster_image = raster_outputs['rgb'].reshape(H, W, 3).cpu().numpy()
        
        # 深度图
        volume_depth = volume_outputs['depth'].reshape(H, W).cpu().numpy()
        raster_depth = raster_outputs['depth'].reshape(H, W).cpu().numpy()
        
        # 归一化深度图
        volume_depth_norm = (volume_depth - volume_depth.min()) / (volume_depth.max() - volume_depth.min() + 1e-8)
        raster_depth_norm = (raster_depth - raster_depth.min()) / (raster_depth.max() - raster_depth.min() + 1e-8)
        
        # 计算差异图
        rgb_diff = np.abs(volume_image - raster_image)
        depth_diff = np.abs(volume_depth_norm - raster_depth_norm)
        
        return {
            'volume_rgb': volume_image,
            'raster_rgb': raster_image,
            'volume_depth': volume_depth_norm,
            'raster_depth': raster_depth_norm,
            'rgb_diff': rgb_diff,
            'depth_diff': depth_diff
        }
        
    def save_comparison_images(self, comparison_data: dict, output_dir: Path):
        """保存对比图像"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存 RGB 图像
        imageio.imwrite(
            output_dir / "volume_rendering.png",
            (np.clip(comparison_data['volume_rgb'], 0, 1) * 255).astype(np.uint8)
        )
        
        imageio.imwrite(
            output_dir / "raster_rendering.png",
            (np.clip(comparison_data['raster_rgb'], 0, 1) * 255).astype(np.uint8)
        )
        
        # 保存深度图
        imageio.imwrite(
            output_dir / "volume_depth.png",
            (comparison_data['volume_depth'] * 255).astype(np.uint8)
        )
        
        imageio.imwrite(
            output_dir / "raster_depth.png",
            (comparison_data['raster_depth'] * 255).astype(np.uint8)
        )
        
        # 保存差异图
        imageio.imwrite(
            output_dir / "rgb_difference.png",
            (np.clip(comparison_data['rgb_diff'] * 10, 0, 1) * 255).astype(np.uint8)  # 放大差异
        )
        
        imageio.imwrite(
            output_dir / "depth_difference.png",
            (np.clip(comparison_data['depth_diff'] * 10, 0, 1) * 255).astype(np.uint8)
        )
        
        logger.info(f"对比图像已保存到: {output_dir}")
        
    def render_animation(self, output_dir: Path, num_frames: int = 30):
        """渲染动画序列"""
        logger.info(f"渲染 {num_frames} 帧动画...")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 生成相机路径
        camera_path = self.generate_camera_path(num_frames)
        
        render_times = []
        
        with tqdm(enumerate(camera_path), total=num_frames, desc="渲染动画") as pbar:
            for frame_idx, (camera_pos, camera_forward) in pbar:
                # 生成光线
                ray_origins, ray_directions = self.generate_ray_batch(
                    camera_pos, camera_forward, subset_ratio=0.5  # 使用一半分辨率提高速度
                )
                
                # 渲染
                start_time = time.time()
                with torch.no_grad():
                    outputs = self.model(ray_origins, ray_directions, mode="inference")
                render_time = time.time() - start_time
                render_times.append(render_time)
                
                # 转换为图像
                H, W = int(self.model_config.image_height * 0.5), int(self.model_config.image_width * 0.5)
                image = outputs['rgb'].reshape(H, W, 3).cpu().numpy()
                image = np.clip(image, 0, 1)
                
                # 保存帧
                frame_path = output_dir / f"frame_{frame_idx:03d}.png"
                imageio.imwrite(frame_path, (image * 255).astype(np.uint8))
                
                # 更新进度条
                avg_render_time = np.mean(render_times)
                fps = 1.0 / avg_render_time if avg_render_time > 0 else 0
                pbar.set_postfix({
                    'avg_time': f'{avg_render_time:.3f}s',
                    'fps': f'{fps:.1f}'
                })
        
        avg_fps = 1.0 / np.mean(render_times)
        logger.info(f"动画渲染完成！平均帧率: {avg_fps:.1f} FPS")
        
        return render_times
        
    def run_full_demo(self):
        """运行完整演示"""
        logger.info("开始 SVRaster 高效渲染演示...")
        
        # 性能基准测试
        benchmark_results = self.benchmark_rendering_modes()
        
        # 质量对比
        camera_pos = np.array([3.0, 1.0, 1.5])
        target = np.array([0.0, 0.0, 0.0])
        camera_forward = target - camera_pos
        camera_forward = camera_forward / np.linalg.norm(camera_forward)
        
        comparison_data = self.render_quality_comparison(camera_pos, camera_forward)
        
        # 保存对比图像
        output_dir = Path("demos/demo_outputs/svraster_rendering")
        self.save_comparison_images(comparison_data, output_dir)
        
        # 渲染动画
        animation_times = self.render_animation(
            output_dir / "animation", num_frames=24
        )
        
        # 生成报告
        self._generate_performance_report(
            benchmark_results, comparison_data, animation_times, output_dir
        )
        
    def _generate_performance_report(
        self, benchmark_results: dict, comparison_data: dict, 
        animation_times: list, output_dir: Path
    ):
        """生成性能报告"""
        report = {
            "device": str(self.device),
            "model_config": {
                "resolution": f"{self.model_config.image_width}x{self.model_config.image_height}",
                "base_resolution": self.model_config.base_resolution,
                "sh_degree": self.model_config.sh_degree
            },
            "benchmark_results": {},
            "animation_performance": {
                "num_frames": len(animation_times),
                "avg_frame_time": float(np.mean(animation_times)),
                "avg_fps": float(1.0 / np.mean(animation_times)),
                "min_frame_time": float(np.min(animation_times)),
                "max_frame_time": float(np.max(animation_times))
            },
            "quality_metrics": {
                "rgb_mse": float(np.mean(comparison_data['rgb_diff'] ** 2)),
                "depth_mse": float(np.mean(comparison_data['depth_diff'] ** 2)),
                "rgb_psnr": float(-10 * np.log10(np.mean(comparison_data['rgb_diff'] ** 2) + 1e-8))
            }
        }
        
        # 添加基准测试结果
        for ratio, results in benchmark_results.items():
            report["benchmark_results"][f"resolution_{ratio}"] = {
                "resolution": results['resolution'],
                "num_rays": results['num_rays'],
                "volume_time": results['volume_time'],
                "raster_time": results['raster_time'],
                "volume_fps": results['volume_fps'],
                "raster_fps": results['raster_fps'],
                "speedup": results['speedup']
            }
        
        # 保存报告
        with open(output_dir / "performance_report.json", "w") as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"性能报告已保存到: {output_dir / 'performance_report.json'}")


def main():
    """主函数"""
    print("=" * 70)
    print("SVRaster 高效渲染演示")
    print("=" * 70)
    
    try:
        # 创建渲染演示
        demo = SVRasterRenderingDemo()
        
        # 设置模型和渲染器
        demo.setup_model_and_renderers()
        
        # 运行完整演示
        demo.run_full_demo()
        
        print("\n🎉 SVRaster 高效渲染演示完成！")
        print("\n渲染特点:")
        print("✅ 使用 VoxelRasterizer 进行实时光栅化")
        print("✅ GPU 加速高性能渲染")
        print("✅ 多分辨率性能测试")
        print("✅ 训练/推理模式质量对比")
        print("✅ 实时动画序列渲染")
        print("\n输出文件:")
        print("📁 demos/demo_outputs/svraster_rendering/")
        print("   ├── volume_rendering.png      # 体积渲染结果")
        print("   ├── raster_rendering.png      # 光栅化渲染结果")
        print("   ├── *_depth.png               # 深度图")
        print("   ├── *_difference.png          # 差异图")
        print("   ├── animation/                # 动画序列")
        print("   └── performance_report.json   # 性能报告")
        
    except KeyboardInterrupt:
        print("\n⏹️  渲染被用户中断")
    except Exception as e:
        print(f"\n❌ 渲染失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
