"""
SVRaster 架构重构建议

解决光栅化渲染逻辑应该放在 SVRasterRenderer 中的问题
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from pathlib import Path


class SVRasterRendererRefactored:
    """
    重构后的 SVRaster 渲染器
    
    将光栅化渲染逻辑从模型中移到渲染器中，实现更好的职责分离
    """
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 模型只负责存储体素数据
        self.model: Optional['SVRasterModel'] = None
        
        # 渲染器负责实际的渲染逻辑
        self.volume_renderer = None  # 用于高质量渲染
        self.rasterizer = None      # 用于快速推理
        
    def load_model(self, checkpoint_path: str):
        """加载模型"""
        # 加载模型（只包含体素数据，不包含渲染逻辑）
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model = SVRasterModelMinimal(checkpoint['model_config'])
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # 初始化渲染器
        from .volume_renderer import VolumeRenderer
        from .true_rasterizer import TrueVoxelRasterizer
        
        self.volume_renderer = VolumeRenderer(self.model.config)
        self.rasterizer = TrueVoxelRasterizer(self.model.config)
        
    def render_single_view(
        self,
        camera_pose: torch.Tensor,
        intrinsics: torch.Tensor,
        mode: str = "rasterization",  # "rasterization" 或 "volume"
        image_size: Optional[Tuple[int, int]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        渲染单个视角
        
        Args:
            camera_pose: 相机位姿矩阵
            intrinsics: 相机内参矩阵
            mode: 渲染模式 - "rasterization" 或 "volume"
            image_size: 图像尺寸
            
        Returns:
            渲染结果
        """
        if self.model is None:
            raise RuntimeError("模型未加载")
            
        # 设置图像尺寸
        if image_size is None:
            width, height = self.config.image_width, self.config.image_height
        else:
            width, height = image_size
        
        # 获取体素数据
        voxels = self.model.get_all_voxels()
        
        if mode == "rasterization":
            # 使用光栅化渲染 - 快速推理
            return self._render_rasterization(voxels, camera_pose, intrinsics, width, height)
        elif mode == "volume":
            # 使用体积渲染 - 高质量渲染
            return self._render_volume(voxels, camera_pose, intrinsics, width, height)
        else:
            raise ValueError(f"不支持的渲染模式: {mode}")
    
    def _render_rasterization(
        self,
        voxels: Dict[str, torch.Tensor],
        camera_pose: torch.Tensor,
        intrinsics: torch.Tensor,
        width: int,
        height: int
    ) -> Dict[str, torch.Tensor]:
        """
        光栅化渲染实现 - 在渲染器中实现，不在模型中
        """
        # 直接使用光栅化器
        viewport_size = (width, height)
        
        outputs = self.rasterizer(
            voxels,
            camera_pose,
            intrinsics,
            viewport_size,
        )
        
        return outputs
    
    def _render_volume(
        self,
        voxels: Dict[str, torch.Tensor],
        camera_pose: torch.Tensor,
        intrinsics: torch.Tensor,
        width: int,
        height: int
    ) -> Dict[str, torch.Tensor]:
        """
        体积渲染实现 - 高质量渲染
        """
        # 生成光线
        rays_o, rays_d = self._generate_rays(camera_pose, intrinsics, width, height)
        
        # 重塑为批次格式
        rays_o = rays_o.reshape(-1, 3)
        rays_d = rays_d.reshape(-1, 3)
        
        # 批量渲染
        batch_size = self.config.render_batch_size
        rgb_list = []
        depth_list = []
        
        for i in range(0, rays_o.shape[0], batch_size):
            batch_rays_o = rays_o[i:i + batch_size]
            batch_rays_d = rays_d[i:i + batch_size]
            
            # 使用体积渲染器
            outputs = self.volume_renderer(
                voxels,
                batch_rays_o,
                batch_rays_d
            )
            
            rgb_list.append(outputs['rgb'])
            if 'depth' in outputs:
                depth_list.append(outputs['depth'])
        
        # 合并结果
        rgb = torch.cat(rgb_list, dim=0).reshape(height, width, 3)
        results = {'rgb': rgb}
        
        if depth_list:
            depth = torch.cat(depth_list, dim=0).reshape(height, width)
            results['depth'] = depth
        
        return results
    
    def render_path_comparison(
        self,
        camera_poses: List[torch.Tensor],
        intrinsics: torch.Tensor,
        output_dir: str
    ):
        """
        渲染路径对比 - 展示两种渲染模式的差异
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        for i, pose in enumerate(camera_poses):
            # 光栅化渲染
            raster_outputs = self.render_single_view(pose, intrinsics, mode="rasterization")
            # 体积渲染
            volume_outputs = self.render_single_view(pose, intrinsics, mode="volume")
            
            # 保存对比图像
            self._save_comparison(raster_outputs, volume_outputs, f"{output_dir}/frame_{i:04d}")
    
    def _generate_rays(self, camera_pose, intrinsics, width, height):
        """生成光线 - 共用方法"""
        # ... 实现细节同原来的方法
        pass
    
    def _save_comparison(self, raster_outputs, volume_outputs, filename_prefix):
        """保存对比图像"""
        import imageio
        
        # 光栅化结果
        raster_rgb = (raster_outputs['rgb'].cpu().numpy() * 255).astype('uint8')
        imageio.imwrite(f"{filename_prefix}_rasterization.png", raster_rgb)
        
        # 体积渲染结果
        volume_rgb = (volume_outputs['rgb'].cpu().numpy() * 255).astype('uint8')
        imageio.imwrite(f"{filename_prefix}_volume.png", volume_rgb)
        
        # 差异图
        diff = torch.abs(raster_outputs['rgb'] - volume_outputs['rgb'])
        diff_rgb = (diff.cpu().numpy() * 255).astype('uint8')
        imageio.imwrite(f"{filename_prefix}_difference.png", diff_rgb)


class SVRasterModelMinimal:
    """
    最小化的 SVRaster 模型
    
    只负责存储和管理体素数据，不包含渲染逻辑
    """
    
    def __init__(self, config):
        self.config = config
        self.voxels = None  # 体素数据存储
        
    def get_all_voxels(self) -> Dict[str, torch.Tensor]:
        """获取所有体素数据"""
        return self.voxels.get_all_voxels()
    
    def forward(self, *args, **kwargs):
        """
        移除 forward 方法中的渲染逻辑
        模型只负责数据存储，不负责渲染
        """
        raise NotImplementedError(
            "模型不再包含渲染逻辑，请使用 SVRasterRenderer 进行渲染"
        )


def architecture_benefits():
    """
    新架构的优势
    """
    return """
    🎯 重构后的架构优势：

    1. **职责清晰分离**:
       - SVRasterModel: 只负责体素数据存储和管理
       - SVRasterRenderer: 负责所有渲染逻辑（体积渲染 + 光栅化）

    2. **灵活的渲染模式**:
       - 可以在运行时选择渲染模式
       - 便于对比不同渲染方法的效果
       - 便于性能调优

    3. **代码复用性**:
       - 渲染器可以复用于不同的模型
       - 减少代码重复
       - 更好的测试和维护

    4. **部署友好**:
       - 生产环境可以只加载渲染器
       - 模型文件更轻量
       - 便于优化部署

    5. **扩展性**:
       - 容易添加新的渲染算法
       - 便于集成 GPU 加速
       - 支持多种输出格式

    📝 使用示例：
    ```python
    # 创建渲染器
    renderer = SVRasterRendererRefactored(config)
    
    # 加载模型
    renderer.load_model("model.pth")
    
    # 快速推理（光栅化）
    fast_result = renderer.render_single_view(pose, intrinsics, mode="rasterization")
    
    # 高质量渲染（体积渲染）
    quality_result = renderer.render_single_view(pose, intrinsics, mode="volume")
    
    # 对比渲染
    renderer.render_path_comparison(poses, intrinsics, "comparison_output")
    ```

    🔄 迁移步骤：
    1. 将光栅化逻辑从 SVRasterModel.forward() 移到 SVRasterRenderer
    2. 简化 SVRasterModel，只保留数据管理
    3. 在渲染器中实现不同的渲染模式
    4. 更新所有调用代码
    """


if __name__ == "__main__":
    print(architecture_benefits())
