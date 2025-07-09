"""
SVRaster One 配置类

定义可微分光栅化渲染器的所有配置参数。
"""

from dataclasses import dataclass, field
from typing import Tuple, Optional
import torch


@dataclass
class VoxelConfig:
    """体素配置"""

    # 体素网格配置
    grid_resolution: int = 256  # 体素网格分辨率
    voxel_size: float = 0.01  # 体素大小
    max_voxels: int = 1000000  # 最大体素数量

    # 稀疏性配置
    sparsity_threshold: float = 0.01  # 稀疏性阈值
    adaptive_subdivision: bool = True  # 自适应细分
    subdivision_threshold: float = 0.1  # 细分阈值

    # Morton 编码配置
    use_morton_ordering: bool = True  # 使用 Morton 排序
    morton_bits: int = 21  # Morton 编码位数


@dataclass
class RenderingConfig:
    """渲染配置"""

    # 光栅化配置
    image_width: int = 800
    image_height: int = 600
    background_color: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    # 可微分光栅化参数
    soft_rasterization: bool = True  # 软光栅化
    temperature: float = 0.1  # 软光栅化温度参数
    sigma: float = 1.0  # 高斯核标准差

    # 深度排序配置
    depth_sorting: str = "back_to_front"  # back_to_front, front_to_back, none
    use_soft_sorting: bool = True  # 软排序

    # Alpha 混合配置
    alpha_blending: bool = True  # Alpha 混合
    alpha_threshold: float = 0.01  # Alpha 阈值


@dataclass
class TrainingConfig:
    """训练配置"""

    # 损失函数权重
    rgb_loss_weight: float = 1.0
    depth_loss_weight: float = 0.1
    density_reg_weight: float = 0.01
    sparsity_weight: float = 0.001

    # 优化器配置
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4

    # 训练参数
    batch_size: int = 4096
    num_epochs: int = 1000
    save_interval: int = 100

    # 混合精度训练
    use_amp: bool = True
    grad_clip: float = 1.0


@dataclass
class CUDAConfig:
    """CUDA 配置"""

    # CUDA 内核配置
    block_size: int = 256  # CUDA 块大小
    max_blocks: int = 65535  # 最大块数量

    # 内存配置
    memory_pool_size: int = 1024 * 1024 * 1024  # 1GB 内存池
    use_memory_pool: bool = True

    # 性能配置
    enable_profiling: bool = False
    sync_cuda: bool = False


@dataclass
class SVRasterOneConfig:
    """SVRaster One 主配置类"""

    # 子配置
    voxel: VoxelConfig = field(default_factory=VoxelConfig)
    rendering: RenderingConfig = field(default_factory=RenderingConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    cuda: CUDAConfig = field(default_factory=CUDAConfig)

    # 设备配置
    device: str = "auto"  # auto, cuda, cpu

    # 日志配置
    log_level: str = "INFO"
    save_logs: bool = True
    log_dir: str = "logs"

    def __post_init__(self):
        """后初始化验证"""
        # 自动选择设备
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # 验证配置
        if self.voxel.grid_resolution <= 0:
            raise ValueError("grid_resolution must be positive")

        if self.rendering.image_width <= 0 or self.rendering.image_height <= 0:
            raise ValueError("image dimensions must be positive")

        if self.training.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "voxel": self.voxel.__dict__,
            "rendering": self.rendering.__dict__,
            "training": self.training.__dict__,
            "cuda": self.cuda.__dict__,
            "device": self.device,
            "log_level": self.log_level,
            "save_logs": self.save_logs,
            "log_dir": self.log_dir,
        }

    @classmethod
    def from_dict(cls, config_dict: dict) -> "SVRasterOneConfig":
        """从字典创建配置"""
        config = cls()

        if "voxel" in config_dict:
            for key, value in config_dict["voxel"].items():
                setattr(config.voxel, key, value)

        if "rendering" in config_dict:
            for key, value in config_dict["rendering"].items():
                setattr(config.rendering, key, value)

        if "training" in config_dict:
            for key, value in config_dict["training"].items():
                setattr(config.training, key, value)

        if "cuda" in config_dict:
            for key, value in config_dict["cuda"].items():
                setattr(config.cuda, key, value)

        # 设置其他属性
        for key in ["device", "log_level", "save_logs", "log_dir"]:
            if key in config_dict:
                setattr(config, key, config_dict[key])

        return config
