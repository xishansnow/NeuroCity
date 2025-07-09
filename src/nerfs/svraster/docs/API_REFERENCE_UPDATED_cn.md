# SVRaster API 参考文档（更新版）

本文档基于最新的源代码重构，反映 SVRaster 1.0.0 的当前 API 结构。

## 📦 包概述

SVRaster 是一个高效的稀疏体素神经辐射场实现，支持训练和推理两个阶段，提供 CUDA 加速。

### 🎯 核心架构

SVRaster 采用**双渲染器架构**：
- **训练阶段**: `VolumeRenderer` - 使用体积渲染进行梯度优化
- **推理阶段**: `VoxelRasterizer` - 使用光栅化进行快速渲染

## 📚 API 目录

### 🏗️ 核心组件 (Core)

#### SVRasterConfig
```python
@dataclass
class SVRasterConfig:
    """SVRaster 主配置类"""
    # 场景表示
    max_octree_levels: int = 8
    base_resolution: int = 128  
    scene_bounds: Tuple[float, ...] = (-1.0, -1.0, -1.0, 1.0, 1.0, 1.0)
    
    # 体素属性
    density_activation: str = "exp"
    color_activation: str = "sigmoid"
    sh_degree: int = 2
    
    # 训练参数
    learning_rate: float = 1e-3
    weight_decay: float = 1e-6
```

#### SVRasterModel
```python
class SVRasterModel(nn.Module):
    """SVRaster 主模型类"""
    
    def __init__(self, config: SVRasterConfig)
    def forward(self, ray_origins: torch.Tensor, ray_directions: torch.Tensor) -> Dict[str, torch.Tensor]
    def get_voxels(self) -> Dict[str, torch.Tensor]
    def save_checkpoint(self, path: str)
    @classmethod
    def load_checkpoint(cls, path: str) -> 'SVRasterModel'
```

#### SVRasterLoss
```python
class SVRasterLoss:
    """SVRaster 损失函数类"""
    
    def __init__(self, config: SVRasterConfig)
    def compute_rgb_loss(self, pred_rgb: torch.Tensor, gt_rgb: torch.Tensor) -> torch.Tensor
    def compute_depth_loss(self, pred_depth: torch.Tensor, gt_depth: torch.Tensor) -> torch.Tensor
    def compute_sparsity_loss(self, voxel_densities: torch.Tensor) -> torch.Tensor
```

#### AdaptiveSparseVoxels
```python
class AdaptiveSparseVoxels:
    """自适应稀疏体素结构"""
    
    def __init__(self, config: SVRasterConfig)
    def subdivide_voxels(self, density_threshold: float)
    def prune_voxels(self, density_threshold: float) 
    def get_morton_codes(self) -> torch.Tensor
```

### 🎓 训练组件 (Training)

#### SVRasterTrainerConfig
```python
@dataclass
class SVRasterTrainerConfig:
    """训练器配置"""
    num_epochs: int = 100
    batch_size: int = 1
    learning_rate: float = 1e-3
    weight_decay: float = 1e-6
    save_every: int = 10
    validate_every: int = 5
    use_amp: bool = True
    log_dir: str = "logs"
```

#### SVRasterTrainer
```python
class SVRasterTrainer:
    """SVRaster 训练器 - 与 VolumeRenderer 紧密耦合"""
    
    def __init__(self, model: SVRasterModel, volume_renderer: VolumeRenderer, config: SVRasterTrainerConfig)
    def train(self, train_dataset: Dataset, val_dataset: Optional[Dataset] = None)
    def validate(self, val_dataset: Dataset) -> Dict[str, float]
    def save_checkpoint(self, path: str)
    def load_checkpoint(self, path: str)
```

#### VolumeRenderer
```python
class VolumeRenderer:
    """体积渲染器（训练专用）"""
    
    def __init__(self, config: SVRasterConfig)
    def __call__(self, voxels: Dict[str, torch.Tensor], ray_origins: torch.Tensor, 
                 ray_directions: torch.Tensor) -> Dict[str, torch.Tensor]
    def render_rays(self, ray_origins: torch.Tensor, ray_directions: torch.Tensor) -> Dict[str, torch.Tensor]
```

### 🎨 渲染组件 (Rendering)

#### SVRasterRendererConfig
```python
@dataclass
class SVRasterRendererConfig:
    """渲染器配置"""
    image_width: int = 800
    image_height: int = 600
    render_batch_size: int = 4096
    background_color: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    output_format: str = "png"
```

#### SVRasterRenderer
```python
class SVRasterRenderer:
    """SVRaster 渲染器 - 与 VoxelRasterizer 紧密耦合"""
    
    def __init__(self, model: SVRasterModel, rasterizer: VoxelRasterizer, config: SVRasterRendererConfig)
    def render(self, camera_pose: torch.Tensor, image_size: Tuple[int, int]) -> torch.Tensor
    def batch_render(self, camera_poses: List[torch.Tensor]) -> List[torch.Tensor]
    def render_video(self, camera_poses: List[torch.Tensor], output_path: str)
```

#### VoxelRasterizerConfig
```python
@dataclass
class VoxelRasterizerConfig:
    """真实体素光栅化器配置"""
    background_color: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    near_plane: float = 0.1
    far_plane: float = 100.0
    density_activation: str = "exp"
    color_activation: str = "sigmoid"
```

#### VoxelRasterizer
```python
class VoxelRasterizer:
    """真实体素光栅化器（推理专用）"""
    
    def __init__(self, config: VoxelRasterizerConfig)
    def __call__(self, voxels: Dict[str, torch.Tensor], camera_matrix: torch.Tensor,
                 intrinsics: torch.Tensor, viewport_size: Tuple[int, int]) -> Dict[str, torch.Tensor]
    def rasterize_voxels(self, voxels: Dict[str, torch.Tensor]) -> torch.Tensor
```

### 📊 数据组件 (Dataset)

#### SVRasterDatasetConfig
```python
@dataclass
class SVRasterDatasetConfig:
    """数据集配置"""
    data_dir: str = "data/nerf_synthetic/lego"
    image_width: int = 800
    image_height: int = 800
    camera_angle_x: float = 0.6911112070083618
    downscale_factor: int = 1
    num_rays_train: int = 1024
    num_rays_val: int = 512
```

#### SVRasterDataset
```python
class SVRasterDataset(Dataset):
    """SVRaster 数据集类"""
    
    def __init__(self, config: SVRasterDatasetConfig)
    def __len__(self) -> int
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]
    def get_rays(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]
    def load_images(self) -> torch.Tensor
    def load_poses(self) -> torch.Tensor
```

### 🚀 GPU 加速组件 (CUDA)

#### SVRasterGPU
```python
class SVRasterGPU:
    """GPU 优化的 SVRaster 模型"""
    
    def __init__(self, config: SVRasterConfig)
    def forward_gpu(self, ray_origins: torch.Tensor, ray_directions: torch.Tensor) -> Dict[str, torch.Tensor]
    def optimize_memory(self)
    def get_cuda_stats(self) -> Dict[str, Any]
```

#### SVRasterGPUTrainer
```python
class SVRasterGPUTrainer:
    """GPU 优化的训练器"""
    
    def __init__(self, model: SVRasterGPU, config: SVRasterConfig)
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]
    def mixed_precision_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor
```

#### EMAModel
```python
class EMAModel:
    """指数移动平均模型"""
    
    def __init__(self, model: nn.Module, decay: float = 0.999)
    def update(self, model: nn.Module)
    def apply_shadow(self, model: nn.Module)
    def restore(self, model: nn.Module)
```

### 🔧 工具函数 (Utilities)

#### 球谐函数
```python
def eval_sh_basis(degree: int, dirs: torch.Tensor) -> torch.Tensor:
    """计算球谐函数基"""
```

#### Morton 编码
```python
def morton_encode_3d(x: int, y: int, z: int) -> int:
    """3D Morton 编码"""

def morton_decode_3d(morton_code: int) -> Tuple[int, int, int]:
    """3D Morton 解码"""
```

#### 八叉树操作
```python
def octree_subdivision(octree_nodes: torch.Tensor) -> torch.Tensor:
    """八叉树细分"""

def octree_pruning(octree_nodes: torch.Tensor, threshold: float) -> torch.Tensor:
    """八叉树剪枝"""
```

#### 渲染工具
```python
def ray_direction_dependent_ordering(ray_dirs: torch.Tensor) -> torch.Tensor:
    """视角相关排序"""

def depth_peeling(depths: torch.Tensor, colors: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """深度剥离"""

def voxel_pruning(voxel_densities: torch.Tensor, threshold: float) -> torch.Tensor:
    """体素剪枝"""
```

## 🔄 重构变化总结

### 主要架构变化

1. **双渲染器架构**：
   - `VolumeRenderer` - 训练时使用体积渲染
   - `VoxelRasterizer` - 推理时使用光栅化

2. **紧密耦合设计**：
   - `SVRasterTrainer` 与 `VolumeRenderer` 紧密耦合
   - `SVRasterRenderer` 与 `VoxelRasterizer` 紧密耦合

3. **配置系统**：
   - 每个组件都有对应的配置类
   - 配置参数更加细化和专业化

4. **GPU 优化**：
   - 独立的 GPU 优化组件
   - CUDA 扩展可选加载

### 移除的组件

- `VoxelRasterizer` -> 重命名为 `VoxelRasterizer`
- `InteractiveRenderer` -> 合并到 `SVRasterRenderer`
- 一些旧的工具函数和配置选项

### 新增组件

- `VoxelRasterizer` - 专门的推理光栅化器
- `VolumeRenderer` - 专门的训练体积渲染器
- 更完善的配置系统
- 改进的 GPU 优化组件

## 🚀 使用示例

### 完整训练流程
```python
import nerfs.svraster as svraster

# 1. 配置
model_config = svraster.SVRasterConfig(max_octree_levels=8, base_resolution=128)
dataset_config = svraster.SVRasterDatasetConfig(data_dir="data/nerf_synthetic/lego")
trainer_config = svraster.SVRasterTrainerConfig(num_epochs=100)

# 2. 创建组件
model = svraster.SVRasterModel(model_config)
dataset = svraster.SVRasterDataset(dataset_config)
volume_renderer = svraster.VolumeRenderer(model_config)
trainer = svraster.SVRasterTrainer(model, volume_renderer, trainer_config)

# 3. 训练
trainer.train(dataset)
```

### 完整推理流程
```python
# 1. 加载模型
model = svraster.SVRasterModel.load_checkpoint("checkpoint.pth")

# 2. 配置渲染器
raster_config = svraster.VoxelRasterizerConfig()
renderer_config = svraster.SVRasterRendererConfig()

# 3. 创建渲染组件
rasterizer = svraster.VoxelRasterizer(raster_config)
renderer = svraster.SVRasterRenderer(model, rasterizer, renderer_config)

# 4. 渲染
camera_pose = torch.eye(4)
image = renderer.render(camera_pose, image_size=(800, 800))
```

这个 API 文档反映了最新的代码重构，提供了清晰的组件分离和使用指南。
