# SVRaster API 参考文档

本文档自动生成，包含 SVRaster 模块的所有公共 API。

## 📚 目录

### 🏗️ 类 (Classes)

- [AdaptiveSparseVoxels](#adaptivesparsevoxels)
- [EMAModel](#emamodel)
- [InteractiveRenderer](#interactiverenderer)
- [SVRasterConfig](#svrasterconfig)
- [SVRasterDataset](#svrasterdataset)
- [SVRasterDatasetConfig](#svrasterdatasetconfig)
- [SVRasterGPU](#svrastergpu)
- [SVRasterGPUTrainer](#svrastergputrainer)
- [SVRasterLoss](#svrasterloss)
- [SVRasterModel](#svrastermodel)
- [SVRasterOptimizedKernels](#svrasteroptimizedkernels)
- [SVRasterRenderer](#svrasterrenderer)
- [SVRasterRendererConfig](#svrasterrendererconfig)
- [SVRasterTrainer](#svrastertrainer)
- [SVRasterTrainerConfig](#svrastertrainerconfig)
- [VoxelRasterizer](#voxelrasterizer)

### 🔧 函数 (Functions)

- [compact1by2](#compact1by2)
- [compute_morton_order](#compute_morton_order)
- [compute_octree_level](#compute_octree_level)
- [compute_ray_aabb_intersection](#compute_ray_aabb_intersection)
- [compute_ray_samples](#compute_ray_samples)
- [compute_ssim](#compute_ssim)
- [compute_voxel_bounds](#compute_voxel_bounds)
- [create_svraster_dataloader](#create_svraster_dataloader)
- [create_svraster_dataset](#create_svraster_dataset)
- [create_svraster_trainer](#create_svraster_trainer)
- [depth_peeling](#depth_peeling)
- [eval_sh_basis](#eval_sh_basis)
- [get_compute_capability](#get_compute_capability)
- [get_cuda_version](#get_cuda_version)
- [get_octree_neighbors](#get_octree_neighbors)
- [main](#main)
- [morton_decode_3d](#morton_decode_3d)
- [morton_decode_batch](#morton_decode_batch)
- [morton_encode_3d](#morton_encode_3d)
- [morton_encode_batch](#morton_encode_batch)
- [octree_pruning](#octree_pruning)
- [octree_subdivision](#octree_subdivision)
- [part1by2](#part1by2)
- [part1by2](#part1by2)
- [part1by2_vectorized](#part1by2_vectorized)
- [part1by2_vectorized](#part1by2_vectorized)
- [ray_direction_dependent_ordering](#ray_direction_dependent_ordering)
- [render_rays](#render_rays)
- [sort_by_morton_order](#sort_by_morton_order)
- [volume_rendering_integration](#volume_rendering_integration)
- [voxel_pruning](#voxel_pruning)
- [voxel_to_world_coords](#voxel_to_world_coords)
- [world_to_voxel_coords](#world_to_voxel_coords)

## 🏗️ 类 (Classes)

### AdaptiveSparseVoxels

自适应稀疏体素表示类

这个类实现了基于八叉树的多分辨率自适应稀疏体素表示，是 SVRaster 的核心组件。
它管理不同八叉树层级的稀疏体素，只存储叶子节点，不存储完整的八叉树结构。

主要功能：
1. 初始化基础体素网格
2. 根据梯度或密度进行自适应细分
3. 基于密度阈值进行体素剪枝
4. 计算 Morton 码用于深度排序
5. 管理多层级体素的参量值

**模块路径**: `src/nerfs/svraster/core.py`

#### 方法 (Methods)

##### `__init__(self, config: SVRasterConfig)`

*暂无文档*

---

##### `add_voxels(self, positions, sizes, densities, colors, level = 0)`

向指定层级添加体素。
Args:
    positions: [N, 3] 体素中心
    sizes: [N] 体素尺寸
    densities: [N] 体素密度
    colors: [N, 3 * (sh_degree+1)^2] 体素颜色/SH系数
    level: int, 层级索引

---

##### `get_all_voxels(self) -> dict[str, torch.Tensor]`

获取所有层级的体素数据

将所有层级的体素数据合并为一个字典，包含位置、大小、密度、
颜色、层级和 Morton 码。

Returns:
    包含所有体素数据的字典

---

##### `get_total_voxel_count(self) -> int`

获取所有层级的体素总数

Returns:
    体素总数

---

##### `parameters(self) -> list[torch.Tensor]`

Get all optimizable parameters.

---

##### `prune_voxels(self, threshold: Optional[float] = None)`

移除低密度体素

根据密度阈值移除不重要的体素，减少内存使用并提高渲染效率。
密度低于阈值的体素被认为是透明的，对最终渲染结果贡献很小。

Args:
    threshold: 密度阈值，None 时使用配置中的默认值

---

##### `subdivide_voxels(self, subdivision_mask: torch.Tensor, level_idx: int)`

根据细分掩码细分体素

将选中的体素细分为 8 个子体素，每个子体素的大小是父体素的一半。
细分后的子体素被添加到下一层级，父体素被移除。

Args:
    subdivision_mask: 细分掩码，True 表示需要细分的体素
    level_idx: 当前层级索引

---



### EMAModel

Exponential Moving Average model wrapper.

Maintains moving averages of model parameters during training.
This can help produce more stable and better performing models.

**模块路径**: `src/nerfs/svraster/cuda/ema.py`

#### 方法 (Methods)

##### `__init__(self, model: torch.nn.Module, decay: float = 0.999, device: Optional[torch.device] = None)`

Initialize EMA model.

Args:
    model: The model whose parameters we want to maintain averages for
    decay: The decay rate for the moving average (default: 0.999)
    device: The device to store the EMA parameters on

---

##### `apply_shadow(self, model: torch.nn.Module) -> None`

Apply the moving averages to the model parameters.

Args:
    model: The model whose parameters to update with the moving averages

---

##### `load_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> None`

Load a state dictionary into the EMA model.

Args:
    state_dict: The state dictionary to load

---

##### `restore_original(self, model: torch.nn.Module) -> None`

Restore the original model parameters.

Args:
    model: The model whose parameters to restore

---

##### `state_dict(self) -> Dict[str, torch.Tensor]`

Get the state dictionary of the EMA model.

---

##### `update(self, model: torch.nn.Module) -> None`

Update moving averages of model parameters.

Args:
    model: The model whose parameters to update the moving averages with

---



### InteractiveRenderer

交互式渲染器

提供实时相机控制和渲染功能

**模块路径**: `src/nerfs/svraster/renderer.py`

#### 方法 (Methods)

##### `__init__(self, renderer: SVRasterRenderer, initial_pose: torch.Tensor, intrinsics: torch.Tensor, image_size: Optional[Tuple[int, int]] = None)`

*暂无文档*

---

##### `move_camera(self, direction: str, distance: Optional[float] = None) -> torch.Tensor`

移动相机

Args:
    direction: 移动方向 ("forward", "backward", "left", "right", "up", "down")
    distance: 移动距离，None 时使用默认速度

Returns:
    渲染结果

---

##### `render_current_view(self) -> torch.Tensor`

渲染当前视角

Returns:
    渲染的RGB图像

---

##### `rotate_camera(self, yaw: float = 0, pitch: float = 0) -> torch.Tensor`

旋转相机

Args:
    yaw: 偏航角（弧度）
    pitch: 俯仰角（弧度）

Returns:
    渲染结果

---

##### `set_pose(self, pose: torch.Tensor) -> torch.Tensor`

设置相机位姿

Args:
    pose: 新的相机位姿

Returns:
    渲染结果

---



### SVRasterConfig

SVRaster configuration.

Attributes:
    # Scene representation
    max_octree_levels: Maximum octree levels
    base_resolution: Base grid resolution
    scene_bounds: Scene bounds (min_x, min_y, min_z, max_x, max_y, max_z)

    # Voxel properties
    density_activation: Density activation function
    color_activation: Color activation function
    sh_degree: Spherical harmonics degree

    # Training
    learning_rate: Learning rate
    weight_decay: Weight decay
    gradient_clip_val: Gradient clipping value
    use_ema: Whether to use EMA model
    ema_decay: EMA decay rate
    steps_per_epoch: Number of steps per epoch
    num_epochs: Number of epochs
    save_best: Whether to save best model

    # Dataset
    image_width: Image width
    image_height: Image height
    camera_angle_x: Camera horizontal FoV in radians
    data_dir: Data directory
    num_rays_train: Number of rays per training batch
    downscale_factor: Image downscale factor for training

    # Rendering
    ray_samples_per_voxel: Number of samples per voxel along ray
    depth_peeling_layers: Number of depth peeling layers
    morton_ordering: Whether to use Morton ordering for depth sorting
    background_color: Background color (RGB)
    near_plane: Near plane distance
    far_plane: Far plane distance
    use_view_dependent_color: Whether to use view-dependent color
    use_opacity_regularization: Whether to use opacity regularization
    opacity_reg_weight: Opacity regularization weight
    use_ssim_loss: Whether to use SSIM loss
    ssim_loss_weight: SSIM loss weight
    use_distortion_loss: Whether to use distortion loss
    distortion_loss_weight: Distortion loss weight
    use_pointwise_rgb_loss: Whether to use pointwise RGB loss
    pointwise_rgb_loss_weight: Pointwise RGB loss weight

    # New attributes for modern optimizations
    volume_size: tuple[int, int, int] = (128, 128, 128)
    feature_dim: int = 32
    num_planes: int = 3
    plane_channels: int = 16
    hidden_dim: int = 64
    num_layers: int = 4
    skip_connections: list[int] = (2,)
    activation: str = "relu"
    output_activation: str = "sigmoid"
    num_samples: int = 128
    num_importance_samples: int = 64
    learning_rate_decay_steps: int = 20000
    learning_rate_decay_mult: float = 0.1
    use_amp: bool = True
    grad_scaler_init_scale: float = 65536.0
    grad_scaler_growth_factor: float = 2.0
    grad_scaler_backoff_factor: float = 0.5
    grad_scaler_growth_interval: int = 2000
    use_non_blocking: bool = True
    set_grad_to_none: bool = True
    chunk_size: int = 8192
    cache_volume_samples: bool = True
    default_device: str = "cuda"
    pin_memory: bool = True
    batch_size: int = 4096
    num_workers: int = 4
    checkpoint_dir: str = "checkpoints"
    save_every_n_steps: int = 1000
    keep_last_n_checkpoints: int = 5

**模块路径**: `src/nerfs/svraster/core.py`

#### 方法 (Methods)

##### `checkpoint_path(self) -> Path`

Get checkpoint directory as Path object.

---



### SVRasterDataset

SVRaster dataset with modern PyTorch features.

Features:
- Efficient data loading and preprocessing
- Memory-optimized ray generation
- Automatic mixed precision support
- CUDA acceleration with CPU fallback
- Flexible data augmentation

**模块路径**: `src/nerfs/svraster/dataset.py`

#### 方法 (Methods)

##### `__init__(self, config: SVRasterDatasetConfig, split: str = 'train')`

Initialize dataset.

Args:
    config: Dataset configuration
    split: Dataset split ("train", "val", or "test")

---

##### `get_dataset_info(self) -> dict[str, Any]`

Get dataset information.

---



### SVRasterDatasetConfig

SVRaster dataset configuration.

Attributes:
    data_dir: Data directory path
    image_width: Image width
    image_height: Image height
    camera_angle_x: Camera horizontal FoV in radians
    train_split: Training split ratio
    val_split: Validation split ratio
    test_split: Test split ratio
    downscale_factor: Image downscale factor
    color_space: Color space ("linear" or "srgb")
    num_rays_train: Number of rays per training batch
    num_rays_val: Number of rays per validation batch
    patch_size: Size of image patches for ray sampling

**模块路径**: `src/nerfs/svraster/dataset.py`



### SVRasterGPU

Enhanced GPU-optimized SVRaster model using CUDA kernels and advanced algorithms

**模块路径**: `src/nerfs/svraster/cuda/svraster_gpu.py`

#### 方法 (Methods)

##### `__init__(self, config)`

*暂无文档*

---

##### `adaptive_subdivision(self, subdivision_criteria: torch.Tensor) -> None`

Adaptively subdivide voxels based on criteria

---

##### `benchmark_performance(self, num_rays: int = 1000, num_iterations: int = 10) -> Dict[str, float]`

性能基准测试

Args:
    num_rays: 测试光线数量
    num_iterations: 测试迭代次数
    
Returns:
    性能统计结果

---

##### `export_performance_report(self, filepath: str) -> None`

导出性能报告

---

##### `forward(self, ray_origins: torch.Tensor, ray_directions: torch.Tensor) -> Dict[str, torch.Tensor]`

Forward pass using GPU-optimized kernels

---

##### `get_voxel_statistics(self) -> Dict[str, Any]`

Get statistics about voxel distribution

---

##### `optimize_for_production(self) -> None`

生产环境优化

---

##### `parameters(self)`

Return all trainable parameters as a generator (similar to nn.Module)

---

##### `print_performance_stats(self)`

Print performance statistics

---

##### `voxel_pruning(self, pruning_threshold: Optional[float] = None) -> None`

Prune low-density voxels

---



### SVRasterGPUTrainer

GPU-optimized trainer for SVRaster with modern PyTorch features.

**模块路径**: `src/nerfs/svraster/cuda/svraster_gpu.py`

#### 方法 (Methods)

##### `__init__(self, model: SVRasterGPU, config: SVRasterConfig)`

*暂无文档*

---

##### `save_checkpoint(self, filepath: str)`

Save checkpoint

---

##### `train_step(self, batch: dict[str, torch.Tensor]) -> dict[str, float]`

Perform a single training step.

---



### SVRasterLoss

Modern loss functions for SVRaster.

Features:
- Multiple loss terms with configurable weights
- SSIM loss for perceptual quality
- Distortion loss for geometry regularization
- Opacity regularization
- Pointwise RGB loss

**模块路径**: `src/nerfs/svraster/core.py`

#### 方法 (Methods)

##### `__call__(self, outputs: dict[str, torch.Tensor], targets: dict[str, torch.Tensor], model: SVRasterModel | None = None) -> dict[str, torch.Tensor]`

Compute all loss terms.

Args:
    outputs: Model outputs
    targets: Ground truth targets
    model: Optional model for regularization

Returns:
    Dictionary containing all loss terms and total loss

---

##### `__init__(self, config: SVRasterConfig)`

Initialize loss functions.

Args:
    config: Model configuration

---



### SVRasterModel

SVRaster model with modern PyTorch features.

Features:
- Efficient sparse voxel representation
- Automatic mixed precision training
- Memory-optimized operations
- CUDA acceleration with CPU fallback
- Real-time rendering capabilities

**模块路径**: `src/nerfs/svraster/core.py`

#### 方法 (Methods)

##### `__init__(self, config: SVRasterConfig)`

*暂无文档*

---

##### `evaluate(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]`

Evaluate model with modern optimizations.

Args:
    batch: Dictionary containing evaluation data

Returns:
    Dictionary containing evaluation metrics

---

##### `forward(self, ray_origins: torch.Tensor, ray_directions: torch.Tensor, camera_params: dict[str, torch.Tensor] | None = None) -> dict[str, torch.Tensor]`

Forward pass with modern optimizations.

Args:
    ray_origins: Ray origins [N, 3]
    ray_directions: Ray directions [N, 3]
    camera_params: Optional camera parameters

Returns:
    Dictionary containing rendered outputs

---

##### `render_image(self, camera_pose: torch.Tensor, camera_intrinsics: torch.Tensor, image_size: tuple[int, int], device: torch.device | None = None) -> dict[str, torch.Tensor]`

Render a full image efficiently.

Args:
    camera_pose: Camera-to-world transform [4, 4]
    camera_intrinsics: Camera intrinsics [3, 3]
    image_size: Output image size (H, W)
    device: Optional device to render on

Returns:
    Dictionary containing rendered image and auxiliary outputs

---

##### `train_step(self, batch: dict[str, torch.Tensor], optimizer: torch.optim.Optimizer) -> dict[str, torch.Tensor]`

Perform a single training step with modern optimizations.

Args:
    batch: Dictionary containing training data
    optimizer: PyTorch optimizer

Returns:
    Dictionary containing loss values

---



### SVRasterOptimizedKernels

SVRaster优化核心算法集合
实现高效的体素遍历、Morton码排序和内存管理

**模块路径**: `src/nerfs/svraster/cuda/svraster_optimized_kernels.py`

#### 方法 (Methods)

##### `__init__(self, device: torch.device)`

*暂无文档*

---

##### `cleanup_memory_pool(self)`

清理内存池

---

##### `get_performance_stats(self) -> Dict[str, float]`

获取性能统计信息

---

##### `memory_efficient_allocation(self, size: int, dtype: torch.dtype = torch.float32) -> torch.Tensor`

内存高效的张量分配
使用内存池避免频繁的分配和释放

---

##### `optimized_morton_sorting(self, positions: torch.Tensor, scene_bounds: torch.Tensor, precision_bits: int = 21) -> torch.Tensor`

优化的Morton码计算和排序

Args:
    positions: 位置坐标 [N, 3]
    scene_bounds: 场景边界 [6] (min_x, min_y, min_z, max_x, max_y, max_z)
    precision_bits: 精度位数
    
Returns:
    Morton码 [N]

---

##### `optimized_ray_voxel_intersection(self, ray_origins: torch.Tensor, ray_directions: torch.Tensor, voxel_positions: torch.Tensor, voxel_sizes: torch.Tensor, use_spatial_hash: bool = True) -> Dict[str, torch.Tensor]`

优化的光线-体素相交测试

Args:
    ray_origins: 光线起点 [N, 3]
    ray_directions: 光线方向 [N, 3]
    voxel_positions: 体素位置 [V, 3]
    voxel_sizes: 体素大小 [V]
    use_spatial_hash: 是否使用空间哈希加速
    
Returns:
    相交结果字典

---

##### `reset_performance_counters(self)`

重置性能计数器

---



### SVRasterRenderer

SVRaster 渲染器

负责加载训练好的模型并进行高质量渲染

**模块路径**: `src/nerfs/svraster/renderer.py`

#### 方法 (Methods)

##### `__init__(self, config: SVRasterRendererConfig)`

*暂无文档*

---

##### `get_model_info(self) -> Dict[str, Any]`

获取模型信息

Returns:
    模型信息字典

---

##### `interactive_render(self, initial_pose: torch.Tensor, intrinsics: torch.Tensor, image_size: Optional[Tuple[int, int]] = None) -> InteractiveRenderer`

启动交互式渲染模式

Args:
    initial_pose: 初始相机位姿
    intrinsics: 相机内参
    image_size: 图像尺寸

Returns:
    交互式渲染器实例

---

##### `load_model(self, checkpoint_path: Union[str, Path]) -> None`

加载训练好的模型

Args:
    checkpoint_path: 模型检查点路径

---

##### `render_360_video(self, center: torch.Tensor, radius: float, intrinsics: torch.Tensor, num_frames: int = 120, output_path: Union[str, Path] = '360_video.mp4', fps: int = 30) -> str`

渲染 360 度环绕视频

Args:
    center: 环绕中心点 [3]
    radius: 环绕半径
    intrinsics: 相机内参
    num_frames: 视频帧数
    output_path: 输出视频路径
    fps: 帧率

Returns:
    视频文件路径

---

##### `render_path(self, camera_poses: List[torch.Tensor], intrinsics: torch.Tensor, output_dir: Union[str, Path], image_size: Optional[Tuple[int, int]] = None, save_video: bool = True, fps: int = 30) -> List[str]`

渲染相机路径

Args:
    camera_poses: 相机位姿列表
    intrinsics: 相机内参
    output_dir: 输出目录
    image_size: 图像尺寸
    save_video: 是否保存视频
    fps: 视频帧率

Returns:
    生成的图像文件路径列表

---

##### `render_single_view(self, camera_pose: torch.Tensor, intrinsics: torch.Tensor, image_size: Optional[Tuple[int, int]] = None) -> Dict[str, torch.Tensor]`

渲染单个视角

Args:
    camera_pose: 相机位姿矩阵 [4, 4] (world to camera)
    intrinsics: 相机内参矩阵 [3, 3] 或 [4, 4]
    image_size: 图像尺寸 (width, height)，None 时使用配置中的尺寸

Returns:
    渲染结果字典，包含 'rgb', 'depth' 等

---



### SVRasterRendererConfig

SVRaster 渲染器配置

**模块路径**: `src/nerfs/svraster/renderer.py`

#### 方法 (Methods)



### SVRasterTrainer

Main trainer class for SVRaster model.

**模块路径**: `src/nerfs/svraster/trainer.py`

#### 方法 (Methods)

##### `__init__(self, model_config: SVRasterConfig, trainer_config: SVRasterTrainerConfig, train_dataset: SVRasterDataset, val_dataset: Optional[SVRasterDataset] = None) -> None`

*暂无文档*

---

##### `load_checkpoint(self, checkpoint_path: str)`

Load model checkpoint.

---

##### `train(self)`

Main training loop.

---



### SVRasterTrainerConfig

Configuration for SVRaster trainer.

**模块路径**: `src/nerfs/svraster/trainer.py`

#### 方法 (Methods)



### VoxelRasterizer

体素光栅化器

实现了基于光线投射的体素光栅化算法，支持自适应采样和深度剥离。
不包含任何可训练参数，只负责渲染过程。

**模块路径**: `src/nerfs/svraster/core.py`

#### 方法 (Methods)

##### `__call__(self, voxels: dict[str, torch.Tensor], ray_origins: torch.Tensor, ray_directions: torch.Tensor, camera_params: Optional[dict[str, torch.Tensor]] = None) -> dict[str, torch.Tensor]`

执行光栅化过程

Args:
    voxels: 包含体素属性的字典
    ray_origins: 光线起点 [N, 3]
    ray_directions: 光线方向 [N, 3]
    camera_params: 可选的相机参数

Returns:
    包含渲染结果的字典

---

##### `__init__(self, config: SVRasterConfig)`

*暂无文档*

---



## 🔧 函数 (Functions)

### `compact1by2(n)`

Compact bits by removing two zeros between each bit.

**模块路径**: `src/nerfs/svraster/utils/morton_utils.py`

---

### `compute_morton_order(positions: torch.Tensor, scene_bounds: tuple[float, float, float, float, float, float], grid_resolution: int) -> torch.Tensor`

Compute Morton order for a set of 3D positions.

Args:
    positions: Tensor of 3D positions [N, 3]
    scene_bounds: Scene bounds (min_x, min_y, min_z, max_x, max_y, max_z)
    grid_resolution: Grid resolution for discretization
    
Returns:
    Morton codes for the positions [N]

**模块路径**: `src/nerfs/svraster/utils/morton_utils.py`

---

### `compute_octree_level(voxel_size: float, base_size: float) -> int`

Compute octree level based on voxel size.

Args:
    voxel_size: Size of the voxel
    base_size: Base voxel size at level 0
    
Returns:
    Octree level

**模块路径**: `src/nerfs/svraster/utils/octree_utils.py`

---

### `compute_ray_aabb_intersection(ray_origin: torch.Tensor, ray_direction: torch.Tensor, box_min: torch.Tensor, box_max: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]`

Compute ray-AABB intersection.

Args:
    ray_origin: Ray origin [3] or [N, 3]
    ray_direction: Ray direction [3] or [N, 3]
    box_min: Box minimum corner [3] or [M, 3]
    box_max: Box maximum corner [3] or [M, 3]
    
Returns:
    tuple of (t_near, t_far, valid_mask)

**模块路径**: `src/nerfs/svraster/utils/rendering_utils.py`

---

### `compute_ray_samples(ray_origins: torch.Tensor, ray_directions: torch.Tensor, near: float, far: float, num_samples: int = 64, perturb: bool = True) -> dict[str, torch.Tensor]`

Compute sample points along rays.

**模块路径**: `src/nerfs/svraster/utils/rendering_utils.py`

---

### `compute_ssim(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11) -> torch.Tensor`

Compute SSIM between two images

**模块路径**: `src/nerfs/svraster/core.py`

---

### `compute_voxel_bounds(voxel_positions: torch.Tensor, voxel_sizes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]`

Compute bounding boxes for voxels.

Args:
    voxel_positions: Voxel center positions [N, 3]
    voxel_sizes: Voxel sizes [N]
    
Returns:
    tuple of (box_min, box_max) each [N, 3]

**模块路径**: `src/nerfs/svraster/utils/voxel_utils.py`

---

### `create_svraster_dataloader(config: SVRasterDatasetConfig, split: str = 'train', batch_size: int = 1, shuffle: bool = True, num_workers: int = 4) -> DataLoader`

Create a DataLoader for SVRaster dataset.

**模块路径**: `src/nerfs/svraster/dataset.py`

---

### `create_svraster_dataset(config: SVRasterDatasetConfig, split: str = 'train') -> SVRasterDataset`

Create a SVRaster dataset.

**模块路径**: `src/nerfs/svraster/dataset.py`

---

### `create_svraster_trainer(model_config: SVRasterConfig, trainer_config: SVRasterTrainerConfig, train_dataset: SVRasterDataset, val_dataset: Optional[SVRasterDataset] = None) -> SVRasterTrainer`

Create a SVRaster trainer.

**模块路径**: `src/nerfs/svraster/trainer.py`

---

### `depth_peeling(voxel_positions: torch.Tensor, voxel_sizes: torch.Tensor, ray_origin: torch.Tensor, ray_direction: torch.Tensor, num_layers: int = 4) -> list[torch.Tensor]`

Perform depth peeling for correct transparency rendering.

Args:
    voxel_positions: Voxel positions [N, 3]
    voxel_sizes: Voxel sizes [N]
    ray_origin: Ray origin [3]
    ray_direction: Ray direction [3]
    num_layers: Number of depth layers to peel
    
Returns:
    list of voxel indices for each depth layer

**模块路径**: `src/nerfs/svraster/utils/rendering_utils.py`

---

### `eval_sh_basis(degree: int, dirs: torch.Tensor) -> torch.Tensor`

计算球谐函数基（支持 0~3 阶），输入方向 shape [..., 3]，输出 shape [..., num_sh_coeffs]

**模块路径**: `src/nerfs/svraster/core.py`

---

### `get_compute_capability()`

*暂无文档*

**模块路径**: `src/nerfs/svraster/cuda/setup.py`

---

### `get_cuda_version()`

*暂无文档*

**模块路径**: `src/nerfs/svraster/cuda/setup.py`

---

### `get_octree_neighbors(voxel_position: torch.Tensor, voxel_size: float, all_positions: torch.Tensor, all_sizes: torch.Tensor) -> torch.Tensor`

Find neighboring voxels in the octree.

Args:
    voxel_position: Position of the query voxel [3]
    voxel_size: Size of the query voxel
    all_positions: All voxel positions [N, 3]
    all_sizes: All voxel sizes [N]
    
Returns:
    Indices of neighboring voxels

**模块路径**: `src/nerfs/svraster/utils/octree_utils.py`

---

### `main()`

*暂无文档*

**模块路径**: `src/nerfs/svraster/cuda/setup.py`

---

### `morton_decode_3d(morton_code: int) -> tuple[int, int, int]`

Decode Morton code back to 3D coordinates.

Args:
    morton_code: Morton code as integer
    
Returns:
    tuple of (x, y, z) coordinates

**模块路径**: `src/nerfs/svraster/utils/morton_utils.py`

---

### `morton_decode_batch(morton_codes: torch.Tensor) -> torch.Tensor`

Decode batch of Morton codes back to 3D coordinates.

Args:
    morton_codes: Tensor of Morton codes [N]
    
Returns:
    Tensor of coordinates [N, 3]

**模块路径**: `src/nerfs/svraster/utils/morton_utils.py`

---

### `morton_encode_3d(x: int, y: int, z: int) -> int`

Encode 3D coordinates into Morton code.

改进的 Morton 编码实现，支持更高的分辨率：
- 每个坐标分量使用 21 位，总共支持 2097151³ 个位置
- 完全满足 SVRaster 的 65536³ 分辨率需求

Args:
    x, y, z: 3D coordinates
    
Returns:
    Morton code as integer

**模块路径**: `src/nerfs/svraster/utils/morton_utils.py`

---

### `morton_encode_batch(coords: torch.Tensor) -> torch.Tensor`

Encode batch of 3D coordinates into Morton codes.

使用向量化操作进行批量 Morton 码计算，提高性能。

Args:
    coords: Tensor of shape [N, 3] with integer coordinates
    
Returns:
    Tensor of Morton codes [N]

**模块路径**: `src/nerfs/svraster/utils/morton_utils.py`

---

### `octree_pruning(voxel_densities: torch.Tensor, threshold: float = 0.001) -> torch.Tensor`

Create pruning mask for low-density voxels.

Args:
    voxel_densities: Voxel density values [N]
    threshold: Pruning threshold
    
Returns:
    Boolean mask for voxels to keep [N]

**模块路径**: `src/nerfs/svraster/utils/octree_utils.py`

---

### `octree_subdivision(voxel_positions: torch.Tensor, voxel_sizes: torch.Tensor, subdivision_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]`

Subdivide voxels into 8 child voxels.

Args:
    voxel_positions: Voxel center positions [N, 3]
    voxel_sizes: Voxel sizes [N]
    subdivision_mask: Boolean mask for voxels to subdivide [N]
    
Returns:
    tuple of (child_positions, child_sizes)

**模块路径**: `src/nerfs/svraster/utils/octree_utils.py`

---

### `part1by2(n)`

*暂无文档*

**模块路径**: `src/nerfs/svraster/core.py`

---

### `part1by2(n)`

Separate bits by inserting two zeros between each bit.

**模块路径**: `src/nerfs/svraster/utils/morton_utils.py`

---

### `part1by2_vectorized(n)`

*暂无文档*

**模块路径**: `src/nerfs/svraster/core.py`

---

### `part1by2_vectorized(n)`

*暂无文档*

**模块路径**: `src/nerfs/svraster/utils/morton_utils.py`

---

### `ray_direction_dependent_ordering(voxel_positions: torch.Tensor, morton_codes: torch.Tensor, ray_direction: torch.Tensor) -> torch.Tensor`

Sort voxels using ray direction-dependent Morton ordering.

Args:
    voxel_positions: Voxel positions [N, 3]
    morton_codes: Morton codes for voxels [N]
    ray_direction: Mean ray direction [3]
    
Returns:
    Sort indices for correct depth ordering

**模块路径**: `src/nerfs/svraster/utils/rendering_utils.py`

---

### `render_rays(ray_origins: torch.Tensor, ray_directions: torch.Tensor, near: float, far: float, num_samples: int = 64, perturb: bool = True) -> dict[str, torch.Tensor]`

Render rays using volume rendering.

**模块路径**: `src/nerfs/svraster/utils/rendering_utils.py`

---

### `sort_by_morton_order(positions: torch.Tensor, scene_bounds: tuple[float, float, float, float, float, float], grid_resolution: int) -> tuple[torch.Tensor, torch.Tensor]`

Sort positions by Morton order.

Args:
    positions: Tensor of 3D positions [N, 3]
    scene_bounds: Scene bounds
    grid_resolution: Grid resolution for discretization
    
Returns:
    tuple of (sorted_positions, sort_indices)

**模块路径**: `src/nerfs/svraster/utils/morton_utils.py`

---

### `volume_rendering_integration(densities: torch.Tensor, colors: torch.Tensor, distances: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]`

Perform volume rendering integration.

Args:
    densities: Density values along ray [N]
    colors: Color values along ray [N, 3]
    distances: Distance intervals [N]
    
Returns:
    tuple of (integrated_color, integrated_alpha)

**模块路径**: `src/nerfs/svraster/utils/rendering_utils.py`

---

### `voxel_pruning(voxel_densities: torch.Tensor, voxel_positions: torch.Tensor, voxel_sizes: torch.Tensor, threshold: float = 0.001) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]`

Prune voxels with low density.

Args:
    voxel_densities: Voxel density values [N]
    voxel_positions: Voxel positions [N, 3]
    voxel_sizes: Voxel sizes [N]
    threshold: Pruning threshold
    
Returns:
    tuple of (pruned_densities, pruned_positions, pruned_sizes)

**模块路径**: `src/nerfs/svraster/utils/voxel_utils.py`

---

### `voxel_to_world_coords(voxel_coords: torch.Tensor, scene_bounds: tuple[float, float, float, float, float, float], grid_resolution: int) -> torch.Tensor`

Convert voxel coordinates to world coordinates.

Args:
    voxel_coords: Voxel coordinates [N, 3]
    scene_bounds: Scene bounds (min_x, min_y, min_z, max_x, max_y, max_z)
    grid_resolution: Grid resolution
    
Returns:
    World coordinates [N, 3]

**模块路径**: `src/nerfs/svraster/utils/voxel_utils.py`

---

### `world_to_voxel_coords(world_coords: torch.Tensor, scene_bounds: tuple[float, float, float, float, float, float], grid_resolution: int) -> torch.Tensor`

Convert world coordinates to voxel coordinates.

Args:
    world_coords: World coordinates [N, 3]
    scene_bounds: Scene bounds (min_x, min_y, min_z, max_x, max_y, max_z)
    grid_resolution: Grid resolution
    
Returns:
    Voxel coordinates [N, 3]

**模块路径**: `src/nerfs/svraster/utils/voxel_utils.py`

---


---
*本文档生成于 2025-07-05 23:36:46*
