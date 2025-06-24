# Grid-NeRF: 大规模城市场景的网格引导神经辐射场

本包实现了网格引导神经辐射场（Grid-NeRF），这是一种使用分层体素网格引导渲染过程来实现大规模城市环境神经渲染的可扩展方法。

## 概述

Grid-NeRF通过以下方式解决大规模城市场景渲染的挑战：
- 使用分层体素网格组织场景几何
- 采用多分辨率网格结构进行高效采样
- 使用网格引导特征优化神经网络
- 支持大规模数据集的分布式训练

## 关键特性

- **分层网格结构**：具有递增分辨率的多级体素网格
- **网格引导神经网络**：利用网格特征进行高效渲染的MLP
- **可扩展训练**：支持分布式训练的大规模城市数据集
- **快速渲染**：基于网格采样的高效体积渲染
- **多数据集支持**：内置支持KITTI-360和自定义数据集
- **综合评估**：包括PSNR、SSIM和LPIPS等指标

## 安装

### 前置要求

- Python 3.8+
- PyTorch 1.12+
- CUDA 11.0+（用于GPU加速）

### 安装依赖

```bash
pip install torch torchvision torchaudio
pip install opencv-python matplotlib numpy
pip install pyyaml tqdm tensorboard
pip install lpips  # 可选，用于LPIPS指标
```

### 包安装

```bash
# 从项目根目录
cd src/grid_nerf
pip install -e .
```

## 快速开始

### 基本使用

```python
from src.grid_nerf import GridNeRF, GridNeRFConfig, GridNeRFTrainer
from src.grid_nerf import create_dataset, quick_setup

# 使用默认设置快速配置
model, trainer, dataset = quick_setup(
    data_path="path/to/your/data",
    output_dir="./outputs",
    device="cuda"
)

# 开始训练
trainer.train(train_dataset=dataset)
```

### 自定义配置

```python
from src.grid_nerf import GridNeRFConfig, GridNeRF

# 创建自定义配置
config = GridNeRFConfig(
    # 场景边界
    scene_bounds=(-100, -100, -10, 100, 100, 50),
    
    # 网格配置
    grid_levels=4,
    base_resolution=64,
    resolution_multiplier=2,
    grid_feature_dim=32,
    
    # 网络架构
    density_layers=3,
    density_hidden_dim=256,
    color_layers=2,
    color_hidden_dim=128,
    
    # 训练参数
    batch_size=1024,
    num_epochs=200,
    grid_lr=1e-2,
    mlp_lr=5e-4
)

# 创建模型
model = GridNeRF(config)
```

## 架构

### 分层网格结构

分层网格包含多个分辨率递增的级别：

```
级别0: 64³体素（粗略）
级别1: 128³体素 
级别2: 256³体素
级别3: 512³体素（精细）
```

每个级别存储特征向量，这些向量被结合起来引导神经网络。

### 网格引导MLP

神经网络包含：
- **密度网络**：从网格特征和位置预测体积密度
- **颜色网络**：从网格特征、位置和视角方向预测RGB颜色
- **位置编码**：空间坐标的正弦编码
- **方向编码**：视角方向的编码

### 体积渲染

渲染器执行：
1. **光线采样**：沿相机光线采样点
2. **网格采样**：从分层网格提取特征
3. **网络评估**：在采样点预测密度和颜色
4. **体积积分**：使用alpha混合合成最终像素颜色

## 训练

### 单GPU训练

```python
from src.grid_nerf import GridNeRFTrainer, GridNeRFConfig
from src.grid_nerf.dataset import create_dataset

# 设置配置
config = GridNeRFConfig(
    batch_size=1024,
    num_epochs=100,
    grid_lr=1e-2,
    mlp_lr=5e-4
)

# 创建训练器
trainer = GridNeRFTrainer(
    config=config,
    output_dir="./outputs",
    device=torch.device("cuda")
)

# 加载数据集
train_dataset = create_dataset(
    data_path="path/to/data",
    split='train',
    config=config
)

# 开始训练
trainer.train(train_dataset=train_dataset)
```

### 多GPU训练

```python
import torch.multiprocessing as mp
from src.grid_nerf.trainer import main_worker

# 配置
config = GridNeRFConfig(batch_size=2048)  # 为多GPU扩展
data_config = {
    'train_data_path': 'path/to/data',
    'train_kwargs': {}
}

# 启动分布式训练
num_gpus = 4
mp.spawn(
    main_worker,
    args=(num_gpus, config, "./outputs", data_config),
    nprocs=num_gpus,
    join=True
)
```

## 数据集格式

### 目录结构

```
data/
├── images/
│   ├── 000000.png
│   ├── 000001.png
│   └── ...
├── poses/
│   ├── 000000.npy  # 相机姿态矩阵（4x4）
│   ├── 000001.npy
│   └── ...
├── intrinsics.npy  # 相机内参（3x3）
└── scene_bounds.npy  # 场景边界[x_min, y_min, z_min, x_max, y_max, z_max]
```

### KITTI-360数据集

```python
from src.grid_nerf.dataset import KITTI360GridDataset

dataset = KITTI360GridDataset(
    data_root="/path/to/kitti360",
    sequence="2013_05_28_drive_0000_sync",
    frame_range=(0, 1000),
    image_scale=0.5
)
```

### 自定义数据集

```python
from src.grid_nerf.dataset import GridNeRFDataset

dataset = GridNeRFDataset(
    data_root="/path/to/custom/data",
    split="train",
    image_scale=1.0,
    load_lidar=False
)
```

## 配置选项

### 场景配置

```python
config = GridNeRFConfig(
    # 场景边界（米）
    scene_bounds=(-200, -200, -20, 200, 200, 80),
    
    # 分层网格设置
    grid_levels=4,               # 网格级别数
    base_resolution=32,          # 基础分辨率
    resolution_multiplier=2,     # 分辨率倍数
    grid_feature_dim=16,         # 每个体素的特征维度
    
    # 网络架构
    density_layers=3,            # 密度网络层数
    density_hidden_dim=128,      # 密度网络隐藏维度
    color_layers=2,              # 颜色网络层数
    color_hidden_dim=64,         # 颜色网络隐藏维度
    
    # 位置编码
    pos_encode_levels=8,         # 位置编码级别
    dir_encode_levels=4,         # 方向编码级别
)
```

### 训练配置

```python
config = GridNeRFConfig(
    # 训练参数
    batch_size=2048,             # 光线批量大小
    num_epochs=300,              # 训练轮数
    src/nerfs/dnmp_nerf
    grid_lr=5e-3,                # 网格特征学习率
    mlp_lr=1e-3,                 # MLP网络学习率
    
    # 采样
    num_samples=64,              # 每光线采样数
    num_importance=0,            # 重要性采样数（0表示不使用）
    
    # 正则化
    sparsity_loss_weight=1e-4,   # 稀疏性损失权重
    tv_loss_weight=1e-6,         # 总变分损失权重
    
    # 调度
    lr_decay_steps=50000,        # 学习率衰减步数
    lr_decay_factor=0.5,         # 学习率衰减因子
)
```

## 高级功能

### 网格优化

```python
# 自适应网格细化
def adaptive_grid_refinement(grid_features, density_threshold=0.01):
    """基于密度阈值自适应细化网格"""
    high_density_regions = grid_features > density_threshold
    refined_grid = refine_grid_regions(grid_features, high_density_regions)
    return refined_grid

# 网格压缩
def compress_grid_features(grid_features, compression_ratio=0.5):
    """压缩网格特征以节省内存"""
    compressed_features = quantize_features(grid_features, compression_ratio)
    return compressed_features
```

### 多尺度采样

```python
def multiscale_sampling(rays, scene_bounds, grid_levels):
    """基于场景复杂度的多尺度采样"""
    sampling_density = compute_sampling_density(rays, scene_bounds)
    
    for level in range(grid_levels):
        level_samples = adapt_samples_to_level(sampling_density, level)
        yield level_samples
```

## 渲染管道

### 光线生成

```python
def generate_rays(height, width, intrinsics, camera_pose):
    """为给定相机生成光线"""
    # 创建像素坐标网格
    i, j = torch.meshgrid(
        torch.arange(width),
        torch.arange(height),
        indexing='xy'
    )
    
    # 转换为相机坐标
    directions = torch.stack([
        (i - intrinsics[0, 2]) / intrinsics[0, 0],
        -(j - intrinsics[1, 2]) / intrinsics[1, 1],
        -torch.ones_like(i)
    ], dim=-1)
    
    # 转换为世界坐标
    rays_d = torch.sum(directions[..., None, :] * camera_pose[:3, :3], dim=-1)
    rays_o = camera_pose[:3, 3].expand(rays_d.shape)
    
    return rays_o, rays_d
```

### 网格插值

```python
def interpolate_grid_features(positions, grid_features, grid_bounds):
    """从网格插值特征"""
    # 将位置归一化到网格坐标
    normalized_pos = normalize_coordinates(positions, grid_bounds)
    
    # 三线性插值
    interpolated_features = trilinear_interpolation(
        grid_features, normalized_pos
    )
    
    return interpolated_features
```

## 评估和可视化

### 指标计算

```python
from src.grid_nerf.evaluation import compute_metrics

# 渲染测试图像
test_images = renderer.render_test_images(model, test_dataset)

# 计算指标
metrics = compute_metrics(
    pred_images=test_images,
    target_images=ground_truth_images
)

print(f"PSNR: {metrics['psnr']:.2f} dB")
print(f"SSIM: {metrics['ssim']:.4f}")
print(f"LPIPS: {metrics['lpips']:.4f}")
```

### 可视化工具

```python
from src.grid_nerf.visualization import visualize_grid, create_flythrough

# 可视化网格结构
visualize_grid(
    grid_features=model.grid_features,
    level=2,
    output_path="grid_visualization.png"
)

# 创建飞行浏览视频
create_flythrough(
    model=model,
    camera_path=spiral_path,
    output_path="flythrough.mp4",
    fps=30
)
```

## 性能优化

### 内存优化

```python
# 分块渲染
def chunk_render(model, rays, chunk_size=4096):
    """分块渲染以节省GPU内存"""
    results = []
    for i in range(0, rays.shape[0], chunk_size):
        chunk = rays[i:i+chunk_size]
        with torch.no_grad():
            result = model.render_rays(chunk)
        results.append(result)
    return torch.cat(results, dim=0)

# 网格特征缓存
class GridFeatureCache:
    def __init__(self, max_cache_size=8):
        self.cache = {}
        self.max_size = max_cache_size
    
    def get_features(self, grid_level, grid_coords):
        cache_key = (grid_level, tuple(grid_coords))
        if cache_key not in self.cache:
            if len(self.cache) >= self.max_size:
                # 移除最久未使用的条目
                oldest_key = min(self.cache.keys())
                del self.cache[oldest_key]
            
            self.cache[cache_key] = self._load_features(grid_level, grid_coords)
        
        return self.cache[cache_key]
```

### 计算优化

```python
# 并行网格查询
def parallel_grid_query(positions, grid_features, num_workers=4):
    """并行查询网格特征"""
    import concurrent.futures
    
    chunk_size = len(positions) // num_workers
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        
        for i in range(0, len(positions), chunk_size):
            chunk = positions[i:i+chunk_size]
            future = executor.submit(interpolate_grid_features, chunk, grid_features)
            futures.append(future)
        
        results = [future.result() for future in futures]
    
    return torch.cat(results, dim=0)
```

## 实验和基准

### 城市场景基准

```python
# KITTI-360基准测试
def run_kitti360_benchmark():
    config = GridNeRFConfig(
        scene_bounds=(-100, -100, -5, 100, 100, 15),
        grid_levels=5,
        base_resolution=128,
        batch_size=4096,
        num_epochs=500
    )
    
    dataset = KITTI360GridDataset(
        data_root="/path/to/kitti360",
        sequence="2013_05_28_drive_0000_sync"
    )
    
    trainer = GridNeRFTrainer(config)
    trainer.train(dataset)
    
    # 评估
    metrics = trainer.evaluate(dataset.test_split)
    return metrics

# 合成数据基准测试
def run_synthetic_benchmark():
    # 生成合成城市场景
    synthetic_data = generate_synthetic_city(
        city_size=(1000, 1000, 100),
        num_buildings=500,
        num_views=1000
    )
    
    # 训练和评估
    model = GridNeRF(config)
    metrics = train_and_evaluate(model, synthetic_data)
    return metrics
```

### 性能分析

```python
def profile_training_performance():
    """分析训练性能"""
    import time
    import psutil
    
    start_time = time.time()
    start_memory = psutil.virtual_memory().used
    
    # 训练一个epoch
    trainer.train_epoch(train_dataset)
    
    end_time = time.time()
    end_memory = psutil.virtual_memory().used
    
    print(f"训练时间: {end_time - start_time:.2f}秒")
    print(f"内存使用: {(end_memory - start_memory) / 1e9:.2f} GB")
```

## 故障排除

### 常见问题

**内存不足**
```python
# 解决方案：减少批量大小和网格分辨率
config.batch_size = 1024
config.base_resolution = 32
config.grid_levels = 3
```

**训练收敛慢**
```python
# 解决方案：调整学习率和网格初始化
config.grid_lr = 1e-2
config.mlp_lr = 5e-4
config.grid_init_scale = 0.1
```

**渲染质量差**
```python
# 解决方案：增加网格分辨率和网络容量
config.base_resolution = 128
config.grid_levels = 5
config.density_hidden_dim = 256
```

### 调试工具

```python
# 可视化网格激活
def visualize_grid_activations(model, test_rays):
    activations = model.get_grid_activations(test_rays)
    plt.imshow(activations.cpu().numpy())
    plt.title("网格激活")
    plt.show()

# 检查网格利用率
def check_grid_utilization(grid_features):
    active_voxels = (grid_features.abs() > 1e-6).float().mean()
    print(f"活跃体素比例: {active_voxels:.2%}")
```

## 应用场景

### 城市规划

```python
# 城市场景重建用于规划
urban_config = GridNeRFConfig(
    scene_bounds=(-500, -500, -10, 500, 500, 100),
    grid_levels=6,
    base_resolution=256
)

# 训练大规模城市模型
urban_model = train_urban_model(urban_config, city_dataset)

# 生成不同视角的城市视图
planning_views = generate_planning_views(urban_model)
```

### 自动驾驶仿真

```python
# 自动驾驶场景仿真
driving_config = GridNeRFConfig(
    scene_bounds=(-200, -200, -5, 200, 200, 20),
    temporal_consistency=True,
    vehicle_dynamics=True
)

# 训练动态场景模型
driving_model = train_driving_model(driving_config, kitti_dataset)

# 生成仿真数据
simulation_data = generate_simulation_views(driving_model)
```

## 许可证

MIT许可证

## 引用

```bibtex
@article{chen2022gridnerf,
  title={Grid-guided Neural Radiance Fields for Large Urban Scenes},
  author={Chen, Anpei and others},
  journal={Computer Vision and Pattern Recognition},
  year={2022}
}
```

## 贡献

欢迎贡献！请遵循以下步骤：

1. Fork项目
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开Pull Request

## 支持

如有问题或需要支持，请：
- 创建GitHub Issue
- 查看文档：https://grid-nerf.readthedocs.io
- 联系开发团队：grid-nerf@example.com 