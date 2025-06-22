# Mega-NeRF: 大规模NeRF的可扩展构建用于虚拟飞行浏览

本包实现了Mega-NeRF，这是一种用于可扩展构建大规模神经辐射场的方法，能够实现真实世界环境的虚拟飞行浏览。

## 概述

Mega-NeRF通过以下方式解决了将NeRF应用到大规模场景的挑战：
- **空间分割**：将大场景分解为可管理的子区域
- **几何感知数据分割**：智能地将训练数据分布到各个子模块
- **并行训练**：同时训练多个子模块以提高效率
- **时间连贯性**：确保子模块边界间的平滑渲染

## 特性

### 核心组件

- **MegaNeRF模型**：带有空间分解的主模型
- **MegaNeRF子模块**：用于场景区域的独立NeRF网络
- **空间分割器**：基于网格和几何感知的分割策略
- **体积渲染器**：带有分层采样的高效基于光线的渲染
- **训练流水线**：顺序和并行训练模式

### 核心能力

- ✅ 大规模场景重建（城市规模）
- ✅ 可配置网格大小的空间分解
- ✅ 使用相机位置的几何感知分割
- ✅ 子模块的并行训练
- ✅ 处理不同光照条件的外观嵌入
- ✅ 带有缓存的交互式渲染
- ✅ 飞行浏览序列的视频生成
- ✅ 多种数据格式支持（COLMAP, LLFF, NeRF格式）

## 安装

```bash
# 克隆仓库
git clone <repository-url>
cd NeuroCity

# 安装依赖
pip install -r requirements.txt

# 以开发模式安装包
pip install -e .
```

## 快速开始

### 1. 训练Mega-NeRF模型

```bash
python src/mega-nerf/train_mega_nerf.py \
    --data_root /path/to/your/dataset \
    --exp_name my_meganerf_experiment \
    --num_submodules 8 \
    --grid_size 4,2 \
    --iterations_per_submodule 10000 \
    --training_mode sequential
```

### 2. 渲染新视图

```bash
python src/mega-nerf/render_mega_nerf.py \
    --model_path experiments/my_meganerf_experiment/final_model.pth \
    --output_dir renders/spiral_path \
    --render_type spiral \
    --num_frames 120 \
    --create_video
```

### 3. 使用Python API

```python
from src.mega_nerf import (
    MegaNeRF, MegaNeRFConfig, MegaNeRFTrainer,
    MegaNeRFDataset, GridPartitioner
)

# 设置配置
config = MegaNeRFConfig(
    num_submodules=8,
    grid_size=(4, 2),
    hidden_dim=256,
    num_layers=8
)

# 创建空间分割器
partitioner = GridPartitioner(
    scene_bounds=(-100, -100, -10, 100, 100, 50),
    grid_size=(4, 2),
    overlap_factor=0.15
)

# 创建数据集
dataset = MegaNeRFDataset(
    data_root="path/to/data",
    partitioner=partitioner
)

# 创建并训练模型
model = MegaNeRF(config)
trainer = MegaNeRFTrainer(config, model, dataset, "output_dir")
trainer.train_sequential()
```

## 配置

### 模型配置

```python
config = MegaNeRFConfig(
    # 场景分解
    num_submodules=8,           # 子模块数量
    grid_size=(4, 2),           # 2D网格分解
    overlap_factor=0.15,        # 子模块间重叠
    
    # 网络架构
    hidden_dim=256,             # 隐藏层维度
    num_layers=8,               # 网络层数
    use_viewdirs=True,          # 使用视角方向
    
    # 训练参数
    batch_size=1024,            # 光线批量大小
    learning_rate=5e-4,         # 学习率
    max_iterations=500000,      # 最大迭代次数
    
    # 采样参数
    num_coarse=256,             # 每光线粗采样数
    num_fine=512,               # 每光线精细采样数
    near=0.1,                   # 近平面
    far=1000.0,                 # 远平面
    
    # 场景边界 (x_min, y_min, z_min, x_max, y_max, z_max)
    scene_bounds=(-100, -100, -10, 100, 100, 50)
)
```

### 分割策略

#### 网格分割器
```python
partitioner = GridPartitioner(
    scene_bounds=scene_bounds,
    grid_size=(4, 2),           # 4x2网格
    overlap_factor=0.15         # 15%重叠
)
```

#### 几何感知分割器
```python
partitioner = GeometryAwarePartitioner(
    scene_bounds=scene_bounds,
    camera_positions=camera_positions,
    num_partitions=8,
    use_kmeans=True             # 使用k-means聚类
)
```

## 训练模式

### 顺序训练
逐个训练子模块：
```bash
python train_mega_nerf.py --training_mode sequential
```

### 并行训练
同时训练多个子模块：
```bash
python train_mega_nerf.py --training_mode parallel --num_parallel_workers 4
```

## 数据格式

### 支持的格式

1. **COLMAP**：标准COLMAP稀疏重建
2. **LLFF**：局部光场融合格式
3. **NeRF**：原始NeRF transforms.json格式
4. **合成**：生成的合成数据

### 数据结构
```
dataset/
├── images/              # 输入图像
├── sparse/             # COLMAP稀疏重建（可选）
├── transforms.json     # NeRF格式（可选）
├── poses_bounds.npy    # LLFF格式（可选）
└── train.txt          # 训练分割（可选）
```

## 渲染选项

### 渲染类型

1. **单视图**：渲染单个视点
2. **螺旋路径**：围绕场景中心的圆形相机路径
3. **自定义路径**：用户定义的相机轨迹

### 渲染示例

```python
from src.mega_nerf import MegaNeRFRenderer

# 创建渲染器
renderer = MegaNeRFRenderer(model, config)

# 渲染单张图像
rgb_image = renderer.render_image(
    height=800,
    width=800,
    camera_matrix=K,
    camera_pose=c2w
)

# 渲染视频序列
video_frames = renderer.render_video(
    camera_path=spiral_poses,
    height=800,
    width=800
)
```

## 性能优化

### 内存管理

```python
# 子模块缓存
class SubmoduleCache:
    def __init__(self, max_cached=4):
        self.max_cached = max_cached
        self.cache = {}
        self.access_time = {}
    
    def get_submodule(self, submodule_id):
        if submodule_id not in self.cache:
            if len(self.cache) >= self.max_cached:
                # 移除最久未使用的子模块
                lru_id = min(self.access_time, key=self.access_time.get)
                del self.cache[lru_id]
                del self.access_time[lru_id]
            
            # 加载子模块
            self.cache[submodule_id] = self._load_submodule(submodule_id)
        
        self.access_time[submodule_id] = time.time()
        return self.cache[submodule_id]
```

### 分布式训练

```python
# 多GPU并行训练
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

def setup_distributed(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def train_distributed(rank, world_size, config):
    setup_distributed(rank, world_size)
    
    model = MegaNeRF(config).to(rank)
    model = DistributedDataParallel(model, device_ids=[rank])
    
    # 训练逻辑
    trainer = MegaNeRFTrainer(config, model, dataset, output_dir)
    trainer.train()
```

## 应用示例

### 城市场景重建

```python
# 大规模城市数据集
config = MegaNeRFConfig(
    num_submodules=16,
    grid_size=(4, 4),
    scene_bounds=(-1000, -1000, -50, 1000, 1000, 100),
    max_iterations=1000000
)

# 训练
trainer = MegaNeRFTrainer(config, model, dataset, "city_reconstruction")
trainer.train_parallel(num_workers=8)
```

### 自然场景

```python
# 自然环境数据
config = MegaNeRFConfig(
    num_submodules=8,
    grid_size=(2, 4),
    use_appearance_embedding=True,
    appearance_dim=32
)
```

## 评估指标

### 定量评估

```python
from src.mega_nerf.evaluation import evaluate_model

metrics = evaluate_model(
    model=model,
    test_dataset=test_dataset,
    output_dir="evaluation_results"
)

print(f"PSNR: {metrics['psnr']:.2f}")
print(f"SSIM: {metrics['ssim']:.3f}")
print(f"LPIPS: {metrics['lpips']:.3f}")
print(f"渲染时间: {metrics['render_time']:.2f}秒")
```

### 定性评估

```python
# 生成对比图像
from src.mega_nerf.visualization import create_comparison_grid

comparison_grid = create_comparison_grid(
    ground_truth_images=gt_images,
    rendered_images=rendered_images,
    save_path="comparison.png"
)
```

## 故障排除

### 常见问题

**子模块边界伪影**
```python
# 增加重叠因子
config.overlap_factor = 0.2

# 使用更平滑的混合权重
config.blend_weights = 'gaussian'
```

**内存不足**
```python
# 减少批量大小
config.batch_size = 512

# 减少同时训练的子模块数
config.num_parallel_workers = 2
```

**训练收敛慢**
```python
# 调整学习率
config.learning_rate = 1e-3

# 使用学习率调度
config.lr_scheduler = 'cosine'
config.lr_decay_steps = 250000
```

## 技术细节

### 空间分割算法

```python
def create_spatial_partition(scene_bounds, grid_size, overlap_factor):
    """创建空间分割"""
    x_min, y_min, z_min, x_max, y_max, z_max = scene_bounds
    
    # 计算网格大小
    grid_x, grid_y = grid_size
    cell_width = (x_max - x_min) / grid_x
    cell_height = (y_max - y_min) / grid_y
    
    # 计算重叠大小
    overlap_x = cell_width * overlap_factor
    overlap_y = cell_height * overlap_factor
    
    partitions = []
    for i in range(grid_x):
        for j in range(grid_y):
            # 计算单元边界
            cell_x_min = x_min + i * cell_width - overlap_x
            cell_x_max = x_min + (i + 1) * cell_width + overlap_x
            cell_y_min = y_min + j * cell_height - overlap_y
            cell_y_max = y_min + (j + 1) * cell_height + overlap_y
            
            partition = {
                'id': i * grid_y + j,
                'bounds': (cell_x_min, cell_y_min, z_min, 
                          cell_x_max, cell_y_max, z_max),
                'center': ((cell_x_min + cell_x_max) / 2,
                          (cell_y_min + cell_y_max) / 2,
                          (z_min + z_max) / 2)
            }
            partitions.append(partition)
    
    return partitions
```

### 混合权重计算

```python
def compute_blend_weights(point, partitions, overlap_factor):
    """计算混合权重"""
    weights = []
    
    for partition in partitions:
        bounds = partition['bounds']
        center = partition['center']
        
        # 计算到分割中心的距离
        distance = np.linalg.norm(point - center)
        
        # 计算分割半径
        radius = np.linalg.norm([
            (bounds[3] - bounds[0]) / 2,
            (bounds[4] - bounds[1]) / 2,
            (bounds[5] - bounds[2]) / 2
        ])
        
        # 计算权重（高斯衰减）
        if distance <= radius:
            weight = np.exp(-0.5 * (distance / (radius * overlap_factor)) ** 2)
        else:
            weight = 0.0
        
        weights.append(weight)
    
    # 归一化权重
    total_weight = sum(weights)
    if total_weight > 0:
        weights = [w / total_weight for w in weights]
    
    return weights
```

## 许可证

MIT许可证

## 引用

```bibtex
@inproceedings{turki2022meganerf,
  title={Mega-NeRF: Scalable Construction of Large-Scale NeRFs for Virtual Fly-Throughs},
  author={Turki, Haithem and others},
  booktitle={CVPR},
  year={2022}
}
```

## 贡献

欢迎贡献！请参阅CONTRIBUTING.md获取详细信息。

## 支持

如有问题或建议，请创建GitHub Issue或参与讨论。 