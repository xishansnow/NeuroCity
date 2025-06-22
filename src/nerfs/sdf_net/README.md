# DeepSDF Networks (SDFNet)

基于论文 [DeepSDF: Learning Continuous Signed Distance Functions for Shape Representation](https://arxiv.org/abs/1901.05103) 的PyTorch实现。

## 概述

DeepSDF 学习一个连续函数 `f: (R³, Z) → R`，其中 Z 是形状的潜在编码，输出是有符号距离值。这种表示方法通过zero-level set重建3D表面，支持形状的潜在空间插值和编辑。

## 主要特性

- **连续SDF表示**: 学习空间中任意点的有符号距离值
- **潜在编码**: 支持形状的潜在空间表示和插值
- **高质量重建**: 通过zero-level set提取高质量表面
- **几何初始化**: 专门的网络权重初始化策略
- **Eikonal约束**: 梯度惩罚确保SDF性质

## 架构组件

### 核心模型 (`core.py`)

#### `SDFNetwork`
- 基础SDF网络模型
- 使用跳跃连接的深度网络
- 几何感知的权重初始化
- 支持权重归一化

#### `LatentSDFNetwork`
- 带潜在编码的SDF网络
- 支持形状编码和解码
- 可用于形状插值和生成

#### `MultiScaleSDFNetwork`
- 多尺度SDF网络
- 支持不同分辨率的SDF预测
- 渐进式细节建模

### 数据处理 (`dataset.py`)

#### `SDFDataset`
- 从3D网格生成SDF训练数据
- 支持表面、近表面、和均匀采样
- 自动计算SDF值
- 支持距离值截断

#### `SyntheticSDFDataset`
- 合成几何形状SDF数据集
- 支持球体、立方体、圆柱体等
- 解析SDF计算

#### `LatentSDFDataset`
- 支持潜在编码的SDF数据集
- 预训练潜在编码加载
- 形状编码自动生成

### 训练器 (`trainer.py`)

#### `SDFTrainer`
- 完整的SDF训练管道
- 支持Eikonal约束
- 自动网格提取和评估
- 潜在编码优化

### 工具函数 (`utils/`)

- **SDF处理**: SDF计算、网格提取、表面重建
- **可视化**: SDF场可视化、等值面提取
- **评估指标**: Chamfer距离、法向量一致性等

## 快速开始

### 1. 安装依赖

```bash
pip install torch torchvision
pip install trimesh scikit-image
pip install tensorboard scipy
```

### 2. 基本使用

```python
from src.models.sdf_net import SDFNetwork, SDFDataset, SDFTrainer

# 创建模型
model = SDFNetwork(
    dim_input=3,
    dim_latent=256,
    dim_hidden=512,
    num_layers=8,
    skip_connections=[4]
)

# 创建数据集
dataset = SDFDataset(
    data_root='path/to/data',
    split='train',
    num_points=100000
)

# 创建训练器
trainer = SDFTrainer(
    model=model,
    train_dataloader=dataloader,
    device='cuda'
)

# 开始训练
trainer.train(num_epochs=100)
```

### 3. 网格重建

```python
# 使用训练好的模型重建网格
latent_code = torch.randn(1, 256)  # 随机潜在编码

mesh_data = model.extract_mesh(
    latent_code=latent_code,
    resolution=256,
    threshold=0.0
)

# 保存网格
import trimesh
mesh = trimesh.Trimesh(
    vertices=mesh_data['vertices'],
    faces=mesh_data['faces']
)
mesh.export('reconstructed_mesh.obj')
```

## 配置示例

```python
config = {
    'model': {
        'dim_input': 3,
        'dim_latent': 256,
        'dim_hidden': 512,
        'num_layers': 8,
        'skip_connections': [4],
        'geometric_init': True,
        'weight_norm': True
    },
    'dataset': {
        'data_root': 'data/shapenet',
        'num_points': 100000,
        'surface_sampling': 0.4,
        'near_surface_sampling': 0.4,
        'uniform_sampling': 0.2,
        'clamp_distance': 0.1
    },
    'optimizer': {
        'type': 'adam',
        'lr': 1e-4,
        'weight_decay': 0
    },
    'trainer': {
        'device': 'cuda',
        'lambda_gp': 0.1,  # Eikonal约束权重
        'log_dir': 'logs/sdf_net',
        'checkpoint_dir': 'checkpoints/sdf_net'
    }
}

trainer = create_trainer_from_config(config)
```

## 数据格式

### 输入数据结构
```
data_root/
├── meshes/
│   ├── shape1.obj
│   ├── shape2.obj
│   └── ...
├── latent_codes/
│   └── codes.pth
├── train.json
├── val.json
└── test.json
```

### 训练样本格式
```python
sample = {
    'points': torch.Tensor,      # [N, 3] 查询点坐标
    'sdf': torch.Tensor,         # [N, 1] SDF值
    'latent_code': torch.Tensor, # [dim_latent] 潜在编码
    'shape_id': str             # 形状标识符
}
```

## 核心技术

### 几何初始化
```python
# 输出层初始化 - 近似单位球SDF
torch.nn.init.normal_(
    layer.weight, 
    mean=np.sqrt(np.pi) / np.sqrt(layer.in_features), 
    std=0.0001
)
torch.nn.init.constant_(layer.bias, -bias)

# 隐藏层初始化 - Xavier变种
torch.nn.init.normal_(
    layer.weight, 
    0.0, 
    np.sqrt(2) / np.sqrt(layer.out_features)
)
```

### Eikonal约束
```python
def compute_gradient_penalty(self, points, latent_code, lambda_gp=0.1):
    """确保 ||∇f|| = 1"""
    points.requires_grad_(True)
    sdf = self.forward(points, latent_code)
    
    gradients = torch.autograd.grad(
        outputs=sdf,
        inputs=points,
        grad_outputs=torch.ones_like(sdf),
        create_graph=True
    )[0]
    
    gradient_norm = gradients.norm(2, dim=-1)
    penalty = lambda_gp * ((gradient_norm - 1) ** 2).mean()
    return penalty
```

### 跳跃连接
```python
# 在指定层添加跳跃连接
if layer_idx in self.skip_in:
    x = torch.cat([x, inputs], dim=-1) / np.sqrt(2)
```

## 高级功能

### 潜在编码优化
```python
latent_sdf = LatentSDFNetwork(
    dim_latent=256,
    num_shapes=1000  # 预定义形状数量
)

# 获取形状的潜在编码
shape_code = latent_sdf.get_latent_code(shape_id=42)

# 形状插值
code1 = latent_sdf.get_latent_code(0)
code2 = latent_sdf.get_latent_code(1)
interpolated_code = 0.5 * code1 + 0.5 * code2
```

### 多尺度训练
```python
multiscale_model = MultiScaleSDFNetwork(
    scales=[1.0, 0.5, 0.25]
)

# 多尺度预测
multi_sdf = multiscale_model.forward_multiscale(points, latent_code)
```

### 自适应采样
```python
class AdaptiveSDFSampler:
    def __init__(self, model, num_iterations=5):
        self.model = model
        self.num_iterations = num_iterations
    
    def sample_points(self, latent_code, num_points):
        """在SDF梯度大的区域增加采样密度"""
        points = torch.randn(num_points, 3)
        
        for _ in range(self.num_iterations):
            points.requires_grad_(True)
            sdf = self.model(points, latent_code)
            gradients = torch.autograd.grad(sdf.sum(), points)[0]
            
            # 在梯度大的区域重新采样
            gradient_magnitude = gradients.norm(dim=-1)
            high_gradient_mask = gradient_magnitude > gradient_magnitude.median()
            
            # 更新高梯度区域的点
            points[high_gradient_mask] += torch.randn_like(points[high_gradient_mask]) * 0.01
        
        return points
```

## 评估指标

- **Chamfer Distance**: 点云之间的双向最近邻距离
- **Normal Consistency**: 法向量一致性评估
- **Edge Length Ratio**: 网格边长比例评估
- **Volume Difference**: 体积差异比较

## 性能优化

### 分层训练
```python
# 先训练粗糙网络，再fine-tune细节
coarse_trainer = SDFTrainer(model, coarse_data, resolution=64)
coarse_trainer.train(epochs=50)

fine_trainer = SDFTrainer(model, fine_data, resolution=256)
fine_trainer.train(epochs=50)
```

### 内存高效推理
```python
def extract_mesh_memory_efficient(self, latent_code, resolution=256, chunk_size=100000):
    """内存高效的网格提取"""
    # 分块处理大规模点云
    for chunk in point_chunks:
        sdf_chunk = self.predict_sdf(chunk, latent_code, chunk_size)
        # 合并结果...
```

## 应用场景

- **3D形状补全**: 从部分观测重建完整形状
- **形状插值**: 在潜在空间中进行形状变形
- **3D生成**: 从潜在编码生成新形状
- **形状编辑**: 通过潜在编码操作编辑形状

## 引用

如果使用本实现，请引用原始论文：

```bibtex
@inproceedings{park2019deepsdf,
  title={DeepSDF: Learning continuous signed distance functions for shape representation},
  author={Park, Jeong Joon and Florence, Peter and Straub, Julian and Newcombe, Richard and Lovegrove, Steven},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={165--174},
  year={2019}
}
```

## 许可证

本实现遵循原项目的许可证条款。 