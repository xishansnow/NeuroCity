# NeRF CLI 使用示例

本文档展示了各个 NeRF 模块的命令行界面使用方法。

## 通用用法

所有 CLI 程序都支持 `train` 和 `render` 两个子命令：

```bash
# 训练模型
python src/nerfs/[module_name]/cli.py train [options]

# 渲染图像/视频
python src/nerfs/[module_name]/cli.py render [options]
```

## 1. Mega-NeRF CLI

### 训练 Mega-NeRF 模型

```bash
python src/nerfs/mega_nerf/cli.py train \
    --data-dir /path/to/data \
    --dataset-type nerf \
    --num-submodules 8 \
    --grid-size 4 2 \
    --hidden-dim 256 \
    --num-layers 8 \
    --num-epochs 100 \
    --batch-size 1024 \
    --learning-rate 5e-4 \
    --checkpoint-dir ./checkpoints \
    --output-dir ./outputs \
    --device cuda
```

### 渲染 Mega-NeRF 模型

```bash
# 渲染单张图像
python src/nerfs/mega_nerf/cli.py render \
    --checkpoint ./checkpoints/model.pth \
    --width 512 \
    --height 512 \
    --output-dir ./renders \
    --device cuda

# 渲染视频序列
python src/nerfs/mega_nerf/cli.py render \
    --checkpoint ./checkpoints/model.pth \
    --camera-poses ./camera_poses.json \
    --render-video \
    --fps 30 \
    --width 512 \
    --height 512 \
    --output-dir ./renders \
    --device cuda
```

## 2. SV-Raster CLI

### 训练 SV-Raster 模型

```bash
python src/nerfs/svraster/cli.py train \
    --data-dir /path/to/data \
    --dataset-type nerf \
    --hidden-dim 256 \
    --num-layers 8 \
    --num-samples 64 \
    --num-epochs 100 \
    --batch-size 1024 \
    --learning-rate 5e-4 \
    --checkpoint-dir ./checkpoints \
    --output-dir ./outputs \
    --device cuda
```

### 渲染 SV-Raster 模型

```bash
python src/nerfs/svraster/cli.py render \
    --checkpoint ./checkpoints/model.pth \
    --width 512 \
    --height 512 \
    --output-dir ./renders \
    --device cuda
```

## 3. Plenoxels CLI

### 训练 Plenoxels 模型

```bash
python src/nerfs/plenoxels/cli.py train \
    --data-dir /path/to/data \
    --dataset-type blender \
    --grid-resolution 128 \
    --feature-dim 32 \
    --num-samples 64 \
    --num-epochs 100 \
    --batch-size 1024 \
    --learning-rate 1e-2 \
    --checkpoint-dir ./checkpoints \
    --output-dir ./outputs \
    --device cuda
```

### 渲染 Plenoxels 模型

```bash
python src/nerfs/plenoxels/cli.py render \
    --checkpoint ./checkpoints/model.pth \
    --width 512 \
    --height 512 \
    --output-dir ./renders \
    --device cuda
```

## 4. Block-NeRF CLI

### 训练 Block-NeRF 模型

```bash
python src/nerfs/block_nerf/cli.py train \
    --data-dir /path/to/data \
    --dataset-type waymo \
    --num-blocks 8 \
    --block-size 50.0 \
    --hidden-dim 256 \
    --num-layers 8 \
    --num-samples 64 \
    --num-epochs 100 \
    --batch-size 1024 \
    --learning-rate 5e-4 \
    --checkpoint-dir ./checkpoints \
    --output-dir ./outputs \
    --device cuda
```

### 渲染 Block-NeRF 模型

```bash
python src/nerfs/block_nerf/cli.py render \
    --checkpoint ./checkpoints/model.pth \
    --width 512 \
    --height 512 \
    --output-dir ./renders \
    --device cuda
```

## 5. Inf-NeRF CLI

### 训练 Inf-NeRF 模型

```bash
python src/nerfs/inf_nerf/cli.py train \
    --data-dir /path/to/data \
    --dataset-type nerf \
    --hidden-dim 256 \
    --num-layers 8 \
    --num-samples 64 \
    --use-lod \
    --num-epochs 100 \
    --batch-size 1024 \
    --learning-rate 5e-4 \
    --checkpoint-dir ./checkpoints \
    --output-dir ./outputs \
    --device cuda
```

### 渲染 Inf-NeRF 模型

```bash
python src/nerfs/inf_nerf/cli.py render \
    --checkpoint ./checkpoints/model.pth \
    --width 512 \
    --height 512 \
    --output-dir ./renders \
    --device cuda
```

## 6. Instant-NGP CLI

### 训练 Instant-NGP 模型

```bash
python src/nerfs/instant_ngp/cli.py train \
    --data-dir /path/to/data \
    --dataset-type blender \
    --num-levels 16 \
    --base-resolution 16 \
    --finest-resolution 512 \
    --num-epochs 20 \
    --batch-size 8192 \
    --learning-rate 1e-2 \
    --learning-rate-hash 1e-1 \
    --checkpoint-dir ./checkpoints \
    --output-dir ./outputs \
    --device cuda
```

### 渲染 Instant-NGP 模型

```bash
python src/nerfs/instant_ngp/cli.py render \
    --checkpoint ./checkpoints/model.pth \
    --width 512 \
    --height 512 \
    --output-dir ./renders \
    --device cuda
```

## 数据集格式

### NeRF 格式
```
data/
├── transforms.json
└── images/
    ├── image_000.png
    ├── image_001.png
    └── ...
```

### COLMAP 格式
```
data/
├── images/
├── sparse/
└── cameras.txt
```

### LLFF 格式
```
data/
├── images/
├── poses_bounds.npy
└── ...
```

## 相机位姿文件格式

用于视频渲染的相机位姿文件应为 JSON 格式：

```json
{
    "poses": [
        [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 3.0, 0.0, 0.0, 0.0, 1.0],
        ...
    ],
    "intrinsics": [
        [400.0, 0.0, 256.0],
        [0.0, 400.0, 256.0],
        [0.0, 0.0, 1.0]
    ]
}
```

## 环境要求

- Python 3.10+
- PyTorch 2.0+
- CUDA (可选，用于 GPU 加速)
- 其他依赖见各模块的 requirements.txt

## 故障排除

### 常见问题

1. **CUDA 内存不足**
   - 减少 `--batch-size`
   - 减少 `--num-samples`
   - 使用 `--device cpu`

2. **导入错误**
   - 确保已激活正确的 conda 环境
   - 检查依赖包是否正确安装

3. **数据路径错误**
   - 确保数据目录存在且格式正确
   - 检查文件权限

### 调试模式

```bash
# 启用详细日志
export PYTHONPATH=.
python -u src/nerfs/[module_name]/cli.py train --data-dir /path/to/data
```

## 性能优化

1. **GPU 内存优化**
   - 使用混合精度训练
   - 调整批次大小
   - 使用梯度累积

2. **训练速度优化**
   - 使用多 GPU 训练
   - 调整学习率调度
   - 使用数据预加载

3. **渲染速度优化**
   - 减少采样数量
   - 使用分块渲染
   - 调整图像分辨率 