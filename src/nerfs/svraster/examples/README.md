# NeRF Examples

这个目录包含了各种 NeRF 实现的示例脚本。

## 可用示例

### Grid-NeRF
- **文件**: `grid_nerf_example.py`
- **描述**: 展示如何使用 Grid-NeRF 进行大规模城市场景的训练和评估
- **用法**:
  ```bash
  # 基础示例
  python examples/grid_nerf_example.py --example basic
  
  # KITTI-360 训练
  python examples/grid_nerf_example.py --example kitti360 --data_path /path/to/kitti360 --output_dir ./outputs/kitti360
  
  # 自定义数据集
  python examples/grid_nerf_example.py --example custom --data_path /path/to/data --output_dir ./outputs/custom
  
  # 模型评估
  python examples/grid_nerf_example.py --example eval --model_path ./model.pth --data_path /path/to/test --output_dir ./eval_results
  
  # 多GPU训练
  python examples/grid_nerf_example.py --example multi_gpu --data_path /path/to/data --num_gpus 4
  
  # 生成螺旋视频
  python examples/grid_nerf_example.py --example spiral --model_path ./model.pth --output_dir ./spiral_video
  ```

### Classic NeRF
- **文件**: `classic_nerf_example.py`
- **描述**: 经典 NeRF 实现的示例
- **用法**:
  ```bash
  # 基础示例
  python examples/classic_nerf_example.py --example basic
  
  # 训练
  python examples/classic_nerf_example.py --example train --data_path /path/to/data --output_dir ./outputs/classic
  ```

### Instant NGP
- **文件**: `instant_ngp_example.py`
- **描述**: 快速 NeRF 训练的 Instant NGP 实现
- **用法**:
  ```bash
  # 基础示例
  python examples/instant_ngp_example.py --example basic
  
  # 快速训练
  python examples/instant_ngp_example.py --example train --data_path /path/to/data --output_dir ./outputs/ngp
  ```

### Nerfacto
- **文件**: `nerfacto_example.py`
- **描述**: Nerfacto 方法的示例
- **用法**:
  ```bash
  # 基础示例
  python examples/nerfacto_example.py --example basic
  
  # 训练
  python examples/nerfacto_example.py --example train --data_path /path/to/data --output_dir ./outputs/nerfacto
  ```

## 数据格式

大多数示例期望数据按以下格式组织： 