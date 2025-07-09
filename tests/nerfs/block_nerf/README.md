# Block-NeRF Test Suite

这个目录包含 Block-NeRF 实现的完整测试套件，包括单元测试、集成测试和性能测试。

## 测试结构

```
tests/nerfs/block_nerf/
├── __init__.py                 # 测试工具和配置
├── conftest.py                # pytest 配置
├── test_core.py               # 核心模型测试
├── test_trainer.py            # 训练器测试
├── test_renderer.py           # 渲染器测试
├── test_dataset.py            # 数据集测试
├── test_integrations.py       # 集成测试
├── test_runner.py             # 测试运行脚本
└── README.md                  # 本文件
```

## 测试类别

### 单元测试
- **test_core.py**: 测试 Block-NeRF 核心组件
  - `BlockNeRFConfig`: 配置管理
  - `BlockNeRFModel`: 核心模型
  - `BlockNeRFLoss`: 损失函数
  - 工具函数

- **test_trainer.py**: 测试训练管道
  - `BlockNeRFTrainer`: 训练器
  - `BlockNeRFTrainerConfig`: 训练配置
  - 优化器和学习率调度
  - 混合精度训练
  - 检查点保存和加载

- **test_renderer.py**: 测试渲染管道
  - `BlockNeRFRenderer`: 渲染器
  - `BlockNeRFRendererConfig`: 渲染配置
  - 体渲染算法
  - 分层采样
  - 图像渲染

- **test_dataset.py**: 测试数据处理
  - `BlockNeRFDataset`: 数据集
  - `BlockNeRFDatasetConfig`: 数据配置
  - 数据加载和预处理
  - 数据增强
  - 批处理

### 集成测试
- **test_integrations.py**: 端到端工作流测试
  - 完整训练流程
  - 训练到推理管道
  - 多块训练
  - 外观嵌入学习
  - 曝光条件控制

## 运行测试

### 快速开始
```bash
# 运行所有测试
cd tests/nerfs/block_nerf
python test_runner.py

# 只运行快速测试（跳过慢速测试）
python test_runner.py --fast

# 运行特定模块的测试
python test_runner.py --module core
python test_runner.py --module trainer
python test_runner.py --module renderer
```

### 使用 pytest 直接运行
```bash
# 运行所有测试
pytest tests/nerfs/block_nerf/ -v

# 只运行单元测试
pytest tests/nerfs/block_nerf/ -m "unit" -v

# 只运行集成测试
pytest tests/nerfs/block_nerf/ -m "integration" -v

# 跳过慢速测试
pytest tests/nerfs/block_nerf/ -m "not slow" -v

# 包含 CUDA 测试（需要 GPU）
pytest tests/nerfs/block_nerf/ --runcuda -v

# 生成覆盖率报告
pytest tests/nerfs/block_nerf/ --cov=src.nerfs.block_nerf --cov-report=html -v
```

### 测试选项

- `--fast`: 只运行快速测试，跳过标记为 `slow` 的测试
- `--cuda`: 包含需要 CUDA 的测试
- `--integration`: 只运行集成测试
- `--unit`: 只运行单元测试
- `--coverage`: 生成代码覆盖率报告
- `--verbose`: 详细输出
- `--module MODULE`: 运行特定模块的测试

## 测试配置

### 测试数据配置
测试使用的默认配置在 `__init__.py` 中定义：

```python
TEST_CONFIG = {
    "device": "auto",  # auto, cpu, cuda
    "precision": "float32",
    "batch_size": 4,
    "num_rays": 1024,
    "scene_bounds": (-10, -10, -2, 10, 10, 2),
    "block_size": 5.0,
    "max_blocks": 8,
    "appearance_dim": 32,
    "num_epochs": 2,
    "learning_rate": 5e-4,
}
```

### 自定义测试配置
可以通过环境变量覆盖默认配置：

```bash
export BLOCKNERF_TEST_DEVICE=cuda
export BLOCKNERF_TEST_BATCH_SIZE=8
export BLOCKNERF_TEST_NUM_RAYS=2048
```

## 测试工具

### 便利函数
- `get_test_device()`: 获取最佳可用设备
- `create_test_data()`: 创建合成测试数据
- `create_test_camera()`: 创建测试相机参数
- `skip_if_no_cuda()`: 在没有 CUDA 时跳过测试
- `skip_if_slow()`: 跳过慢速测试的装饰器

### Mock 数据
测试使用 mock 数据和图像来避免依赖外部数据集：

```python
# 创建测试数据
test_data = create_test_data(batch_size=4, num_rays=1024)

# 创建测试相机
camera = create_test_camera()
```

## 性能测试

### 基准测试
```bash
# 运行性能基准测试
pytest tests/nerfs/block_nerf/ -k "benchmark" --benchmark-only
```

### 内存使用测试
```bash
# 使用 memory_profiler 监控内存使用
pytest tests/nerfs/block_nerf/ --profile-memory
```

## 调试测试

### 详细输出
```bash
# 获取详细的错误信息
pytest tests/nerfs/block_nerf/ -vvv --tb=long

# 在第一个失败时停止
pytest tests/nerfs/block_nerf/ -x

# 运行特定测试
pytest tests/nerfs/block_nerf/test_core.py::TestBlockNeRFModel::test_model_forward -v
```

### 使用 pdb 调试
```bash
# 在失败时进入 pdb
pytest tests/nerfs/block_nerf/ --pdb

# 在测试开始时进入 pdb
pytest tests/nerfs/block_nerf/ --pdbcls=IPython.terminal.debugger:Pdb --pdb
```

## 持续集成

### GitHub Actions
测试可以配置为在 GitHub Actions 中运行：

```yaml
name: Block-NeRF Tests
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.8
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest pytest-cov
    - name: Run tests
      run: |
        cd tests/nerfs/block_nerf
        python test_runner.py --fast --coverage
```

## 最佳实践

### 编写测试
1. **测试命名**: 使用描述性的测试名称
2. **测试隔离**: 每个测试应该独立运行
3. **Mock外部依赖**: 使用 mock 来隔离外部依赖
4. **参数化测试**: 使用 `@pytest.mark.parametrize` 测试多种情况
5. **适当的断言**: 使用具体的断言消息

### 性能考虑
1. **小规模测试**: 使用小的网络和批量大小进行快速测试
2. **标记慢速测试**: 使用 `@skip_if_slow()` 标记慢速测试
3. **GPU测试**: 使用 `@skip_if_no_cuda()` 标记 GPU 专用测试
4. **内存管理**: 在测试后清理大型张量

### 错误处理
1. **预期异常**: 使用 `pytest.raises()` 测试异常情况
2. **边界条件**: 测试边界条件和极端输入
3. **数值稳定性**: 检查 NaN 和 Inf 值
4. **形状检查**: 验证张量形状的正确性

## 贡献

当添加新功能时，请确保：

1. 添加相应的单元测试
2. 更新集成测试（如果需要）
3. 运行完整测试套件
4. 检查代码覆盖率
5. 更新文档

## 故障排除

### 常见问题

1. **CUDA 内存不足**:
   ```bash
   # 减少批量大小
   export BLOCKNERF_TEST_BATCH_SIZE=2
   export BLOCKNERF_TEST_NUM_RAYS=512
   ```

2. **测试超时**:
   ```bash
   # 只运行快速测试
   python test_runner.py --fast
   ```

3. **依赖问题**:
   ```bash
   # 安装测试依赖
   pip install pytest pytest-cov pytest-mock
   ```

4. **Mock 错误**:
   确保所有外部依赖都正确 mock，特别是图像加载和数据集访问。

## 更多信息

- [pytest 文档](https://docs.pytest.org/)
- [torch.testing 文档](https://pytorch.org/docs/stable/testing.html)
- [Block-NeRF 论文](https://arxiv.org/abs/2112.00525)
- [项目 README](../../../README.md)
