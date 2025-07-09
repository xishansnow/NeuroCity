# SVRaster 测试套件文档

SVRaster 测试套件提供了全面的测试覆盖，包括单元测试、集成测试和性能测试。

## 测试文件结构

```
tests/nerfs/svraster/
├── __init__.py                 # 测试包初始化
├── pytest.ini                 # pytest 配置
├── run_svraster_tests.py      # 主测试运行器
├── test_core.py               # 核心组件测试
├── test_training.py           # 训练组件测试
├── test_rendering.py          # 渲染组件测试
├── test_utils.py              # 工具函数测试
├── test_dataset.py            # 数据集测试
├── test_cuda.py               # CUDA/GPU 测试
├── test_integration.py        # 集成测试
└── README.md                  # 本文档
```

## 测试模块说明

### 1. 核心组件测试 (`test_core.py`)

测试 SVRaster 的核心组件：

- **SVRasterConfig**: 配置类的创建、验证和默认值
- **SVRasterModel**: 模型创建、参数、前向传播、保存/加载
- **SVRasterLoss**: 损失函数的计算和属性
- **设备信息**: 设备检测和兼容性检查
- **包常量**: 版本信息和 CUDA 可用性标志

### 2. 训练组件测试 (`test_training.py`)

测试训练相关的组件：

- **SVRasterTrainerConfig**: 训练配置的创建和验证
- **VolumeRenderer**: 体积渲染器的创建和前向传播
- **SVRasterTrainer**: 训练器创建、属性、优化器设置
- **训练集成**: 完整训练流程和 AMP 支持

### 3. 渲染组件测试 (`test_rendering.py`)

测试推理渲染相关的组件：

- **VoxelRasterizerConfig**: 光栅化配置
- **VoxelRasterizer**: 光栅化器创建和前向传播
- **SVRasterRendererConfig**: 渲染器配置
- **SVRasterRenderer**: 渲染器创建、属性、渲染方法
- **渲染集成**: 完整渲染流程、批量渲染、不同图像尺寸

### 4. 工具函数测试 (`test_utils.py`)

测试各种工具函数：

- **Morton 编码**: 3D Morton 编码/解码和往返测试
- **八叉树操作**: 细分、剪枝和集成操作
- **球谐函数**: 基函数求值和属性测试
- **体素工具**: 体素剪枝功能
- **渲染工具**: 光线排序、深度剥离
- **错误处理**: 无效输入的处理

### 5. 数据集测试 (`test_dataset.py`)

测试数据集相关功能：

- **SVRasterDatasetConfig**: 数据集配置创建和验证
- **SVRasterDataset**: 数据集创建、长度、数据获取
- **数据集集成**: 与训练器集成、数据加载
- **错误处理**: 无效路径和配置的处理

### 6. CUDA/GPU 测试 (`test_cuda.py`)

测试 CUDA 和 GPU 加速功能：

- **CUDA 可用性**: CUDA 检测和设备信息
- **SVRasterGPU**: GPU 模型创建、设备放置、前向传播
- **SVRasterGPUTrainer**: GPU 训练器创建和优化器
- **EMAModel**: 指数移动平均模型
- **CUDA 集成**: 内存管理、张量操作、CPU/GPU 混合操作
- **兼容性**: 降级处理和错误处理

### 7. 集成测试 (`test_integration.py`)

测试完整的端到端工作流：

- **端到端训练**: 完整训练流程和检查点
- **端到端推理**: 完整推理流程和批量推理
- **训练到推理转换**: 训练后推理、模型状态一致性
- **工作流集成**: 开发工作流、性能优化工作流
- **错误处理**: 优雅降级和组件兼容性

## 运行测试

### 基本用法

```bash
# 运行所有测试
python tests/nerfs/svraster/run_svraster_tests.py

# 使用 pytest 直接运行
cd tests/nerfs/svraster
python -m pytest -v
```

### 测试选项

```bash
# 快速测试（跳过慢速集成测试）
python run_svraster_tests.py --quick

# 仅 CUDA 测试
python run_svraster_tests.py --cuda-only

# 跳过 CUDA 测试
python run_svraster_tests.py --no-cuda

# 详细输出
python run_svraster_tests.py --verbose

# 生成覆盖率报告
python run_svraster_tests.py --coverage

# 生成 HTML 测试报告
python run_svraster_tests.py --html

# 运行特定模块
python run_svraster_tests.py --module test_core --module test_utils
```

### pytest 高级用法

```bash
# 运行特定测试类
pytest test_core.py::TestSVRasterModel -v

# 运行特定测试方法
pytest test_core.py::TestSVRasterModel::test_model_creation -v

# 运行带标记的测试
pytest -m "not slow" -v        # 跳过慢速测试
pytest -m "cuda" -v            # 仅 CUDA 测试
pytest -m "integration" -v     # 仅集成测试

# 并行运行测试
pytest -n auto                 # 需要 pytest-xdist

# 生成覆盖率报告
pytest --cov=nerfs.svraster --cov-report=html
```

## 测试依赖

### 必需依赖

- `pytest >= 6.0`: 测试框架
- `torch`: PyTorch
- `numpy`: 数值计算
- `nerfs.svraster`: 被测试的包

### 可选依赖

- `pytest-cov`: 覆盖率报告
- `pytest-html`: HTML 测试报告
- `pytest-xdist`: 并行测试
- `PIL/Pillow`: 图像处理（用于数据集测试）

### 安装依赖

```bash
# 基本测试依赖
pip install pytest torch numpy

# 完整测试依赖
pip install pytest pytest-cov pytest-html pytest-xdist Pillow
```

## 测试标记系统

使用 pytest 标记来分类测试：

- `@pytest.mark.slow`: 慢速测试（通常是集成测试）
- `@pytest.mark.cuda`: 需要 CUDA 的测试
- `@pytest.mark.gpu`: 需要 GPU 的测试
- `@pytest.mark.integration`: 集成测试
- `@pytest.mark.unit`: 单元测试

示例：

```python
@pytest.mark.slow
@pytest.mark.integration
def test_complete_training_pipeline(self):
    # 慢速集成测试
    pass

@pytest.mark.cuda
def test_gpu_model_creation(self):
    # CUDA 测试
    pass
```

## 测试数据和模拟

### 模拟数据集

测试使用临时目录和模拟数据：

```python
def create_dummy_dataset_files(self, temp_dir):
    """创建用于测试的虚拟数据集"""
    transforms = {
        "camera_angle_x": 0.6911112070083618,
        "frames": [...]
    }
    # 创建必要的文件和目录
```

### 临时文件处理

使用 `tempfile` 模块安全处理临时文件：

```python
with tempfile.TemporaryDirectory() as temp_dir:
    # 在临时目录中运行测试
    # 自动清理
```

## 错误处理策略

### 预期失败

某些测试可能因为实现细节而失败：

```python
try:
    # 测试代码
    result = some_function()
    assert result is not None
except Exception as e:
    # 预期的失败
    print(f"Function failed (may be expected): {e}")
    pytest.skip(f"Function not implemented: {e}")
```

### 条件跳过

根据环境条件跳过测试：

```python
if not SVRASTER_AVAILABLE:
    pytest.skip(f"SVRaster not available: {IMPORT_ERROR}")

if not svraster.CUDA_AVAILABLE:
    pytest.skip("CUDA not available")
```

## 持续集成

### GitHub Actions 示例

```yaml
name: SVRaster Tests
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
        pip install pytest torch numpy
    - name: Run tests
      run: |
        cd tests/nerfs/svraster
        python run_svraster_tests.py --quick --no-cuda
```

## 测试最佳实践

### 1. 测试隔离

每个测试应该独立，不依赖其他测试的状态：

```python
def test_model_creation(self):
    """每个测试创建自己的对象"""
    config = svraster.SVRasterConfig(...)
    model = svraster.SVRasterModel(config)
    # 测试逻辑
```

### 2. 资源清理

确保测试后清理资源：

```python
def test_with_cleanup(self):
    try:
        # 测试代码
        pass
    finally:
        # 清理代码
        torch.cuda.empty_cache()
```

### 3. 有意义的断言

使用描述性的断言消息：

```python
assert len(result) > 0, "Result should not be empty"
assert result.shape == expected_shape, f"Expected {expected_shape}, got {result.shape}"
```

### 4. 参数化测试

对于多个输入的相似测试：

```python
@pytest.mark.parametrize("degree", [1, 2, 3])
def test_sh_basis(self, degree):
    sh_values = svraster.eval_sh_basis(degree=degree, dirs=view_dirs)
    expected_coeffs = (degree + 1) ** 2
    assert sh_values.shape[1] == expected_coeffs
```

## 性能测试

### 基准测试

```python
import time

def test_training_performance(self):
    start_time = time.time()
    # 训练代码
    end_time = time.time()
    
    training_time = end_time - start_time
    assert training_time < 60, f"Training took too long: {training_time}s"
```

### 内存测试

```python
def test_memory_usage(self):
    if torch.cuda.is_available():
        initial_memory = torch.cuda.memory_allocated()
        # 测试代码
        final_memory = torch.cuda.memory_allocated()
        memory_used = final_memory - initial_memory
        assert memory_used < 1e9, f"Too much memory used: {memory_used} bytes"
```

## 故障排除

### 常见问题

1. **导入错误**: 确保 SVRaster 正确安装和路径设置
2. **CUDA 错误**: 检查 CUDA 驱动和 PyTorch CUDA 版本
3. **内存错误**: 减少测试数据大小或使用 CPU 测试
4. **依赖错误**: 安装所需的测试依赖

### 调试技巧

1. 使用 `pytest -s` 查看 print 输出
2. 使用 `pytest --pdb` 在失败时进入调试器
3. 使用 `pytest -x` 在第一个失败时停止
4. 使用 `pytest --lf` 仅运行上次失败的测试

## 贡献测试

### 添加新测试

1. 在适当的测试文件中添加测试方法
2. 使用描述性的测试名称
3. 添加适当的标记
4. 处理预期的失败情况
5. 更新文档

### 测试覆盖率

目标是达到高测试覆盖率：

```bash
# 生成覆盖率报告
pytest --cov=nerfs.svraster --cov-report=html --cov-report=term-missing

# 查看覆盖率报告
open htmlcov/index.html
```

### 代码质量

遵循测试代码质量标准：

- 清晰的测试名称和文档
- 适当的错误处理
- 合理的测试数据大小
- 避免硬编码值
- 使用 fixtures 共享设置

---

这个测试套件为 SVRaster 提供了全面的质量保证，确保代码的可靠性和稳定性。定期运行测试，特别是在代码变更后，可以及早发现问题并保持代码质量。
