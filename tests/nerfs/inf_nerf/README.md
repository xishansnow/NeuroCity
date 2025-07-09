# InfNeRF 测试套件

这个目录包含 InfNeRF 的完整测试套件，用于验证模型的功能、性能和稳定性。

## 目录结构

```
tests/nerfs/inf_nerf/
├── __init__.py              # 测试包初始化
├── conftest.py              # pytest 配置和夹具
├── test_core.py             # 核心模型测试
├── test_trainer.py          # 训练器测试
├── test_renderer.py         # 渲染器测试
├── test_dataset.py          # 数据集测试
├── test_utils.py            # 工具函数测试
├── test_integration.py      # 集成测试
├── run_tests.py             # 测试运行脚本
└── README.md                # 本文档
```

## 测试类型

### 1. 单元测试 (Unit Tests)

- **test_core.py**: 测试 InfNeRF 核心组件
  - InfNeRFConfig 配置类
  - OctreeNode 八叉树节点
  - LoDAwareNeRF 网络
  - HashEncoder 哈希编码器
  - SphericalHarmonicsEncoder 球谐编码器
  - InfNeRF 主模型

- **test_trainer.py**: 测试训练器功能
  - InfNeRFTrainerConfig 配置
  - InfNeRFTrainer 训练器
  - 损失计算
  - 优化器管理
  - 检查点保存/加载
  - 训练状态管理

- **test_renderer.py**: 测试渲染器功能
  - InfNeRFRendererConfig 配置
  - InfNeRFRenderer 渲染器
  - 图像渲染
  - 批量渲染
  - 视频生成
  - 光线生成

- **test_dataset.py**: 测试数据集功能
  - InfNeRFDatasetConfig 配置
  - InfNeRFDataset 数据集
  - 数据加载
  - 数据预处理
  - 相机参数处理

- **test_utils.py**: 测试工具函数
  - 八叉树工具
  - LoD 工具
  - 渲染工具
  - 体积渲染工具

### 2. 集成测试 (Integration Tests)

- **test_integration.py**: 端到端测试
  - 完整训练流程
  - 完整渲染流程
  - 模型序列化
  - 性能基准测试
  - 错误处理
  - 多GPU集成

## 运行测试

### 环境要求

- Python 3.8+
- PyTorch 1.8+
- NumPy
- pytest
- pytest-cov (可选，用于覆盖率)

### 基本用法

```bash
# 检查测试环境
python tests/nerfs/inf_nerf/run_tests.py --check-env

# 运行所有测试
python tests/nerfs/inf_nerf/run_tests.py

# 运行特定类型测试
python tests/nerfs/inf_nerf/run_tests.py --type unit
python tests/nerfs/inf_nerf/run_tests.py --type integration
python tests/nerfs/inf_nerf/run_tests.py --type performance

# 运行特定测试文件
python tests/nerfs/inf_nerf/run_tests.py --file test_core.py

# 详细输出
python tests/nerfs/inf_nerf/run_tests.py --verbose

# 生成覆盖率报告
python tests/nerfs/inf_nerf/run_tests.py --coverage

# 指定设备
python tests/nerfs/inf_nerf/run_tests.py --device cpu
python tests/nerfs/inf_nerf/run_tests.py --device cuda
```

### 直接使用 pytest

```bash
# 运行所有测试
pytest tests/nerfs/inf_nerf/

# 运行特定测试文件
pytest tests/nerfs/inf_nerf/test_core.py

# 运行特定测试类
pytest tests/nerfs/inf_nerf/test_core.py::TestInfNeRF

# 运行特定测试方法
pytest tests/nerfs/inf_nerf/test_core.py::TestInfNeRF::test_model_creation

# 标记测试
pytest tests/nerfs/inf_nerf/ -m "slow"
pytest tests/nerfs/inf_nerf/ -m "gpu"
pytest tests/nerfs/inf_nerf/ -m "performance"

# 并行运行
pytest tests/nerfs/inf_nerf/ -n auto

# 生成覆盖率报告
pytest tests/nerfs/inf_nerf/ --cov=src.nerfs.inf_nerf --cov-report=html
```

## 测试标记

- `@pytest.mark.slow`: 慢速测试，需要较长时间
- `@pytest.mark.gpu`: 需要 GPU 的测试
- `@pytest.mark.integration`: 集成测试
- `@pytest.mark.performance`: 性能测试

## 测试夹具 (Fixtures)

### 配置夹具

- `small_config`: 小型配置，用于快速测试
- `medium_config`: 中等配置，用于标准测试
- `trainer_config`: 训练器配置
- `renderer_config`: 渲染器配置
- `volume_renderer_config`: 体积渲染器配置

### 数据夹具

- `synthetic_dataset`: 合成数据集
- `mock_checkpoint`: 模拟检查点文件
- `temp_dir`: 临时目录

### 设备夹具

- `device`: 测试设备 (CPU/GPU)

## 测试数据

测试使用合成数据，包括：

1. **合成相机轨迹**: 围绕原点的圆形轨迹
2. **合成图像**: 简单的球体渲染
3. **模拟检查点**: 包含模型状态和配置

## 性能基准

测试包含以下性能基准：

- **训练性能**: 每次迭代时间 < 1秒
- **渲染性能**: 渲染时间与像素数量成比例
- **内存使用**: 模型内存 < 10GB

## 错误处理测试

测试覆盖以下错误情况：

- 无效配置参数
- 无效输入数据
- 设备不匹配
- 文件不存在
- 数据格式错误

## 覆盖率目标

目标代码覆盖率：

- 核心模型: > 90%
- 训练器: > 85%
- 渲染器: > 85%
- 工具函数: > 80%
- 整体覆盖率: > 85%

## 持续集成

测试套件设计用于持续集成环境：

- 支持并行测试
- 自动设备检测
- 详细的错误报告
- 覆盖率报告生成

## 故障排除

### 常见问题

1. **导入错误**
   ```bash
   # 确保在项目根目录运行
   export PYTHONPATH="${PYTHONPATH}:$(pwd)"
   ```

2. **CUDA 错误**
   ```bash
   # 强制使用 CPU
   python tests/nerfs/inf_nerf/run_tests.py --device cpu
   ```

3. **内存不足**
   ```bash
   # 使用较小的配置
   # 在 conftest.py 中调整 small_config
   ```

4. **测试超时**
   ```bash
   # 增加超时时间
   pytest tests/nerfs/inf_nerf/ --timeout=300
   ```

### 调试测试

```bash
# 详细输出
pytest tests/nerfs/inf_nerf/ -v -s

# 在失败时停止
pytest tests/nerfs/inf_nerf/ -x

# 显示局部变量
pytest tests/nerfs/inf_nerf/ --tb=long

# 使用 pdb 调试
pytest tests/nerfs/inf_nerf/ --pdb
```

## 贡献指南

添加新测试时请遵循以下规范：

1. **测试命名**: 使用描述性的测试名称
2. **测试组织**: 按功能分组测试类
3. **夹具使用**: 尽可能使用现有夹具
4. **标记使用**: 适当使用测试标记
5. **文档**: 为复杂测试添加文档字符串

### 测试模板

```python
def test_feature_name():
    """测试功能描述"""
    # 准备测试数据
    # 执行被测试的功能
    # 验证结果
    assert condition
```

## 更新日志

- **v1.0.0**: 初始测试套件
  - 核心模型测试
  - 训练器测试
  - 渲染器测试
  - 集成测试
  - 性能基准测试 