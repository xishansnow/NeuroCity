# Plenoxels Test Suite

这个目录包含了 Plenoxels 神经渲染实现的全面测试套件。

## 📁 文件结构

```
tests/nerfs/plenoxels/
├── __init__.py                          # Python包初始化
├── README.md                            # 本文件
├── run_all_tests.py                     # 主测试运行器
├── test_categories.py                   # 分类测试运行器
├── test_plenoxels_comprehensive.py     # 综合功能测试
├── test_plenoxels_cuda.py              # CUDA基础测试
├── test_plenoxels_cuda_extensions.py   # CUDA扩展测试
└── test_plenoxels_integration.py       # 集成测试
```

## 🚀 运行测试

### 1. 运行所有测试
```bash
# 从项目根目录
python tests/nerfs/plenoxels/run_all_tests.py
```

### 2. 分类测试（推荐）
```bash
# 运行分类测试，获得详细的测试类别报告
python tests/nerfs/plenoxels/test_categories.py
```

### 3. 运行特定测试模块
```bash
# 综合功能测试
python -m unittest tests.nerfs.plenoxels.test_plenoxels_comprehensive -v

# CUDA扩展测试
python -m unittest tests.nerfs.plenoxels.test_plenoxels_cuda_extensions -v

# 集成测试
python -m unittest tests.nerfs.plenoxels.test_plenoxels_integration -v

# CUDA基础测试
python -m unittest tests.nerfs.plenoxels.test_plenoxels_cuda -v
```

# 原始测试
python -m unittest tests.nerfs.plenoxels.test_plenoxels -v
```

### 4. 运行单个测试类
```bash
python -m unittest tests.nerfs.plenoxels.test_plenoxels_comprehensive.TestPlenoxelConfig -v
```

## 🧪 测试模块说明

### `test_plenoxels_comprehensive.py`
- **配置类测试**: PlenoxelConfig, TrainingConfig, InferenceConfig
- **核心组件测试**: VoxelGrid, SphericalHarmonics, PlenoxelModel
- **训练器测试**: PlenoxelTrainer 功能验证
- **渲染器测试**: PlenoxelRenderer 功能验证
- **数据集测试**: PlenoxelDataset 数据处理
- **工具函数测试**: 各种辅助函数

### `test_plenoxels_cuda_extensions.py` - CUDA扩展深度测试
- **CUDA内核测试**: volume rendering, trilinear interpolation
- **性能基准测试**: 不同配置下的性能对比
- **内存效率测试**: GPU内存使用验证
- **梯度计算测试**: 反向传播正确性
- **数值精度验证**: 与PyTorch参考实现对比
- **完整单元测试**: 基于unittest框架的结构化测试

### `test_plenoxels_cuda.py` - CUDA基础功能测试
- **CUDA环境验证**: 检查CUDA可用性和设备信息
- **基础功能测试**: 核心CUDA操作验证（简化版）
- **快速验证**: 适用于开发阶段的快速功能检查
- **性能对比**: 基础的GPU vs CPU性能测试
- **错误诊断**: 提供详细的错误信息和诊断

### `test_plenoxels_integration.py`
- **端到端工作流**: 完整的训练和推理流程
- **检查点管理**: 模型保存和加载
- **配置序列化**: YAML配置文件处理
- **错误处理**: 异常情况和边界条件
- **性能测试**: 不同配置下的性能测试

## 🔧 CUDA测试文件对比

### 使用建议

| 测试文件 | 适用场景 | 运行时间 | 测试深度 |
|---------|----------|----------|----------|
| `test_plenoxels_cuda.py` | 快速验证、开发调试 | 短 (~30秒) | 基础功能 |
| `test_plenoxels_cuda_extensions.py` | 发布前测试、性能优化 | 长 (~2-5分钟) | 深度测试 |

### 功能对比

**共同功能**:
- Ray-voxel 交集计算
- 体积渲染
- 三线性插值
- 基础性能测试

**`test_plenoxels_cuda.py` 特有**:
- 简化的测试流程
- 直观的输出格式
- 快速的故障诊断

**`test_plenoxels_cuda_extensions.py` 特有**:
- 完整的单元测试框架
- 数值精度验证（与PyTorch对比）
- 详细的性能基准测试
- 内存效率和梯度计算测试
- 结构化的测试报告

### 运行建议
```bash
# 开发阶段快速验证
python tests/nerfs/plenoxels/test_plenoxels_cuda.py

# 发布前全面测试
python -m unittest tests.nerfs.plenoxels.test_plenoxels_cuda_extensions -v
```

## 📊 测试覆盖范围

- ✅ **配置管理**: 参数验证、序列化、默认值
- ✅ **核心算法**: 体素网格、球谐函数、体积渲染
- ✅ **训练流程**: 优化器、损失函数、检查点
- ✅ **推理流程**: 图像渲染、视频生成
- ✅ **数据处理**: 数据集加载、预处理、批处理
- ✅ **CUDA加速**: 内核正确性、性能、内存管理
- ✅ **错误处理**: 异常捕获、边界条件、恢复机制
- ✅ **性能验证**: 基准测试、内存使用、吞吐量

## �️ 开发指南
- 输出格式设置

## 🎯 持续集成

测试套件设计为在 CI/CD 环境中运行：
- 自动检测可用硬件 (CUDA)
- 跳过不适用的测试
- 生成详细的测试报告
- 支持并行测试执行

## ⚠️ 注意事项

1. **CUDA依赖**: 某些测试需要CUDA环境
2. **内存需求**: 大型网格测试可能需要足够的GPU内存
3. **运行时间**: 完整测试套件可能需要几分钟
4. **临时文件**: 集成测试会创建临时文件，测试后会自动清理

## 🐛 故障排除

### 导入错误
```bash
# 确保从正确目录运行
cd /path/to/NeuroCity
python tests/nerfs/plenoxels/run_all_tests.py
```

### 添加新测试
1. 选择合适的测试模块（comprehensive, cuda, integration）
2. 添加测试方法，包含清晰的文档字符串
3. 包含正面和负面测试案例
4. 运行测试验证功能正确性

### 运行特定测试类别
```bash
# 使用分类测试器查看各类别状态
python tests/nerfs/plenoxels/test_categories.py
```

### 故障排除

#### 导入错误
```bash
# 确保从项目根目录运行测试
cd /home/xishansnow/3DVision/NeuroCity
python tests/nerfs/plenoxels/run_all_tests.py
```

#### CUDA错误
```bash
# 检查CUDA可用性
python -c "import torch; print(torch.cuda.is_available())"
```

#### 依赖问题
```bash
# 安装测试依赖
pip install -r requirements.txt
```

## 📈 维护指南

- **核心测试**: `test_plenoxels_comprehensive.py` 包含最重要的功能测试
- **CUDA测试**: 仅在CUDA可用时运行，会自动跳过如果不可用
- **集成测试**: 测试完整工作流，可能需要较长运行时间
- **测试发现**: 使用 `test_categories.py` 快速了解各测试类别状态

## 🔄 更新日志

- **v2.0.0**: 清理冗余文件，简化测试结构
- **v1.5.0**: 完成测试迁移到独立目录
- **v1.0.0**: 初始测试套件创建
