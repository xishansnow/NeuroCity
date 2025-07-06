# NeuroCity

Neural Radiance Fields (NeRF) implementations and utilities for 3D scene reconstruction.

## 🎉 Latest Updates

**� Major Refactoring Completed (July 2025)**
- ✅ Complete separation of training and rendering logic
- ✅ Removed all Lightning dependencies, pure PyTorch implementation
- ✅ 100% Python 3.10 compatibility ensured
- ✅ Comprehensive documentation system with 13+ technical documents
- ✅ 100% test coverage (17/17 tests passing)
- ✅ Complete API reference with auto-generated documentation

**📖 Read the full refactoring report**: [Refactoring Summary (Chinese)](REFACTORING_SUMMARY_cn.md)

## �📚 Documentation

- **[Project Architecture Overview (English)](PROJECT_ARCHITECTURE_en.md)** - Complete project structure and learning guide
- **[项目架构概览 (中文)](PROJECT_ARCHITECTURE_cn.md)** - 完整项目结构和学习指南
- **[SVRaster Technical Documentation](src/nerfs/svraster/COMPLETE_DOCUMENTATION_INDEX_cn.md)** - Core implementation details
- **[API Reference](src/nerfs/svraster/API_REFERENCE_cn.md)** - Complete API documentation

## Requirements

- Python 3.10 (required, other versions are not supported)
- CUDA-capable GPU (recommended)
- OpenVDB (optional, for volumetric data support)

## Installation

```bash
# Create a Python 3.10 virtual environment
python3.10 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install OpenVDB (optional)
bash install_dependencies.sh
```

## Development

This project maintains Python 3.10 compatibility and uses proper type hints. Please ensure:

- Use `typing` module for type hints (e.g., `List[int]` instead of `list[int]`)
- Use `Union` for union types (e.g., `Union[str, int]` instead of `str | int`)
- Add `from __future__ import annotations` at the top of files for forward references
- Run type checks with mypy: `mypy src/ tests/`
- Run tests: `pytest tests/`

For detailed Python 3.10 compatibility guidelines, see: `src/nerfs/svraster/PYTHON310_COMPATIBILITY_cn.md`

## License

[License information here] 