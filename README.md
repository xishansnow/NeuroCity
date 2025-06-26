# NeuroCity

Neural Radiance Fields (NeRF) implementations and utilities for 3D scene reconstruction.

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

This project uses modern Python features and type hints. Please ensure:

- Use Python 3.10 type hints (e.g., `list[int]` instead of `List[int]`)
- Use pipe operator for union types (e.g., `str | int` instead of `Union[str, int]`)
- Run type checks with mypy: `mypy src/ tests/`
- Run tests: `pytest tests/`

## License

[License information here] 