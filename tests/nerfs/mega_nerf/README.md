# Mega-NeRF Test Suite

This directory contains comprehensive tests for the Mega-NeRF implementation.

## Test Structure

```
tests/nerfs/mega_nerf/
├── __init__.py              # Test package initialization
├── conftest.py              # Pytest fixtures and configuration
├── test_core.py             # Core module tests
├── test_trainer.py          # Trainer module tests
├── test_renderer.py         # Renderer module tests
├── test_dataset.py          # Dataset module tests
├── test_utils.py            # Utility module tests
├── test_integration.py      # Integration tests
├── run_tests.py             # Test runner script
├── pytest.ini              # Pytest configuration
└── README.md               # This file
```

## Test Categories

### Unit Tests
- **Core Tests** (`test_core.py`): Test the core Mega-NeRF components
  - `MegaNeRFConfig`: Configuration validation and defaults
  - `PositionalEncoding`: Positional encoding functionality
  - `MegaNeRFSubmodule`: Individual submodule behavior
  - `MegaNeRF`: Main model architecture

- **Trainer Tests** (`test_trainer.py`): Test training functionality
  - `MegaNeRFTrainerConfig`: Trainer configuration
  - `MegaNeRFTrainer`: Training loop and optimization

- **Renderer Tests** (`test_renderer.py`): Test rendering functionality
  - `MegaNeRFRendererConfig`: Renderer configuration
  - `MegaNeRFRenderer`: Image and video rendering

- **Dataset Tests** (`test_dataset.py`): Test data handling
  - `MegaNeRFDatasetConfig`: Dataset configuration
  - `MegaNeRFDataset`: Data loading and preprocessing
  - `CameraInfo`: Camera information handling

### Integration Tests
- **End-to-End Tests** (`test_integration.py`): Full pipeline tests
  - Complete training cycles
  - Rendering pipelines
  - Model serialization
  - Performance benchmarks

### Performance Tests
- **Benchmark Tests**: Performance and memory usage tests
- **Scalability Tests**: Multi-GPU and large-scale tests

## Running Tests

### Quick Start
```bash
# Run all tests
python tests/nerfs/mega_nerf/run_tests.py

# Run unit tests only
python tests/nerfs/mega_nerf/run_tests.py --test-type unit

# Run with coverage
python tests/nerfs/mega_nerf/run_tests.py --coverage

# Run specific test file
python -m pytest tests/nerfs/mega_nerf/test_core.py -v
```

### Advanced Usage
```bash
# Run integration tests
python tests/nerfs/mega_nerf/run_tests.py --test-type integration

# Run performance tests
python tests/nerfs/mega_nerf/run_tests.py --test-type performance

# Run on specific device
python tests/nerfs/mega_nerf/run_tests.py --device cuda

# Run with specific markers
python tests/nerfs/mega_nerf/run_tests.py --markers gpu slow

# Run in parallel
python tests/nerfs/mega_nerf/run_tests.py --parallel 4

# Set custom timeout
python tests/nerfs/mega_nerf/run_tests.py --timeout 600
```

### Direct Pytest Usage
```bash
# Run all tests
pytest tests/nerfs/mega_nerf/

# Run with markers
pytest tests/nerfs/mega_nerf/ -m "unit and not slow"

# Run with coverage
pytest tests/nerfs/mega_nerf/ --cov=src.nerfs.mega_nerf --cov-report=html

# Run specific test
pytest tests/nerfs/mega_nerf/test_core.py::TestMegaNeRF::test_forward -v
```

## Test Fixtures

The test suite provides several fixtures in `conftest.py`:

### Device Fixtures
- `device`: Returns the appropriate device (cuda/cpu)
- `model`: Creates a Mega-NeRF model for testing
- `mega_nerf_config`: Basic model configuration
- `trainer_config`: Basic trainer configuration
- `renderer_config`: Basic renderer configuration
- `dataset_config`: Basic dataset configuration

### Data Fixtures
- `synthetic_cameras`: Synthetic camera data
- `sample_rays`: Sample ray data
- `sample_points`: Sample 3D points
- `sample_viewdirs`: Sample view directions
- `sample_camera_pose`: Sample camera pose
- `sample_intrinsics`: Sample camera intrinsics

### Utility Fixtures
- `temp_dir`: Temporary directory for testing
- `test_data_dir`: Test data directory with synthetic data

## Test Data

The test suite uses synthetic data to avoid requiring large datasets:

- **Synthetic Cameras**: Generated circular camera trajectory
- **Synthetic Images**: Mock image data for testing
- **Synthetic Poses**: Generated camera poses and intrinsics

## Coverage Goals

- **Unit Tests**: 90%+ line coverage
- **Integration Tests**: 80%+ functional coverage
- **Error Handling**: 100% error path coverage

## Performance Benchmarks

The test suite includes performance benchmarks:

- **Training Speed**: Iterations per second
- **Memory Usage**: Peak memory consumption
- **Rendering Speed**: Frames per second
- **Scalability**: Multi-GPU performance

## Error Handling

Tests cover various error scenarios:

- **Invalid Inputs**: Malformed data and configurations
- **Missing Files**: Non-existent data and checkpoints
- **Resource Limits**: Memory and time constraints
- **Device Errors**: CUDA and CPU fallbacks

## Continuous Integration

The test suite is designed for CI/CD:

- **Fast Unit Tests**: < 30 seconds
- **Integration Tests**: < 5 minutes
- **Performance Tests**: < 10 minutes
- **Parallel Execution**: Multi-process support

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```bash
   # Reduce batch size in tests
   export MEGA_NERF_TEST_BATCH_SIZE=64
   ```

2. **Test Timeouts**
   ```bash
   # Increase timeout
   python tests/nerfs/mega_nerf/run_tests.py --timeout 600
   ```

3. **Import Errors**
   ```bash
   # Install dependencies
   pip install -r requirements.txt
   conda activate neurocity
   ```

4. **Missing Test Data**
   ```bash
   # Generate test data
   python tests/nerfs/mega_nerf/generate_test_data.py
   ```

### Debug Mode
```bash
# Run with debug output
python tests/nerfs/mega_nerf/run_tests.py --verbose

# Run single test with debugger
python -m pytest tests/nerfs/mega_nerf/test_core.py::TestMegaNeRF::test_forward -s --pdb
```

## Contributing

When adding new tests:

1. **Follow Naming Convention**: `test_*.py` files, `Test*` classes, `test_*` methods
2. **Use Fixtures**: Leverage existing fixtures in `conftest.py`
3. **Add Markers**: Use appropriate pytest markers
4. **Update Coverage**: Ensure new code is covered by tests
5. **Document**: Add docstrings to test methods

### Test Template
```python
@pytest.mark.unit
def test_new_feature():
    """Test description."""
    # Arrange
    config = MegaNeRFConfig()
    
    # Act
    result = some_function(config)
    
    # Assert
    assert result is not None
    assert result.shape == expected_shape
```

## Environment Requirements

- Python 3.10+
- PyTorch 2.0+
- NumPy 1.20+
- pytest 7.0+
- pytest-cov (for coverage)
- CUDA (optional, for GPU tests)

## License

This test suite is part of the NeuroCity project and follows the same license. 