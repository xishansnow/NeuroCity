#!/usr/bin/env python3
"""
SVRaster Test Environment Check

This script checks if the test environment is properly set up for running SVRaster tests.
"""

import sys
import os
from pathlib import Path

def check_python_version():
    """Check Python version"""
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 7):
        print("⚠ Warning: Python 3.7+ recommended")
        return False
    else:
        print("✓ Python version OK")
        return True

def check_package_imports():
    """Check if required packages can be imported"""
    packages = {
        'torch': 'PyTorch',
        'numpy': 'NumPy', 
        'pytest': 'pytest'
    }
    
    results = {}
    for package, name in packages.items():
        try:
            __import__(package)
            print(f"✓ {name} available")
            results[package] = True
        except ImportError:
            print(f"✗ {name} not available")
            results[package] = False
    
    return results

def check_svraster():
    """Check SVRaster availability"""
    src_path = Path(__file__).parent.parent.parent.parent / "src"
    sys.path.insert(0, str(src_path))
    
    try:
        import nerfs.svraster as svraster
        print(f"✓ SVRaster available (version: {svraster.__version__})")
        
        # Check device info
        device_info = svraster.get_device_info()
        print(f"  PyTorch version: {device_info['torch_version']}")
        print(f"  CUDA available: {device_info['cuda_available']}")
        print(f"  SVRaster CUDA: {device_info['svraster_cuda']}")
        
        if device_info['cuda_available']:
            print(f"  GPU devices: {device_info['device_count']}")
        
        return True
        
    except ImportError as e:
        print(f"✗ SVRaster not available: {e}")
        return False

def check_optional_packages():
    """Check optional packages for enhanced testing"""
    optional_packages = {
        'pytest_cov': 'pytest-cov (for coverage)',
        'pytest_html': 'pytest-html (for HTML reports)',
        'pytest_xdist': 'pytest-xdist (for parallel testing)',
        'PIL': 'Pillow (for image processing in dataset tests)'
    }
    
    print("\nOptional packages:")
    for package, description in optional_packages.items():
        try:
            __import__(package)
            print(f"✓ {description}")
        except ImportError:
            print(f"○ {description} (optional)")

def check_test_files():
    """Check if test files exist"""
    test_dir = Path(__file__).parent
    test_files = [
        "test_core.py",
        "test_training.py", 
        "test_rendering.py",
        "test_utils.py",
        "test_dataset.py",
        "test_cuda.py",
        "test_integration.py",
        "run_svraster_tests.py"
    ]
    
    print("\nTest files:")
    all_exist = True
    for test_file in test_files:
        if (test_dir / test_file).exists():
            print(f"✓ {test_file}")
        else:
            print(f"✗ {test_file}")
            all_exist = False
    
    return all_exist

def main():
    """Main check function"""
    print("SVRaster Test Environment Check")
    print("=" * 50)
    
    # Check Python version
    python_ok = check_python_version()
    print()
    
    # Check required packages
    print("Required packages:")
    package_results = check_package_imports()
    print()
    
    # Check SVRaster
    svraster_ok = check_svraster()
    print()
    
    # Check optional packages
    check_optional_packages()
    print()
    
    # Check test files
    test_files_ok = check_test_files()
    print()
    
    # Summary
    print("Summary:")
    print("=" * 20)
    
    all_required = all([
        python_ok,
        package_results.get('torch', False),
        package_results.get('numpy', False), 
        package_results.get('pytest', False),
        svraster_ok,
        test_files_ok
    ])
    
    if all_required:
        print("✓ Environment ready for testing!")
        print("\nYou can now run tests with:")
        print("  python run_svraster_tests.py")
        print("  python run_svraster_tests.py --quick")
        print("  python -m pytest -v")
        return 0
    else:
        print("✗ Environment not ready for testing")
        print("\nPlease install missing dependencies:")
        
        if not package_results.get('torch', False):
            print("  pip install torch")
        if not package_results.get('numpy', False):
            print("  pip install numpy")
        if not package_results.get('pytest', False):
            print("  pip install pytest")
        if not svraster_ok:
            print("  Make sure SVRaster is properly installed")
        
        return 1

if __name__ == "__main__":
    sys.exit(main())
