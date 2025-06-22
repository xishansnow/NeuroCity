#!/usr/bin/env python3
"""
NeuroCity Test Runner

This script runs all tests in the NeuroCity test suite.
It can run tests for specific modules or all tests at once.

Usage:
    python tests/run_tests.py              # Run all tests
    python tests/run_tests.py nerfs        # Run NeRF tests only
    python tests/run_tests.py demos        # Run demo tests only
    python tests/run_tests.py --list       # List all available tests
"""

import sys
import os
import argparse
import subprocess
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / 'src'))

def list_available_tests():
    """List all available test modules and files."""
    test_dir = Path(__file__).parent
    
    print("Available test modules:")
    for module_dir in test_dir.iterdir():
        if module_dir.is_dir() and not module_dir.name.startswith('__'):
            test_files = list(module_dir.glob('test_*.py')) + list(module_dir.glob('*_test.py'))
            if test_files:
                print(f"\n  {module_dir.name}:")
                for test_file in test_files:
                    print(f"    - {test_file.name}")

def run_test_file(test_file):
    """Run a single test file."""
    print(f"\n{'='*60}")
    print(f"Running {test_file}")
    print('='*60)
    
    try:
        result = subprocess.run([sys.executable, str(test_file)], 
                              capture_output=True, text=True, cwd=project_root)
        
        if result.stdout:
            print("STDOUT:")
            print(result.stdout)
        
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        
        if result.returncode == 0:
            print(f"✅ {test_file.name} PASSED")
            return True
        else:
            print(f"❌ {test_file.name} FAILED (exit code: {result.returncode})")
            return False
            
    except Exception as e:
        print(f"❌ {test_file.name} ERROR: {e}")
        return False

def run_module_tests(module_name):
    """Run all tests in a specific module."""
    test_dir = Path(__file__).parent / module_name
    
    if not test_dir.exists():
        print(f"❌ Test module '{module_name}' not found")
        return False
    
    test_files = list(test_dir.glob('test_*.py')) + list(test_dir.glob('*_test.py'))
    
    if not test_files:
        print(f"⚠️  No test files found in '{module_name}' module")
        return True
    
    print(f"\n🚀 Running {len(test_files)} test(s) in '{module_name}' module")
    
    passed = 0
    failed = 0
    
    for test_file in test_files:
        if run_test_file(test_file):
            passed += 1
        else:
            failed += 1
    
    print(f"\n📊 {module_name} module results: {passed} passed, {failed} failed")
    return failed == 0

def run_all_tests():
    """Run all tests in all modules."""
    test_dir = Path(__file__).parent
    
    modules = []
    for module_dir in test_dir.iterdir():
        if module_dir.is_dir() and not module_dir.name.startswith('__'):
            test_files = list(module_dir.glob('test_*.py')) + list(module_dir.glob('*_test.py'))
            if test_files:
                modules.append(module_dir.name)
    
    if not modules:
        print("⚠️  No test modules found")
        return True
    
    print(f"🚀 Running all tests in {len(modules)} module(s): {', '.join(modules)}")
    
    total_passed = 0
    total_failed = 0
    failed_modules = []
    
    for module in modules:
        if run_module_tests(module):
            total_passed += 1
        else:
            total_failed += 1
            failed_modules.append(module)
    
    print(f"\n{'='*60}")
    print("📊 FINAL RESULTS")
    print('='*60)
    print(f"Modules passed: {total_passed}")
    print(f"Modules failed: {total_failed}")
    
    if failed_modules:
        print(f"Failed modules: {', '.join(failed_modules)}")
        return False
    else:
        print("🎉 All tests passed!")
        return True

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='NeuroCity Test Runner')
    parser.add_argument('module', nargs='?', help='Test module to run (nerfs, demos, datagen, gfv, neuralvdb)')
    parser.add_argument('--list', action='store_true', help='List all available tests')
    
    args = parser.parse_args()
    
    if args.list:
        list_available_tests()
        return
    
    if args.module:
        success = run_module_tests(args.module)
    else:
        success = run_all_tests()
    
    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main() 