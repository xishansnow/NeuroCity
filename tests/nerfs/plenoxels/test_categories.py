#!/usr/bin/env python3
"""
Simple Test Runner for Plenoxels Tests

This script runs different categories of tests and provides clear feedback
on what's working and what needs to be fixed.
"""

import os
import sys
import subprocess
from pathlib import Path

def run_test_category(test_module, test_class=None, description=""):
    """Run a specific test category and report results"""
    print(f"\n{'='*60}")
    print(f"🧪 TESTING: {description}")
    print(f"{'='*60}")
    
    if test_class:
        test_target = f"{test_module}.{test_class}"
    else:
        test_target = test_module
    
    try:
        result = subprocess.run(
            [sys.executable, "-m", "unittest", test_target, "-v"],
            cwd=Path(__file__).parent,
            capture_output=True,
            text=True,
            timeout=60
        )
        
        print(f"Exit code: {result.returncode}")
        if result.stdout:
            print("STDOUT:")
            print(result.stdout)
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
            
        if result.returncode == 0:
            print(f"✅ {description} - PASSED")
            return True
        else:
            print(f"❌ {description} - FAILED")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"⏰ {description} - TIMEOUT")
        return False
    except Exception as e:
        print(f"💥 {description} - ERROR: {e}")
        return False

def main():
    """Run all test categories"""
    print("🚀 Plenoxels Test Suite - Category Runner")
    
    # Change to the tests directory
    os.chdir(Path(__file__).parent)
    
    # Test categories to run
    test_categories = [
        ("test_plenoxels_comprehensive", "TestPlenoxelConfig", "Configuration Classes"),
        ("test_plenoxels_comprehensive", "TestVoxelGrid", "Voxel Grid Functions"),
        ("test_plenoxels_comprehensive", "TestSphericalHarmonics", "Spherical Harmonics"),
        ("test_plenoxels_comprehensive", "TestUtilityFunctions", "Utility Functions"),
        ("test_plenoxels_comprehensive", "TestPlenoxelModel", "Model Classes"),
        ("test_plenoxels_comprehensive", "TestPlenoxelRenderer", "Renderer Classes"),
        ("test_plenoxels_comprehensive", "TestPlenoxelTrainer", "Trainer Classes"),
        ("test_plenoxels_comprehensive", "TestPlenoxelDataset", "Dataset Classes"),
    ]
    
    results = []
    
    for test_module, test_class, description in test_categories:
        success = run_test_category(test_module, test_class, description)
        results.append((description, success))
    
    # Summary
    print(f"\n{'='*60}")
    print("📊 TEST SUMMARY")
    print(f"{'='*60}")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for description, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{description:<30} {status}")
    
    print(f"\nOverall: {passed}/{total} test categories passed")
    
    if passed == total:
        print("🎉 All test categories are working!")
        return 0
    else:
        print("🔧 Some test categories need fixes")
        return 1

if __name__ == "__main__":
    sys.exit(main())
