#!/usr/bin/env python3
"""
Simple test runner for Block-NeRF CUDA tests
"""
import subprocess
import sys
import os

def run_test(test_file, description):
    """Run a single test file"""
    print(f"\n{'='*50}")
    print(f"Running {description}")
    print(f"{'='*50}")
    
    if not os.path.exists(test_file):
        print(f"❌ Test file not found: {test_file}")
        return False
    
    try:
        result = subprocess.run([sys.executable, test_file], 
                              capture_output=False, text=True)
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Error running {test_file}: {e}")
        return False

def main():
    """Main test runner"""
    print("🧪 Block-NeRF CUDA Test Runner")
    
    # Available tests
    tests = [
        ("quick_test.py", "Quick CUDA Test"),
        ("verify_environment.py", "Environment Verification"),
        ("comprehensive_test.py", "Comprehensive Tests"),
        ("test_unit.py", "Unit Tests"),
        ("test_full_resolution.py", "Full Resolution Test"),
    ]
    
    results = {}
    
    for test_file, description in tests:
        if os.path.exists(test_file):
            results[test_file] = run_test(test_file, description)
        else:
            print(f"⚠️  Skipping {test_file} - file not found")
            results[test_file] = None
    
    # Summary
    print(f"\n{'='*50}")
    print("📊 Test Summary")
    print(f"{'='*50}")
    
    passed = 0
    total = 0
    
    for test_file, result in results.items():
        if result is not None:
            total += 1
            if result:
                passed += 1
                print(f"✅ {test_file}")
            else:
                print(f"❌ {test_file}")
        else:
            print(f"⚠️  {test_file} (skipped)")
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)