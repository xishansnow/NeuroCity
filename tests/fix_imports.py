#!/usr/bin/env python3
"""
Script to fix import paths in test files after migration to tests/ directory.

This script updates relative imports to absolute imports that reference the 
source modules correctly.
"""

import os
import re
from pathlib import Path

def fix_imports_in_file(file_path):
    """Fix imports in a single test file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Add sys.path setup if not already present
    if 'sys.path.append' not in content:
        # Find the first import and add path setup before it
        import_match = re.search(r'^(from|import)\s+', content, re.MULTILINE)
        if import_match:
            insert_pos = import_match.start()
            path_setup = """import sys
import os
# Add the src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

"""
            content = content[:insert_pos] + path_setup + content[insert_pos:]
    
    # Fix relative imports for NeRF modules
    patterns = [
        # Fix "from . import" to proper module imports
        (r'from \. import \((.*?)\)', r'from nerfs.{module} import (\1)'),
        (r'from \.([a-z_]+) import', r'from nerfs.{module}.\1 import'),
        (r'from \. import ([a-z_A-Z0-9_, ]+)', r'from nerfs.{module} import \1'),
    ]
    
    # Determine module name from file path
    file_name = Path(file_path).stem
    if file_name.startswith('test_'):
        module_name = file_name[5:]  # Remove 'test_' prefix
        
        for pattern, replacement in patterns:
            content = re.sub(pattern, replacement.format(module=module_name), content, flags=re.MULTILINE | re.DOTALL)
    
    # Write back the fixed content
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"Fixed imports in {file_path}")

def main():
    """Fix imports in all test files."""
    test_dir = Path(__file__).parent
    
    # Find all test files
    test_files = []
    for subdir in ['nerfs', 'demos']:
        subdir_path = test_dir / subdir
        if subdir_path.exists():
            test_files.extend(subdir_path.glob('test_*.py'))
            test_files.extend(subdir_path.glob('*_test.py'))
    
    print(f"Found {len(test_files)} test files to fix:")
    for file_path in test_files:
        print(f"  - {file_path}")
    
    # Fix imports in each file
    for file_path in test_files:
        fix_imports_in_file(file_path)
    
    print("\nImport fixing complete!")

if __name__ == '__main__':
    main() 