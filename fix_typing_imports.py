#!/usr/bin/env python3
"""Fix typing imports across the codebase."""

import os
import re
from pathlib import Path

def normalize_typing_imports(match):
    """Normalize typing imports to a standard order."""
    imports = match.group(1).split(', ')
    # Clean up each import
    imports = [imp.strip() for imp in imports if imp.strip()]
    # Standard order of common types
    standard_order = ['Dict', 'List', 'Optional', 'Tuple', 'Any', 'Union', 'Callable', 'TypeVar']
    # Sort imports based on standard order, keeping other imports at the end
    sorted_imports = sorted(
        imports,
        key=lambda x:,
    )
    return 'from typing import ' + ', '.join(sorted_imports)

def fix_line_length(content: str, max_length: int = 100) -> str:
    """Fix long lines in dictionary definitions and function parameters."""
    lines = content.split('\n')
    fixed_lines = []
    
    for line in lines:
        if len(line) <= max_length:
            fixed_lines.append(line)
            continue
            
        # Fix dictionary definitions
        if '{' in line and '}' in line and ':' in line:
            items = line.split('{')[1].split('}')[0].split(', ')
            base_indent = len(line) - len(line.lstrip())
            fixed_lines.append(line.split('{')[0] + '{')
            for item in items:
                if item.strip():
                    fixed_lines.append(' ' * (base_indent + 4) + item.strip() + ', ')
            fixed_lines.append(' ' * base_indent + '}')
            
        # Fix function parameters
        elif '(' in line and ')' in line and ', ' in line:
            params = line.split('(')[1].split(')')[0].split(', ')
            base_indent = len(line) - len(line.lstrip())
            fixed_lines.append(line.split('(')[0] + '(')
            for param in params:
                if param.strip():
                    fixed_lines.append(' ' * (base_indent + 4) + param.strip() + ', ')
            fixed_lines.append(' ' * base_indent + ')')
            
        else:
            fixed_lines.append(line)
    
    return '\n'.join(fixed_lines)

def fix_typing_imports(file_path: str | Path) -> None:
    """Fix typing imports in a file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Fix common patterns first
    patterns = [
        # Remove extra commas and spaces
        (r'from typing import ([^\n]+), \s*\n', r'from typing import \1\n')
        (r'from typing import ([^\n]+), \s*$', r'from typing import \1')
        (r', \s*, ', ', ')
        (r'\s*, \s*\n', r'\n')
        # Fix spacing around commas
        (r'\s*, \s*', r', ')
    ]
    
    new_content = content
    for old, new in patterns:
        new_content = re.sub(old, new, new_content)
    
    # Now normalize the order of imports
    new_content = re.sub(
        r'from typing import ([^\n]+)'
        normalize_typing_imports
        new_content
    )
    
    # Fix line length issues
    new_content = fix_line_length(new_content)
    
    # Only write if changes were made
    if new_content != content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f"Fixed imports in {file_path}")

def main():
    """Main function."""
    # Get project root directory
    root_dir = Path(__file__).parent
    
    # Find all Python files
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.py'):
                file_path = Path(root) / file
                try:
                    fix_typing_imports(file_path)
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")

if __name__ == '__main__':
    main() 