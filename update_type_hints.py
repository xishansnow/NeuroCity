#!/usr/bin/env python3
"""Script to update type hints to Python 3.10 style."""

import os
import re
from pathlib import Path

def update_type_hints(file_path: str | Path) -> None:
    """Update type hints in a file to Python 3.10 style."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Update imports
    content = re.sub(
        r'from typing import (.*?)(.*?)\n', r'from typing import \1\2\n', content
    )
    
    # Update X | Y to X | Y
    content = re.sub(
        r'Union\[(
            [\w\[\],
            ]+?,
        )
    )
    
    # Update NotRequired to Optional
    content = re.sub(
        r'NotRequired\[([\w\[\], ]+?)\]', r'Optional[\1]', content
    )
    
    # Update Dict, List, Tuple to lowercase
    content = re.sub(r'\bDict\[', 'dict[', content)
    content = re.sub(r'\bList\[', 'list[', content)
    content = re.sub(r'\bTuple\[', 'tuple[', content)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)

def main():
    """Main function."""
    workspace_root = Path('/home/xishansnow/uav_planner_ws/NeuroCity')
    python_files = workspace_root.rglob('*.py')
    
    for file_path in python_files:
        if '.git' not in str(file_path):
            print(f'Updating {file_path}...')
            update_type_hints(file_path)

if __name__ == '__main__':
    main() 