#!/usr/bin/env python3
"""Convert typing imports to modern Python 3.9+ built-in types."""

import os
import re
import sys
from pathlib import Path
from typing import Dict, Set, Tuple

# Mapping from old typing imports to new built-in types
TYPE_REPLACEMENTS = {
    'Dict': 'dict',
    'List': 'list', 
    'Set': 'set',
    'Tuple': 'tuple',
    'FrozenSet': 'frozenset',
}

# Types that need to stay in typing
KEEP_IN_TYPING = {
    'Optional', 'Union', 'Any', 'Callable', 'TypeVar', 'Generic', 'Protocol',
    'Literal', 'Final', 'ClassVar', 'Type', 'NoReturn', 'Awaitable', 'Coroutine',
    'AsyncGenerator', 'AsyncIterator', 'AsyncIterable', 'Generator', 'Iterator',
    'Iterable', 'Mapping', 'MutableMapping', 'Sequence', 'MutableSequence'
}

def analyze_typing_imports(content: str) -> Tuple[Set[str], Set[str]]:
    """Analyze which typing imports can be replaced and which should remain."""
    # Find all typing imports
    typing_pattern = r'from typing import ([^\n]+)'
    matches = re.findall(typing_pattern, content)
    
    all_imports = set()
    for match in matches:
        imports = [imp.strip() for imp in match.split(',')]
        all_imports.update(imports)
    
    # Separate replaceable and non-replaceable imports
    replaceable = set()
    keep_typing = set()
    
    for imp in all_imports:
        if imp in TYPE_REPLACEMENTS:
            replaceable.add(imp)
        elif imp in KEEP_IN_TYPING:
            keep_typing.add(imp)
        else:
            # Unknown import, keep in typing to be safe
            keep_typing.add(imp)
    
    return replaceable, keep_typing

def replace_type_annotations(content: str, replaceable: Set[str]) -> str:
    """Replace type annotations in the content."""
    new_content = content
    
    # Replace type annotations in the code (not in imports)
    for old_type, new_type in TYPE_REPLACEMENTS.items():
        if old_type in replaceable:
            # Replace in type annotations, being careful about context
            patterns = [
                # Function parameters and return types
                (rf'\b{old_type}\[', f'{new_type}['),
                # Variable annotations
                (rf': {old_type}\[', f': {new_type}['),
                (rf': {old_type}$', f': {new_type}'),
                (rf': {old_type}\s*=', f': {new_type} ='),
                # In Union, Optional, etc.
                (rf'Union\[([^,\]]*){old_type}', rf'Union[\1{new_type}'),
                (rf'{old_type}([,\]])', rf'{new_type}\1'),
                # Standalone type (at word boundaries)
                (rf'\b{old_type}\b(?!\[)', new_type),
            ]
            
            for pattern, replacement in patterns:
                new_content = re.sub(pattern, replacement, new_content)
    
    return new_content

def update_imports(content: str, replaceable: Set[str], keep_typing: Set[str]) -> str:
    """Update the import statements."""
    new_content = content
    
    # Remove old typing import line
    typing_pattern = r'from typing import [^\n]+\n?'
    new_content = re.sub(typing_pattern, '', new_content)
    
    # Add new typing import if needed
    if keep_typing:
        sorted_typing = sorted(keep_typing)
        if len(sorted_typing) == 1:
            typing_import = f'from typing import {sorted_typing[0]}\n'
        else:
            typing_import = f'from typing import {", ".join(sorted_typing)}\n'
        
        # Find a good place to insert the import (after other imports)
        import_pattern = r'((?:from [^\n]+ import [^\n]+\n|import [^\n]+\n)*)'
        match = re.match(import_pattern, new_content)
        if match:
            import_section = match.group(1)
            rest = new_content[len(import_section):]
            new_content = import_section + typing_import + rest
        else:
            # If no imports found, add at the beginning
            new_content = typing_import + new_content
    
    return new_content

def fix_function_syntax(content: str) -> str:
    """Fix missing colons in function definitions."""
    lines = content.split('\n')
    fixed_lines = []
    
    for i, line in enumerate(lines):
        if (line.strip().endswith(')') and 
            not line.strip().endswith('):') and 
            i + 1 < len(lines) and 
            lines[i + 1].strip().startswith('"""')):
            # This is likely a function definition missing a colon
            fixed_lines.append(line + ':')
        else:
            fixed_lines.append(line)
    
    return '\n'.join(fixed_lines)

def convert_typing_imports(file_path: Path) -> bool:
    """Convert typing imports in a file. Returns True if changes were made."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return False
    
    # Check if file has typing imports
    if 'from typing import' not in content:
        return False
    
    # Analyze what needs to be replaced
    replaceable, keep_typing = analyze_typing_imports(content)
    
    if not replaceable:
        # No replaceable types found
        return False
    
    # Replace type annotations
    new_content = replace_type_annotations(content, replaceable)
    
    # Update import statements
    new_content = update_imports(new_content, replaceable, keep_typing)
    
    # Fix function syntax issues
    new_content = fix_function_syntax(new_content)
    
    # Clean up any duplicate newlines
    new_content = re.sub(r'\n\n\n+', '\n\n', new_content)
    
    # Only write if changes were made
    if new_content != content:
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            print(f"✓ Converted {file_path}")
            return True
        except Exception as e:
            print(f"Error writing {file_path}: {e}")
            return False
    
    return False

def find_python_files(root_dir: Path) -> list[Path]:
    """Find all Python files in the directory tree."""
    python_files = []
    for root, dirs, files in os.walk(root_dir):
        # Skip common directories that don't need processing
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'node_modules']]
        
        for file in files:
            if file.endswith('.py'):
                python_files.append(Path(root) / file)
    
    return python_files

def main():
    """Main function to convert typing imports across the codebase."""
    # Get project root directory
    root_dir = Path(__file__).parent
    
    print("🔄 Converting typing imports to modern Python types...")
    print(f"Scanning directory: {root_dir}")
    
    # Find all Python files
    python_files = find_python_files(root_dir)
    print(f"Found {len(python_files)} Python files")
    
    # Process each file
    converted_count = 0
    for file_path in python_files:
        if convert_typing_imports(file_path):
            converted_count += 1
    
    print(f"\n✅ Conversion complete! Modified {converted_count} files")
    
    if converted_count > 0:
        print("\nChanges made:")
        print("• Dict → dict")
        print("• List → list")
        print("• Set → set")
        print("• Tuple → tuple")
        print("• FrozenSet → frozenset")
        print("\nTypes kept in typing module:")
        print("• Optional, Union, Any, Callable, etc.")

if __name__ == '__main__':
    main()
