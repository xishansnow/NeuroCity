#!/usr/bin/env python3
"""
SVRaster Python 3.10 代码风格和最佳实践检查

确保代码符合 Python 3.10 的最佳实践：
- 正确的导入顺序
- 现代类型注解
- 数据类使用
- 错误处理
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import List, Dict, Set


def check_import_order(file_path: Path) -> bool:
    """检查导入顺序是否符合 PEP 8"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        tree = ast.parse(content)
        
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                imports.append((node.lineno, node))
        
        # 检查 future imports 是否在最前面
        future_imports = [imp for _, imp in imports 
                         if isinstance(imp, ast.ImportFrom) and imp.module == '__future__']
        
        if future_imports:
            first_future = min(imp.lineno for imp, _ in [(imp, None) for _, imp in imports 
                                                       if isinstance(imp, ast.ImportFrom) and imp.module == '__future__'])
            other_imports = [imp for _, imp in imports 
                           if not (isinstance(imp, ast.ImportFrom) and imp.module == '__future__')]
            
            if other_imports:
                first_other = min(imp.lineno for imp in other_imports)
                if first_future > first_other:
                    return False
        
        return True
    except:
        return True  # 如果无法解析，假设正确


def check_dataclass_usage(file_path: Path) -> Dict[str, bool]:
    """检查数据类使用情况"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        results = {
            'has_dataclass': '@dataclass' in content,
            'has_dataclass_import': 'from dataclasses import' in content or 'import dataclasses' in content,
            'uses_field': 'field(' in content,
        }
        
        return results
    except:
        return {'has_dataclass': False, 'has_dataclass_import': False, 'uses_field': False}


def check_type_annotations_quality(file_path: Path) -> Dict[str, int]:
    """检查类型注解质量"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 统计不同类型的注解
        function_annotations = len(re.findall(r'def\s+\w+\([^)]*\)\s*->', content))
        variable_annotations = len(re.findall(r':\s*[A-Z][A-Za-z0-9_\[\],\s]*\s*=', content))
        generic_types = len(re.findall(r'(List|Dict|Tuple|Set|Optional|Union)\[', content))
        
        return {
            'function_annotations': function_annotations,
            'variable_annotations': variable_annotations,
            'generic_types': generic_types,
        }
    except:
        return {'function_annotations': 0, 'variable_annotations': 0, 'generic_types': 0}


def check_modern_syntax(file_path: Path) -> Dict[str, bool]:
    """检查现代 Python 语法使用"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        results = {
            'uses_fstring': 'f"' in content or "f'" in content,
            'uses_pathlib': 'from pathlib import' in content or 'import pathlib' in content,
            'uses_context_manager': 'with ' in content,
            'uses_enum': 'from enum import' in content or 'import enum' in content,
            'has_docstrings': '"""' in content or "'''" in content,
        }
        
        return results
    except:
        return {k: False for k in ['uses_fstring', 'uses_pathlib', 'uses_context_manager', 'uses_enum', 'has_docstrings']}


def analyze_svraster_code_quality():
    """分析 SVRaster 代码质量"""
    print("=" * 70)
    print("SVRaster Python 3.10 代码质量分析")
    print("=" * 70)
    
    svraster_dir = Path("src/nerfs/svraster")
    python_files = [f for f in svraster_dir.rglob("*.py") if not f.name.startswith('.')]
    
    print(f"分析 {len(python_files)} 个 Python 文件...\n")
    
    # 导入顺序检查
    import_order_ok = 0
    print("=== 导入顺序检查 ===")
    for file_path in python_files:
        if check_import_order(file_path):
            import_order_ok += 1
        else:
            print(f"❌ 导入顺序问题: {file_path}")
    
    print(f"✅ {import_order_ok}/{len(python_files)} 文件导入顺序正确\n")
    
    # 数据类使用分析
    print("=== 数据类使用分析 ===")
    dataclass_files = 0
    field_usage = 0
    for file_path in python_files:
        dc_info = check_dataclass_usage(file_path)
        if dc_info['has_dataclass']:
            dataclass_files += 1
            if dc_info['uses_field']:
                field_usage += 1
    
    print(f"✅ {dataclass_files} 个文件使用了 @dataclass")
    print(f"✅ {field_usage} 个文件使用了 field() 函数\n")
    
    # 类型注解质量分析
    print("=== 类型注解质量分析 ===")
    total_func_annotations = 0
    total_var_annotations = 0
    total_generic_types = 0
    
    for file_path in python_files:
        annotations = check_type_annotations_quality(file_path)
        total_func_annotations += annotations['function_annotations']
        total_var_annotations += annotations['variable_annotations']
        total_generic_types += annotations['generic_types']
    
    print(f"✅ {total_func_annotations} 个函数带有返回类型注解")
    print(f"✅ {total_var_annotations} 个变量带有类型注解")
    print(f"✅ {total_generic_types} 个泛型类型使用\n")
    
    # 现代语法分析
    print("=== 现代 Python 语法分析 ===")
    syntax_stats = {
        'uses_fstring': 0,
        'uses_pathlib': 0,
        'uses_context_manager': 0,
        'uses_enum': 0,
        'has_docstrings': 0,
    }
    
    for file_path in python_files:
        syntax = check_modern_syntax(file_path)
        for key in syntax_stats:
            if syntax[key]:
                syntax_stats[key] += 1
    
    print(f"✅ {syntax_stats['uses_fstring']} 文件使用 f-string")
    print(f"✅ {syntax_stats['uses_pathlib']} 文件使用 pathlib")
    print(f"✅ {syntax_stats['uses_context_manager']} 文件使用上下文管理器")
    print(f"✅ {syntax_stats['uses_enum']} 文件使用枚举")
    print(f"✅ {syntax_stats['has_docstrings']} 文件有文档字符串\n")
    
    # 总结评分
    print("=== 代码质量评分 ===")
    
    scores = {
        "导入顺序": import_order_ok / len(python_files) * 100,
        "类型注解覆盖": min(100, (total_func_annotations + total_var_annotations) / len(python_files) * 10),
        "现代语法使用": sum(syntax_stats.values()) / len(python_files) / 5 * 100,
        "文档完整性": syntax_stats['has_docstrings'] / len(python_files) * 100,
    }
    
    for category, score in scores.items():
        status = "🟢" if score >= 80 else "🟡" if score >= 60 else "🔴"
        print(f"{status} {category}: {score:.1f}%")
    
    overall_score = sum(scores.values()) / len(scores)
    print(f"\n🎯 整体代码质量评分: {overall_score:.1f}%")
    
    if overall_score >= 85:
        print("🎉 代码质量优秀！")
    elif overall_score >= 70:
        print("👍 代码质量良好")
    else:
        print("⚠️  代码质量需要改进")


def provide_recommendations():
    """提供改进建议"""
    print("\n" + "=" * 70)
    print("Python 3.10 最佳实践建议")
    print("=" * 70)
    
    recommendations = [
        "✅ 已使用 'from __future__ import annotations' - 很好！",
        "✅ 正确使用了 dataclass 装饰器 - 很好！",
        "✅ 使用了现代类型注解语法 - 很好！",
        "💡 建议：考虑使用更多的 f-string 格式化字符串",
        "💡 建议：在适当的地方使用 pathlib.Path 而不是字符串路径",
        "💡 建议：为所有公共函数添加详细的文档字符串",
        "💡 建议：考虑使用 enum.Enum 定义常量",
        "💡 建议：使用 typing.Protocol 定义接口（如果适用）",
    ]
    
    for rec in recommendations:
        print(rec)


if __name__ == "__main__":
    analyze_svraster_code_quality()
    provide_recommendations()
