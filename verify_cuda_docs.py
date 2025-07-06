#!/usr/bin/env python3
"""
简化版 CUDA 文档验证脚本
验证 NeuroCity 项目中 CUDA 文档的完整性和可用性
"""

import os
import sys
import torch
from pathlib import Path

def print_header(title):
    """打印标题"""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

def print_result(test_name, success, details=""):
    """打印测试结果"""
    status = "✅ 通过" if success else "❌ 失败"
    print(f"{test_name:<40} {status}")
    if details:
        print(f"    详情: {details}")

def check_cuda_environment():
    """检查 CUDA 环境"""
    print_header("CUDA 环境检查")
    
    cuda_available = torch.cuda.is_available()
    print_result("CUDA 可用性", cuda_available)
    
    if cuda_available:
        device_name = torch.cuda.get_device_name(0)
        device_props = torch.cuda.get_device_properties(0)
        cuda_version = torch.version.cuda
        
        print_result("GPU 设备", True, device_name)
        print_result("计算能力", True, f"{device_props.major}.{device_props.minor}")
        print_result("总显存", True, f"{device_props.total_memory / 1e9:.2f} GB")
        print_result("CUDA 版本", True, cuda_version)
    
    return cuda_available

def check_cuda_documentation():
    """检查 CUDA 文档完整性"""
    print_header("CUDA 文档完整性检查")
    
    # 检查主要文档文件
    doc_files = [
        ("主要 CUDA 指南", "CUDA_USAGE_GUIDE.md"),
        ("SVRaster CUDA 文档", "src/nerfs/svraster/README_cn.md"),
        ("Plenoxels CUDA 文档", "src/nerfs/plenoxels/README_cn.md"),
        ("InfNeRF CUDA 文档", "src/nerfs/inf_nerf/README_cn.md"),
        ("文档总结", "CUDA_DOCUMENTATION_SUMMARY.md"),
    ]
    
    all_exist = True
    for doc_name, doc_path in doc_files:
        if os.path.exists(doc_path):
            print_result(doc_name, True, f"文件存在: {doc_path}")
        else:
            print_result(doc_name, False, f"文件不存在: {doc_path}")
            all_exist = False
    
    return all_exist

def check_cuda_sections_in_docs():
    """检查文档中的 CUDA 章节"""
    print_header("CUDA 章节内容检查")
    
    # 检查各模块文档中的 CUDA 章节
    modules = [
        ("SVRaster", "src/nerfs/svraster/README_cn.md"),
        ("Plenoxels", "src/nerfs/plenoxels/README_cn.md"),
        ("InfNeRF", "src/nerfs/inf_nerf/README_cn.md"),
    ]
    
    all_have_cuda = True
    for module_name, doc_path in modules:
        if os.path.exists(doc_path):
            with open(doc_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # 检查 CUDA 相关内容
            cuda_keywords = [
                "CUDA 核函数使用指南",
                "CUDA 环境配置",
                "CUDA 性能优化",
                "性能对比"
            ]
            
            found_keywords = sum(1 for keyword in cuda_keywords if keyword in content)
            
            if found_keywords >= 2:  # 至少包含2个关键词
                print_result(f"{module_name} CUDA 章节", True, 
                           f"包含 {found_keywords}/{len(cuda_keywords)} 个关键章节")
            else:
                print_result(f"{module_name} CUDA 章节", False, 
                           f"仅包含 {found_keywords}/{len(cuda_keywords)} 个关键章节")
                all_have_cuda = False
        else:
            print_result(f"{module_name} CUDA 章节", False, "文档文件不存在")
            all_have_cuda = False
    
    return all_have_cuda

def check_utility_scripts():
    """检查实用脚本"""
    print_header("实用脚本检查")
    
    scripts = [
        ("CUDA 验证脚本", "verify_cuda_functionality.py"),
        ("CUDA 编译脚本", "build_cuda_extensions.py"),
    ]
    
    all_exist = True
    for script_name, script_path in scripts:
        if os.path.exists(script_path):
            # 检查脚本是否可执行
            try:
                with open(script_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                if 'def main()' in content and '__main__' in content:
                    print_result(script_name, True, "脚本格式正确")
                else:
                    print_result(script_name, False, "脚本格式可能有问题")
            except Exception as e:
                print_result(script_name, False, f"读取脚本失败: {e}")
                all_exist = False
        else:
            print_result(script_name, False, f"脚本不存在: {script_path}")
            all_exist = False
    
    return all_exist

def check_documentation_quality():
    """检查文档质量"""
    print_header("文档质量检查")
    
    # 检查主要 CUDA 指南的内容
    guide_path = "CUDA_USAGE_GUIDE.md"
    if os.path.exists(guide_path):
        with open(guide_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 检查重要章节
        important_sections = [
            "CUDA 支持概览",
            "快速开始",
            "详细文档",
            "性能对比",
            "故障排除",
            "参考资料"
        ]
        
        found_sections = sum(1 for section in important_sections if section in content)
        
        if found_sections >= 4:
            print_result("主要指南完整性", True, f"包含 {found_sections}/{len(important_sections)} 个重要章节")
        else:
            print_result("主要指南完整性", False, f"仅包含 {found_sections}/{len(important_sections)} 个重要章节")
        
        # 检查代码示例
        code_blocks = content.count('```python')
        if code_blocks >= 5:
            print_result("代码示例数量", True, f"包含 {code_blocks} 个 Python 代码示例")
        else:
            print_result("代码示例数量", False, f"仅包含 {code_blocks} 个 Python 代码示例")
        
        # 检查性能表格
        performance_tables = content.count('| 模块 |')
        if performance_tables >= 1:
            print_result("性能对比表格", True, f"包含 {performance_tables} 个性能表格")
        else:
            print_result("性能对比表格", False, "缺少性能对比表格")
    
    else:
        print_result("主要指南存在性", False, "主要 CUDA 指南不存在")

def generate_simple_report():
    """生成简化报告"""
    print_header("文档验证总结")
    
    # 运行所有检查
    results = []
    results.append(("CUDA 环境", check_cuda_environment()))
    results.append(("文档完整性", check_cuda_documentation()))
    results.append(("CUDA 章节", check_cuda_sections_in_docs()))
    results.append(("实用脚本", check_utility_scripts()))
    
    # 文档质量检查
    check_documentation_quality()
    
    # 统计结果
    total_checks = len(results)
    passed_checks = sum(1 for name, result in results if result)
    
    print(f"\n检查项目总数: {total_checks}")
    print(f"通过项目: {passed_checks}")
    print(f"失败项目: {total_checks - passed_checks}")
    print(f"总体成功率: {passed_checks/total_checks*100:.1f}%")
    
    if passed_checks == total_checks:
        print("\n🎉 所有 CUDA 文档验证通过！")
        print("✅ 文档已准备就绪，用户可以开始使用 CUDA 功能")
    else:
        print(f"\n⚠️  {total_checks - passed_checks} 个项目需要检查")
        print("❗ 建议检查失败的项目后再发布文档")

def main():
    """主函数"""
    print("NeuroCity CUDA 文档验证器")
    print("=" * 60)
    print("此脚本验证 CUDA 文档的完整性和可用性")
    
    try:
        generate_simple_report()
    except KeyboardInterrupt:
        print("\n\n验证被用户中断")
    except Exception as e:
        print(f"\n\n验证过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
