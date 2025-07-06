"""
Spherical Harmonics utilities for SVRaster

This module implements spherical harmonics basis functions used for
view-dependent color representation in the sparse voxel radiance fields.
"""

from __future__ import annotations

import torch


def eval_sh_basis(degree: int, dirs: torch.Tensor) -> torch.Tensor:
    """
    计算球谐函数基（支持 0~3 阶），输入方向 shape [..., 3]，输出 shape [..., num_sh_coeffs]
    
    Args:
        degree: SH degree (0-3)
        dirs: Direction vectors [..., 3], must be normalized
        
    Returns:
        SH basis values [..., num_sh_coeffs]
    """
    # dirs: [..., 3], 必须归一化
    x, y, z = dirs.unbind(-1)
    sh_list = []
    sh_list.append(torch.ones_like(x))  # l=0, m=0
    if degree >= 1:
        sh_list += [y, z, x]  # l=1, m=-1,0,1
    if degree >= 2:
        sh_list += [
            x * y,  # l=2, m=-2
            y * z,  # l=2, m=-1
            3 * z**2 - 1,  # l=2, m=0
            x * z,  # l=2, m=1
            x**2 - y**2,  # l=2, m=2
        ]
    if degree >= 3:
        sh_list += [
            y * (3 * x**2 - y**2),  # l=3, m=-3
            x * y * z,  # l=3, m=-2
            y * (5 * z**2 - 1),  # l=3, m=-1
            z * (5 * z**2 - 3),  # l=3, m=0
            x * (5 * z**2 - 1),  # l=3, m=1
            (x**2 - y**2) * z,  # l=3, m=2
            x * (x**2 - 3 * y**2),  # l=3, m=3
        ]
    return torch.stack(sh_list, dim=-1)
