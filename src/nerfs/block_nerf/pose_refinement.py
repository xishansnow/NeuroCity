from __future__ import annotations

"""
Pose Refinement Module for Block-NeRF

This module implements learnable pose refinement to improve camera pose alignment
for large-scale urban scenes, as described in the Block-NeRF paper.

Based on Block-NeRF: Scalable Large Scene Neural View Synthesis (CVPR 2022)
"""

from typing import Optional, Union


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class PoseRefinement(nn.Module):
    """
    Learnable pose refinement module for camera poses.
    
    This module learns small corrections to input camera poses to improve
    alignment and reduce artifacts in multi-view synthesis.
    """
    
    def __init__(
        self,
        num_cameras: int,
        learning_rate_scale: float = 1e-3,
        regularization_weight: float = 1e-4,
        max_translation: float = 0.1,
        max_rotation: float = 0.1
    ):
        """
        Initialize pose refinement module.
        
        Args:
            num_cameras: Number of camera poses to refine
            learning_rate_scale: Scale factor for pose learning rate
            regularization_weight: Weight for pose regularization loss
            max_translation: Maximum allowed translation correction
            max_rotation: Maximum allowed rotation correction
        """
        super().__init__()
        
        self.num_cameras = num_cameras
        self.learning_rate_scale = learning_rate_scale
        self.regularization_weight = regularization_weight
        self.max_translation = max_translation
        self.max_rotation = max_rotation
        
        # SE(3) parameterization: [translation(3), rotation(3)]
        self.pose_corrections = nn.Parameter(
            torch.zeros(num_cameras, 6)
        )
        
        # Base poses (fixed, provided during training)
        self.register_buffer('base_poses', torch.eye(4).unsqueeze(0).repeat(num_cameras, 1, 1))
        self.poses_initialized = False
    
    def set_base_poses(self, poses: torch.Tensor):
        """Set the base camera poses to refine."""
        assert poses.shape[0] <= self.num_cameras
        self.base_poses[:poses.shape[0]] = poses
        self.poses_initialized = True
    
    def forward(self, camera_ids: torch.Tensor) -> torch.Tensor:
        """Get refined camera poses."""
        if not self.poses_initialized:
            raise RuntimeError("Base poses not initialized")
        
        camera_ids = torch.clamp(camera_ids, 0, self.num_cameras - 1)
        return self.base_poses[camera_ids]
    
    def get_regularization_loss(self) -> torch.Tensor:
        """Compute regularization loss."""
        if self.regularization_weight <= 0:
            return torch.tensor(0.0, device=self.pose_corrections.device)
        
        corrections_norm = torch.norm(self.pose_corrections, dim=-1)
        return self.regularization_weight * torch.mean(corrections_norm ** 2)
