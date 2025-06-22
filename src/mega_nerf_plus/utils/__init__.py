"""
Utility modules for Mega-NeRF++

This package contains utility functions and classes for:
- Photogrammetric computations
- Coordinate transformations
- Evaluation metrics
- Visualization tools
"""

from .photogrammetry_utils import (
    PhotogrammetricUtils,
    CameraPoseEstimator,
    BundleAdjustment,
    ImageMatching,
    FeatureExtractor
)

from .evaluation_utils import (
    EvaluationMetrics,
    PerceptualMetrics,
    GeometricMetrics,
    compute_psnr,
    compute_ssim,
    compute_lpips
)

from .visualization_utils import (
    SceneVisualizer,
    TrainingVisualizer,
    InteractiveViewer,
    plot_training_curves,
    render_camera_trajectory
)

from .coordinate_utils import (
    CoordinateTransforms,
    CameraUtils,
    transform_points,
    project_points,
    unproject_points
)

__all__ = [
    # Photogrammetric utilities
    "PhotogrammetricUtils",
    "CameraPoseEstimator",
    "BundleAdjustment",
    "ImageMatching",
    "FeatureExtractor",
    
    # Evaluation metrics
    "EvaluationMetrics",
    "PerceptualMetrics", 
    "GeometricMetrics",
    "compute_psnr",
    "compute_ssim",
    "compute_lpips",
    
    # Visualization
    "SceneVisualizer",
    "TrainingVisualizer",
    "InteractiveViewer",
    "plot_training_curves",
    "render_camera_trajectory",
    
    # Coordinate utilities
    "CoordinateTransforms",
    "CameraUtils",
    "transform_points",
    "project_points",
    "unproject_points"
] 