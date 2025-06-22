"""
Utility functions for Instant NGP implementation.

This module provides various utility functions used throughout the Instant NGP
implementation including coordinate transformations, encoding functions,
and optimization utilities.
"""

from .hash_utils import (
    morton_encode_3d,
    morton_decode_3d,
    compute_hash_grid_size
)

from .coordinate_utils import (
    contract_to_unisphere,
    uncontract_from_unisphere,
    spherical_to_cartesian,
    cartesian_to_spherical
)

from .regularization_utils import (
    compute_tv_loss,
    compute_entropy_loss,
    compute_distortion_loss
)

from .sampling_utils import (
    adaptive_sampling,
    importance_sampling,
    stratified_sampling,
    uniform_sampling
)

from .geometry_utils import (
    estimate_normals,
    compute_surface_points,
    mesh_extraction
)

from .visualization_utils import (
    visualize_hash_grid,
    plot_training_curves,
    render_turntable
)

__all__ = [
    # Hash utilities
    'morton_encode_3d',
    'morton_decode_3d', 
    'compute_hash_grid_size',
    
    # Coordinate transformations
    'contract_to_unisphere',
    'uncontract_from_unisphere',
    'spherical_to_cartesian',
    'cartesian_to_spherical',
    
    # Regularization
    'compute_tv_loss',
    'compute_entropy_loss',
    'compute_distortion_loss',
    
    # Sampling
    'adaptive_sampling',
    'importance_sampling',
    'stratified_sampling',
    'uniform_sampling',
    
    # Geometry
    'estimate_normals',
    'compute_surface_points',
    'mesh_extraction',
    
    # Visualization
    'visualize_hash_grid',
    'plot_training_curves',
    'render_turntable'
] 