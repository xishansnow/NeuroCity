"""
DNMP-NeRF: Urban Radiance Field Representation with Deformable Neural Mesh Primitives

This package implements the DNMP (Deformable Neural Mesh Primitives) method for efficient
urban-scale neural radiance field rendering, as described in:
"Urban Radiance Field Representation with Deformable Neural Mesh Primitives" (ICCV 2023)

Key components:
- Deformable Neural Mesh Primitives (DNMP)
- Mesh Auto-Encoder for latent space representation
- Rasterization-based rendering
- Two-stage training pipeline (geometry + radiance)
"""

from .core import (
    DNMPConfig, DeformableNeuralMeshPrimitive, DNMPRenderer, DNMPLoss, DNMP
)

from .mesh_autoencoder import (
    MeshEncoder, MeshDecoder, MeshAutoEncoder, LatentCode
)

from .rasterizer import (
    DNMPRasterizer, MeshRasterizer, VertexInterpolator, RasterizationConfig
)

from .dataset import (
    DNMPDataset, UrbanSceneDataset, KITTI360Dataset, WaymoDataset
)

from .trainer import (
    DNMPTrainer, GeometryTrainer, RadianceTrainer, TwoStageTrainer
)

from .utils import (
    mesh_utils, voxel_utils, geometry_utils, rendering_utils, evaluation_utils
)

__version__ = "0.1.0"
__author__ = "DNMP-NeRF Team"

__all__ = [
    # Core components
    "DNMPConfig", "DeformableNeuralMeshPrimitive", "DNMPRenderer", "DNMPLoss", "DNMP", # Mesh AutoEncoder
    "MeshEncoder", "MeshDecoder", "MeshAutoEncoder", "LatentCode", # Rasterization
    "DNMPRasterizer", "MeshRasterizer", "VertexInterpolator", "RasterizationConfig", # Datasets
    "DNMPDataset", "UrbanSceneDataset", "KITTI360Dataset", "WaymoDataset", # Training
    "DNMPTrainer", "GeometryTrainer", "RadianceTrainer", "TwoStageTrainer", # Utilities
    "mesh_utils", "voxel_utils", "geometry_utils", "rendering_utils", "evaluation_utils"
] 