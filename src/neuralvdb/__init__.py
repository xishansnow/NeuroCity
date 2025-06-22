"""
NeuralVDB: Efficient Sparse Volumetric Neural Representations

This package implements NeuralVDB, a method for efficient sparse volumetric 
neural representations based on hierarchical data structures and neural networks.

Key Features:
- Sparse voxel representation with octree structure
- Hierarchical neural encoding
- Adaptive resolution
- Memory optimization
- SDF and occupancy field support
- Large-scale urban scene processing

References:
- NeuralVDB: Efficient Sparse Volumetric Neural Representations
- OpenVDB data structure
- Neural SDF/Occupancy representations
"""

from .core import (
    NeuralVDB, NeuralVDBConfig, AdvancedNeuralVDB, AdvancedNeuralVDBConfig
)
from .octree import (
    OctreeNode, AdaptiveOctreeNode, SparseVoxelGrid, AdvancedSparseVoxelGrid
)
from .networks import (
    FeatureNetwork, OccupancyNetwork, MultiScaleFeatureNetwork, 
    AdvancedOccupancyNetwork, PositionalEncoding, MLP
)
from .trainer import (
    NeuralVDBTrainer, AdvancedNeuralVDBTrainer, NeuralSDFTrainer
)
from .dataset import (
    NeuralVDBDataset, VoxelDataset
)
from .utils import (
    create_sample_data, load_training_data, save_vdb_data, load_vdb_data
)
from .viewer import (
    VDBViewer, visualize_training_data, visualize_predictions
)
from .generator import (
    TileCityGenerator, SimpleVDBGenerator
)

__version__ = "1.0.0"

__all__ = [
    # Core classes
    'NeuralVDB',
    'NeuralVDBConfig', 
    'AdvancedNeuralVDB',
    'AdvancedNeuralVDBConfig',
    
    # Octree structures
    'OctreeNode',
    'AdaptiveOctreeNode',
    'SparseVoxelGrid',
    'AdvancedSparseVoxelGrid',
    
    # Neural networks
    'FeatureNetwork',
    'OccupancyNetwork',
    'MultiScaleFeatureNetwork',
    'AdvancedOccupancyNetwork',
    'PositionalEncoding',
    'MLP',
    
    # Training
    'NeuralVDBTrainer',
    'AdvancedNeuralVDBTrainer',
    'NeuralSDFTrainer',
    
    # Data handling
    'NeuralVDBDataset',
    'VoxelDataset',
    
    # Utilities
    'create_sample_data',
    'load_training_data',
    'save_vdb_data',
    'load_vdb_data',
    
    # Visualization
    'VDBViewer',
    'visualize_training_data',
    'visualize_predictions',
    
    # Generation
    'TileCityGenerator',
    'SimpleVDBGenerator'
] 