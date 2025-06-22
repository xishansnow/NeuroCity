"""
NeuralVDB Interface for Plenoxels

This module provides integration between Plenoxels and NeuralVDB for efficient
storage and retrieval of sparse voxel data. NeuralVDB enables compression and
optimization of large-scale volumetric neural data.

Key Features:
- Export Plenoxels data to NeuralVDB format
- Load from NeuralVDB files into Plenoxels models
- Efficient sparse storage with compression
- Support for hierarchical levels of detail
- Streaming and chunked data access
"""

import os
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path
import logging
import json
from dataclasses import dataclass, asdict

try:
    import openvdb as vdb
    VDB_AVAILABLE = True
except ImportError:
    VDB_AVAILABLE = False
    logging.warning("OpenVDB not available. NeuralVDB functionality will be limited.")

from .core import VoxelGrid, PlenoxelConfig, SphericalHarmonics

logger = logging.getLogger(__name__)


@dataclass
class NeuralVDBConfig:
    """Configuration for NeuralVDB integration."""
    
    # Storage settings
    compression_level: int = 6  # ZIP compression level (0-9)
    half_precision: bool = True  # Use half precision for storage
    chunk_size: Tuple[int, int, int] = (64, 64, 64)  # Chunk size for large grids
    
    # Optimization settings
    tolerance: float = 1e-4  # Tolerance for sparse storage
    background_value: float = 0.0  # Background value for empty voxels
    
    # Metadata
    include_metadata: bool = True
    include_training_info: bool = True
    
    # Hierarchical storage
    use_lod: bool = False  # Level of detail
    lod_levels: int = 3    # Number of LOD levels


class NeuralVDBManager:
    """Manager for NeuralVDB operations with Plenoxels."""
    
    def __init__(self, config: NeuralVDBConfig = None):
        """
        Initialize NeuralVDB manager.
        
        Args:
            config: NeuralVDB configuration
        """
        if not VDB_AVAILABLE:
            raise ImportError("OpenVDB is required for NeuralVDB functionality. "
                            "Install with: pip install openvdb")
        
        self.config = config or NeuralVDBConfig()
        self.grids = {}  # Cache for loaded grids
        
    def export_plenoxel_to_vdb(self,
                              voxel_grid: VoxelGrid,
                              output_path: str,
                              model_config: PlenoxelConfig = None) -> bool:
        """
        Export Plenoxel voxel grid to NeuralVDB format.
        
        Args:
            voxel_grid: Plenoxel voxel grid to export
            output_path: Output VDB file path
            model_config: Plenoxel model configuration
            
        Returns:
            Success flag
        """
        try:
            # Convert tensors to numpy arrays
            density_np = voxel_grid.density.detach().cpu().numpy()
            sh_coeffs_np = voxel_grid.sh_coeffs.detach().cpu().numpy()
            
            # Create VDB grids
            grids = []
            
            # Export density grid
            density_grid = self._create_density_grid(density_np, voxel_grid)
            grids.append(density_grid)
            
            # Export SH coefficient grids
            sh_grids = self._create_sh_grids(sh_coeffs_np, voxel_grid)
            grids.extend(sh_grids)
            
            # Add metadata
            if self.config.include_metadata:
                metadata_grid = self._create_metadata_grid(voxel_grid, model_config)
                grids.append(metadata_grid)
            
            # Write to file
            vdb.write(output_path, grids)
            
            logger.info(f"Successfully exported Plenoxel data to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export to VDB: {str(e)}")
            return False
    
    def load_plenoxel_from_vdb(self,
                              vdb_path: str,
                              device: torch.device = None) -> Tuple[VoxelGrid, PlenoxelConfig]:
        """
        Load Plenoxel voxel grid from NeuralVDB format.
        
        Args:
            vdb_path: Path to VDB file
            device: Target device for loaded data
            
        Returns:
            Tuple of (VoxelGrid, PlenoxelConfig)
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        try:
            # Read VDB file
            grids = vdb.read(vdb_path)
            grid_dict = {grid.name: grid for grid in grids}
            
            # Load metadata
            model_config = self._load_metadata(grid_dict)
            
            # Load density grid
            density_tensor = self._load_density_grid(grid_dict, device)
            
            # Load SH coefficient grids
            sh_coeffs_tensor = self._load_sh_grids(grid_dict, device, model_config.sh_degree)
            
            # Create VoxelGrid object
            voxel_grid = VoxelGrid(
                resolution=tuple(density_tensor.shape),
                scene_bounds=model_config.scene_bounds,
                sh_degree=model_config.sh_degree
            ).to(device)
            
            # Set loaded data
            voxel_grid.density.data = density_tensor
            voxel_grid.sh_coeffs.data = sh_coeffs_tensor
            
            logger.info(f"Successfully loaded Plenoxel data from {vdb_path}")
            return voxel_grid, model_config
            
        except Exception as e:
            logger.error(f"Failed to load from VDB: {str(e)}")
            raise
    
    def _create_density_grid(self, density_np: np.ndarray, voxel_grid: VoxelGrid) -> vdb.FloatGrid:
        """Create VDB grid for density data."""
        # Apply sparsity threshold
        sparse_density = np.where(
            np.exp(density_np) > self.config.tolerance,
            density_np.astype(np.float32 if not self.config.half_precision else np.float16),
            self.config.background_value
        )
        
        # Create VDB grid
        density_grid = vdb.FloatGrid()
        density_grid.name = "density"
        density_grid.gridClass = vdb.GridClass.FOG_VOLUME
        
        # Set transform to match scene bounds
        transform = self._create_transform(voxel_grid)
        density_grid.transform = transform
        
        # Fill grid with data
        accessor = density_grid.getAccessor()
        D, H, W = sparse_density.shape
        
        for z in range(D):
            for y in range(H):
                for x in range(W):
                    value = sparse_density[z, y, x]
                    if abs(value - self.config.background_value) > self.config.tolerance:
                        accessor.setValueOn((x, y, z), float(value))
        
        # Compress
        density_grid.pruneInactive()
        if self.config.compression_level > 0:
            density_grid.compress()
        
        return density_grid
    
    def _create_sh_grids(self, sh_coeffs_np: np.ndarray, voxel_grid: VoxelGrid) -> List[vdb.Vec3fGrid]:
        """Create VDB grids for spherical harmonics coefficients."""
        D, H, W, n_channels, n_coeffs = sh_coeffs_np.shape
        grids = []
        
        # Create one grid per SH coefficient
        for coeff_idx in range(n_coeffs):
            sh_grid = vdb.Vec3fGrid()
            sh_grid.name = f"sh_coeff_{coeff_idx}"
            sh_grid.gridClass = vdb.GridClass.UNKNOWN
            
            # Set transform
            transform = self._create_transform(voxel_grid)
            sh_grid.transform = transform
            
            # Fill grid with RGB data for this coefficient
            accessor = sh_grid.getAccessor()
            
            for z in range(D):
                for y in range(H):
                    for x in range(W):
                        # Get RGB values for this coefficient
                        rgb_values = sh_coeffs_np[z, y, x, :, coeff_idx]  # [3]
                        
                        # Check if significant
                        if np.any(np.abs(rgb_values) > self.config.tolerance):
                            if self.config.half_precision:
                                rgb_values = rgb_values.astype(np.float16)
                            accessor.setValueOn((x, y, z), tuple(rgb_values.astype(float)))
            
            # Compress
            sh_grid.pruneInactive()
            if self.config.compression_level > 0:
                sh_grid.compress()
            
            grids.append(sh_grid)
        
        return grids
    
    def _create_metadata_grid(self, voxel_grid: VoxelGrid, model_config: PlenoxelConfig) -> vdb.StringGrid:
        """Create metadata grid with configuration information."""
        metadata_grid = vdb.StringGrid()
        metadata_grid.name = "metadata"
        
        # Collect metadata
        metadata = {
            "resolution": list(voxel_grid.resolution),
            "scene_bounds": list(voxel_grid.scene_bounds.cpu().numpy()),
            "sh_degree": voxel_grid.sh_degree,
            "num_sh_coeffs": voxel_grid.num_sh_coeffs,
            "export_config": asdict(self.config)
        }
        
        if model_config:
            metadata["model_config"] = asdict(model_config)
        
        # Store as JSON string
        metadata_json = json.dumps(metadata, indent=2)
        
        # Add to grid (store at origin)
        accessor = metadata_grid.getAccessor()
        accessor.setValueOn((0, 0, 0), metadata_json)
        
        return metadata_grid
    
    def _create_transform(self, voxel_grid: VoxelGrid) -> vdb.Transform:
        """Create VDB transform from voxel grid parameters."""
        # Calculate voxel size
        scene_bounds = voxel_grid.scene_bounds.cpu().numpy()
        resolution = voxel_grid.resolution
        
        voxel_size = np.array([
            (scene_bounds[3] - scene_bounds[0]) / resolution[2],  # X
            (scene_bounds[4] - scene_bounds[1]) / resolution[1],  # Y
            (scene_bounds[5] - scene_bounds[2]) / resolution[0]   # Z
        ])
        
        # Create transform
        transform = vdb.createLinearTransform(voxel_size[0])
        
        # Set translation to scene minimum
        translation = vdb.Vec3d(scene_bounds[0], scene_bounds[1], scene_bounds[2])
        transform.postTranslate(translation)
        
        return transform
    
    def _load_metadata(self, grid_dict: Dict) -> PlenoxelConfig:
        """Load metadata from VDB grids."""
        if "metadata" not in grid_dict:
            raise ValueError("No metadata found in VDB file")
        
        metadata_grid = grid_dict["metadata"]
        accessor = metadata_grid.getAccessor()
        
        # Get metadata JSON string
        metadata_json = accessor.getValue((0, 0, 0))
        metadata = json.loads(metadata_json)
        
        # Extract model config
        if "model_config" in metadata:
            model_config_dict = metadata["model_config"]
            model_config = PlenoxelConfig(**model_config_dict)
        else:
            # Create basic config from available data
            model_config = PlenoxelConfig(
                grid_resolution=tuple(metadata["resolution"]),
                scene_bounds=tuple(metadata["scene_bounds"]),
                sh_degree=metadata["sh_degree"]
            )
        
        return model_config
    
    def _load_density_grid(self, grid_dict: Dict, device: torch.device) -> torch.Tensor:
        """Load density grid from VDB."""
        if "density" not in grid_dict:
            raise ValueError("No density grid found in VDB file")
        
        density_grid = grid_dict["density"]
        
        # Get grid bounds
        bbox = density_grid.evalActiveVoxelBoundingBox()
        min_coord = bbox.min()
        max_coord = bbox.max()
        
        # Calculate dimensions
        dims = (max_coord.z() - min_coord.z() + 1,
                max_coord.y() - min_coord.y() + 1,
                max_coord.x() - min_coord.x() + 1)
        
        # Initialize tensor
        density_tensor = torch.full(dims, self.config.background_value, device=device)
        
        # Fill from VDB grid
        accessor = density_grid.getConstAccessor()
        for z in range(dims[0]):
            for y in range(dims[1]):
                for x in range(dims[2]):
                    coord = (min_coord.x() + x, min_coord.y() + y, min_coord.z() + z)
                    if accessor.isValueOn(coord):
                        value = accessor.getValue(coord)
                        density_tensor[z, y, x] = value
        
        return density_tensor
    
    def _load_sh_grids(self, grid_dict: Dict, device: torch.device, sh_degree: int) -> torch.Tensor:
        """Load SH coefficient grids from VDB."""
        num_coeffs = SphericalHarmonics.get_num_coeffs(sh_degree)
        
        # Find SH coefficient grids
        sh_grid_names = [f"sh_coeff_{i}" for i in range(num_coeffs)]
        missing_grids = [name for name in sh_grid_names if name not in grid_dict]
        
        if missing_grids:
            raise ValueError(f"Missing SH grids: {missing_grids}")
        
        # Get dimensions from first grid
        first_grid = grid_dict[sh_grid_names[0]]
        bbox = first_grid.evalActiveVoxelBoundingBox()
        min_coord = bbox.min()
        max_coord = bbox.max()
        
        dims = (max_coord.z() - min_coord.z() + 1,
                max_coord.y() - min_coord.y() + 1,
                max_coord.x() - min_coord.x() + 1)
        
        # Initialize tensor [D, H, W, 3, num_coeffs]
        sh_tensor = torch.zeros((*dims, 3, num_coeffs), device=device)
        
        # Load each coefficient
        for coeff_idx, grid_name in enumerate(sh_grid_names):
            sh_grid = grid_dict[grid_name]
            accessor = sh_grid.getConstAccessor()
            
            for z in range(dims[0]):
                for y in range(dims[1]):
                    for x in range(dims[2]):
                        coord = (min_coord.x() + x, min_coord.y() + y, min_coord.z() + z)
                        if accessor.isValueOn(coord):
                            rgb_value = accessor.getValue(coord)
                            sh_tensor[z, y, x, :, coeff_idx] = torch.tensor(rgb_value, device=device)
        
        return sh_tensor
    
    def optimize_vdb_storage(self, vdb_path: str, output_path: str = None) -> bool:
        """
        Optimize VDB file for better compression and access patterns.
        
        Args:
            vdb_path: Input VDB file path
            output_path: Output path (overwrites input if None)
            
        Returns:
            Success flag
        """
        if output_path is None:
            output_path = vdb_path
        
        try:
            # Load grids
            grids = vdb.read(vdb_path)
            
            # Optimize each grid
            optimized_grids = []
            for grid in grids:
                # Prune inactive voxels
                grid.pruneInactive()
                
                # Compress
                grid.compress()
                
                # TODO: Add more optimization techniques
                # - Hierarchical levels of detail
                # - Quantization
                # - Spatial clustering
                
                optimized_grids.append(grid)
            
            # Write optimized file
            vdb.write(output_path, optimized_grids)
            
            logger.info(f"Optimized VDB file saved to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to optimize VDB file: {str(e)}")
            return False
    
    def create_hierarchical_lod(self,
                               voxel_grid: VoxelGrid,
                               output_dir: str,
                               levels: int = 3) -> List[str]:
        """
        Create hierarchical levels of detail for large scenes.
        
        Args:
            voxel_grid: Source voxel grid
            output_dir: Output directory for LOD files
            levels: Number of LOD levels
            
        Returns:
            List of created file paths
        """
        os.makedirs(output_dir, exist_ok=True)
        created_files = []
        
        current_grid = voxel_grid
        
        for level in range(levels):
            # Create filename
            filename = f"lod_{level}.vdb"
            filepath = os.path.join(output_dir, filename)
            
            # Export current level
            success = self.export_plenoxel_to_vdb(current_grid, filepath)
            if success:
                created_files.append(filepath)
            
            # Create next level (downsample by factor of 2)
            if level < levels - 1:
                current_grid = self._downsample_voxel_grid(current_grid)
        
        return created_files
    
    def _downsample_voxel_grid(self, voxel_grid: VoxelGrid) -> VoxelGrid:
        """Downsample voxel grid by factor of 2."""
        # Get current data
        density = voxel_grid.density
        sh_coeffs = voxel_grid.sh_coeffs
        
        # Downsample using average pooling
        density_down = torch.nn.functional.avg_pool3d(
            density.unsqueeze(0).unsqueeze(0),
            kernel_size=2,
            stride=2
        ).squeeze(0).squeeze(0)
        
        # Downsample SH coefficients
        D, H, W, n_channels, n_coeffs = sh_coeffs.shape
        sh_coeffs_reshaped = sh_coeffs.view(D, H, W, -1).permute(3, 0, 1, 2).unsqueeze(0)
        sh_coeffs_down = torch.nn.functional.avg_pool3d(
            sh_coeffs_reshaped,
            kernel_size=2,
            stride=2
        ).squeeze(0).permute(1, 2, 3, 0).view(*density_down.shape, n_channels, n_coeffs)
        
        # Create new voxel grid
        downsampled_grid = VoxelGrid(
            resolution=tuple(density_down.shape),
            scene_bounds=voxel_grid.scene_bounds,
            sh_degree=voxel_grid.sh_degree
        ).to(voxel_grid.density.device)
        
        downsampled_grid.density.data = density_down
        downsampled_grid.sh_coeffs.data = sh_coeffs_down
        
        return downsampled_grid
    
    def get_storage_stats(self, vdb_path: str) -> Dict[str, Any]:
        """
        Get storage statistics for a VDB file.
        
        Args:
            vdb_path: Path to VDB file
            
        Returns:
            Dictionary with storage statistics
        """
        try:
            grids = vdb.read(vdb_path)
            
            stats = {
                "file_size_mb": os.path.getsize(vdb_path) / (1024 * 1024),
                "num_grids": len(grids),
                "grids": {}
            }
            
            for grid in grids:
                grid_stats = {
                    "name": grid.name,
                    "grid_class": str(grid.gridClass),
                    "active_voxel_count": grid.activeVoxelCount(),
                    "memory_usage_mb": grid.memUsage() / (1024 * 1024)
                }
                
                if hasattr(grid, 'evalActiveVoxelBoundingBox'):
                    bbox = grid.evalActiveVoxelBoundingBox()
                    grid_stats["bounding_box"] = {
                        "min": (bbox.min().x(), bbox.min().y(), bbox.min().z()),
                        "max": (bbox.max().x(), bbox.max().y(), bbox.max().z())
                    }
                
                stats["grids"][grid.name] = grid_stats
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get storage stats: {str(e)}")
            return {}


def save_plenoxel_as_neuralvdb(voxel_grid: VoxelGrid,
                              output_path: str,
                              model_config: PlenoxelConfig = None,
                              vdb_config: NeuralVDBConfig = None) -> bool:
    """
    Convenience function to save Plenoxel data as NeuralVDB.
    
    Args:
        voxel_grid: Plenoxel voxel grid
        output_path: Output VDB file path
        model_config: Plenoxel model configuration
        vdb_config: NeuralVDB configuration
        
    Returns:
        Success flag
    """
    manager = NeuralVDBManager(vdb_config)
    return manager.export_plenoxel_to_vdb(voxel_grid, output_path, model_config)


def load_plenoxel_from_neuralvdb(vdb_path: str,
                                device: torch.device = None) -> Tuple[VoxelGrid, PlenoxelConfig]:
    """
    Convenience function to load Plenoxel data from NeuralVDB.
    
    Args:
        vdb_path: Path to VDB file
        device: Target device
        
    Returns:
        Tuple of (VoxelGrid, PlenoxelConfig)
    """
    manager = NeuralVDBManager()
    return manager.load_plenoxel_from_vdb(vdb_path, device) 