"""
NeuralVDB Integration Example for Plenoxels

This example demonstrates how to use the NeuralVDB interface to save and load
Plenoxel data for efficient external storage.

Usage:
    python -m src.plenoxels.neuralvdb_example --mode save
    python -m src.plenoxels.neuralvdb_example --mode load
    python -m src.plenoxels.neuralvdb_example --mode stats
"""

import argparse
import torch
import numpy as np
import os
from pathlib import Path

from .core import VoxelGrid, PlenoxelConfig, PlenoxelModel
from .neuralvdb_interface import (
    NeuralVDBManager, NeuralVDBConfig, save_plenoxel_as_neuralvdb, load_plenoxel_from_neuralvdb
)
from .trainer import PlenoxelTrainer, PlenoxelTrainerConfig
from .dataset import PlenoxelDataset, PlenoxelDatasetConfig


def create_demo_voxel_grid(device: torch.device = None) -> VoxelGrid:
    """Create a demo voxel grid for testing."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create a simple voxel grid
    resolution = (64, 64, 64)
    scene_bounds = [-1.0, -1.0, -1.0, 1.0, 1.0, 1.0]
    sh_degree = 2
    
    voxel_grid = VoxelGrid(
        resolution=resolution, scene_bounds=scene_bounds, sh_degree=sh_degree
    ).to(device)
    
    # Initialize with some demo data
    # Create a sphere-like density distribution
    center = torch.tensor([32, 32, 32], dtype=torch.float32, device=device)
    coords = torch.stack(torch.meshgrid(
        torch.arange(
            64,
            device=device,
        )
    ), dim=-1).float()
    
    # Distance from center
    distances = torch.norm(coords - center, dim=-1)
    
    # Create density field (negative log-space for proper density)
    max_distance = 20.0
    density = -torch.clamp(distances / max_distance, 0, 1) * 10
    
    # Set density
    voxel_grid.density.data = density
    
    # Initialize SH coefficients with some color variation
    num_coeffs = voxel_grid.num_sh_coeffs
    sh_coeffs = torch.zeros(*resolution, 3, num_coeffs, device=device)
    
    # Base color (DC component)
    sh_coeffs[:, :, :, 0, 0] = 0.5  # Red
    sh_coeffs[:, :, :, 1, 0] = 0.3  # Green  
    sh_coeffs[:, :, :, 2, 0] = 0.8  # Blue
    
    # Add some variation based on position
    if num_coeffs > 1:
        # Add directional variation
        sh_coeffs[:, :, :, :, 1] = coords[:, :, :, 0:1].expand(-1, -1, -1, 3) / 64.0 * 0.2
        
    voxel_grid.sh_coeffs.data = sh_coeffs
    
    return voxel_grid


def save_example(output_path: str):
    """Example of saving Plenoxel data to NeuralVDB."""
    print("🚀 Creating demo voxel grid...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    voxel_grid = create_demo_voxel_grid(device)
    
    # Create model config
    model_config = PlenoxelConfig(
        grid_resolution=(
            64,
            64,
            64,
        )
    )
    
    # Create VDB config for optimized storage
    vdb_config = NeuralVDBConfig(
        compression_level=8, # Higher compression
        half_precision=True, tolerance=1e-5, # More aggressive sparsity
        include_metadata=True
    )
    
    print(f"💾 Saving to {output_path}...")
    success = save_plenoxel_as_neuralvdb(
        voxel_grid=voxel_grid, output_path=output_path, model_config=model_config, vdb_config=vdb_config
    )
    
    if success:
        # Get file size
        file_size = os.path.getsize(output_path) / (1024 * 1024)
        print(f"✅ Successfully saved VDB file ({file_size:.2f} MB)")
        
        # Get storage statistics
        manager = NeuralVDBManager(vdb_config)
        stats = manager.get_storage_stats(output_path)
        print(f"📊 Storage stats:")
        print(f"   - File size: {stats.get('file_size_mb', 0):.2f} MB")
        print(f"   - Number of grids: {stats.get('num_grids', 0)}")
        
        for grid_name, grid_stats in stats.get('grids', {}).items():
            print(f"   - {grid_name}:")
            print(f"     * Active voxels: {grid_stats.get('active_voxel_count', 0)}")
            print(f"     * Memory usage: {grid_stats.get('memory_usage_mb', 0):.2f} MB")
    else:
        print("❌ Failed to save VDB file")


def load_example(vdb_path: str):
    """Example of loading Plenoxel data from NeuralVDB."""
    if not os.path.exists(vdb_path):
        print(f"❌ VDB file not found: {vdb_path}")
        return
    
    print(f"📂 Loading from {vdb_path}...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        voxel_grid, model_config = load_plenoxel_from_neuralvdb(vdb_path, device)
        
        print("✅ Successfully loaded VDB file")
        print(f"📐 Grid resolution: {voxel_grid.resolution}")
        print(f"🎯 Scene bounds: {voxel_grid.scene_bounds}")
        print(f"🌟 SH degree: {voxel_grid.sh_degree}")
        print(f"💾 Device: {voxel_grid.density.device}")
        
        # Print some statistics
        density_stats = {
            'min': voxel_grid.density.min(
            )
        }
        print(f"📊 Density stats: {density_stats}")
        
        # Check non-zero elements
        non_empty_voxels = (torch.exp(voxel_grid.density) > 0.01).sum().item()
        total_voxels = voxel_grid.density.numel()
        sparsity = 1.0 - (non_empty_voxels / total_voxels)
        print(f"🎭 Sparsity: {sparsity:.1%} ({non_empty_voxels}/{total_voxels} non-empty)")
        
    except Exception as e:
        print(f"❌ Failed to load VDB file: {str(e)}")


def stats_example(vdb_path: str):
    """Show detailed statistics for a VDB file."""
    if not os.path.exists(vdb_path):
        print(f"❌ VDB file not found: {vdb_path}")
        return
    
    print(f"📊 Getting statistics for {vdb_path}...")
    
    manager = NeuralVDBManager()
    stats = manager.get_storage_stats(vdb_path)
    
    if not stats:
        print("❌ Failed to get statistics")
        return
    
    print("📈 VDB File Statistics:")
    print(f"   File size: {stats['file_size_mb']:.2f} MB")
    print(f"   Number of grids: {stats['num_grids']}")
    print()
    
    for grid_name, grid_stats in stats['grids'].items():
        print(f"🔹 Grid: {grid_name}")
        print(f"   Class: {grid_stats['grid_class']}")
        print(f"   Active voxels: {grid_stats['active_voxel_count']:, }")
        print(f"   Memory usage: {grid_stats['memory_usage_mb']:.2f} MB")
        
        if 'bounding_box' in grid_stats:
            bbox = grid_stats['bounding_box']
            print(f"   Bounding box: {bbox['min']} to {bbox['max']}")
        print()


def hierarchical_lod_example(output_dir: str):
    """Example of creating hierarchical levels of detail."""
    print("🏗️ Creating hierarchical LOD example...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create a larger demo grid
    resolution = (128, 128, 128)
    scene_bounds = [-2.0, -2.0, -2.0, 2.0, 2.0, 2.0]
    sh_degree = 2
    
    voxel_grid = VoxelGrid(
        resolution=resolution, scene_bounds=scene_bounds, sh_degree=sh_degree
    ).to(device)
    
    # Create more complex geometry
    center = torch.tensor([64, 64, 64], dtype=torch.float32, device=device)
    coords = torch.stack(torch.meshgrid(
        torch.arange(
            128,
            device=device,
        )
    ), dim=-1).float()
    
    # Multiple spheres
    centers = [
        torch.tensor(
            [40,
            40,
            40],
            device=device,
        )
    
    density = torch.full(resolution, -10.0, device=device)
    
    for center in centers:
        distances = torch.norm(coords - center, dim=-1)
        sphere_density = -torch.clamp(distances / 15.0, 0, 1) * 8
        density = torch.maximum(density, sphere_density)
    
    voxel_grid.density.data = density
    
    # Initialize SH coefficients
    num_coeffs = voxel_grid.num_sh_coeffs
    sh_coeffs = torch.rand(*resolution, 3, num_coeffs, device=device) * 0.5
    voxel_grid.sh_coeffs.data = sh_coeffs
    
    # Create hierarchical LOD
    manager = NeuralVDBManager(NeuralVDBConfig(compression_level=9))
    created_files = manager.create_hierarchical_lod(
        voxel_grid=voxel_grid, output_dir=output_dir, levels=4
    )
    
    print(f"✅ Created {len(created_files)} LOD levels:")
    for i, filepath in enumerate(created_files):
        file_size = os.path.getsize(filepath) / (1024 * 1024)
        print(f"   Level {i}: {filepath} ({file_size:.2f} MB)")


def optimize_example(input_path: str, output_path: str):
    """Example of optimizing a VDB file."""
    if not os.path.exists(input_path):
        print(f"❌ Input VDB file not found: {input_path}")
        return
    
    print(f"⚡ Optimizing {input_path} -> {output_path}...")
    
    # Get original stats
    manager = NeuralVDBManager()
    original_stats = manager.get_storage_stats(input_path)
    original_size = original_stats.get('file_size_mb', 0)
    
    # Optimize
    success = manager.optimize_vdb_storage(input_path, output_path)
    
    if success:
        # Get optimized stats
        optimized_stats = manager.get_storage_stats(output_path)
        optimized_size = optimized_stats.get('file_size_mb', 0)
        
        compression_ratio = original_size / optimized_size if optimized_size > 0 else 1.0
        
        print(f"✅ Optimization complete:")
        print(f"   Original size: {original_size:.2f} MB")
        print(f"   Optimized size: {optimized_size:.2f} MB")
        print(f"   Compression ratio: {compression_ratio:.2f}x")
    else:
        print("❌ Optimization failed")


def main():
    parser = argparse.ArgumentParser(description="NeuralVDB Integration Example")
    parser.add_argument(
        '--mode',
        choices=['save',
        'load',
        'stats',
        'lod',
        'optimize'],
        default='save',
        help='Mode to run',
    )
    parser.add_argument('--vdb_path', default='demo_plenoxel.vdb', help='Path to VDB file')
    parser.add_argument('--output_path', help='Output path for optimization')
    parser.add_argument('--lod_dir', default='lod_output', help='Directory for LOD output')
    
    args = parser.parse_args()
    
    print("🎯 NeuralVDB Integration Example for Plenoxels")
    print("=" * 50)
    
    try:
        if args.mode == 'save':
            save_example(args.vdb_path)
            
        elif args.mode == 'load':
            load_example(args.vdb_path)
            
        elif args.mode == 'stats':
            stats_example(args.vdb_path)
            
        elif args.mode == 'lod':
            hierarchical_lod_example(args.lod_dir)
            
        elif args.mode == 'optimize':
            output_path = args.output_path or args.vdb_path.replace('.vdb', '_optimized.vdb')
            optimize_example(args.vdb_path, output_path)
            
    except ImportError as e:
        print(f"❌ Import error: {str(e)}")
        print("💡 Make sure OpenVDB is installed: pip install openvdb")
    except Exception as e:
        print(f"❌ Error: {str(e)}")


if __name__ == "__main__":
    main() 