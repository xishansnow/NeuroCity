from __future__ import annotations

"""
Block Manager for Block-NeRF

This module manages multiple Block-NeRF instances, handles block placement, and coordinates training and inference across blocks.
"""

from typing import Optional, Union


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import os
from pathlib import Path

from .block_nerf_model import BlockNeRF
from .visibility_network import VisibilityNetwork

class BlockManager:
    """
    Manages multiple Block-NeRF instances for large-scale scene reconstruction
    """
    
    def __init__(
        self,
        scene_bounds: Tuple[Tuple[float,
        float],
        Tuple[float,
        float],
        Tuple[float,
        float]],
        block_size: float = 75.0,
        overlap_ratio: float = 0.5,
        device: str = 'cuda',
    ):
        """
        Initialize Block Manager
        
        Args:
            scene_bounds: ((x_min, x_max), (y_min, y_max), (z_min, z_max))
            block_size: Size of each block (radius in meters)
            overlap_ratio: Overlap ratio between adjacent blocks
            device: Device to use for computation
        """
        self.scene_bounds = scene_bounds
        self.block_size = block_size
        self.overlap_ratio = overlap_ratio
        self.device = device
        
        # Storage for blocks
        self.blocks: Dict[str, BlockNeRF] = {}
        self.block_centers: Dict[str, torch.Tensor] = {}
        self.block_metadata: Dict[str, dict] = {}
        
        # Visibility network (shared across blocks)
        self.visibility_network = VisibilityNetwork().to(device)
        
        # Generate block layout
        self._generate_block_layout()
        
    def _generate_block_layout(self):
        """Generate block centers based on scene bounds and block size"""
        (x_min, x_max), (y_min, y_max), (z_min, z_max) = self.scene_bounds
        
        # Calculate step size with overlap
        step_size = self.block_size * 2 * (1 - self.overlap_ratio)
        
        # Generate grid of block centers
        x_centers = np.arange(x_min, x_max + step_size, step_size)
        y_centers = np.arange(y_min, y_max + step_size, step_size)
        z_centers = np.arange(z_min, z_max + step_size, step_size)
        
        # Create block centers
        block_id = 0
        for x in x_centers:
            for y in y_centers:
                for z in z_centers:
                    block_name = f"block_{block_id:04d}"
                    center = torch.tensor([x, y, z], dtype=torch.float32, device=self.device)
                    
                    self.block_centers[block_name] = center
                    self.block_metadata[block_name] = {
                        'id': block_id, 'center': [x, y, z], 'radius': self.block_size, 'active': False, 'trained': False
                    }
                    
                    block_id += 1
        
        print(f"Generated {len(self.block_centers)} blocks for scene")
        
    def create_block(
        self,
        block_name: str,
        network_config: dict,
        num_appearance_embeddings: int = 1000
    ):
        """
        Create a new Block-NeRF instance
        
        Args:
            block_name: Name/ID of the block
            network_config: Configuration for the NeRF network
            num_appearance_embeddings: Number of appearance embeddings
            
        Returns:
            Created Block-NeRF instance
        """
        if block_name not in self.block_centers:
            raise ValueError(f"Block {block_name} not found in layout")
        
        center = self.block_centers[block_name]
        
        block = BlockNeRF(
            network_config=network_config, block_center=center, block_radius=self.block_size, num_appearance_embeddings=num_appearance_embeddings
        ).to(self.device)
        
        self.blocks[block_name] = block
        self.block_metadata[block_name]['active'] = True
        
        return block
    
    def get_blocks_for_position(
        self,
        position: torch.Tensor,
        max_distance: Optional[float] = None    
    ):
        """
        Get block names that contain or are near a given position
        
        Args:
            position: 3D position (3, )
            max_distance: Maximum distance to consider (default: block_size * 2)
            
        Returns:
            list of block names
        """
        if max_distance is None:
            max_distance = self.block_size * 2
        
        relevant_blocks = []
        
        for block_name, center in self.block_centers.items():
            distance = torch.norm(position - center).item()
            if distance <= max_distance:
                relevant_blocks.append(block_name)
        
        return relevant_blocks
    
    def get_blocks_for_camera(
        self,
        camera_position: torch.Tensor,
        camera_direction: torch.Tensor,
        max_distance: float = 100.0,
        use_visibility: bool = True
    ):
        """
        Get relevant blocks for a camera viewpoint
        
        Args:
            camera_position: Camera position (3, )
            camera_direction: Camera viewing direction (3, )
            max_distance: Maximum distance to consider
            use_visibility: Whether to use visibility filtering
            
        Returns:
            list of relevant block names
        """
        relevant_blocks = []
        
        # Get blocks within distance
        nearby_blocks = self.get_blocks_for_position(camera_position, max_distance)
        
        if not use_visibility:
            return [name for name in nearby_blocks if name in self.blocks]
        
        # Use visibility network for filtering
        block_centers = torch.stack([self.block_centers[name] for name in nearby_blocks])
        block_radii = torch.tensor([self.block_size] * len(nearby_blocks), device=self.device)
        
        visibility_mask = self.visibility_network.filter_blocks_by_visibility(
            camera_position, block_centers, block_radii
        )
        
        filtered_blocks = [nearby_blocks[i] for i in range(len(nearby_blocks)) 
                          if visibility_mask[i] and nearby_blocks[i] in self.blocks]
        
        return filtered_blocks
    
    def get_training_blocks_for_image(
        self,
        camera_position: torch.Tensor,
        image_bounds: Optional[Tuple[torch.Tensor,
        torch.Tensor]] = None
    ):
        """
        Get blocks that should be trained on a given image
        
        Args:
            camera_position: Camera position (3, )
            image_bounds: Optional bounds of what's visible in the image
            
        Returns:
            list of block names for training
        """
        # For training, use a larger radius to ensure good coverage
        training_radius = self.block_size * 1.5
        
        training_blocks = []
        
        for block_name, center in self.block_centers.items():
            distance = torch.norm(camera_position - center).item()
            
            # Include blocks within training radius
            if distance <= training_radius:
                training_blocks.append(block_name)
        
        return training_blocks
    
    def compute_block_weights(
        self,
        camera_position: torch.Tensor,
        block_names: List[str],
        power: float = 2.0  
    ):
        """
        Compute interpolation weights for blocks based on distance
        
        Args:
            camera_position: Camera position (3, )
            block_names: list of block names
            power: Power for inverse distance weighting
            
        Returns:
            Normalized weights (len(block_names), )
        """
        if not block_names:
            return torch.tensor([], device=self.device)
        
        distances = []
        for block_name in block_names:
            center = self.block_centers[block_name]
            distance = torch.norm(camera_position - center).item()
            distances.append(max(distance, 0.1))  # Avoid division by zero
        
        distances = torch.tensor(distances, device=self.device)
        weights = 1.0 / (distances ** power)
        weights = weights / weights.sum()
        
        return weights
    
    def save_block_layout(self, save_path: str):
        """Save block layout and metadata"""
        layout_data = {
            'scene_bounds': self.scene_bounds, 'block_size': self.block_size, 'overlap_ratio': self.overlap_ratio, 'blocks': {
            }
        }
        
        for block_name, metadata in self.block_metadata.items():
            layout_data['blocks'][block_name] = metadata
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(layout_data, f, indent=2)
        
        print(f"Saved block layout to {save_path}")
    
    def load_block_layout(self, load_path: str):
        """Load block layout and metadata"""
        with open(load_path, 'r') as f:
            layout_data = json.load(f)
        
        self.scene_bounds = layout_data['scene_bounds']
        self.block_size = layout_data['block_size']
        self.overlap_ratio = layout_data['overlap_ratio']
        
        # Recreate block centers and metadata
        self.block_centers = {}
        self.block_metadata = layout_data['blocks']
        
        for block_name, metadata in self.block_metadata.items():
            center = torch.tensor(metadata['center'], dtype=torch.float32, device=self.device)
            self.block_centers[block_name] = center
        
        print(f"Loaded block layout from {load_path}")
    
    def save_blocks(self, save_dir: str):
        """Save all active blocks"""
        save_dir = Path(save_dir)  # type: ignore
        save_dir.mkdir(parents=True, exist_ok=True)
        
        for block_name, block in self.blocks.items():
            block_path = save_dir / f"{block_name}.pth"
            torch.save({
                'model_state_dict': block.state_dict(
                )
            }, block_path)
        
        # Save visibility network
        visibility_path = save_dir / "visibility_network.pth"
        torch.save(self.visibility_network.state_dict(), visibility_path)
        
        print(f"Saved {len(self.blocks)} blocks to {save_dir}")
    
    def load_blocks(self, load_dir: str, network_config: dict):
        """Load blocks from directory"""
        load_dir = Path(load_dir)
        
        # Load visibility network
        visibility_path = load_dir / "visibility_network.pth"
        if visibility_path.exists():
            self.visibility_network.load_state_dict(
                torch.load(visibility_path)
            )
        
        # Load individual blocks
        for block_file in load_dir.glob("block_*.pth"):
            block_name = block_file.stem
            
            checkpoint = torch.load(block_file, map_location=self.device)
            
            # Create block
            block = self.create_block(block_name, network_config)
            block.load_state_dict(checkpoint['model_state_dict'])
            
            # Update metadata
            if 'metadata' in checkpoint:
                self.block_metadata[block_name].update(checkpoint['metadata'])
                self.block_metadata[block_name]['trained'] = True
        
        print(f"Loaded {len(self.blocks)} blocks from {load_dir}")
    
    def get_scene_statistics(self) -> dict:
        """Get statistics about the scene and blocks"""
        total_blocks = len(self.block_centers)
        active_blocks = sum(1 for meta in self.block_metadata.values() if meta['active'])
        trained_blocks = sum(1 for meta in self.block_metadata.values() if meta['trained'])
        
        # Compute scene volume
        (x_min, x_max), (y_min, y_max), (z_min, z_max) = self.scene_bounds
        scene_volume = (x_max - x_min) * (y_max - y_min) * (z_max - z_min)
        
        return {
            'total_blocks': total_blocks, 'active_blocks': active_blocks, 'trained_blocks': trained_blocks, 'scene_bounds': self.scene_bounds, 'scene_volume': scene_volume, 'block_size': self.block_size, 'overlap_ratio': self.overlap_ratio
        }
    
    def optimize_block_layout(self, camera_positions: torch.Tensor) -> dict:
        """
        Optimize block layout based on camera distribution
        
        Args:
            camera_positions: Camera positions from training data (N, 3)
            
        Returns:
            Optimization statistics
        """
        # Analyze camera distribution
        camera_mean = camera_positions.mean(dim=0)
        camera_std = camera_positions.std(dim=0)
        
        # Find blocks that are never visited
        visited_blocks = set()
        for pos in camera_positions:
            nearby_blocks = self.get_blocks_for_position(pos, self.block_size * 2)
            visited_blocks.update(nearby_blocks)
        
        unvisited_blocks = set(self.block_centers.keys()) - visited_blocks
        
        # Mark unvisited blocks as inactive
        for block_name in unvisited_blocks:
            self.block_metadata[block_name]['active'] = False
        
        optimization_stats = {
            'camera_mean': camera_mean.tolist(
            )
        }
        
        return optimization_stats

    def forward(
        self, rays_o: torch.Tensor, rays_d: torch.Tensor, **kwargs
    ) -> Dict[str, torch.Tensor]:
        # Implementation of forward method
        pass

    def get_block_info(self) -> Dict[str, Union[torch.Tensor, float, List[float]]]:
        # Implementation of get_block_info method
        pass 