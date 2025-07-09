"""
Block Manager for Block-NeRF

This module provides block management functionality for Block-NeRF,
handling the spatial decomposition and coordination of multiple blocks.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np

from .core import BlockNeRFConfig, BlockNeRFModel
from .visibility_network import VisibilityNetwork

# Type aliases
Tensor = torch.Tensor


class BlockManager:
    """
    Manages multiple Block-NeRF instances for large-scale scene reconstruction.

    This component handles:
    - Scene spatial decomposition into blocks
    - Block creation and management
    - Block selection for training and inference
    - Visibility determination
    """

    def __init__(
        self,
        scene_bounds: tuple[float, float, float, float, float, float],
        block_size: float = 75.0,
        overlap_ratio: float = 0.1,
        device: str = "cuda",
        use_visibility_network: bool = True,
    ):
        """
        Initialize Block Manager.

        Args:
            scene_bounds: Scene boundaries (x_min, y_min, z_min, x_max, y_max, z_max)
            block_size: Size of each block
            overlap_ratio: Overlap ratio between adjacent blocks
            device: Device for computation
            use_visibility_network: Whether to use visibility network
        """
        self.scene_bounds = scene_bounds
        self.block_size = block_size
        self.overlap_ratio = overlap_ratio
        self.device = torch.device(device)

        # Block storage
        self.blocks: dict[str, BlockNeRFModel] = {}
        self.block_centers: dict[str, Tensor] = {}
        self.block_metadata: dict[str, dict] = {}

        # Visibility network
        if use_visibility_network:
            self.visibility_network = VisibilityNetwork().to(self.device)
        else:
            self.visibility_network = None

        # Generate initial block layout
        self._generate_block_layout()

    def _generate_block_layout(self) -> None:
        """Generate block centers based on scene bounds and block size."""
        (x_min, y_min, z_min, x_max, y_max, z_max) = self.scene_bounds

        # Calculate step size with overlap
        step_size = self.block_size * (1 - self.overlap_ratio)

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
                        "id": block_id,
                        "center": [x, y, z],
                        "radius": self.block_size,
                        "active": False,
                        "trained": False,
                    }

                    block_id += 1

        print(f"Generated {len(self.block_centers)} blocks for scene")

    def create_block(
        self,
        block_name: str,
        model_config: BlockNeRFConfig,
    ) -> BlockNeRFModel:
        """
        Create a new Block-NeRF instance.

        Args:
            block_name: Name/ID of the block
            model_config: Configuration for the NeRF model

        Returns:
            Created Block-NeRF instance
        """
        if block_name not in self.block_centers:
            raise ValueError(f"Block {block_name} not found in layout")

        center = self.block_centers[block_name]

        # Create block model
        block = BlockNeRFModel(model_config).to(self.device)
        block.set_block_center(center)

        self.blocks[block_name] = block
        self.block_metadata[block_name]["active"] = True

        return block

    def get_blocks_for_position(
        self,
        position: Tensor,
        max_distance: float | None = None,
    ) -> list[str]:
        """
        Get block names that contain or are near a given position.

        Args:
            position: 3D position (3,)
            max_distance: Maximum distance to consider

        Returns:
            List of block names
        """
        if max_distance is None:
            max_distance = self.block_size * 1.5

        relevant_blocks = []

        for block_name, center in self.block_centers.items():
            distance = torch.norm(position - center).item()
            if distance <= max_distance:
                relevant_blocks.append(block_name)

        return relevant_blocks

    def get_blocks_for_camera(
        self,
        camera_position: Tensor,
        camera_direction: Tensor,
        max_distance: float = 200.0,
        use_visibility: bool = True,
    ) -> list[str]:
        """
        Get blocks visible from a camera position.

        Args:
            camera_position: Camera position (3,)
            camera_direction: Camera viewing direction (3,)
            max_distance: Maximum distance to consider
            use_visibility: Whether to use visibility network

        Returns:
            List of visible block names
        """
        visible_blocks = []

        for block_name, center in self.block_centers.items():
            # Distance culling
            distance = torch.norm(camera_position - center).item()
            if distance > max_distance:
                continue

            # Frustum culling (simplified)
            to_block = center - camera_position
            to_block_normalized = to_block / torch.norm(to_block)
            dot_product = torch.dot(camera_direction, to_block_normalized).item()

            # Only consider blocks in front of camera (rough frustum)
            if dot_product < 0.1:  # Adjust threshold as needed
                continue

            # Visibility network check
            if use_visibility and self.visibility_network is not None:
                with torch.no_grad():
                    visibility = self.visibility_network(
                        camera_position.unsqueeze(0),
                        camera_direction.unsqueeze(0),
                        center.unsqueeze(0),
                    )
                if visibility.item() < 0.1:  # Visibility threshold
                    continue

            visible_blocks.append(block_name)

        return visible_blocks

    def get_training_blocks_for_image(
        self,
        camera_position: Tensor,
        image_bounds: tuple[Tensor, Tensor] | None = None,
    ) -> list[str]:
        """
        Get blocks relevant for training with a specific image.

        Args:
            camera_position: Camera position (3,)
            image_bounds: Optional image frustum bounds

        Returns:
            List of training block names
        """
        # For training, we typically want blocks closer to the camera
        training_distance = self.block_size * 3.0

        training_blocks = []
        for block_name, center in self.block_centers.items():
            distance = torch.norm(camera_position - center).item()
            if distance <= training_distance:
                training_blocks.append(block_name)

        return training_blocks

    def compute_block_weights(
        self,
        camera_position: Tensor,
        block_names: list[str],
        power: float = 2.0,
    ) -> Tensor:
        """
        Compute interpolation weights for blocks based on distance.

        Args:
            camera_position: Camera position (3,)
            block_names: List of block names
            power: Power for inverse distance weighting

        Returns:
            Normalized weights (len(block_names),)
        """
        if not block_names:
            return torch.tensor([], device=self.device)

        distances = []
        for block_name in block_names:
            center = self.block_centers[block_name]
            distance = torch.norm(camera_position - center).item()
            distances.append(max(distance, 0.1))  # Avoid division by zero

        distances = torch.tensor(distances, device=self.device)
        weights = 1.0 / (distances**power)
        weights = weights / weights.sum()

        return weights

    def save_block_layout(self, save_path: str) -> None:
        """Save block layout and metadata."""
        layout_data = {
            "scene_bounds": self.scene_bounds,
            "block_size": self.block_size,
            "overlap_ratio": self.overlap_ratio,
            "blocks": {},
        }

        for block_name, metadata in self.block_metadata.items():
            layout_data["blocks"][block_name] = metadata.copy()
            # Convert tensor to list for JSON serialization
            if block_name in self.block_centers:
                center = self.block_centers[block_name].cpu().tolist()
                layout_data["blocks"][block_name]["center"] = center

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(layout_data, f, indent=2)

        print(f"Saved block layout to {save_path}")

    def load_block_layout(self, load_path: str) -> None:
        """Load block layout and metadata."""
        with open(load_path, "r") as f:
            layout_data = json.load(f)

        self.scene_bounds = layout_data["scene_bounds"]
        self.block_size = layout_data["block_size"]
        self.overlap_ratio = layout_data["overlap_ratio"]

        # Recreate block centers and metadata
        self.block_centers = {}
        self.block_metadata = layout_data["blocks"]

        for block_name, metadata in self.block_metadata.items():
            center = torch.tensor(metadata["center"], dtype=torch.float32, device=self.device)
            self.block_centers[block_name] = center

        print(f"Loaded block layout from {load_path}")

    def save_blocks(self, save_dir: str) -> None:
        """Save all active blocks."""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        for block_name, block in self.blocks.items():
            block_path = save_dir / f"{block_name}.pth"
            torch.save(
                {
                    "model_state_dict": block.state_dict(),
                    "metadata": self.block_metadata[block_name],
                },
                block_path,
            )

        # Save visibility network
        if self.visibility_network is not None:
            visibility_path = save_dir / "visibility_network.pth"
            torch.save(self.visibility_network.state_dict(), visibility_path)

        # Save block layout
        layout_path = save_dir / "block_layout.json"
        self.save_block_layout(str(layout_path))

        print(f"Saved {len(self.blocks)} blocks to {save_dir}")

    def load_blocks(
        self,
        load_dir: str,
        model_config: BlockNeRFConfig,
    ) -> None:
        """Load blocks from directory."""
        load_dir = Path(load_dir)

        # Load block layout first
        layout_file = load_dir / "block_layout.json"
        if layout_file.exists():
            self.load_block_layout(str(layout_file))

        # Load visibility network
        visibility_path = load_dir / "visibility_network.pth"
        if visibility_path.exists() and self.visibility_network is not None:
            self.visibility_network.load_state_dict(
                torch.load(visibility_path, map_location=self.device)
            )

        # Load individual blocks
        for block_file in load_dir.glob("block_*.pth"):
            block_name = block_file.stem

            checkpoint = torch.load(block_file, map_location=self.device)

            # Create block
            block = self.create_block(block_name, model_config)
            block.load_state_dict(checkpoint["model_state_dict"])

            # Update metadata
            if "metadata" in checkpoint:
                self.block_metadata[block_name].update(checkpoint["metadata"])
                self.block_metadata[block_name]["trained"] = True

        print(f"Loaded {len(self.blocks)} blocks from {load_dir}")

    def get_scene_statistics(self) -> Dict:
        """Get statistics about the scene and blocks."""
        total_blocks = len(self.block_centers)
        active_blocks = len(self.blocks)
        trained_blocks = sum(
            1 for meta in self.block_metadata.values() if meta.get("trained", False)
        )

        return {
            "total_blocks": total_blocks,
            "active_blocks": active_blocks,
            "trained_blocks": trained_blocks,
            "scene_bounds": self.scene_bounds,
            "block_size": self.block_size,
            "overlap_ratio": self.overlap_ratio,
        }

    def get_state_dict(self) -> Dict:
        """Get state dictionary for checkpointing."""
        state_dict = {
            "scene_bounds": self.scene_bounds,
            "block_size": self.block_size,
            "overlap_ratio": self.overlap_ratio,
            "block_metadata": self.block_metadata,
        }

        # Save block centers as lists
        block_centers_list = {}
        for name, center in self.block_centers.items():
            block_centers_list[name] = center.cpu().tolist()
        state_dict["block_centers"] = block_centers_list

        return state_dict

    def load_state_dict(self, state_dict: Dict) -> None:
        """Load state dictionary from checkpoint."""
        self.scene_bounds = state_dict["scene_bounds"]
        self.block_size = state_dict["block_size"]
        self.overlap_ratio = state_dict["overlap_ratio"]
        self.block_metadata = state_dict["block_metadata"]

        # Restore block centers
        self.block_centers = {}
        for name, center_list in state_dict["block_centers"].items():
            center = torch.tensor(center_list, dtype=torch.float32, device=self.device)
            self.block_centers[name] = center

    def set_train_mode(self) -> None:
        """Set all blocks to training mode."""
        for block in self.blocks.values():
            block.train()
        if self.visibility_network is not None:
            self.visibility_network.train()

    def set_eval_mode(self) -> None:
        """Set all blocks to evaluation mode."""
        for block in self.blocks.values():
            block.eval()
        if self.visibility_network is not None:
            self.visibility_network.eval()


def create_block_manager(
    config_or_bounds,
    block_size: float = 75.0,
    overlap_ratio: float = 0.1,
    device: str = "cuda",
    use_visibility_network: bool = True,
) -> BlockManager:
    """
    Create a block manager with specified parameters.

    Args:
        config_or_bounds: Either a BlockNeRFConfig object or scene bounds tuple
        block_size: Block size in meters (ignored if config provided)
        overlap_ratio: Overlap ratio between blocks (ignored if config provided)
        device: Device for computation
        use_visibility_network: Whether to use visibility network

    Returns:
        BlockManager instance
    """
    # Handle both config object and direct parameters
    if hasattr(config_or_bounds, "scene_bounds"):
        # It's a BlockNeRFConfig object
        config = config_or_bounds
        return BlockManager(
            scene_bounds=config.scene_bounds,
            block_size=config.block_size,
            overlap_ratio=config.overlap_ratio,
            device=device,
            use_visibility_network=use_visibility_network,
        )
    else:
        # It's scene bounds tuple
        scene_bounds = config_or_bounds
        return BlockManager(
            scene_bounds=scene_bounds,
            block_size=block_size,
            overlap_ratio=overlap_ratio,
            device=device,
            use_visibility_network=use_visibility_network,
        )
