"""
Block Rasterizer for Block-NeRF Inference

This module provides efficient rasterization functionality for Block-NeRF inference,
following the pattern established in SVRaster where rasterization is used
during inference for speed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .core import BlockNeRFConfig, BlockNeRFModel

# Type aliases
Tensor = torch.Tensor
TensorDict = dict[str, Tensor]


@dataclass
class BlockRasterizerConfig:
    """Configuration for Block-NeRF block rasterizer."""

    # Block management
    max_blocks_per_ray: int = 8
    block_overlap_threshold: float = 0.1
    visibility_threshold: float = 0.01

    # Sampling configuration
    samples_per_block: int = 32
    adaptive_sampling: bool = True
    early_termination: bool = True

    # Performance optimization
    use_block_culling: bool = True
    use_morton_ordering: bool = True
    use_cached_features: bool = True

    # Ray marching
    step_size: float = 0.5
    max_steps: int = 512
    termination_threshold: float = 0.99

    # Quality settings
    use_alpha_blending: bool = True
    use_depth_peeling: bool = False
    antialiasing: bool = True


class BlockRasterizer(nn.Module):
    """
    Block Rasterizer for Block-NeRF inference.

    This component is tightly coupled with inference rendering and provides
    efficient block-based rasterization for real-time performance.
    """

    def __init__(self, config: BlockRasterizerConfig):
        super().__init__()
        self.config = config

        # Cache for block features
        if config.use_cached_features:
            self.feature_cache = {}

    def select_blocks_for_ray(
        self,
        ray_origins: Tensor,
        ray_directions: Tensor,
        block_centers: list[Tensor],
        block_radii: list[float],
    ) -> tuple[list[int], Tensor]:
        """Select relevant blocks for each ray."""
        batch_size = ray_origins.shape[0]
        num_blocks = len(block_centers)

        # Compute ray-block intersections
        selected_blocks = []
        intersection_distances = []

        for i in range(batch_size):
            ray_origin = ray_origins[i]
            ray_direction = ray_directions[i]

            ray_blocks = []
            ray_distances = []

            for block_idx, (block_center, block_radius) in enumerate(
                zip(block_centers, block_radii)
            ):
                # Ray-sphere intersection test
                oc = ray_origin - block_center
                a = torch.dot(ray_direction, ray_direction)
                b = 2.0 * torch.dot(oc, ray_direction)
                c = torch.dot(oc, oc) - block_radius**2

                discriminant = b * b - 4 * a * c

                if discriminant >= 0:
                    # Ray intersects block
                    sqrt_discriminant = torch.sqrt(discriminant)
                    t1 = (-b - sqrt_discriminant) / (2 * a)
                    t2 = (-b + sqrt_discriminant) / (2 * a)

                    # Use closest positive intersection
                    t = t1 if t1 > 0 else t2
                    if t > 0:
                        distance = torch.norm(ray_origin + t * ray_direction - block_center)
                        ray_blocks.append(block_idx)
                        ray_distances.append(distance.item())

            # Sort blocks by distance and limit to max_blocks_per_ray
            if ray_blocks:
                sorted_indices = sorted(range(len(ray_blocks)), key=lambda k: ray_distances[k])
                ray_blocks = [
                    ray_blocks[i] for i in sorted_indices[: self.config.max_blocks_per_ray]
                ]

            selected_blocks.append(ray_blocks)

        return selected_blocks, None  # Placeholder for distances

    def compute_block_weights(
        self,
        positions: Tensor,
        block_centers: list[Tensor],
        block_radii: list[float],
        selected_blocks: list[list[int]],
    ) -> Tensor:
        """Compute interpolation weights for blocks."""
        batch_size = positions.shape[0]
        max_blocks = max(len(blocks) for blocks in selected_blocks) if selected_blocks else 1

        weights = torch.zeros(batch_size, max_blocks, device=positions.device)

        for i in range(batch_size):
            pos = positions[i]
            blocks = selected_blocks[i] if i < len(selected_blocks) else []

            if not blocks:
                continue

            block_distances = []
            for block_idx in blocks:
                center = block_centers[block_idx]
                distance = torch.norm(pos - center)
                block_distances.append(distance)

            # Convert to weights using inverse distance weighting
            if block_distances:
                block_distances = torch.stack(block_distances)
                # Add small epsilon to avoid division by zero
                inv_distances = 1.0 / (block_distances + 1e-6)
                block_weights = inv_distances / inv_distances.sum()

                weights[i, : len(block_weights)] = block_weights

        return weights

    def march_rays(
        self,
        models: list[BlockNeRFModel],
        ray_origins: Tensor,
        ray_directions: Tensor,
        block_centers: list[Tensor],
        block_radii: list[float],
        appearance_ids: Tensor,
        exposure_values: Tensor,
        near: Tensor | None = None,
        far: Tensor | None = None,
    ) -> TensorDict:
        """March rays through blocks using rasterization."""
        batch_size = ray_origins.shape[0]
        device = ray_origins.device

        # Initialize outputs
        rgb_output = torch.zeros(batch_size, 3, device=device)
        depth_output = torch.zeros(batch_size, device=device)
        alpha_output = torch.zeros(batch_size, device=device)

        # Set near and far planes
        if near is None:
            near = torch.full((batch_size,), 0.1, device=device)
        if far is None:
            far = torch.full((batch_size,), 1000.0, device=device)

        # Select blocks for each ray
        selected_blocks, _ = self.select_blocks_for_ray(
            ray_origins, ray_directions, block_centers, block_radii
        )

        # Ray marching
        for i in range(batch_size):
            ray_origin = ray_origins[i]
            ray_direction = ray_directions[i]
            ray_blocks = selected_blocks[i] if i < len(selected_blocks) else []

            if not ray_blocks:
                # No blocks selected, set background color
                if hasattr(self.config, "white_background") and self.config.white_background:
                    rgb_output[i] = 1.0
                continue

            # Ray marching parameters
            t_start = near[i].item()
            t_end = far[i].item()
            step_size = self.config.step_size

            # Initialize ray state
            ray_rgb = torch.zeros(3, device=device)
            ray_alpha = 0.0
            ray_depth = 0.0
            transmittance = 1.0

            # March along ray
            t = t_start
            step_count = 0

            while t < t_end and step_count < self.config.max_steps and transmittance > 1e-4:
                # Current position
                pos = ray_origin + t * ray_direction

                # Compute block weights
                block_weights = self.compute_block_weights(
                    pos.unsqueeze(0), block_centers, block_radii, [ray_blocks]
                )[0]

                # Sample from relevant blocks
                total_density = 0.0
                total_color = torch.zeros(3, device=device)
                total_weight = 0.0

                for j, block_idx in enumerate(ray_blocks):
                    if (
                        j >= len(block_weights)
                        or block_weights[j] < self.config.visibility_threshold
                    ):
                        continue

                    model = models[block_idx]

                    # Forward pass through block
                    with torch.no_grad():
                        outputs = model(
                            pos.unsqueeze(0),
                            ray_direction.unsqueeze(0),
                            appearance_ids[i : i + 1],
                            exposure_values[i : i + 1],
                        )

                    density = outputs["density"][0].item()
                    color = outputs["color"][0]
                    weight = block_weights[j].item()

                    total_density += density * weight
                    total_color += color * weight
                    total_weight += weight

                # Normalize color
                if total_weight > 0:
                    total_color = total_color / total_weight

                # Compute alpha for this step
                alpha = 1.0 - torch.exp(-total_density * step_size)
                alpha = alpha.clamp(0.0, 1.0)

                # Accumulate color and alpha
                ray_rgb += transmittance * alpha * total_color
                ray_alpha += transmittance * alpha
                ray_depth += transmittance * alpha * t

                # Update transmittance
                transmittance *= 1.0 - alpha

                # Early termination
                if self.config.early_termination and ray_alpha > self.config.termination_threshold:
                    break

                # Advance along ray
                t += step_size
                step_count += 1

            # Store results
            rgb_output[i] = ray_rgb
            depth_output[i] = ray_depth / max(ray_alpha, 1e-6)  # Normalize depth by alpha
            alpha_output[i] = ray_alpha

        # Add background for pixels with low alpha
        if hasattr(self.config, "white_background") and self.config.white_background:
            rgb_output = rgb_output + (1.0 - alpha_output.unsqueeze(-1))

        return {
            "rgb": rgb_output,
            "depth": depth_output,
            "alpha": alpha_output,
        }

    def forward(
        self,
        models: list[BlockNeRFModel],
        ray_origins: Tensor,
        ray_directions: Tensor,
        block_centers: list[Tensor],
        block_radii: list[float],
        appearance_ids: Tensor,
        exposure_values: Tensor,
        **kwargs,
    ) -> TensorDict:
        """Forward pass through block rasterizer."""
        return self.march_rays(
            models,
            ray_origins,
            ray_directions,
            block_centers,
            block_radii,
            appearance_ids,
            exposure_values,
            **kwargs,
        )


def create_block_rasterizer(config: BlockRasterizerConfig | None = None) -> BlockRasterizer:
    """Create a block rasterizer with default configuration."""
    if config is None:
        config = BlockRasterizerConfig()
    return BlockRasterizer(config)
