"""
from __future__ import annotations

Integration Example for Block-NeRF CUDA Extension

This example demonstrates how to integrate the CUDA-accelerated Block-NeRF
implementation into a complete rendering pipeline.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional
import time

# Try to import the CUDA extension
try:
    import block_nerf_cuda
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    print("Warning: block_nerf_cuda extension not available")


class BlockNeRFRenderer:
    """
    Complete Block-NeRF rendering pipeline using CUDA acceleration
    """
    
    def __init__(self, device: str = 'cuda'):
        """
        Initialize the Block-NeRF renderer
        
        Args:
            device: Device to use ('cuda' or 'cpu')
        """
        self.device = torch.device(device)
        
        if not torch.cuda.is_available() and device == 'cuda':
            print("Warning: CUDA not available, falling back to CPU")
            self.device = torch.device('cpu')
        
        # Scene parameters
        self.scene_bounds = torch.tensor([[-10, -10, -10], [10, 10, 10]], device=self.device)
        self.blocks = []
        self.camera_poses = []
        
        # Rendering parameters
        self.samples_per_ray = 64
        self.max_blocks_per_ray = 32
        self.visibility_threshold = 0.1
        
        print(f"Block-NeRF Renderer initialized on device: {self.device}")
    
    def create_scene_blocks(self, num_blocks: int = 64, block_size: float = 2.0):
        """
        Create a grid of blocks for the scene
        
        Args:
            num_blocks: Number of blocks per dimension
            block_size: Size of each block
        """
        # Create grid of block centers
        blocks_per_dim = int(np.cbrt(num_blocks))
        x = torch.linspace(-8, 8, blocks_per_dim, device=self.device)
        y = torch.linspace(-8, 8, blocks_per_dim, device=self.device)
        z = torch.linspace(-8, 8, blocks_per_dim, device=self.device)
        
        xx, yy, zz = torch.meshgrid(x, y, z, indexing='ij')
        
        # Flatten and create block centers
        self.block_centers = torch.stack([xx.flatten(), yy.flatten(), zz.flatten()], dim=1)
        self.block_radii = torch.ones(self.block_centers.shape[0], device=self.device) * (block_size / 2)
        self.block_active = torch.ones(self.block_centers.shape[0], dtype=torch.int32, device=self.device)
        
        # Create synthetic neural network weights for each block
        self.block_features = torch.randn(self.block_centers.shape[0], 32, device=self.device)
        
        print(f"Created {self.block_centers.shape[0]} blocks")
    
    def setup_cameras(self, num_cameras: int = 20, radius: float = 12.0):
        """
        Setup cameras in a circular pattern around the scene
        
        Args:
            num_cameras: Number of cameras to place
            radius: Radius of the camera circle
        """
        angles = torch.linspace(0, 2 * np.pi, num_cameras, device=self.device)
        
        # Create camera positions in a circle
        camera_x = radius * torch.cos(angles)
        camera_y = radius * torch.sin(angles)
        camera_z = torch.zeros_like(camera_x)
        
        self.camera_positions = torch.stack([camera_x, camera_y, camera_z], dim=1)
        
        # Create camera orientations (looking towards center)
        self.camera_targets = torch.zeros_like(self.camera_positions)
        
        print(f"Setup {num_cameras} cameras")
    
    def generate_rays(self, camera_idx: int, image_width: int = 64, image_height: int = 64) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate rays for a specific camera
        
        Args:
            camera_idx: Index of the camera
            image_width: Width of the image
            image_height: Height of the image
        
        Returns:
            Ray origins and directions
        """
        camera_pos = self.camera_positions[camera_idx]
        camera_target = self.camera_targets[camera_idx]
        
        # Create ray directions (simplified pinhole camera model)
        i, j = torch.meshgrid(
            torch.linspace(-1, 1, image_width, device=self.device),
            torch.linspace(-1, 1, image_height, device=self.device),
            indexing='ij'
        )
        
        # Camera forward direction
        forward = camera_target - camera_pos
        forward = forward / torch.norm(forward)
        
        # Camera right and up directions
        up = torch.tensor([0, 0, 1], device=self.device, dtype=torch.float32)
        right = torch.cross(forward, up)
        right = right / torch.norm(right)
        up = torch.cross(right, forward)
        
        # Create ray directions
        ray_dirs = (i.unsqueeze(-1) * right.unsqueeze(0).unsqueeze(0) + 
                   j.unsqueeze(-1) * up.unsqueeze(0).unsqueeze(0) + 
                   forward.unsqueeze(0).unsqueeze(0))
        
        ray_dirs = ray_dirs / torch.norm(ray_dirs, dim=-1, keepdim=True)
        
        # Create ray origins (all from camera position)
        ray_origins = camera_pos.unsqueeze(0).unsqueeze(0).expand(image_height, image_width, -1)
        
        # Flatten for processing
        ray_origins = ray_origins.reshape(-1, 3)
        ray_dirs = ray_dirs.reshape(-1, 3)
        
        return ray_origins, ray_dirs
    
    def render_image(self, camera_idx: int, image_width: int = 64, image_height: int = 64) -> torch.Tensor:
        """
        Render an image from a specific camera viewpoint
        
        Args:
            camera_idx: Index of the camera
            image_width: Width of the image
            image_height: Height of the image
        
        Returns:
            Rendered image tensor
        """
        if not CUDA_AVAILABLE:
            return self._render_image_cpu(camera_idx, image_width, image_height)
        
        # Generate rays
        ray_origins, ray_dirs = self.generate_rays(camera_idx, image_width, image_height)
        num_rays = ray_origins.shape[0]
        
        # Setup ray parameters
        ray_near = torch.ones(num_rays, device=self.device) * 0.1
        ray_far = torch.ones(num_rays, device=self.device) * 20.0
        
        # Select blocks for each ray
        selected_blocks, num_selected = block_nerf_cuda.block_selection(
            ray_origins,
            ray_dirs,
            ray_near,
            ray_far,
            self.block_centers,
            self.block_radii,
            self.block_active,
            self.max_blocks_per_ray
        )
        
        # Simulate volume rendering with synthetic colors
        colors = torch.zeros(num_rays, 3, device=self.device)
        
        for ray_idx in range(num_rays):
            num_blocks_for_ray = num_selected[ray_idx]
            
            if num_blocks_for_ray > 0:
                # Get selected blocks for this ray
                ray_blocks = selected_blocks[ray_idx, :num_blocks_for_ray]
                
                # Synthetic color based on block features
                block_colors = torch.sigmoid(self.block_features[ray_blocks, :3])
                
                # Simple alpha blending
                alpha = 0.1
                for i in range(num_blocks_for_ray):
                    colors[ray_idx] += (1 - alpha) * alpha * block_colors[i]
                    alpha *= 0.9
        
        # Reshape to image
        image = colors.reshape(image_height, image_width, 3)
        
        return image
    
    def _render_image_cpu(self, camera_idx: int, image_width: int = 64, image_height: int = 64) -> torch.Tensor:
        """
        CPU fallback rendering (simplified)
        """
        # Generate rays
        ray_origins, ray_dirs = self.generate_rays(camera_idx, image_width, image_height)
        
        # Simple synthetic rendering
        colors = torch.zeros(ray_origins.shape[0], 3, device=self.device)
        
        # Add some pattern based on ray direction
        colors[:, 0] = torch.abs(ray_dirs[:, 0])  # Red based on X direction
        colors[:, 1] = torch.abs(ray_dirs[:, 1])  # Green based on Y direction
        colors[:, 2] = torch.abs(ray_dirs[:, 2])  # Blue based on Z direction
        
        # Reshape to image
        image = colors.reshape(image_height, image_width, 3)
        
        return image
    
    def render_all_views(self, image_width: int = 64, image_height: int = 64) -> List[torch.Tensor]:
        """
        Render images from all camera viewpoints
        
        Args:
            image_width: Width of each image
            image_height: Height of each image
        
        Returns:
            List of rendered images
        """
        images = []
        
        print(f"Rendering {len(self.camera_positions)} views...")
        
        for camera_idx in range(len(self.camera_positions)):
            start_time = time.time()
            
            image = self.render_image(camera_idx, image_width, image_height)
            images.append(image)
            
            render_time = time.time() - start_time
            print(f"  Camera {camera_idx}: {render_time:.3f}s")
        
        return images
    
    def compute_visibility_map(self) -> torch.Tensor:
        """
        Compute visibility map for all blocks from all cameras
        
        Returns:
            Visibility scores for each block
        """
        if not CUDA_AVAILABLE:
            return torch.ones(self.block_centers.shape[0], device=self.device)
        
        visibility = block_nerf_cuda.block_visibility(
            self.camera_positions,
            self.block_centers,
            self.block_radii,
            self.block_active,
            self.visibility_threshold
        )
        
        return visibility
    
    def save_images(self, images: List[torch.Tensor], output_dir: str = "renders"):
        """
        Save rendered images to disk
        
        Args:
            images: List of images to save
            output_dir: Output directory
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        for i, image in enumerate(images):
            # Convert to numpy and clip values
            image_np = image.cpu().numpy()
            image_np = np.clip(image_np, 0, 1)
            
            # Save as PNG
            plt.figure(figsize=(6, 6))
            plt.imshow(image_np)
            plt.axis('off')
            plt.title(f"View {i}")
            plt.savefig(f"{output_dir}/view_{i:03d}.png", dpi=100, bbox_inches='tight')
            plt.close()
        
        print(f"Saved {len(images)} images to {output_dir}/")


def run_integration_example():
    """Run the complete integration example"""
    print("🚀 Block-NeRF CUDA Integration Example")
    print("=" * 60)
    
    # Create renderer
    renderer = BlockNeRFRenderer()
    
    # Setup scene
    renderer.create_scene_blocks(num_blocks=125, block_size=2.0)  # 5x5x5 grid
    renderer.setup_cameras(num_cameras=8, radius=15.0)
    
    # Compute visibility map
    print("\nComputing visibility map...")
    visibility = renderer.compute_visibility_map()
    print(f"Visibility range: [{visibility.min():.3f}, {visibility.max():.3f}]")
    
    # Render a few views
    print("\nRendering images...")
    images = renderer.render_all_views(image_width=128, image_height=128)
    
    # Save results
    renderer.save_images(images[:4])  # Save first 4 views
    
    print("\n🎉 Integration example completed successfully!")
    
    # Performance summary
    print("\n📊 Performance Summary:")
    print(f"  - Blocks: {renderer.block_centers.shape[0]}")
    print(f"  - Cameras: {len(renderer.camera_positions)}")
    print(f"  - Images rendered: {len(images)}")
    print(f"  - CUDA acceleration: {'✅' if CUDA_AVAILABLE else '❌'}")


if __name__ == "__main__":
    run_integration_example()
