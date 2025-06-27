from __future__ import annotations

from typing import Optional
#!/usr/bin/env python3
"""
Block-NeRF Rendering Script

This script renders novel views using trained Block-NeRF models.
"""

import argparse
import os
import sys
import torch
import numpy as np
import json
import cv2
from pathlib import Path
from tqdm import tqdm
import imageio

# Add the parent directory to the path to import our modules
sys.path.append(str(Path(__file__).parent.parent.parent))

from .block_manager import BlockManager
from .block_compositor import BlockCompositor

class BlockNeRFRenderer:
    """
    Renderer for Block-NeRF models
    """
    
    def __init__(
        self,
        block_manager: BlockManager,
        compositor: BlockCompositor,
        device: str = 'cuda'
    ):
        """
        Initialize renderer
        
        Args:
            block_manager: Block manager with loaded models
            compositor: Block compositor for combining renderings
            device: Device for computation
        """
        self.block_manager = block_manager
        self.compositor = compositor
        self.device = device
        
        # Set models to eval mode
        self._set_eval_mode()
    
    def _set_eval_mode(self):
        """Set all models to evaluation mode"""
        for block in self.block_manager.blocks.values():
            block.eval()
        self.block_manager.visibility_network.eval()
        self.compositor.training = False
    
    def render_image(
        self,
        camera_pose: torch.Tensor,
        intrinsics: torch.Tensor,
        height: int,
        width: int,
        appearance_id: int = 0,
        exposure_value: float = 1.0,
        chunk_size: int = 1024
    ):
        """
        Render a single image
        
        Args:
            camera_pose: Camera pose matrix (4, 4)
            intrinsics: Camera intrinsics (3, 3)
            height: Image height
            width: Image width
            appearance_id: Appearance embedding ID
            exposure_value: Exposure value
            chunk_size: Chunk size for ray processing
            
        Returns:
            Dictionary with rendered outputs
        """
        camera_position = camera_pose[:3, 3]
        
        # Get relevant blocks
        relevant_blocks = self.block_manager.get_blocks_for_camera(
            camera_position, torch.zeros(3, device=self.device), use_visibility=True
        )
        
        if not relevant_blocks:
            print("Warning: No relevant blocks found for camera position")
            # Return black image
            return {
                'rgb': np.zeros(
                )
            }
        
        print(f"Rendering with {len(relevant_blocks)} blocks: {relevant_blocks}")
        
        # Generate rays
        ray_origins, ray_directions = self._generate_rays(height, width, intrinsics, camera_pose)
        ray_origins = ray_origins.reshape(-1, 3)
        ray_directions = ray_directions.reshape(-1, 3)
        
        # Prepare inputs
        num_rays = ray_origins.shape[0]
        appearance_ids = torch.full(
        )
        exposure_values = torch.full(
        )
        
        # Render in chunks
        rgb_chunks = []
        depth_chunks = []
        opacity_chunks = []
        
        for i in tqdm(range(0, num_rays, chunk_size), desc="Rendering"):
            end_i = min(i + chunk_size, num_rays)
            
            chunk_origins = ray_origins[i:end_i]
            chunk_directions = ray_directions[i:end_i]
            chunk_appearance_ids = appearance_ids[i:end_i]
            chunk_exposure_values = exposure_values[i:end_i]
            
            with torch.no_grad():
                chunk_output = self._render_rays(
                    chunk_origins, chunk_directions, camera_position, chunk_appearance_ids, chunk_exposure_values, relevant_blocks
                )
            
            rgb_chunks.append(chunk_output['rgb'].cpu().numpy())
            depth_chunks.append(chunk_output['depth'].cpu().numpy())
            opacity_chunks.append(chunk_output['opacity'].cpu().numpy())
        
        # Combine chunks
        rgb = np.concatenate(rgb_chunks, axis=0).reshape(height, width, 3)
        depth = np.concatenate(depth_chunks, axis=0).reshape(height, width)
        opacity = np.concatenate(opacity_chunks, axis=0).reshape(height, width)
        
        return {
            'rgb': rgb, 'depth': depth, 'opacity': opacity
        }
    
    def _generate_rays(
        self,
        height: int,
        width: int,
        intrinsics: torch.Tensor,
        pose: torch.Tensor
    ):
        """Generate rays for a camera"""
        device = self.device
        
        i, j = torch.meshgrid(
            torch.arange(width, device=device), torch.arange(height, device=device), indexing='xy'
        )
        
        # Pixel coordinates to camera coordinates
        dirs = torch.stack([
            (
                i - intrinsics[0,
                2],
            )
        ], dim=-1).float()
        
        # Transform ray directions to world coordinates
        ray_directions = torch.sum(dirs[..., None, :] * pose[:3, :3], dim=-1)
        
        # Ray origins (camera center in world coordinates)
        ray_origins = pose[:3, 3].expand(ray_directions.shape)
        
        return ray_origins, ray_directions
    
    def _render_rays(
        self,
        ray_origins: torch.Tensor,
        ray_directions: torch.Tensor,
        camera_position: torch.Tensor,
        appearance_ids: torch.Tensor,
        exposure_values: torch.Tensor,
        block_names: list[str]
    ):
        """Render a batch of rays"""
        blocks = [self.block_manager.blocks[name] for name in block_names]
        block_centers = [self.block_manager.block_centers[name] for name in block_names]
        
        return self.compositor.render_with_blocks(
            blocks=blocks, block_names=block_names, block_centers=block_centers, camera_position=camera_position, ray_origins=ray_origins, ray_directions=ray_directions, appearance_ids=appearance_ids, exposure_values=exposure_values, near=0.1, far=100.0, num_samples=64
        )
    
    def render_path(
        self,
        poses: torch.Tensor,
        intrinsics: torch.Tensor,
        height: int,
        width: int,
        output_dir: str,
        appearance_id: int = 0,
        exposure_value: float = 1.0,
        save_depth: bool = True
    ):
        """
        Render a camera path
        
        Args:
            poses: Camera poses (N, 4, 4)
            intrinsics: Camera intrinsics (3, 3)
            height: Image height
            width: Image width
            output_dir: Output directory
            appearance_id: Appearance embedding ID
            exposure_value: Exposure value
            save_depth: Whether to save depth maps
            
        Returns:
            list of output file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        rgb_dir = output_dir / 'rgb'
        rgb_dir.mkdir(exist_ok=True)
        
        if save_depth:
            depth_dir = output_dir / 'depth'
            depth_dir.mkdir(exist_ok=True)
        
        output_files = []
        
        for i, pose in enumerate(tqdm(poses, desc="Rendering path")):
            # Render image
            outputs = self.render_image(
                camera_pose=pose, intrinsics=intrinsics, height=height, width=width, appearance_id=appearance_id, exposure_value=exposure_value
            )
            
            # Save RGB
            rgb = (outputs['rgb'] * 255).astype(np.uint8)
            rgb_path = rgb_dir / f'{i:04d}.png'
            imageio.imwrite(rgb_path, rgb)
            output_files.append(str(rgb_path))
            
            # Save depth
            if save_depth:
                depth = outputs['depth']
                depth_normalized = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
                depth_colored = cv2.applyColorMap(
                    (depth_normalized * 255).astype(np.uint8), cv2.COLORMAP_PLASMA
                )
                depth_path = depth_dir / f'{i:04d}.png'
                cv2.imwrite(str(depth_path), depth_colored)
        
        print(f"Rendered {len(poses)} images to {output_dir}")
        return output_files
    
    def create_video(self, image_paths: list[str], output_path: str, fps: int = 30):
        """Create video from rendered images"""
        if not image_paths:
            print("No images to create video")
            return
        
        # Read first image to get dimensions
        first_img = imageio.imread(image_paths[0])
        height, width = first_img.shape[:2]
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        for img_path in tqdm(image_paths, desc="Creating video"):
            img = cv2.imread(img_path)
            out.write(img)
        
        out.release()
        print(f"Video saved to {output_path}")

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Render Block-NeRF')
    
    # Model arguments
    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help='Path to trained model directory',
    )
    parser.add_argument(
        '--config_path',
        type=str,
        default=None,
        help='Path to model config'
    )
    
    # Rendering arguments
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Output directory for rendered images',
    )
    parser.add_argument(
        '--render_type',
        type=str,
        default='path',
        choices=['single',
        'path',
        'spiral',
        'dataset'],
        help='Type of rendering',
    )
    
    # Camera arguments
    parser.add_argument('--height', type=int, default=480, help='Image height')
    parser.add_argument('--width', type=int, default=640, help='Image width')
    parser.add_argument(
        '--focal',
        type=float,
        default=None,
        help='Focal length'
    )
    
    # Path rendering arguments
    parser.add_argument(
        '--num_frames',
        type=int,
        default=120,
        help='Number of frames for path rendering',
    )
    parser.add_argument('--radius', type=float, default=10.0, help='Radius for spiral path')
    parser.add_argument(
        '--height_offset',
        type=float,
        default=0.0,
        help='Height offset for spiral path',
    )
    
    # Appearance arguments
    parser.add_argument('--appearance_id', type=int, default=0, help='Appearance embedding ID')
    parser.add_argument('--exposure_value', type=float, default=1.0, help='Exposure value')
    
    # Output arguments
    parser.add_argument('--save_depth', action='store_true', help='Save depth maps')
    parser.add_argument(
        '--create_video',
        action='store_true',
        help='Create video from rendered images',
    )
    parser.add_argument('--fps', type=int, default=30, help='Video frame rate')
    
    # System arguments
    parser.add_argument('--device', type=str, default='cuda', help='Device for rendering')
    parser.add_argument(
        '--chunk_size',
        type=int,
        default=1024,
        help='Chunk size for ray processing',
    )
    
    # Dataset arguments (for dataset rendering)
    parser.add_argument(
        '--data_root',
        type=str,
        default=None,
        help='Dataset root directory'
    )
    parser.add_argument('--split', type=str, default='test', help='Dataset split to render')
    
    return parser.parse_args()

def generate_spiral_path(
    center: np.ndarray,
    radius: float,
    num_frames: int,
    height_offset: float = 0.0
):
    """Generate spiral camera path"""
    angles = np.linspace(0, 2 * np.pi, num_frames, endpoint=False)
    
    poses = []
    for angle in angles:
        # Camera position
        x = center[0] + radius * np.cos(angle)
        y = center[1] + radius * np.sin(angle)
        z = center[2] + height_offset
        
        # Look at center
        camera_pos = np.array([x, y, z])
        look_at = center
        up = np.array([0, 0, 1])
        
        # Compute camera rotation
        forward = look_at - camera_pos
        forward = forward / np.linalg.norm(forward)
        
        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)
        
        up = np.cross(right, forward)
        
        # Create pose matrix
        pose = np.eye(4)
        pose[:3, 0] = right
        pose[:3, 1] = up
        pose[:3, 2] = -forward
        pose[:3, 3] = camera_pos
        
        poses.append(pose)
    
    return np.array(poses)

def main():
    """Main rendering function"""
    args = parse_args()
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model configuration
    model_path = Path(args.model_path)
    config_path = Path(args.config_path) if args.config_path else model_path / 'args.json'
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Load block layout
    layout_path = model_path / 'block_layout.json'
    if not layout_path.exists():
        raise FileNotFoundError(f"Block layout not found: {layout_path}")
    
    # Create block manager
    print("Loading block manager...")
    block_manager = BlockManager(
        scene_bounds=[[-100, 100], [-100, 100], [-10, 10]], # Will be overridden
        block_size=config.get(
            'block_size',
            75.0,
        )
    )
    
    # Load block layout and models
    block_manager.load_block_layout(str(layout_path))
    
    network_config = {
        'hidden_dim': config.get(
            'hidden_dim',
            256,
        )
    }
    
    blocks_dir = model_path / 'final_blocks'
    if not blocks_dir.exists():
        blocks_dir = model_path / 'blocks'
    
    block_manager.load_blocks(str(blocks_dir), network_config)
    
    # Create compositor
    compositor = BlockCompositor(
        interpolation_method='inverse_distance', power=2.0, use_appearance_matching=False
    )
    
    # Create renderer
    print("Setting up renderer...")
    renderer = BlockNeRFRenderer(block_manager, compositor, device)
    
    # Setup camera intrinsics
    if args.focal is not None:
        intrinsics = torch.tensor([
            [args.focal, 0, args.width / 2], [0, args.focal, args.height / 2], [0, 0, 1]
        ], dtype=torch.float32, device=device)
    else:
        # Use default intrinsics
        focal = max(args.width, args.height)
        intrinsics = torch.tensor([
            [focal, 0, args.width / 2], [0, focal, args.height / 2], [0, 0, 1]
        ], dtype=torch.float32, device=device)
    
    # Generate camera poses based on render type
    if args.render_type == 'spiral':
        # Generate spiral path around scene center
        scene_stats = block_manager.get_scene_statistics()
        scene_bounds = scene_stats['scene_bounds']
        
        center = np.array([
            (
                scene_bounds[0][0] + scene_bounds[0][1],
            )
        ])
        
        poses = generate_spiral_path(center, args.radius, args.num_frames, args.height_offset)
        poses = torch.from_numpy(poses).float().to(device)
        
    elif args.render_type == 'dataset':
        # Load dataset and use test poses
        if not args.data_root:
            raise ValueError("data_root required for dataset rendering")
        
        dataset = BlockNeRFDataset(
            data_root=args.data_root, split=args.split, img_scale=1.0, use_cache=False
        )
        
        poses = torch.from_numpy(dataset.poses).float().to(device)
        if hasattr(dataset, 'intrinsics') and dataset.intrinsics is not None:
            intrinsics = torch.from_numpy(dataset.intrinsics[0]).float().to(device)
        
    else:
        raise NotImplementedError(f"Render type {args.render_type} not implemented")
    
    # Render
    print(f"Rendering {len(poses)} frames...")
    output_files = renderer.render_path(
        poses=poses, intrinsics=intrinsics, height=args.height, width=args.width, output_dir=args.output_dir, appearance_id=args.appearance_id, exposure_value=args.exposure_value, save_depth=args.save_depth
    )
    
    # Create video if requested
    if args.create_video:
        video_path = Path(args.output_dir) / 'rendered_video.mp4'
        renderer.create_video(output_files, str(video_path), args.fps)
    
    print(f"Rendering completed! Output saved to {args.output_dir}")

if __name__ == '__main__':
    main() 