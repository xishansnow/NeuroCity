#!/usr/bin/env python3
"""
Mega-NeRF Rendering Script

This script renders novel views using trained Mega-NeRF models.
"""

import argparse
import os
import sys
import torch
import numpy as np
import json
import cv2
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import imageio

# Add the parent directory to the path to import our modules
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.mega_nerf import (
    MegaNeRF, MegaNeRFConfig, VolumetricRenderer, InteractiveRenderer,
    CameraDataset, create_sample_camera_path
)


class MegaNeRFRenderer:
    """
    Renderer for Mega-NeRF models
    """
    
    def __init__(self,
                 model: MegaNeRF,
                 config: MegaNeRFConfig,
                 device: str = 'cuda'):
        """
        Initialize renderer
        
        Args:
            model: Trained Mega-NeRF model
            config: Model configuration
            device: Device for computation
        """
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        # Initialize volumetric renderer
        self.volumetric_renderer = VolumetricRenderer(
            num_coarse_samples=config.num_coarse,
            num_fine_samples=config.num_fine,
            near=config.near,
            far=config.far,
            use_hierarchical_sampling=True
        )
        
        # Initialize interactive renderer
        self.interactive_renderer = InteractiveRenderer(
            model=model,
            base_renderer=self.volumetric_renderer,
            cache_size=100
        )
        
        # Set model to eval mode
        self.model.eval()
    
    def render_single_view(self,
                          camera_info,
                          appearance_id: Optional[int] = None,
                          chunk_size: int = 1024) -> Dict[str, np.ndarray]:
        """
        Render a single view
        
        Args:
            camera_info: Camera information
            appearance_id: Appearance embedding ID
            chunk_size: Chunk size for ray processing
            
        Returns:
            Dictionary with rendered outputs
        """
        return self.volumetric_renderer.render_image(
            self.model, camera_info, chunk_size, appearance_id
        )
    
    def render_path(self,
                   camera_path: List[np.ndarray],
                   intrinsics: np.ndarray,
                   height: int,
                   width: int,
                   output_dir: str,
                   appearance_id: Optional[int] = None,
                   save_depth: bool = True) -> List[str]:
        """
        Render a camera path
        
        Args:
            camera_path: List of camera poses
            intrinsics: Camera intrinsics
            height: Image height
            width: Image width
            output_dir: Output directory
            appearance_id: Appearance embedding ID
            save_depth: Whether to save depth maps
            
        Returns:
            List of output file paths
        """
        from src.mega_nerf.mega_nerf_dataset import CameraInfo
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        rgb_dir = output_dir / 'rgb'
        rgb_dir.mkdir(exist_ok=True)
        
        if save_depth:
            depth_dir = output_dir / 'depth'
            depth_dir.mkdir(exist_ok=True)
        
        output_files = []
        
        for i, pose in enumerate(tqdm(camera_path, desc="Rendering path")):
            # Create camera info
            camera_info = CameraInfo(
                transform_matrix=pose,
                intrinsics=intrinsics,
                image_path=f"render_{i:04d}.png",
                image_id=i,
                width=width,
                height=height
            )
            
            # Render image
            outputs = self.render_single_view(camera_info, appearance_id)
            
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
                    (depth_normalized * 255).astype(np.uint8),
                    cv2.COLORMAP_PLASMA
                )
                depth_path = depth_dir / f'{i:04d}.png'
                cv2.imwrite(str(depth_path), depth_colored)
        
        print(f"Rendered {len(camera_path)} images to {output_dir}")
        return output_files
    
    def create_video(self,
                    image_paths: List[str],
                    output_path: str,
                    fps: int = 30):
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
    parser = argparse.ArgumentParser(description='Render Mega-NeRF')
    
    # Model arguments
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model file')
    parser.add_argument('--config_path', type=str, default=None,
                       help='Path to model config (if different from model directory)')
    
    # Rendering arguments
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for rendered images')
    parser.add_argument('--render_type', type=str, default='spiral',
                       choices=['single', 'path', 'spiral', 'dataset'],
                       help='Type of rendering')
    
    # Camera arguments
    parser.add_argument('--height', type=int, default=600,
                       help='Image height')
    parser.add_argument('--width', type=int, default=800,
                       help='Image width')
    parser.add_argument('--focal', type=float, default=800,
                       help='Focal length')
    
    # Path rendering arguments
    parser.add_argument('--num_frames', type=int, default=120,
                       help='Number of frames for path rendering')
    parser.add_argument('--radius', type=float, default=50.0,
                       help='Radius for spiral path')
    parser.add_argument('--center', type=str, default='0,0,20',
                       help='Center point for spiral path as "x,y,z"')
    parser.add_argument('--height_variation', type=float, default=10.0,
                       help='Height variation for spiral path')
    
    # Appearance arguments
    parser.add_argument('--appearance_id', type=int, default=0,
                       help='Appearance embedding ID')
    
    # Output arguments
    parser.add_argument('--save_depth', action='store_true',
                       help='Save depth maps')
    parser.add_argument('--create_video', action='store_true',
                       help='Create video from rendered images')
    parser.add_argument('--fps', type=int, default=30,
                       help='Video frame rate')
    
    # System arguments
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device for rendering')
    parser.add_argument('--chunk_size', type=int, default=1024,
                       help='Chunk size for ray processing')
    
    # Dataset arguments (for dataset rendering)
    parser.add_argument('--data_root', type=str, default=None,
                       help='Dataset root directory (for dataset rendering)')
    parser.add_argument('--split', type=str, default='test',
                       help='Dataset split to render')
    
    return parser.parse_args()


def load_model_and_config(model_path: str, config_path: Optional[str] = None, device: str = 'cuda'):
    """Load trained model and configuration"""
    model_path = Path(model_path)
    
    # Load model
    if model_path.is_file():
        model_data = torch.load(model_path, map_location=device)
        config = model_data.get('config')
        
        if config is None:
            # Try to load config from same directory
            config_file = model_path.parent / 'config.json'
            if config_file.exists():
                with open(config_file, 'r') as f:
                    config_dict = json.load(f)
                config = MegaNeRFConfig(**config_dict)
            else:
                # Use default config
                config = MegaNeRFConfig()
        
        # Create model
        model = MegaNeRF(config)
        model.load_state_dict(model_data['model_state_dict'])
        
    elif model_path.is_dir():
        # Load from directory
        config_file = config_path if config_path else model_path / 'config.json'
        model_file = model_path / 'final_model.pth'
        
        # Load config
        if Path(config_file).exists():
            with open(config_file, 'r') as f:
                config_dict = json.load(f)
            config = MegaNeRFConfig(**config_dict)
        else:
            config = MegaNeRFConfig()
        
        # Load model
        if model_file.exists():
            model_data = torch.load(model_file, map_location=device)
            model = MegaNeRF(config)
            model.load_state_dict(model_data['model_state_dict'])
        else:
            raise FileNotFoundError(f"Model file not found: {model_file}")
    
    else:
        raise FileNotFoundError(f"Model path not found: {model_path}")
    
    return model, config


def generate_camera_path(render_type: str,
                        num_frames: int,
                        center: np.ndarray,
                        radius: float,
                        height_variation: float,
                        scene_bounds: Tuple[float, ...]) -> List[np.ndarray]:
    """Generate camera path based on render type"""
    
    if render_type == 'spiral':
        return create_sample_camera_path(
            center=center,
            radius=radius,
            num_frames=num_frames,
            height_variation=height_variation
        )
    
    elif render_type == 'path':
        # Create a more complex path
        poses = []
        
        # First segment: circular motion
        for i in range(num_frames // 2):
            angle = i * 2 * np.pi / (num_frames // 2)
            pos = center + np.array([
                radius * np.cos(angle),
                radius * np.sin(angle),
                height_variation * np.sin(angle * 2)
            ])
            
            # Look at center
            target = center
            up = np.array([0, 0, 1])
            
            forward = target - pos
            forward = forward / np.linalg.norm(forward)
            right = np.cross(forward, up)
            right = right / np.linalg.norm(right)
            up = np.cross(right, forward)
            
            pose = np.eye(4)
            pose[:3, 0] = right
            pose[:3, 1] = up
            pose[:3, 2] = -forward
            pose[:3, 3] = pos
            
            poses.append(pose)
        
        # Second segment: linear motion
        start_pos = poses[-1][:3, 3]
        end_pos = center + np.array([0, 0, radius])
        
        for i in range(num_frames // 2):
            t = i / (num_frames // 2)
            pos = start_pos * (1 - t) + end_pos * t
            
            target = center
            up = np.array([0, 0, 1])
            
            forward = target - pos
            forward = forward / np.linalg.norm(forward)
            right = np.cross(forward, up)
            right = right / np.linalg.norm(right)
            up = np.cross(right, forward)
            
            pose = np.eye(4)
            pose[:3, 0] = right
            pose[:3, 1] = up
            pose[:3, 2] = -forward
            pose[:3, 3] = pos
            
            poses.append(pose)
        
        return poses
    
    else:
        raise ValueError(f"Unknown render type: {render_type}")


def main():
    """Main rendering function"""
    args = parse_args()
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model and configuration
    print("Loading model and configuration...")
    model, config = load_model_and_config(args.model_path, args.config_path, device)
    
    print(f"Loaded model with {len(model.submodules)} submodules")
    print(f"Scene bounds: {config.scene_bounds}")
    
    # Create renderer
    print("Setting up renderer...")
    renderer = MegaNeRFRenderer(model, config, device)
    
    # Setup camera intrinsics
    intrinsics = np.array([
        [args.focal, 0, args.width / 2],
        [0, args.focal, args.height / 2],
        [0, 0, 1]
    ])
    
    # Generate camera poses based on render type
    if args.render_type in ['spiral', 'path']:
        # Parse center
        center = np.array([float(x) for x in args.center.split(',')])
        
        # Generate camera path
        camera_path = generate_camera_path(
            render_type=args.render_type,
            num_frames=args.num_frames,
            center=center,
            radius=args.radius,
            height_variation=args.height_variation,
            scene_bounds=config.scene_bounds
        )
        
    elif args.render_type == 'dataset':
        # Load dataset and use test poses
        if not args.data_root:
            raise ValueError("data_root required for dataset rendering")
        
        camera_dataset = CameraDataset(
            data_root=args.data_root,
            split=args.split,
            image_scale=1.0,
            load_images=False
        )
        
        camera_path = [camera.transform_matrix for camera in camera_dataset.cameras]
        if camera_dataset.cameras:
            intrinsics = camera_dataset.cameras[0].intrinsics
        
    elif args.render_type == 'single':
        # Single view rendering
        pose = np.eye(4)
        pose[:3, 3] = [0, 0, 20]  # Default camera position
        camera_path = [pose]
        
    else:
        raise ValueError(f"Unknown render type: {args.render_type}")
    
    # Render
    print(f"Rendering {len(camera_path)} frames...")
    output_files = renderer.render_path(
        camera_path=camera_path,
        intrinsics=intrinsics,
        height=args.height,
        width=args.width,
        output_dir=args.output_dir,
        appearance_id=args.appearance_id,
        save_depth=args.save_depth
    )
    
    # Create video if requested
    if args.create_video and len(output_files) > 1:
        video_path = Path(args.output_dir) / 'rendered_video.mp4'
        renderer.create_video(output_files, str(video_path), args.fps)
    
    # Save rendering info
    render_info = {
        'render_type': args.render_type,
        'num_frames': len(camera_path),
        'image_size': [args.height, args.width],
        'appearance_id': args.appearance_id,
        'model_path': args.model_path,
        'output_files': output_files
    }
    
    with open(Path(args.output_dir) / 'render_info.json', 'w') as f:
        json.dump(render_info, f, indent=2)
    
    print(f"Rendering completed! Output saved to {args.output_dir}")


if __name__ == '__main__':
    main() 