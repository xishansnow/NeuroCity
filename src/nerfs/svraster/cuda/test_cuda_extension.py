#!/usr/bin/env python3
"""
Test script for CUDA extension functionality
"""

import torch
import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_cuda_extension():
    """Test the CUDA extension functionality"""

    # Set library path
    torch_lib_path = (
        "/home/xishansnow/anaconda3/envs/neurocity/lib/python3.10/site-packages/torch/lib"
    )
    if torch_lib_path not in os.environ.get("LD_LIBRARY_PATH", ""):
        os.environ["LD_LIBRARY_PATH"] = os.environ.get("LD_LIBRARY_PATH", "") + ":" + torch_lib_path

    print("Testing CUDA extension...")

    try:
        # Test direct import
        import nerfs.svraster.cuda.renderer.voxel_rasterizer_gpu as voxel_rasterizer_cuda

        print("✓ Direct import successful")

        # Test function availability
        functions = [
            "voxel_rasterization",
            "create_camera_matrix",
            "rays_to_camera_matrix",
            "benchmark",
        ]
        for func in functions:
            if hasattr(voxel_rasterizer_cuda, func):
                print(f"✓ Function {func} available")
            else:
                print(f"✗ Function {func} not available")

        # Test module import
        import nerfs.svraster.cuda.renderer.voxel_rasterizer_gpu as voxel_rasterizer_gpu

        print("✓ Module import successful")
        print(f"✓ CUDA_AVAILABLE: {voxel_rasterizer_gpu.CUDA_AVAILABLE}")

        # Test basic functionality
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"✓ CUDA device available: {torch.cuda.get_device_name(0)}")

            # Create test data
            num_voxels = 100
            voxel_positions = torch.rand(num_voxels, 3, device=device)
            voxel_sizes = torch.rand(num_voxels, device=device) * 0.1
            voxel_densities = torch.randn(num_voxels, device=device)
            voxel_colors = torch.rand(num_voxels, 3, device=device)

            camera_matrix = torch.eye(4, device=device)
            intrinsics = torch.tensor([[800, 0, 400], [0, 800, 300], [0, 0, 1]], device=device)
            viewport_size = torch.tensor([800, 600], dtype=torch.int32, device=device)

            print("✓ Test data created")

            # Test voxel_rasterization function
            try:
                rgb, depth = voxel_rasterizer_cuda.voxel_rasterization(
                    voxel_positions,
                    voxel_sizes,
                    voxel_densities,
                    voxel_colors,
                    camera_matrix,
                    intrinsics,
                    viewport_size,
                    0.1,  # near_plane
                    100.0,  # far_plane
                    torch.tensor([0.0, 0.0, 0.0], device=device),  # background_color
                    "exp",  # density_activation
                    "sigmoid",  # color_activation
                    2,  # sh_degree
                )
                print("✓ voxel_rasterization function executed successfully")
                print(f"  RGB shape: {rgb.shape}")
                print(f"  Depth shape: {depth.shape}")
            except Exception as e:
                print(f"✗ voxel_rasterization failed: {e}")

            # Test create_camera_matrix function
            try:
                pose = torch.eye(4, device=device)
                camera_matrix_result = voxel_rasterizer_cuda.create_camera_matrix(pose)
                print("✓ create_camera_matrix function executed successfully")
                print(f"  Camera matrix shape: {camera_matrix_result.shape}")
            except Exception as e:
                print(f"✗ create_camera_matrix failed: {e}")

            # Test rays_to_camera_matrix function
            try:
                ray_origins = torch.rand(100, 3, device=device)
                ray_directions = torch.randn(100, 3, device=device)
                ray_directions = ray_directions / torch.norm(ray_directions, dim=1, keepdim=True)

                camera_matrix_result, intrinsics_result = (
                    voxel_rasterizer_cuda.rays_to_camera_matrix(ray_origins, ray_directions)
                )
                print("✓ rays_to_camera_matrix function executed successfully")
                print(f"  Camera matrix shape: {camera_matrix_result.shape}")
                print(f"  Intrinsics shape: {intrinsics_result.shape}")
            except Exception as e:
                print(f"✗ rays_to_camera_matrix failed: {e}")

        else:
            print("✗ CUDA not available")

    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_cuda_extension()
