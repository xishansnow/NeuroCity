#!/usr/bin/env python3
"""
Test script to verify voxel_rasterization function accessibility
"""

import torch
import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_voxel_rasterization_access():
    """Test if voxel_rasterization function can be accessed directly"""

    # Set library path
    torch_lib_path = (
        "/home/xishansnow/anaconda3/envs/neurocity/lib/python3.10/site-packages/torch/lib"
    )
    if torch_lib_path not in os.environ.get("LD_LIBRARY_PATH", ""):
        os.environ["LD_LIBRARY_PATH"] = os.environ.get("LD_LIBRARY_PATH", "") + ":" + torch_lib_path

    print("Testing voxel_rasterization function access...")

    try:
        # Test direct import of CUDA extension
        import voxel_rasterizer_cuda

        print("✓ Direct CUDA extension import successful")

        # Test if function exists
        if hasattr(voxel_rasterizer_cuda, "voxel_rasterization"):
            print("✓ voxel_rasterization function found in CUDA extension")
        else:
            print("✗ voxel_rasterization function not found in CUDA extension")
            return False

        # Test module import
        import nerfs.svraster.cuda.renderer.voxel_rasterizer_gpu as vrg

        print("✓ Module import successful")
        print(f"✓ CUDA_AVAILABLE: {vrg.CUDA_AVAILABLE}")

        # Test function access through getter
        voxel_rasterization_func = vrg.get_voxel_rasterization_function()
        if voxel_rasterization_func is not None:
            print("✓ voxel_rasterization function accessible through getter")
        else:
            print("✗ voxel_rasterization function not accessible through getter")
            return False

        # Test direct access to the function
        if hasattr(vrg, "voxel_rasterizer_cuda") and vrg.voxel_rasterizer_cuda is not None:
            if hasattr(vrg.voxel_rasterizer_cuda, "voxel_rasterization"):
                print("✓ voxel_rasterization function accessible directly from module")
            else:
                print("✗ voxel_rasterization function not accessible directly from module")
        else:
            print("✗ voxel_rasterizer_cuda not available in module")

        # Test actual function call
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"✓ CUDA device available: {torch.cuda.get_device_name(0)}")

            # Create test data
            num_voxels = 10
            voxel_positions = torch.rand(num_voxels, 3, device=device)
            voxel_sizes = torch.rand(num_voxels, device=device) * 0.1
            voxel_densities = torch.randn(num_voxels, device=device)
            voxel_colors = torch.rand(num_voxels, 3, device=device)

            camera_matrix = torch.eye(4, device=device)
            intrinsics = torch.tensor([[800, 0, 400], [0, 800, 300], [0, 0, 1]], device=device)
            viewport_size = torch.tensor([800, 600], dtype=torch.int32, device=device)

            print("✓ Test data created")

            # Test function call
            try:
                rgb, depth = voxel_rasterization_func(
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
                print("✓ voxel_rasterization function call successful")
                print(f"  RGB shape: {rgb.shape}")
                print(f"  Depth shape: {depth.shape}")
                return True
            except Exception as e:
                print(f"✗ voxel_rasterization function call failed: {e}")
                return False
        else:
            print("✗ CUDA not available")
            return False

    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_voxel_rasterization_access()
    if success:
        print("\n🎉 All tests passed! voxel_rasterization function is accessible.")
    else:
        print("\n❌ Some tests failed. Please check the errors above.")
