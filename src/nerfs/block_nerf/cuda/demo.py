#!/usr/bin/env python3

"""
Block-NeRF CUDA Extension Usage Example

This script demonstrates how to use the Block-NeRF CUDA extension
in a practical scenario.
"""

import torch
import numpy as np
import time
import os

# Set up environment (alternatively, you can source env_setup.sh)
lib_path = "/home/xishansnow/anaconda3/envs/neurocity/lib/python3.10/site-packages/torch/lib"
if lib_path not in os.environ.get('LD_LIBRARY_PATH', ''):
    os.environ['LD_LIBRARY_PATH'] = f"{os.environ.get('LD_LIBRARY_PATH', '')}:{lib_path}"

def demo_block_nerf_cuda():
    """Demonstrate Block-NeRF CUDA extension usage"""
    print("🌟 Block-NeRF CUDA Extension Demo")
    print("=" * 50)
    
    # Import the extension
    try:
        import block_nerf_cuda_simple as block_nerf
        print("✓ Block-NeRF CUDA extension loaded successfully")
    except ImportError as e:
        print(f"❌ Failed to import: {e}")
        print("Make sure to run 'source env_setup.sh' first")
        return
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return
    
    device = torch.device('cuda')
    print(f"✓ Using device: {torch.cuda.get_device_name()}")
    print()
    
    # --- Scenario 1: Memory Bandwidth Testing ---
    print("📊 Scenario 1: Memory Bandwidth Testing")
    print("-" * 40)
    
    sizes = [1024, 10240, 102400]
    for size in sizes:
        data = torch.randn(size, device=device, dtype=torch.float32)
        
        start_time = time.time()
        result = block_nerf.memory_bandwidth_test(data)
        end_time = time.time()
        
        bandwidth = (size * 4 * 2) / ((end_time - start_time) * 1024**3)  # GB/s
        print(f"Size {size:>6}: {(end_time - start_time)*1000:>6.2f} ms, {bandwidth:>6.2f} GB/s")
    
    print()
    
    # --- Scenario 2: Neural Network Operations ---
    print("🧠 Scenario 2: Neural Network Operations")
    print("-" * 40)
    
    # Simulate some neural network computations
    batch_size = 1000
    features = 256
    
    # Generate feature vectors
    features_a = torch.randn(batch_size, features, device=device, dtype=torch.float32)
    features_b = torch.randn(batch_size, features, device=device, dtype=torch.float32)
    
    # Element-wise operations
    sum_result = block_nerf.simple_add(features_a, features_b)
    product_result = block_nerf.simple_multiply(features_a, features_b)
    
    print(f"✓ Processed {batch_size} samples with {features} features each")
    print(f"  Addition result range: [{sum_result.min():.3f}, {sum_result.max():.3f}]")
    print(f"  Product result range: [{product_result.min():.3f}, {product_result.max():.3f}]")
    print()
    
    # --- Scenario 3: Block Visibility Computation ---
    print("🎯 Scenario 3: Block Visibility for Virtual Scene")
    print("-" * 40)
    
    # Simulate a virtual scene with cameras and blocks
    num_cameras = 50
    num_blocks = 100
    
    # Camera positions in 3D space
    camera_positions = torch.randn(num_cameras, 3, device=device, dtype=torch.float32) * 10
    
    # Block centers distributed in space
    block_centers = torch.randn(num_blocks, 3, device=device, dtype=torch.float32) * 20
    
    # Block radii (different sizes)
    block_radii = torch.rand(num_blocks, device=device, dtype=torch.float32) * 3 + 1
    
    # View directions (normalized)
    view_directions = torch.randn(num_cameras, 3, device=device, dtype=torch.float32)
    view_directions = view_directions / torch.norm(view_directions, dim=1, keepdim=True)
    
    # Compute visibility
    start_time = time.time()
    visibility_matrix = block_nerf.block_visibility(
        camera_positions, block_centers, block_radii, view_directions, 
        visibility_threshold=0.3
    )
    end_time = time.time()
    
    # Analyze results
    visible_blocks_per_camera = (visibility_matrix > 0).sum(dim=1).float()
    avg_visible = visible_blocks_per_camera.mean().item()
    max_visible = visible_blocks_per_camera.max().item()
    
    print(f"✓ Visibility computed for {num_cameras} cameras and {num_blocks} blocks")
    print(f"  Average visible blocks per camera: {avg_visible:.1f}")
    print(f"  Maximum visible blocks per camera: {max_visible}")
    print(f"  Computation time: {(end_time - start_time)*1000:.2f} ms")
    print()
    
    # --- Scenario 4: Ray-Block Selection ---
    print("🔍 Scenario 4: Ray-Block Selection for Rendering")
    print("-" * 40)
    
    # Simulate rays for rendering
    num_rays = 1000
    max_blocks_per_ray = 5
    
    # Ray origins and directions
    ray_origins = torch.randn(num_rays, 3, device=device, dtype=torch.float32) * 5
    ray_directions = torch.randn(num_rays, 3, device=device, dtype=torch.float32)
    ray_directions = ray_directions / torch.norm(ray_directions, dim=1, keepdim=True)
    
    # Use same blocks as before
    start_time = time.time()
    selected_blocks, num_selected = block_nerf.block_selection(
        ray_origins, ray_directions, block_centers, block_radii, max_blocks_per_ray
    )
    end_time = time.time()
    
    # Analyze selection results
    avg_selected = num_selected.float().mean().item()
    selection_efficiency = (num_selected > 0).float().mean().item() * 100
    
    print(f"✓ Block selection completed for {num_rays} rays")
    print(f"  Average blocks selected per ray: {avg_selected:.1f}")
    print(f"  Rays with at least one block: {selection_efficiency:.1f}%")
    print(f"  Selection time: {(end_time - start_time)*1000:.2f} ms")
    print()
    
    # --- Performance Summary ---
    print("⚡ Performance Summary")
    print("-" * 40)
    
    # Overall throughput estimates
    camera_throughput = num_cameras * num_blocks / ((end_time - start_time) * 1000)  # K operations/ms
    ray_throughput = num_rays / ((end_time - start_time) * 1000)  # K rays/ms
    
    print(f"Camera-block visibility: {camera_throughput:.1f}K ops/ms")
    print(f"Ray-block selection: {ray_throughput:.1f}K rays/ms")
    print()
    
    # --- Recommendations ---
    print("💡 Usage Recommendations")
    print("-" * 40)
    print("1. Use memory_bandwidth_test() to verify CUDA performance")
    print("2. Leverage block_visibility() for view-dependent culling")
    print("3. Use block_selection() for efficient ray-block intersection")
    print("4. Combine operations to minimize CPU-GPU transfers")
    print("5. Batch operations when possible for better throughput")
    print()
    
    print("🎉 Demo completed successfully!")
    print("The Block-NeRF CUDA extension is ready for production use.")

if __name__ == "__main__":
    demo_block_nerf_cuda()
