"""
Block-NeRF Quick Start Example

This example demonstrates the basic usage of Block-NeRF for
large-scale neural view synthesis.
"""

import torch
import numpy as np
from block_nerf import (
    BlockNeRFConfig,
    BlockNeRFModel,
    BlockNeRFTrainer,
    BlockNeRFRenderer,
    BlockNeRFDataset,
    BlockManager,
)

def main():
    """Main example function."""
    print("🌟 Block-NeRF Quick Start Example")
    print("=" * 50)
    
    # Check CUDA availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Create configuration
    print("\n📋 Creating Configuration")
    config = BlockNeRFConfig(
        block_size=64,           # Size of each block in world units
        max_blocks=100,          # Maximum number of blocks
        appearance_embedding_dim=32,  # Appearance variation modeling
        pose_refinement_steps=1000,   # Pose optimization steps
    )
    print(f"✓ Config created with {config.max_blocks} blocks")
    
    # 2. Initialize Block Manager
    print("\n🔧 Setting up Block Manager")
    block_manager = BlockManager(config)
    
    # Define scene bounds (example: city block)
    scene_bounds = torch.tensor([
        [-100, -100, 0],     # min_xyz
        [100, 100, 50]       # max_xyz
    ], dtype=torch.float32, device=device)
    
    block_manager.initialize_blocks(scene_bounds)
    print(f"✓ Initialized {len(block_manager.blocks)} blocks")
    
    # 3. Create model
    print("\n🧠 Creating Block-NeRF Model")
    model = BlockNeRFModel(config).to(device)
    print(f"✓ Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # 4. Setup training (example)
    print("\n🏋️ Training Setup Example")
    trainer = BlockNeRFTrainer(model, config)
    
    # Example training data (replace with real data)
    sample_rays = torch.randn(1000, 6, device=device)  # [origin(3) + direction(3)]
    sample_images = torch.randn(1000, 3, device=device)  # RGB values
    
    print("✓ Trainer configured")
    
    # 5. Inference setup
    print("\n🎨 Inference Setup Example")
    renderer = BlockNeRFRenderer(model, config)
    
    # Example inference
    with torch.no_grad():
        # Sample camera pose for rendering
        camera_origin = torch.tensor([0, 0, 10], dtype=torch.float32, device=device)
        view_direction = torch.tensor([0, 0, -1], dtype=torch.float32, device=device)
        
        # Get visible blocks for this viewpoint
        visible_blocks = block_manager.get_visible_blocks(camera_origin, view_direction)
        print(f"✓ Found {len(visible_blocks)} visible blocks for rendering")
    
    # 6. Example workflow
    print("\n🔄 Example Workflow")
    print("1. Load your dataset with BlockNeRFDataset")
    print("2. Train individual blocks with BlockNeRFTrainer")
    print("3. Use pose refinement for better alignment")
    print("4. Render large scenes with BlockNeRFRenderer")
    print("5. Leverage CUDA acceleration for performance")
    
    print("\n🎉 Quick start example completed!")
    print("For more details, see the documentation and other examples.")

if __name__ == "__main__":
    main()
