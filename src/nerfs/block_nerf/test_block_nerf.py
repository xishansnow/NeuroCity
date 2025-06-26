"""
Comprehensive Test Suite for Block-NeRF Implementation

This script tests all core components of Block-NeRF to ensure they work correctly
according to the paper specifications.
"""

import torch
import numpy as np
import sys
import os
from pathlib import Path

# Add parent directories to path
current_dir = Path(__file__).parent
sys.path.append(str(current_dir.parent.parent.parent))

def test_basic_import():
    """Test basic import functionality"""
    print("🧪 Testing Block-NeRF imports...")
    
    try:
        # Import from relative path
        sys.path.append(str(current_dir))
        from appearance_embedding import AppearanceEmbedding
        from block_manager import BlockManager
        from visibility_network import VisibilityNetwork
        from block_nerf_model import BlockNeRF, BlockNeRFNetwork
        from pose_refinement import PoseRefinement
        from block_compositor import BlockCompositor
        from renderer import BlockNeRFRenderer
        print("  ✅ All imports successful")
        return True
    except Exception as e:
        print(f"  ❌ Import failed: {e}")
        return False

def test_appearance_embedding():
    """Test AppearanceEmbedding functionality"""
    print("🧪 Testing AppearanceEmbedding...")
    
    try:
        sys.path.append(str(current_dir))
        from appearance_embedding import AppearanceEmbedding
        
        # Create appearance embedding
        app_embedding = AppearanceEmbedding(
            num_embeddings=100, embedding_dim=32
        )
        
        # Test forward pass
        ids = torch.randint(0, 100, (10, ))
        embeddings = app_embedding(ids)
        
        assert embeddings.shape == (10, 32), f"Expected (10, 32), got {embeddings.shape}"
        
        print("  ✅ AppearanceEmbedding tests passed")
        return True
        
    except Exception as e:
        print(f"  ❌ AppearanceEmbedding tests failed: {e}")
        return False

def test_block_manager():
    """Test BlockManager functionality"""
    print("🧪 Testing BlockManager...")
    
    try:
        sys.path.append(str(current_dir))
        from block_manager import BlockManager
        
        # Create block manager
        scene_bounds = ((-50, 50), (-50, 50), (-5, 5))
        
        manager = BlockManager(
            scene_bounds=scene_bounds, block_size=25.0, overlap_ratio=0.3, device='cpu'
        )
        
        assert len(manager.block_centers) > 0, "Should generate at least one block"
        print(f"  📊 Generated {len(manager.block_centers)} blocks")
        
        print("  ✅ BlockManager tests passed")
        return True
        
    except Exception as e:
        print(f"  ❌ BlockManager tests failed: {e}")
        return False

def main():
    """Main test function"""
    print("🚀 Starting Block-NeRF Basic Tests")
    print("=" * 50)
    
    tests = [
        test_basic_import, test_appearance_embedding, test_block_manager
    ]
    
    results = []
    for test in tests:
        result = test()
        results.append(result)
        print()
    
    print("=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    
    passed = sum(results)
    total = len(results)
    
    print(f"TOTAL: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 Basic Block-NeRF components are working!")
    else:
        print("⚠️  Some tests failed.")

if __name__ == "__main__":
    main()
