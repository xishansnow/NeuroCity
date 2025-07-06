"""Generate test data for Plenoxels tests."""

from src.nerfs.plenoxels.test_utils import create_test_images


def main():
    """Generate test data."""
    # Create test images for train, val, and test splits
    create_test_images("test_demo_scene", num_images=3)  # Train split
    create_test_images("test_demo_scene", num_images=2, start_idx=3)  # Val split
    create_test_images("test_demo_scene", num_images=2, start_idx=5)  # Test split


if __name__ == "__main__":
    main()
