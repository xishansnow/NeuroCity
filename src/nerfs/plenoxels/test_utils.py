"""Test utilities for Plenoxels."""

import numpy as np
import imageio.v3 as imageio
import os


def create_test_image(filename: str, size: int = 64) -> None:
    """Create a test image with a simple pattern.

    Args:
        filename: Path to save the image
        size: Size of the image (both width and height)
    """
    # Create a simple pattern (checkerboard)
    x = np.linspace(0, 1, size)
    y = np.linspace(0, 1, size)
    xx, yy = np.meshgrid(x, y)
    pattern = ((xx + yy) % 0.2 < 0.1).astype(np.float32)

    # Convert to RGB
    image = np.stack([pattern, pattern * 0.5, pattern * 0.25], axis=-1)

    # Ensure directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # Save image
    imageio.imwrite(filename, (image * 255).astype(np.uint8))


def create_test_images(base_dir: str, num_images: int = 7, start_idx: int = 0) -> None:
    """Create test images for the test scene.

    Args:
        base_dir: Base directory to save images
        num_images: Number of images to create
        start_idx: Starting index for image numbering
    """
    # Create images directory
    images_dir = os.path.join(base_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    # Create test images with consistent size
    for i in range(start_idx, start_idx + num_images):
        filename = os.path.join(images_dir, f"image_{i:03d}.png")
        create_test_image(filename)
