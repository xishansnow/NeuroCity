"""Block-NeRF version information."""

__version__ = "1.0.0"
__author__ = "NeuroCity Team"
__email__ = "contact@neurocity.com"
__description__ = "Block-NeRF: Scalable Large Scene Neural View Synthesis"
__url__ = "https://github.com/neurocity/block_nerf"

# Version history
__version_info__ = {
    "1.0.0": {
        "date": "2025-07-07",
        "features": [
            "Block decomposition for scalable scene reconstruction",
            "Appearance embeddings for environmental variations",
            "Learned pose refinement",
            "CUDA acceleration support",
            "Dual rendering architecture (volume + rasterization)",
            "Visibility prediction and block selection"
        ],
        "improvements": [
            "Optimized memory usage",
            "Enhanced rendering quality",
            "Better GPU utilization",
            "Streamlined API"
        ]
    }
}
