# Block-NeRF: Scalable Large Scene Neural View Synthesis

This package implements Block-NeRF, a method for scalable neural view synthesis of large-scale scenes by decomposing them into individually trained NeRF blocks.

## Overview

Block-NeRF addresses the challenges of applying NeRF to large-scale scenes through:
- **Block Decomposition**: Dividing scenes into manageable blocks
- **Appearance Embeddings**: Handling lighting and environmental variations
- **Pose Refinement**: Improving camera pose alignment
- **Visibility Prediction**: Efficient block selection for rendering
- **Block Compositing**: Seamless blending between blocks

## Features

### Core Components

- **BlockNeRF Model**: Main model with block-based decomposition
- **Block Manager**: Handles block organization and selection
- **Visibility Network**: Predicts block visibility for efficient rendering
- **Appearance Embedding**: Controls environmental variations
- **Pose Refinement**: Improves camera pose alignment
- **Block Compositor**: Blends between blocks smoothly

### Key Capabilities

- ✅ City-scale scene reconstruction
- ✅ Dynamic appearance handling
- ✅ Efficient block-based rendering
- ✅ Pose refinement and optimization
- ✅ Exposure and lighting control
- ✅ Seamless block transitions
- ✅ Multi-GPU training support

## Installation





