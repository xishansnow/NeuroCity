#!/bin/bash

# Block-NeRF CUDA Extension Environment Setup
# Source this script before using the extension

# Add PyTorch libraries to LD_LIBRARY_PATH
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/home/xishansnow/anaconda3/envs/neurocity/lib/python3.10/site-packages/torch/lib"

# Set CUDA environment
export TORCH_CUDA_ARCH_LIST="6.1"
export CUDA_VISIBLE_DEVICES=0

echo "✓ Block-NeRF CUDA environment configured"
echo "  LD_LIBRARY_PATH updated"
echo "  CUDA architecture set to 6.1 (GTX 1080 Ti)"
