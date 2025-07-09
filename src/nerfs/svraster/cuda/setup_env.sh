#!/bin/bash
# Environment setup script for SVRaster CUDA Extension
# Source this script before running your Python code

export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/home/xishansnow/anaconda3/envs/neurocity/lib/python3.10/site-packages/torch/lib"
echo "Environment variables set for SVRaster CUDA Extension"
echo "PyTorch library path: /home/xishansnow/anaconda3/envs/neurocity/lib/python3.10/site-packages/torch/lib"
