"""
NeRF Examples Runner

This script provides a unified interface to run all NeRF examples.
"""

import argparse
import sys
import subprocess
from pathlib import Path


def run_grid_nerf_example(args):
    """Run Grid-NeRF example."""
    cmd = [
        sys.executable, "examples/grid_nerf_example.py", "--example", args.example_type
    ]
    
    if args.data_path:
        cmd.extend(["--data_path", args.data_path])
    if args.output_dir:
        cmd.extend(["--output_dir", args.output_dir])
    if args.model_path:
        cmd.extend(["--model_path", args.model_path])
    if args.num_gpus:
        cmd.extend(["--num_gpus", str(args.num_gpus)])
    
    subprocess.run(cmd)


def run_classic_nerf_example(args):
    """Run Classic NeRF example."""
    cmd = [
        sys.executable, "examples/classic_nerf_example.py", "--example", args.example_type
    ]
    
    if args.data_path:
        cmd.extend(["--data_path", args.data_path])
    if args.output_dir:
        cmd.extend(["--output_dir", args.output_dir])
    
    subprocess.run(cmd)


def run_instant_ngp_example(args):
    """Run Instant NGP example."""
    cmd = [
        sys.executable, "examples/instant_ngp_example.py", "--example", args.example_type
    ]
    
    if args.data_path:
        cmd.extend(["--data_path", args.data_path])
    if args.output_dir:
        cmd.extend(["--output_dir", args.output_dir])
    
    subprocess.run(cmd)


def run_nerfacto_example(args):
    """Run Nerfacto example."""
    cmd = [
        sys.executable, "examples/nerfacto_example.py", "--example", args.example_type
    ]
    
    if args.data_path:
        cmd.extend(["--data_path", args.data_path])
    if args.output_dir:
        cmd.extend(["--output_dir", args.output_dir])
    
    subprocess.run(cmd)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="NeRF Examples Runner")
    parser.add_argument(
        "method",
        choices=["grid_nerf",
        "classic_nerf",
        "instant_ngp",
        "nerfacto"],
        help="NeRF method to run",
    )
    parser.add_argument("--example_type", type=str, default="basic", help="Type of example to run")
    parser.add_argument("--data_path", type=str, help="Path to dataset")
    parser.add_argument("--output_dir", type=str, help="Output directory")
    parser.add_argument("--model_path", type=str, help="Path to trained model")
    parser.add_argument("--num_gpus", type=int, help="Number of GPUs for multi-GPU training")
    
    args = parser.parse_args()
    
    # Change to project root directory
    project_root = Path(__file__).parent.parent
    import os
    os.chdir(project_root)
    
    if args.method == "grid_nerf":
        run_grid_nerf_example(args)
    elif args.method == "classic_nerf":
        run_classic_nerf_example(args)
    elif args.method == "instant_ngp":
        run_instant_ngp_example(args)
    elif args.method == "nerfacto":
        run_nerfacto_example(args)


if __name__ == "__main__":
    main() 