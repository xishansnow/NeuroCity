"""
NeRF Examples Runner

This script provides a unified interface to run all NeRF examples with
error handling and module availability checking.
"""

import argparse
import sys
import subprocess
from pathlib import Path
import importlib
import time
import os


def check_module_availability(module_name: str) -> bool:
    """Check if a NeRF module is available and properly implemented."""
    try:
        # Ensure the project root is in the path
        project_root = Path(__file__).parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        
        # Try to import the module
        module = importlib.import_module(f"src.nerfs.{module_name}")
        
        # Additional check: try to access key components
        if hasattr(module, '__all__') and module.__all__:
            # Module has explicit exports, check if they're accessible
            for item in module.__all__[:3]:  # Check first 3 items to avoid long checks
                if not hasattr(module, item):
                    return False
        
        return True
    except (ImportError, AttributeError, ModuleNotFoundError) as e:
        # For debugging purposes, you can uncomment this line:
        # print(f"Debug: Module {module_name} import failed: {e}")
        return False


def run_example_safely(example_script: str, args, method_name: str = None):
    """Run an example script with error handling."""
    cmd = [sys.executable, f"examples/{example_script}", "--example", args.example_type]
    
    if args.data_path:
        cmd.extend(["--data_path", args.data_path])
    if args.output_dir:
        cmd.extend(["--output_dir", args.output_dir])
    if hasattr(args, 'model_path') and args.model_path:
        cmd.extend(["--model_path", args.model_path])
    if hasattr(args, 'num_gpus') and args.num_gpus:
        cmd.extend(["--num_gpus", str(args.num_gpus)])
    
    method_display = method_name or example_script.replace(
        "_example.py",
        "",
    )
    print(f"\n{'='*60}")
    print(f"🚀 Running {method_display} Example")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
        )
        
        elapsed = time.time() - start_time
        
        if result.returncode == 0:
            print(f"✅ {method_display} example completed successfully! ({elapsed:.1f}s)")
            if result.stdout and args.verbose:
                print("Output:")
                print(result.stdout)
        else:
            print(f"❌ {method_display} example failed! ({elapsed:.1f}s)")
            if result.stderr:
                print("Error output:")
                print(result.stderr)
            if result.stdout and args.verbose:
                print("Standard output:")
                print(result.stdout)
        
        return result.returncode == 0
                
    except subprocess.TimeoutExpired:
        print(f"⏰ {method_display} example timed out after 5 minutes")
        return False
    except Exception as e:
        print(f"❌ Failed to run {method_display} example: {e}")
        return False


def get_available_methods():
    """Get list of all available NeRF methods by checking actual module availability."""
    project_root = Path(__file__).parent.parent
    nerfs_dir = project_root / "src" / "nerfs"
    
    available_methods = []
    
    # Check each subdirectory in src/nerfs
    if nerfs_dir.exists():
        for item in nerfs_dir.iterdir():
            if item.is_dir() and not item.name.startswith('__'):
                # Check if the module has an __init__.py file
                if (item / "__init__.py").exists():
                    available_methods.append(item.name)
    
    return sorted(available_methods)


def run_all_examples(args):
    """Run all available examples with summary reporting."""
    print(f"\n{'='*80}")
    print("🎯 RUNNING ALL NERF EXAMPLES")
    print(f"{'='*80}")
    
    # Get available methods dynamically
    available_methods = get_available_methods()
    
    # Define stability info (which methods are considered stable)
    stable_methods = {
        "classic_nerf", "grid_nerf", "instant_ngp", "nerfacto", "plenoxels", "mega_nerf", "svraster"
    }
    
    # Create method info list
    all_methods = []
    for method in available_methods:
        is_stable = method in stable_methods
        method_display = method.replace("_", "-").title()
        all_methods.append((method, method_display, is_stable))
    
    print(f"Found {len(all_methods)} NeRF implementations:")
    for method, display, stable in all_methods:
        status = "✅ Stable" if stable else "⚠️  Experimental"
        print(f"  • {display} ({method}) - {status}")
    
    results = {}
    successful = 0
    failed = 0
    skipped = 0
    
    start_time = time.time()
    
    # Run each example
    for method_key, method_name, is_stable in all_methods:
        # Check if we should skip based on stability
        if args.stable_only and not is_stable:
            print(f"⏭️  Skipping {method_name} (experimental implementation)")
            results[method_name] = "skipped"
            skipped += 1
            continue
        
        # Check if example file exists
        example_file = Path(__file__).parent / f"{method_key}_example.py"
        if not example_file.exists():
            print(f"⚠️  Skipping {method_name} (example file not found)")
            results[method_name] = "no_example"
            skipped += 1
            continue
            
        # Check module availability (but don't skip if ignore_missing is set)
        module_available = check_module_availability(method_key)
        if not module_available and not args.ignore_missing:
            print(f"⚠️  Skipping {method_name} (module not available)")
            results[method_name] = "unavailable"
            skipped += 1
            continue
        elif not module_available:
            print(f"⚠️  {method_name} module may not be fully implemented, but trying anyway...")
        
        # Run the example
        success = run_example_safely(f"{method_key}_example.py", args, method_name)
        
        if success:
            results[method_name] = "success"
            successful += 1
        else:
            results[method_name] = "failed"
            failed += 1
            
        # Add delay between examples to prevent resource conflicts
        if args.delay and (successful + failed) < len([m for m in all_methods if not args.stable_only or m[2]]):
            time.sleep(args.delay)
    
    # Print summary
    total_time = time.time() - start_time
    print(f"\n{'='*80}")
    print("📊 EXECUTION SUMMARY")
    print(f"{'='*80}")
    print(f"Total time: {total_time:.1f}s")
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {failed}")
    print(f"⏭️  Skipped: {skipped}")
    print(f"📋 Total: {successful + failed + skipped}")
    
    print(f"\n{'='*50}")
    print("DETAILED RESULTS:")
    print(f"{'='*50}")
    
    for method_name, status in results.items():
        if status == "success":
            print(f"✅ {method_name}")
        elif status == "failed":
            print(f"❌ {method_name}")
        elif status == "skipped":
            print(f"⏭️  {method_name} (skipped)")
        elif status == "unavailable":
            print(f"⚠️  {method_name} (unavailable)")
        elif status == "no_example":
            print(f"📄 {method_name} (no example file)")
    
    print(f"\n{'='*80}")
    
    # Return summary for potential scripting use
    return {
        "successful": successful, "failed": failed, "skipped": skipped, "results": results
    }


# Individual method runners (updated to remove redundant availability checks)
def run_classic_nerf_example(args):
    """Run Classic NeRF example."""
    return run_example_safely("classic_nerf_example.py", args, "Classic NeRF")


def run_grid_nerf_example(args):
    """Run Grid-NeRF example."""
    return run_example_safely("grid_nerf_example.py", args, "Grid-NeRF")


def run_instant_ngp_example(args):
    """Run Instant NGP example."""
    return run_example_safely("instant_ngp_example.py", args, "Instant-NGP")


def run_nerfacto_example(args) -> int:
    """Run Nerfacto example."""
    return run_example_safely("nerfacto_example.py", args, "Nerfacto")


def run_plenoxels_example(args):
    """Run Plenoxels example."""
    return run_example_safely("plenoxels_example.py", args, "Plenoxels")


def run_mega_nerf_example(args):
    """Run Mega-NeRF example."""
    return run_example_safely("mega_nerf_example.py", args, "Mega-NeRF")


def run_svraster_example(args):
    """Run SVRaster example."""
    return run_example_safely("svraster_example.py", args, "SVRaster")


def run_bungee_nerf_example(args):
    """Run Bungee-NeRF example."""
    return run_example_safely("bungee_nerf_example.py", args, "Bungee-NeRF")


def run_mip_nerf_example(args):
    """Run Mip-NeRF example."""
    return run_example_safely("mip_nerf_example.py", args, "Mip-NeRF")


def run_pyramid_nerf_example(args):
    """Run Pyramid-NeRF example."""
    return run_example_safely("pyramid_nerf_example.py", args, "Pyramid-NeRF")


def run_block_nerf_example(args):
    """Run Block-NeRF example."""
    return run_example_safely("block_nerf_example.py", args, "Block-NeRF")


def run_mega_nerf_plus_example(args):
    """Run Mega-NeRF+ example."""
    return run_example_safely("mega_nerf_plus_example.py", args, "Mega-NeRF+")


def run_cnc_nerf_example(args):
    """Run CNC-NeRF example."""
    return run_example_safely("cnc_nerf_example.py", args, "CNC-NeRF")


def run_dnmp_nerf_example(args):
    """Run DNMP-NeRF example."""
    return run_example_safely("dnmp_nerf_example.py", args, "DNMP-NeRF")


def run_inf_nerf_example(args):
    """Run Inf-NeRF example."""
    return run_example_safely("inf_nerf_example.py", args, "Inf-NeRF")


def run_sdf_net_example(args):
    """Run SDF-Net example."""
    return run_example_safely("sdf_net_example.py", args, "SDF-Net")


def run_occupancy_net_example(args):
    """Run Occupancy Networks example."""
    return run_example_safely("occupancy_net_example.py", args, "Occupancy Networks")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="NeRF Examples Runner")
    
    # Get available methods dynamically
    available_methods = get_available_methods()
    
    parser.add_argument(
        "method",
        choices=["all"] + available_methods,
        help="NeRF method to run",
    )
    
    parser.add_argument(
        "--example_type",
        type=str,
        default="basic",
        help="Type of example to run",
    )
    parser.add_argument("--data_path", type=str, help="Path to dataset")
    parser.add_argument("--output_dir", type=str, help="Output directory")
    parser.add_argument("--model_path", type=str, help="Path to trained model")
    parser.add_argument("--num_gpus", type=int, help="Number of GPUs for multi-GPU training")
    
    # Options for running all examples
    parser.add_argument(
        "--stable_only",
        action="store_true",
        help="When using 'all', only run stable/core NeRF variants",
    )
    parser.add_argument(
        "--ignore_missing",
        action="store_true",
        help="Try to run examples even if modules appear unavailable",
    )
    parser.add_argument(
        "--delay",
        type=int,
        default=2,
        help="Delay in seconds between examples",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed output from each example",
    )
    
    args = parser.parse_args()
    
    # Change to project root directory
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    
    # Handle "all" option
    if args.method == "all":
        return run_all_examples(args)
    
    # Method dispatch for individual methods - build dynamically
    method_runners = {}
    for method in available_methods:
        # Create function name
        func_name = f"run_{method}_example"
        if func_name in globals():
            method_runners[method] = globals()[func_name]
        else:
            # Create a generic runner for methods without specific functions
            method_runners[method] = lambda args, m=method: run_example_safely(f"{m}")
    
    if args.method in method_runners:
        success = method_runners[args.method](args)
        return 0 if success else 1
    else:
        print(f"Unknown method: {args.method}")
        print(f"Available methods: {available_methods + ['all']}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 