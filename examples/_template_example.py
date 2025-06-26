"""
Generic Example Template

This template can be used for NeRF modules that may not be fully implemented yet.
It provides a flexible structure that can handle missing components gracefully.
"""

import os
import sys
import torch
import argparse
from pathlib import Path

# Add the project root to Python path to enable imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def safe_import_module(module_name: str):
    """Safely import a module and return its classes if available."""
    try:
        module = __import__(f"src.nerfs.{module_name}", fromlist=[''])
        
        # Try to find common class patterns
        config_class = None
        model_class = None
        trainer_class = None
        
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if hasattr(attr, '__name__'):
                if 'Config' in attr_name:
                    config_class = attr
                elif 'Model' in attr_name or attr_name.endswith('NeRF'):
                    model_class = attr
                elif 'Trainer' in attr_name:
                    trainer_class = attr
        
        return {
            'module': module, 'config': config_class, 'model': model_class, 'trainer': trainer_class, 'available': True
        }
        
    except ImportError as e:
        return {
            'module': None, 'config': None, 'model': None, 'trainer': None, 'available': False, 'error': str(
                e,
            )
        }


def create_placeholder_example(module_info: dict, module_name: str):
    """Create a placeholder example when the module is not available."""
    
    def basic_example():
        print(f"=== Basic {module_name.title()} Example ===")
        
        if not module_info['available']:
            print(f"❌ {module_name} module not available")
            print(f"Error: {module_info.get('error', 'Unknown error')}")
            print(f"This is a placeholder example for {module_name}")
            print("Please implement the module or check the imports.")
            return None, None
        
        try:
            # Try to create a basic configuration
            if module_info['config']:
                config = module_info['config']()
                print(f"✅ Created {module_name} configuration")
            else:
                print(f"⚠️  No configuration class found for {module_name}")
                config = {}
            
            # Try to create a model
            if module_info['model']:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                model = module_info['model'](config).to(device)
                print(f"✅ Created {module_name} model on {device}")
                print(f"Model parameters: {sum(p.numel() for p in model.parameters()):, }")
            else:
                print(f"⚠️  No model class found for {module_name}")
                model = None
            
            return model, config
            
        except Exception as e:
            print(f"❌ Error creating {module_name} example: {e}")
            return None, None
    
    return basic_example


def main():
    """Main function template."""
    parser = argparse.ArgumentParser(description="Generic NeRF Example Template")
    parser.add_argument(
        "--module",
        type=str,
        required=True,
        help="NeRF module name,
    )
    parser.add_argument("--example", type=str, default="basic", help="Example to run")
    parser.add_argument(
        "--data_path",
        type=str,
        default="./data/nerf_synthetic/lego",
        help="Path to dataset",
    )
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Output directory")
    
    args = parser.parse_args()
    
    # Import the specified module
    module_info = safe_import_module(args.module)
    
    # Create and run the example
    example_func = create_placeholder_example(module_info, args.module)
    model, config = example_func()
    
    if model is not None:
        print(f"✅ {args.module} example completed successfully!")
    else:
        print(f"❌ {args.module} example failed")


if __name__ == "__main__":
    main() 