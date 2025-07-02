#!/usr/bin/env python3
"""
SVRaster Ablation Study Runner

This script runs a simplified version of the SVRaster ablation study
to demonstrate the contribution of different components.

Usage:
    python demos/run_svraster_ablation.py
"""

import sys
import os
import torch
import numpy as np
from pathlib import Path
import json
import time

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from nerfs.svraster.core import SVRasterConfig, SVRasterModel, SVRasterLoss


class QuickAblationStudy:
    """Quick ablation study for SVRaster demonstration."""
    
    def __init__(self, output_dir: str = "quick_ablation_results"):
        """Initialize quick ablation study."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"Using device: {self.device}")
        
        # Simplified test configuration
        self.base_config = SVRasterConfig(
            max_octree_levels=5,  # Reduced for faster training
            base_resolution=8,     # Reduced for faster training
            scene_bounds=(-1.0, -1.0, -1.0, 1.0, 1.0, 1.0),
            ray_samples_per_voxel=4,  # Reduced for faster training
            morton_ordering=True,
            sh_degree=2
        )
        
        # Create simple synthetic data
        self.test_data = self._create_simple_test_data()
    
    def _create_simple_test_data(self):
        """Create simple synthetic test data."""
        # Create a simple scene with a colored sphere
        num_rays = 512  # Reduced for faster training
        image_size = int(np.sqrt(num_rays))
        
        # Create rays pointing towards origin
        u = torch.linspace(-0.5, 0.5, image_size)
        v = torch.linspace(-0.5, 0.5, image_size)
        u, v = torch.meshgrid(u, v, indexing='ij')
        
        # Camera position
        camera_pos = torch.tensor([0.0, 0.0, 2.0])
        
        # Create rays
        ray_origins = camera_pos.unsqueeze(0).repeat(num_rays, 1)
        ray_directions = torch.stack([
            u.flatten(),
            v.flatten(),
            -torch.ones(num_rays)
        ], dim=-1)
        ray_directions = ray_directions / torch.norm(ray_directions, dim=-1, keepdim=True)
        
        # Create target colors (simple pattern)
        target_colors = torch.stack([
            0.5 + 0.5 * torch.sin(2 * np.pi * u.flatten()),
            0.5 + 0.5 * torch.cos(2 * np.pi * v.flatten()),
            0.5 + 0.25 * torch.sin(4 * np.pi * (u + v).flatten())
        ], dim=-1)
        
        return {
            'rays_o': ray_origins,
            'rays_d': ray_directions,
            'colors': target_colors
        }
    
    def _compute_metrics(self, pred_rgb: torch.Tensor, target_rgb: torch.Tensor) -> dict:
        """Compute evaluation metrics."""
        with torch.no_grad():
            # PSNR
            mse = torch.mean((pred_rgb - target_rgb) ** 2)
            psnr = -10.0 * torch.log10(mse + 1e-8)
            
            # SSIM (simplified)
            pred_gray = 0.299 * pred_rgb[:, 0] + 0.587 * pred_rgb[:, 1] + 0.114 * pred_rgb[:, 2]
            target_gray = 0.299 * target_rgb[:, 0] + 0.587 * target_rgb[:, 1] + 0.114 * target_rgb[:, 2]
            
            mu1, mu2 = torch.mean(pred_gray), torch.mean(target_gray)
            sigma1_sq, sigma2_sq = torch.var(pred_gray), torch.var(target_gray)
            sigma12 = torch.mean((pred_gray - mu1) * (target_gray - mu2))
            
            c1, c2 = 0.01 ** 2, 0.03 ** 2
            ssim = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / \
                   ((mu1**2 + mu2**2 + c1) * (sigma1_sq + sigma2_sq + c2))
            
            return {
                'psnr': psnr.item(),
                'ssim': ssim.item(),
                'mse': mse.item()
            }
    
    def _train_model(self, config: SVRasterConfig, experiment_name: str, 
                    num_epochs: int = 3) -> dict:
        """Train model with given config."""
        print(f"Running experiment: {experiment_name}")
        
        # Create model and loss function
        model = SVRasterModel(config).to(self.device)
        loss_fn = SVRasterLoss(config)
        
        # Create optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        # Training loop
        train_losses = []
        start_time = time.time()
        
        for epoch in range(num_epochs):
            model.train()
            
            # Get data
            rays_o = self.test_data['rays_o'].to(self.device)
            rays_d = self.test_data['rays_d'].to(self.device)
            target_colors = self.test_data['colors'].to(self.device)
            
            # Forward pass
            outputs = model(rays_o, rays_d)
            losses = loss_fn(outputs, {'rgb': target_colors}, model)
            loss = losses['total_loss']
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
            
            # Print progress
            if epoch % 1 == 0:
                print(f"  Epoch {epoch}: Loss={loss.item():.4f}")
        
        training_time = time.time() - start_time
        
        # Final evaluation
        model.eval()
        with torch.no_grad():
            outputs = model(rays_o, rays_d)
            metrics = self._compute_metrics(outputs['rgb'], target_colors)
        
        # Get voxel statistics
        try:
            voxel_stats = model.get_voxel_statistics()
        except:
            voxel_stats = {'total_voxels': 0}
        
        return {
            'experiment_name': experiment_name,
            'config': {k: v for k, v in config.__dict__.items() if not k.startswith('_')},
            'train_losses': train_losses,
            'final_metrics': metrics,
            'voxel_stats': voxel_stats,
            'training_time': training_time
        }
    
    def run_loss_ablation(self) -> dict:
        """Quick loss function ablation study."""
        print("\n" + "="*50)
        print("LOSS FUNCTION ABLATION STUDY")
        print("="*50)
        
        experiments = [
            ("RGB_MSE_only", {
                'use_ssim_loss': False,
                'use_distortion_loss': False,
                'use_pointwise_rgb_loss': False,
                'use_opacity_regularization': False
            }),
            ("RGB_MSE_SSIM", {
                'use_ssim_loss': True,
                'use_distortion_loss': False,
                'use_pointwise_rgb_loss': False,
                'use_opacity_regularization': False
            }),
            ("All_Losses", {
                'use_ssim_loss': True,
                'use_distortion_loss': True,
                'use_pointwise_rgb_loss': True,
                'use_opacity_regularization': True
            })
        ]
        
        results = {}
        for exp_name, config_updates in experiments:
            config = SVRasterConfig(**{**self.base_config.__dict__, **config_updates})
            results[exp_name] = self._train_model(config, exp_name)
        
        return results
    
    def run_component_ablation(self) -> dict:
        """Quick component ablation study."""
        print("\n" + "="*50)
        print("COMPONENT ABLATION STUDY")
        print("="*50)
        
        experiments = [
            ("No_Morton_Ordering", {'morton_ordering': False}),
            ("With_Morton_Ordering", {'morton_ordering': True}),
            ("SH_Degree_0", {'sh_degree': 0}),
            ("SH_Degree_2", {'sh_degree': 2}),
            ("Low_Sampling", {'ray_samples_per_voxel': 2}),
            ("High_Sampling", {'ray_samples_per_voxel': 8})
        ]
        
        results = {}
        for exp_name, config_updates in experiments:
            config = SVRasterConfig(**{**self.base_config.__dict__, **config_updates})
            results[exp_name] = self._train_model(config, exp_name)
        
        return results
    
    def run_comprehensive_ablation(self) -> dict:
        """Run comprehensive ablation study."""
        print("Starting SVRaster Quick Ablation Study...")
        print("This study tests the contribution of different components:")
        print("1. Loss functions (RGB, SSIM, Distortion, Pointwise RGB, Opacity)")
        print("2. Morton ordering vs no ordering")
        print("3. Different SH degrees (0, 2)")
        print("4. Different sampling strategies")
        print("="*50)
        
        all_results = {
            'loss_ablation': self.run_loss_ablation(),
            'component_ablation': self.run_component_ablation()
        }
        
        # Save results
        self._save_results(all_results)
        self._print_summary(all_results)
        
        return all_results
    
    def _save_results(self, results: dict):
        """Save ablation results to JSON file."""
        # Convert tensors to lists for JSON serialization
        def convert_for_json(obj):
            if isinstance(obj, torch.Tensor):
                return obj.cpu().numpy().tolist()
            elif isinstance(obj, dict):
                return {k: convert_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_for_json(item) for item in obj]
            else:
                return obj
        
        json_results = convert_for_json(results)
        
        output_file = self.output_dir / "quick_ablation_results.json"
        with open(output_file, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"\nResults saved to {output_file}")
    
    def _print_summary(self, results: dict):
        """Print summary of ablation results."""
        print("\n" + "="*60)
        print("ABLATION STUDY SUMMARY")
        print("="*60)
        
        # Loss ablation summary
        if 'loss_ablation' in results:
            print("\nLoss Function Ablation:")
            print("-" * 30)
            for exp_name, exp_results in results['loss_ablation'].items():
                metrics = exp_results['final_metrics']
                print(f"{exp_name:20s}: PSNR={metrics['psnr']:6.2f}, SSIM={metrics['ssim']:6.4f}")
        
        # Component ablation summary
        if 'component_ablation' in results:
            print("\nComponent Ablation:")
            print("-" * 20)
            for exp_name, exp_results in results['component_ablation'].items():
                metrics = exp_results['final_metrics']
                print(f"{exp_name:20s}: PSNR={metrics['psnr']:6.2f}, SSIM={metrics['ssim']:6.4f}")
        
        print("\n" + "="*60)


def main():
    """Main function to run the ablation study."""
    print("SVRaster Quick Ablation Study")
    print("=" * 60)
    
    # Run ablation study
    ablation_study = QuickAblationStudy()
    results = ablation_study.run_comprehensive_ablation()
    
    print("\nAblation study completed successfully!")
    print("Check the output directory for detailed results.")


if __name__ == "__main__":
    main() 