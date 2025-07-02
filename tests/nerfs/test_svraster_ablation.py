"""
Ablation study tests for SVRaster.

This script implements comprehensive ablation experiments as described in the original SVRaster paper,
testing the contribution of different components and loss functions to the overall performance.
"""

import unittest
import torch
import numpy as np
import tempfile
import os
import time
import json
from pathlib import Path
import sys
from typing import Dict, List, Tuple, Any

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from nerfs.svraster.core import SVRasterConfig, SVRasterModel, SVRasterLoss
from nerfs.svraster.dataset import SVRasterDatasetConfig, SVRasterDataset
from nerfs.svraster.trainer import SVRasterTrainerConfig, SVRasterTrainer


class SVRasterAblationStudy:
    """
    Comprehensive ablation study for SVRaster.
    
    Tests the contribution of:
    1. Different loss functions (RGB, SSIM, Distortion, Pointwise RGB, Opacity)
    2. Morton ordering vs no ordering
    3. Different SH degrees (0, 1, 2, 3)
    4. Adaptive subdivision vs fixed structure
    5. Different voxel sampling strategies
    6. View-dependent vs view-independent rendering
    """
    
    def __init__(self, output_dir: str = "ablation_results"):
        """Initialize ablation study."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.results = {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Standard test configuration
        self.base_config = SVRasterConfig(
            max_octree_levels=6,
            base_resolution=16,
            scene_bounds=(-1.0, -1.0, -1.0, 1.0, 1.0, 1.0),
            ray_samples_per_voxel=8,
            morton_ordering=True,
            sh_degree=2
        )
        
        # Create synthetic test dataset
        self.test_dataset = self._create_synthetic_dataset()
        
    def _create_synthetic_dataset(self):
        """Create synthetic test dataset for ablation study."""
        # Create simple synthetic scene with known geometry
        num_images = 10
        image_size = 64
        num_rays = image_size * image_size
        
        # Create camera poses in a circle around the scene
        angles = torch.linspace(0, 2 * np.pi, num_images)
        camera_positions = torch.stack([
            torch.cos(angles) * 2.0,
            torch.sin(angles) * 2.0,
            torch.ones_like(angles) * 1.5
        ], dim=-1)
        
        # Create rays pointing towards origin
        ray_origins = []
        ray_directions = []
        target_colors = []
        
        for i in range(num_images):
            # Create rays for this camera
            rays_o = camera_positions[i].unsqueeze(0).repeat(num_rays, 1)
            rays_d = -camera_positions[i].unsqueeze(0).repeat(num_rays, 1)
            rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
            
            # Create synthetic target colors (simple pattern)
            u = torch.linspace(0, 1, image_size)
            v = torch.linspace(0, 1, image_size)
            u, v = torch.meshgrid(u, v, indexing='ij')
            
            # Create color pattern
            colors = torch.stack([
                0.5 + 0.5 * torch.sin(2 * np.pi * u.flatten()),
                0.5 + 0.5 * torch.cos(2 * np.pi * v.flatten()),
                0.5 + 0.25 * torch.sin(4 * np.pi * (u + v).flatten())
            ], dim=-1)
            
            ray_origins.append(rays_o)
            ray_directions.append(rays_d)
            target_colors.append(colors)
        
        # Combine all data
        all_rays_o = torch.cat(ray_origins, dim=0)
        all_rays_d = torch.cat(ray_directions, dim=0)
        all_colors = torch.cat(target_colors, dim=0)
        
        # Create dataset
        dataset_config = SVRasterDatasetConfig(
            data_dir="",
            image_height=image_size,
            image_width=image_size
        )
        
        # Create a simple dataset class for testing
        class SyntheticDataset:
            def __init__(self, rays_o, rays_d, colors, batch_size=1024):
                self.rays_o = rays_o
                self.rays_d = rays_d
                self.colors = colors
                self.batch_size = batch_size
                self.num_samples = len(rays_o)
            
            def __len__(self):
                return (self.num_samples + self.batch_size - 1) // self.batch_size
            
            def __getitem__(self, idx):
                start_idx = idx * self.batch_size
                end_idx = min(start_idx + self.batch_size, self.num_samples)
                
                return {
                    'rays_o': self.rays_o[start_idx:end_idx],
                    'rays_d': self.rays_d[start_idx:end_idx],
                    'colors': self.colors[start_idx:end_idx]
                }
        
        return SyntheticDataset(all_rays_o, all_rays_d, all_colors)
    
    def _compute_metrics(self, pred_rgb: torch.Tensor, target_rgb: torch.Tensor) -> Dict[str, float]:
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
            
            # LPIPS approximation (simplified)
            lpips = torch.mean(torch.abs(pred_rgb - target_rgb))
            
            return {
                'psnr': psnr.item(),
                'ssim': ssim.item(),
                'lpips': lpips.item(),
                'mse': mse.item()
            }
    
    def _train_and_evaluate(self, config: SVRasterConfig, experiment_name: str, 
                           num_epochs: int = 5) -> Dict[str, Any]:
        """Train model with given config and evaluate performance."""
        print(f"Running experiment: {experiment_name}")
        
        # Create model and loss function
        model = SVRasterModel(config).to(self.device)
        loss_fn = SVRasterLoss(config)
        
        # Create optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        # Training loop
        train_losses = []
        val_metrics = []
        
        for epoch in range(num_epochs):
            model.train()
            epoch_loss = 0.0
            num_batches = 0
            
            # Training
            for batch_idx in range(len(self.test_dataset)):
                batch = self.test_dataset[batch_idx]
                rays_o = batch['rays_o'].to(self.device)
                rays_d = batch['rays_d'].to(self.device)
                target_colors = batch['colors'].to(self.device)
                
                # Forward pass
                outputs = model(rays_o, rays_d)
                losses = loss_fn(outputs, {'rgb': target_colors}, model)
                loss = losses['total_loss']
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            avg_loss = epoch_loss / num_batches
            train_losses.append(avg_loss)
            
            # Validation
            if epoch % 2 == 0:  # Evaluate every 2 epochs
                model.eval()
                with torch.no_grad():
                    # Evaluate on a subset
                    eval_batch = self.test_dataset[0]  # Use first batch for evaluation
                    rays_o = eval_batch['rays_o'].to(self.device)
                    rays_d = eval_batch['rays_d'].to(self.device)
                    target_colors = eval_batch['colors'].to(self.device)
                    
                    outputs = model(rays_o, rays_d)
                    metrics = self._compute_metrics(outputs['rgb'], target_colors)
                    val_metrics.append(metrics)
                    
                    print(f"  Epoch {epoch}: Loss={avg_loss:.4f}, PSNR={metrics['psnr']:.2f}, SSIM={metrics['ssim']:.4f}")
        
        # Final evaluation
        model.eval()
        with torch.no_grad():
            final_metrics = self._compute_metrics(outputs['rgb'], target_colors)
        
        # Get voxel statistics
        voxel_stats = model.get_voxel_statistics()
        
        return {
            'experiment_name': experiment_name,
            'config': config.__dict__,
            'train_losses': train_losses,
            'val_metrics': val_metrics,
            'final_metrics': final_metrics,
            'voxel_stats': voxel_stats,
            'training_time': time.time()  # Placeholder for timing
        }
    
    def run_loss_ablation(self) -> Dict[str, Any]:
        """Ablation study for different loss functions."""
        print("\n" + "="*60)
        print("LOSS FUNCTION ABLATION STUDY")
        print("="*60)
        
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
            ("RGB_MSE_Distortion", {
                'use_ssim_loss': False,
                'use_distortion_loss': True,
                'use_pointwise_rgb_loss': False,
                'use_opacity_regularization': False
            }),
            ("RGB_MSE_Pointwise", {
                'use_ssim_loss': False,
                'use_distortion_loss': False,
                'use_pointwise_rgb_loss': True,
                'use_opacity_regularization': False
            }),
            ("RGB_MSE_Opacity", {
                'use_ssim_loss': False,
                'use_distortion_loss': False,
                'use_pointwise_rgb_loss': False,
                'use_opacity_regularization': True
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
            results[exp_name] = self._train_and_evaluate(config, exp_name)
        
        return results
    
    def run_morton_ordering_ablation(self) -> Dict[str, Any]:
        """Ablation study for Morton ordering."""
        print("\n" + "="*60)
        print("MORTON ORDERING ABLATION STUDY")
        print("="*60)
        
        experiments = [
            ("No_Morton_Ordering", {'morton_ordering': False}),
            ("With_Morton_Ordering", {'morton_ordering': True})
        ]
        
        results = {}
        for exp_name, config_updates in experiments:
            config = SVRasterConfig(**{**self.base_config.__dict__, **config_updates})
            results[exp_name] = self._train_and_evaluate(config, exp_name)
        
        return results
    
    def run_sh_degree_ablation(self) -> Dict[str, Any]:
        """Ablation study for different SH degrees."""
        print("\n" + "="*60)
        print("SPHERICAL HARMONICS DEGREE ABLATION STUDY")
        print("="*60)
        
        experiments = [
            ("SH_Degree_0", {'sh_degree': 0}),
            ("SH_Degree_1", {'sh_degree': 1}),
            ("SH_Degree_2", {'sh_degree': 2}),
            ("SH_Degree_3", {'sh_degree': 3})
        ]
        
        results = {}
        for exp_name, config_updates in experiments:
            config = SVRasterConfig(**{**self.base_config.__dict__, **config_updates})
            results[exp_name] = self._train_and_evaluate(config, exp_name)
        
        return results
    
    def run_sampling_ablation(self) -> Dict[str, Any]:
        """Ablation study for different sampling strategies."""
        print("\n" + "="*60)
        print("SAMPLING STRATEGY ABLATION STUDY")
        print("="*60)
        
        experiments = [
            ("Low_Sampling", {'ray_samples_per_voxel': 4}),
            ("Medium_Sampling", {'ray_samples_per_voxel': 8}),
            ("High_Sampling", {'ray_samples_per_voxel': 16})
        ]
        
        results = {}
        for exp_name, config_updates in experiments:
            config = SVRasterConfig(**{**self.base_config.__dict__, **config_updates})
            results[exp_name] = self._train_and_evaluate(config, exp_name)
        
        return results
    
    def run_octree_level_ablation(self) -> Dict[str, Any]:
        """Ablation study for different octree levels."""
        print("\n" + "="*60)
        print("OCTREE LEVEL ABLATION STUDY")
        print("="*60)
        
        experiments = [
            ("Low_Levels", {'max_octree_levels': 4}),
            ("Medium_Levels", {'max_octree_levels': 6}),
            ("High_Levels", {'max_octree_levels': 8})
        ]
        
        results = {}
        for exp_name, config_updates in experiments:
            config = SVRasterConfig(**{**self.base_config.__dict__, **config_updates})
            results[exp_name] = self._train_and_evaluate(config, exp_name)
        
        return results
    
    def run_comprehensive_ablation(self) -> Dict[str, Any]:
        """Run all ablation studies."""
        print("Starting comprehensive SVRaster ablation study...")
        
        all_results = {
            'loss_ablation': self.run_loss_ablation(),
            'morton_ordering_ablation': self.run_morton_ordering_ablation(),
            'sh_degree_ablation': self.run_sh_degree_ablation(),
            'sampling_ablation': self.run_sampling_ablation(),
            'octree_level_ablation': self.run_octree_level_ablation()
        }
        
        # Save results
        self._save_results(all_results)
        self._create_visualizations(all_results)
        
        return all_results
    
    def _save_results(self, results: Dict[str, Any]):
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
        
        output_file = self.output_dir / "ablation_results.json"
        with open(output_file, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"Results saved to {output_file}")
    
    def _create_visualizations(self, results: Dict[str, Any]):
        """Create visualization plots for ablation results."""
        try:
            import matplotlib.pyplot as plt
            
            # Create summary plots
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            axes = axes.flatten()
            
            # Loss ablation results
            if 'loss_ablation' in results:
                loss_results = results['loss_ablation']
                exp_names = list(loss_results.keys())
                psnr_values = [loss_results[name]['final_metrics']['psnr'] for name in exp_names]
                ssim_values = [loss_results[name]['final_metrics']['ssim'] for name in exp_names]
                
                axes[0].bar(exp_names, psnr_values)
                axes[0].set_title('Loss Ablation - PSNR')
                axes[0].set_ylabel('PSNR (dB)')
                axes[0].tick_params(axis='x', rotation=45)
                
                axes[1].bar(exp_names, ssim_values)
                axes[1].set_title('Loss Ablation - SSIM')
                axes[1].set_ylabel('SSIM')
                axes[1].tick_params(axis='x', rotation=45)
            
            # SH degree ablation results
            if 'sh_degree_ablation' in results:
                sh_results = results['sh_degree_ablation']
                exp_names = list(sh_results.keys())
                psnr_values = [sh_results[name]['final_metrics']['psnr'] for name in exp_names]
                
                axes[2].bar(exp_names, psnr_values)
                axes[2].set_title('SH Degree Ablation - PSNR')
                axes[2].set_ylabel('PSNR (dB)')
                axes[2].tick_params(axis='x', rotation=45)
            
            # Sampling ablation results
            if 'sampling_ablation' in results:
                sampling_results = results['sampling_ablation']
                exp_names = list(sampling_results.keys())
                psnr_values = [sampling_results[name]['final_metrics']['psnr'] for name in exp_names]
                
                axes[3].bar(exp_names, psnr_values)
                axes[3].set_title('Sampling Ablation - PSNR')
                axes[3].set_ylabel('PSNR (dB)')
                axes[3].tick_params(axis='x', rotation=45)
            
            # Morton ordering ablation results
            if 'morton_ordering_ablation' in results:
                morton_results = results['morton_ordering_ablation']
                exp_names = list(morton_results.keys())
                psnr_values = [morton_results[name]['final_metrics']['psnr'] for name in exp_names]
                
                axes[4].bar(exp_names, psnr_values)
                axes[4].set_title('Morton Ordering Ablation - PSNR')
                axes[4].set_ylabel('PSNR (dB)')
                axes[4].tick_params(axis='x', rotation=45)
            
            # Octree level ablation results
            if 'octree_level_ablation' in results:
                octree_results = results['octree_level_ablation']
                exp_names = list(octree_results.keys())
                psnr_values = [octree_results[name]['final_metrics']['psnr'] for name in exp_names]
                
                axes[5].bar(exp_names, psnr_values)
                axes[5].set_title('Octree Level Ablation - PSNR')
                axes[5].set_ylabel('PSNR (dB)')
                axes[5].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            plt.savefig(self.output_dir / "ablation_summary.png", dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {self.output_dir / 'ablation_summary.png'}")
            
        except ImportError:
            print("Matplotlib not available, skipping visualizations")
        except Exception as e:
            print(f"Error creating visualizations: {e}")


class TestSVRasterAblation(unittest.TestCase):
    """Unit tests for SVRaster ablation study."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.ablation_study = SVRasterAblationStudy(output_dir=self.temp_dir)
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_loss_ablation(self):
        """Test loss function ablation study."""
        results = self.ablation_study.run_loss_ablation()
        
        # Check that all experiments completed
        expected_experiments = [
            "RGB_MSE_only", "RGB_MSE_SSIM", "RGB_MSE_Distortion",
            "RGB_MSE_Pointwise", "RGB_MSE_Opacity", "All_Losses"
        ]
        
        for exp_name in expected_experiments:
            self.assertIn(exp_name, results)
            self.assertIn('final_metrics', results[exp_name])
            self.assertIn('psnr', results[exp_name]['final_metrics'])
            self.assertIn('ssim', results[exp_name]['final_metrics'])
    
    def test_morton_ordering_ablation(self):
        """Test Morton ordering ablation study."""
        results = self.ablation_study.run_morton_ordering_ablation()
        
        expected_experiments = ["No_Morton_Ordering", "With_Morton_Ordering"]
        
        for exp_name in expected_experiments:
            self.assertIn(exp_name, results)
            self.assertIn('final_metrics', results[exp_name])
    
    def test_sh_degree_ablation(self):
        """Test SH degree ablation study."""
        results = self.ablation_study.run_sh_degree_ablation()
        
        expected_experiments = ["SH_Degree_0", "SH_Degree_1", "SH_Degree_2", "SH_Degree_3"]
        
        for exp_name in expected_experiments:
            self.assertIn(exp_name, results)
            self.assertIn('final_metrics', results[exp_name])
    
    def test_comprehensive_ablation(self):
        """Test comprehensive ablation study."""
        results = self.ablation_study.run_comprehensive_ablation()
        
        # Check that all ablation studies are present
        expected_studies = [
            'loss_ablation', 'morton_ordering_ablation', 'sh_degree_ablation',
            'sampling_ablation', 'octree_level_ablation'
        ]
        
        for study_name in expected_studies:
            self.assertIn(study_name, results)
            self.assertIsInstance(results[study_name], dict)
            self.assertGreater(len(results[study_name]), 0)


def run_ablation_study(output_dir: str = "ablation_results"):
    """Run the complete SVRaster ablation study."""
    print("SVRaster Ablation Study")
    print("=" * 60)
    print("This study tests the contribution of different components:")
    print("1. Loss functions (RGB, SSIM, Distortion, Pointwise RGB, Opacity)")
    print("2. Morton ordering vs no ordering")
    print("3. Different SH degrees (0, 1, 2, 3)")
    print("4. Different sampling strategies")
    print("5. Different octree levels")
    print("=" * 60)
    
    ablation_study = SVRasterAblationStudy(output_dir=output_dir)
    results = ablation_study.run_comprehensive_ablation()
    
    print("\n" + "="*60)
    print("ABLATION STUDY COMPLETED")
    print("="*60)
    print(f"Results saved to: {output_dir}")
    print("Check ablation_results.json for detailed results")
    print("Check ablation_summary.png for visualizations")
    
    return results


if __name__ == "__main__":
    # Run ablation study
    run_ablation_study()
    
    # Run unit tests
    unittest.main(argv=[''], exit=False, verbosity=2) 