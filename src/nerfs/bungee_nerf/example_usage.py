"""
Example usage of BungeeNeRF
"""

import torch
import numpy as np
import logging
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .core import BungeeNeRF, BungeeNeRFConfig
from .progressive_encoder import ProgressivePositionalEncoder, MultiScaleEncoder
from .multiscale_renderer import MultiScaleRenderer, LevelOfDetailRenderer
from .trainer import ProgressiveTrainer, MultiScaleTrainer
from .utils import (
    compute_scale_factor, get_level_of_detail,
    create_progressive_schedule, save_bungee_model
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_basic_usage():
    """Example of basic BungeeNeRF usage"""
    logger.info("=== Basic BungeeNeRF Usage ===")
    
    # Create configuration
    config = BungeeNeRFConfig(
        num_stages=4,
        base_resolution=16,
        max_resolution=2048,
        hidden_dim=256,
        num_layers=8,
        learning_rate=5e-4
    )
    
    logger.info(f"Configuration: {config}")
    
    # Create model
    model = BungeeNeRF(config)
    logger.info(f"Model created with {model.count_parameters():,} parameters")
    
    # Show progressive info
    info = model.get_progressive_info()
    logger.info(f"Progressive info: {info}")
    
    # Test forward pass
    batch_size = 4
    rays_o = torch.randn(batch_size, 3)
    rays_d = torch.randn(batch_size, 3)
    rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
    bounds = torch.tensor([[2.0, 6.0]]).expand(batch_size, 2)
    distances = torch.rand(batch_size) * 50 + 10
    
    logger.info(f"Input shapes - rays_o: {rays_o.shape}, rays_d: {rays_d.shape}")
    logger.info(f"Distances: {distances.numpy()}")
    
    outputs = model(rays_o, rays_d, bounds, distances)
    
    logger.info(f"Output RGB shape: {outputs['rgb'].shape}")
    logger.info(f"Output depth shape: {outputs['depth'].shape}")
    logger.info(f"RGB range: [{outputs['rgb'].min():.3f}, {outputs['rgb'].max():.3f}]")
    logger.info(f"Depth range: [{outputs['depth'].min():.3f}, {outputs['depth'].max():.3f}]")


def example_progressive_training():
    """Example of progressive training setup"""
    logger.info("\n=== Progressive Training Setup ===")
    
    # Create progressive schedule
    schedule = create_progressive_schedule(
        num_stages=4,
        steps_per_stage=10000,
        warmup_steps=1000
    )
    
    logger.info(f"Progressive schedule: {schedule}")
    
    # Show stage progression
    test_steps = [0, 5000, 10000, 15000, 25000, 35000]
    for step in test_steps:
        stage = 0
        for transition_step in schedule["stage_transitions"]:
            if step >= transition_step:
                stage += 1
            else:
                break
        stage = min(stage, schedule["num_stages"] - 1)
        logger.info(f"Step {step:5d} -> Stage {stage}")


def example_multiscale_encoding():
    """Example of multi-scale encoding"""
    logger.info("\n=== Multi-scale Encoding ===")
    
    # Create encoders
    progressive_encoder = ProgressivePositionalEncoder(
        num_freqs_base=4,
        num_freqs_max=10,
        include_input=True
    )
    
    multiscale_encoder = MultiScaleEncoder(
        num_scales=4,
        base_freqs=4,
        max_freqs=10
    )
    
    # Test data
    batch_size = 100
    coords = torch.randn(batch_size, 3) * 10  # Scale up coordinates
    distances = torch.rand(batch_size) * 100 + 1  # Distances from 1 to 101
    
    logger.info(f"Input coordinates shape: {coords.shape}")
    logger.info(f"Distance range: [{distances.min():.1f}, {distances.max():.1f}]")
    
    # Progressive encoding at different stages
    for stage in range(4):
        progressive_encoder.set_current_stage(stage)
        encoded = progressive_encoder(coords)
        current_freqs = progressive_encoder.get_current_freqs()
        
        logger.info(f"Stage {stage}: {current_freqs} freqs, output dim: {encoded.shape[-1]}")
    
    # Multi-scale encoding
    for scale in range(4):
        encoded = multiscale_encoder(coords, scale=scale)
        logger.info(f"Scale {scale}: output dim: {encoded.shape[-1]}")
    
    # Adaptive encoding based on distance
    encoded_adaptive = multiscale_encoder(coords, distances=distances)
    logger.info(f"Adaptive encoding: output dim: {encoded_adaptive.shape[-1]}")


def example_level_of_detail():
    """Example of level-of-detail rendering"""
    logger.info("\n=== Level of Detail ===")
    
    # Create LOD renderer
    config = BungeeNeRFConfig()
    lod_renderer = LevelOfDetailRenderer(
        config,
        num_lod_levels=4,
        lod_thresholds=[100.0, 50.0, 25.0, 10.0]
    )
    
    # Test distances
    test_distances = torch.tensor([150.0, 75.0, 30.0, 15.0, 5.0])
    
    for distance in test_distances:
        lod_level = lod_renderer.get_lod_level(distance.unsqueeze(0))
        scale_factor = compute_scale_factor(distance.unsqueeze(0), lod_renderer.lod_thresholds)
        
        logger.info(f"Distance {distance:6.1f} -> LOD level {lod_level.item()}, "
                   f"scale factor {scale_factor.item():.2f}")


def example_multiscale_loss():
    """Example of multi-scale loss computation"""
    logger.info("\n=== Multi-scale Loss ===")
    
    from .utils import compute_multiscale_loss
    
    # Create fake outputs and targets
    batch_size = 100
    outputs = {
        "rgb": torch.rand(batch_size, 3),
        "depth": torch.rand(batch_size) * 10,
        "weights": torch.rand(batch_size, 64)  # Fake attention weights
    }
    
    targets = {
        "rgb": torch.rand(batch_size, 3),
        "depth": torch.rand(batch_size) * 10
    }
    
    distances = torch.rand(batch_size) * 100 + 1
    
    config = BungeeNeRFConfig()
    
    # Compute losses for different stages
    for stage in range(4):
        losses = compute_multiscale_loss(outputs, targets, distances, stage, config)
        
        logger.info(f"Stage {stage} losses:")
        for key, value in losses.items():
            logger.info(f"  {key}: {value.item():.6f}")


def example_model_stages():
    """Example of model progression through stages"""
    logger.info("\n=== Model Stage Progression ===")
    
    config = BungeeNeRFConfig(num_stages=4, hidden_dim=128)
    model = BungeeNeRF(config)
    
    # Test model at different stages
    rays_o = torch.randn(10, 3)
    rays_d = torch.randn(10, 3)
    rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
    bounds = torch.tensor([[2.0, 6.0]]).expand(10, 2)
    distances = torch.rand(10) * 50 + 10
    
    for stage in range(4):
        model.set_current_stage(stage)
        
        info = model.get_progressive_info()
        logger.info(f"Stage {stage}:")
        logger.info(f"  Current stage: {info['current_stage']}")
        logger.info(f"  Progressive blocks: {info['num_progressive_blocks']}")
        logger.info(f"  Positional encoder freqs: {info['pos_encoder_freqs']}")
        
        # Forward pass
        outputs = model(rays_o, rays_d, bounds, distances)
        logger.info(f"  Output RGB mean: {outputs['rgb'].mean().item():.4f}")


def example_save_and_load():
    """Example of saving and loading models"""
    logger.info("\n=== Model Save/Load ===")
    
    # Create and configure model
    config = BungeeNeRFConfig(hidden_dim=64, num_layers=4)
    model = BungeeNeRF(config)
    model.set_current_stage(2)
    
    # Save model
    save_path = "/tmp/example_bungee_model.pth"
    save_bungee_model(
        model,
        config.__dict__,
        save_path,
        stage=2,
        epoch=50
    )
    
    logger.info(f"Model saved to {save_path}")
    
    # Load model
            from .utils import load_bungee_model
    
    new_model = BungeeNeRF(config)
    new_model.set_current_stage(2)  # Set same stage as saved model
    loaded_model, loaded_config = load_bungee_model(new_model, save_path, device="cpu")
    
    logger.info(f"Model loaded successfully")
    logger.info(f"Loaded config keys: {list(loaded_config.keys())}")
    
    # Verify models are equivalent
    original_params = sum(p.numel() for p in model.parameters())
    loaded_params = sum(p.numel() for p in loaded_model.parameters())
    
    logger.info(f"Original model parameters: {original_params:,}")
    logger.info(f"Loaded model parameters: {loaded_params:,}")
    
    # Clean up
    os.remove(save_path)


def example_performance_analysis():
    """Example of performance analysis"""
    logger.info("\n=== Performance Analysis ===")
    
    import time
    
    # Create models with different sizes
    configs = [
        ("Small", BungeeNeRFConfig(hidden_dim=128, num_layers=4, num_samples=32)),
        ("Medium", BungeeNeRFConfig(hidden_dim=256, num_layers=8, num_samples=64)),
        ("Large", BungeeNeRFConfig(hidden_dim=512, num_layers=12, num_samples=128))
    ]
    
    batch_size = 100
    rays_o = torch.randn(batch_size, 3)
    rays_d = torch.randn(batch_size, 3)
    rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
    bounds = torch.tensor([[2.0, 6.0]]).expand(batch_size, 2)
    distances = torch.rand(batch_size) * 50 + 10
    
    for name, config in configs:
        model = BungeeNeRF(config)
        model.eval()
        
        # Warmup
        with torch.no_grad():
            for _ in range(5):
                _ = model(rays_o, rays_d, bounds, distances)
        
        # Timing
        num_runs = 10
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(num_runs):
                outputs = model(rays_o, rays_d, bounds, distances)
        
        avg_time = (time.time() - start_time) / num_runs
        
        logger.info(f"{name} model:")
        logger.info(f"  Parameters: {model.count_parameters():,}")
        logger.info(f"  Forward time: {avg_time*1000:.2f} ms")
        logger.info(f"  Samples per second: {batch_size/avg_time:.0f}")


def main():
    """Run all examples"""
    logger.info("BungeeNeRF Example Usage")
    logger.info("=" * 50)
    
    try:
        example_basic_usage()
        example_progressive_training()
        example_multiscale_encoding()
        example_level_of_detail()
        example_multiscale_loss()
        example_model_stages()
        example_save_and_load()
        example_performance_analysis()
        
        logger.info("\n🎉 All examples completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Example failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
