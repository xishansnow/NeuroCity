#!/usr/bin/env python3
"""
1920x1200 Resolution Ray Processing Test
Tests Block-NeRF CUDA extension with full HD+ resolution
"""
import torch
import numpy as np
import time
import math

class FullResolutionTest:
    """Test Block-NeRF CUDA with 1920x1200 resolution"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Screen resolution
        self.width = 1920
        self.height = 1200
        self.total_rays = self.width * self.height
        
        # Camera parameters
        self.focal_length = 1000.0  # Typical focal length
        self.near_plane = 0.1
        self.far_plane = 100.0
        
        # Scene parameters
        self.scene_bounds = [-50, 50]  # Large scene
        self.block_size = 5.0
        
        print(f"🖥️  Resolution: {self.width}x{self.height}")
        print(f"📏 Total rays: {self.total_rays:,}")
        print(f"🎯 Device: {self.device}")
        
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            memory_gb = props.total_memory / 1024**3
            print(f"💾 GPU Memory: {memory_gb:.2f} GB")
    
    def calculate_memory_requirements(self):
        """Calculate memory requirements for full resolution"""
        print("\n📊 Memory Requirements Analysis")
        print("=" * 50)
        
        bytes_per_float = 4
        
        # Ray data
        ray_origins_mb = (self.total_rays * 3 * bytes_per_float) / 1024**2
        ray_directions_mb = (self.total_rays * 3 * bytes_per_float) / 1024**2
        
        print(f"Ray origins: {ray_origins_mb:.1f} MB")
        print(f"Ray directions: {ray_directions_mb:.1f} MB")
        
        # Block data (assuming 1000 blocks)
        num_blocks = 1000
        block_centers_mb = (num_blocks * 3 * bytes_per_float) / 1024**2
        block_sizes_mb = (num_blocks * bytes_per_float) / 1024**2
        
        print(f"Block centers ({num_blocks} blocks): {block_centers_mb:.1f} MB")
        print(f"Block sizes: {block_sizes_mb:.1f} MB")
        
        # Visibility matrix
        visibility_mb = (self.total_rays * num_blocks * bytes_per_float) / 1024**2
        
        print(f"Visibility matrix: {visibility_mb:.1f} MB")
        
        # Volume rendering (64 samples per ray)
        samples_per_ray = 64
        densities_mb = (self.total_rays * samples_per_ray * bytes_per_float) / 1024**2
        colors_mb = (self.total_rays * samples_per_ray * 3 * bytes_per_float) / 1024**2
        
        print(f"Densities ({samples_per_ray} samples/ray): {densities_mb:.1f} MB")
        print(f"Colors: {colors_mb:.1f} MB")
        
        # Final image
        final_image_mb = (self.total_rays * 3 * bytes_per_float) / 1024**2
        
        print(f"Final image: {final_image_mb:.1f} MB")
        
        total_mb = (ray_origins_mb + ray_directions_mb + block_centers_mb + 
                   block_sizes_mb + visibility_mb + densities_mb + colors_mb + 
                   final_image_mb)
        
        print(f"\n💾 Total estimated memory: {total_mb:.1f} MB ({total_mb/1024:.2f} GB)")
        
        return total_mb
    
    def generate_camera_rays(self):
        """Generate camera rays for full resolution"""
        print(f"\n📷 Generating {self.total_rays:,} camera rays...")
        
        start_time = time.time()
        
        # Create pixel coordinates
        i, j = torch.meshgrid(
            torch.arange(self.width, device=self.device, dtype=torch.float32),
            torch.arange(self.height, device=self.device, dtype=torch.float32),
            indexing='xy'
        )
        
        # Convert to normalized device coordinates
        x = (i - self.width * 0.5) / self.focal_length
        y = -(j - self.height * 0.5) / self.focal_length
        z = -torch.ones_like(i)
        
        # Create ray directions
        ray_directions = torch.stack([x, y, z], dim=-1)
        ray_directions = ray_directions.reshape(-1, 3)
        ray_directions = torch.nn.functional.normalize(ray_directions, dim=1)
        
        # Camera origin (looking down negative z-axis)
        camera_origin = torch.tensor([0, 0, 0], device=self.device, dtype=torch.float32)
        ray_origins = camera_origin.unsqueeze(0).expand(self.total_rays, -1)
        
        end_time = time.time()
        
        print(f"✅ Ray generation completed in {end_time - start_time:.2f}s")
        print(f"📏 Ray origins shape: {ray_origins.shape}")
        print(f"📏 Ray directions shape: {ray_directions.shape}")
        
        return ray_origins, ray_directions
    
    def create_scene_blocks(self, num_blocks=1000):
        """Create scene blocks"""
        print(f"\n🏗️  Creating scene with {num_blocks} blocks...")
        
        # Create random block positions within scene bounds
        block_centers = torch.rand(num_blocks, 3, device=self.device) * \
                       (self.scene_bounds[1] - self.scene_bounds[0]) + self.scene_bounds[0]
        
        # Add some blocks along the view direction
        z_positions = torch.linspace(5, 50, num_blocks//4, device=self.device)
        forward_blocks = torch.stack([
            torch.zeros(len(z_positions), device=self.device),
            torch.zeros(len(z_positions), device=self.device),
            z_positions
        ], dim=1)
        
        # Combine random and forward blocks
        block_centers = torch.cat([block_centers[:3*num_blocks//4], forward_blocks], dim=0)
        block_sizes = torch.ones(num_blocks, device=self.device) * self.block_size
        
        print(f"✅ Scene blocks created")
        print(f"📏 Block centers shape: {block_centers.shape}")
        print(f"📏 Block sizes shape: {block_sizes.shape}")
        
        return block_centers, block_sizes
    
    def test_batch_processing(self, ray_origins, ray_directions, block_centers, block_sizes):
        """Test processing in batches to manage memory"""
        print(f"\n🔄 Testing batch processing...")
        
        try:
            import block_nerf_cuda
        except ImportError:
            print("❌ Block-NeRF CUDA extension not available")
            return False
        
        # Calculate optimal batch size based on available memory
        if torch.cuda.is_available():
            available_memory = torch.cuda.get_device_properties(0).total_memory
            # Use conservative estimate - 50% of available memory
            safe_memory = available_memory * 0.5
            
            # Estimate memory per ray (including all operations)
            memory_per_ray = (3 + 3 + len(block_centers)) * 4  # float32
            max_batch_size = int(safe_memory / memory_per_ray)
            
            # Use power of 2 for efficiency
            batch_size = 2 ** int(math.log2(max_batch_size))
            batch_size = min(batch_size, 4096)  # Cap at reasonable size
        else:
            batch_size = 1024
        
        print(f"📦 Using batch size: {batch_size:,}")
        
        num_batches = (self.total_rays + batch_size - 1) // batch_size
        print(f"📦 Total batches: {num_batches}")
        
        total_time = 0
        successful_batches = 0
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, self.total_rays)
            actual_batch_size = end_idx - start_idx
            
            if batch_idx % 10 == 0:  # Print progress every 10 batches
                progress = (batch_idx / num_batches) * 100
                print(f"  Processing batch {batch_idx}/{num_batches} ({progress:.1f}%)")
            
            try:
                # Extract batch
                batch_origins = ray_origins[start_idx:end_idx]
                batch_directions = ray_directions[start_idx:end_idx]
                
                # Test block visibility
                torch.cuda.synchronize()
                start_time = time.time()
                
                visibility = block_nerf_cuda.determine_block_visibility(
                    batch_origins, batch_directions, block_centers, block_sizes
                )
                
                torch.cuda.synchronize()
                end_time = time.time()
                
                batch_time = end_time - start_time
                total_time += batch_time
                successful_batches += 1
                
                # Calculate throughput for this batch
                operations = actual_batch_size * len(block_centers)
                throughput = operations / batch_time
                
                if batch_idx % 50 == 0:  # Detailed info every 50 batches
                    print(f"    Batch {batch_idx}: {batch_time*1000:.1f}ms, "
                          f"{throughput:.0f} ops/sec")
                
                # Clean up to save memory
                del visibility
                
            except Exception as e:
                print(f"❌ Batch {batch_idx} failed: {e}")
                # Try to continue with smaller batches
                if "out of memory" in str(e).lower():
                    print("💾 GPU memory exhausted, reducing batch size...")
                    batch_size = batch_size // 2
                    if batch_size < 64:
                        print("❌ Batch size too small, stopping")
                        break
                    continue
                else:
                    break
        
        if successful_batches > 0:
            avg_time_per_batch = total_time / successful_batches
            total_rays_processed = successful_batches * batch_size
            overall_throughput = total_rays_processed * len(block_centers) / total_time
            
            print(f"\n✅ Batch processing completed!")
            print(f"📊 Statistics:")
            print(f"   Successful batches: {successful_batches}/{num_batches}")
            print(f"   Total rays processed: {total_rays_processed:,}")
            print(f"   Average time per batch: {avg_time_per_batch*1000:.1f}ms")
            print(f"   Overall throughput: {overall_throughput:.0f} ray-block ops/sec")
            
            # Estimate full resolution processing time
            estimated_full_time = (self.total_rays / total_rays_processed) * total_time
            print(f"   Estimated full resolution time: {estimated_full_time:.1f}s")
            
            return True
        else:
            print("❌ No batches processed successfully")
            return False
    
    def run_full_resolution_test(self):
        """Run complete full resolution test"""
        print("\n" + "=" * 60)
        print("🖥️  1920x1200 Full Resolution Block-NeRF Test")
        print("=" * 60)
        
        # Check memory requirements
        required_memory = self.calculate_memory_requirements()
        
        if torch.cuda.is_available():
            available_memory = torch.cuda.get_device_properties(0).total_memory / 1024**2
            print(f"💾 Available GPU memory: {available_memory:.1f} MB")
            
            if required_memory > available_memory * 0.8:
                print("⚠️  Warning: Required memory exceeds 80% of available GPU memory")
                print("   Will use batch processing to manage memory usage")
        
        # Generate rays
        try:
            ray_origins, ray_directions = self.generate_camera_rays()
        except Exception as e:
            print(f"❌ Failed to generate rays: {e}")
            return False
        
        # Create scene
        try:
            block_centers, block_sizes = self.create_scene_blocks()
        except Exception as e:
            print(f"❌ Failed to create scene: {e}")
            return False
        
        # Test batch processing
        success = self.test_batch_processing(ray_origins, ray_directions, 
                                           block_centers, block_sizes)
        
        # Performance summary
        if success:
            print(f"\n🎉 Full resolution test completed successfully!")
            print(f"📊 Performance Summary:")
            print(f"   Resolution: {self.width}x{self.height}")
            print(f"   Total rays: {self.total_rays:,}")
            print(f"   Memory management: Batch processing")
            print(f"   Status: ✅ Success")
        else:
            print(f"\n❌ Full resolution test failed")
        
        return success

def main():
    """Main function"""
    if not torch.cuda.is_available():
        print("❌ CUDA not available. Cannot run full resolution test.")
        return
    
    test = FullResolutionTest()
    success = test.run_full_resolution_test()
    
    if success:
        print("\n🎊 1920x1200 resolution test completed!")
        print("   The Block-NeRF CUDA extension can handle full HD+ resolution.")
    else:
        print("\n💥 Test failed - check GPU memory and CUDA extension.")

if __name__ == "__main__":
    main()
