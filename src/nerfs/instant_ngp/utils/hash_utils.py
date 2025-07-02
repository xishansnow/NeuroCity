"""
Hash utilities for Instant NGP implementation.

This module provides functions for Morton encoding, hash table operations, and other hash-related utilities used in the multiresolution hash encoding.
"""

import torch
import numpy as np

def morton_encode_3d(coords: torch.Tensor) -> torch.Tensor:
    """
    Morton (Z-order) encoding for 3D coordinates.
    
    改进的 Morton 编码实现，支持更高的分辨率：
    - 每个坐标分量使用 21 位，总共支持 2097151³ 个位置
    - 完全满足高分辨率需求
    
    Args:
        coords: [N, 3] integer coordinates
        
    Returns:
        [N] Morton codes
    """
    def expand_bits(v):
        """Expand bits for Morton encoding."""
        # 支持更大的坐标值，使用更高效的位交错
        # 将 21 位输入扩展为 63 位输出（3 * 21 = 63）
        v = v & 0x1fffff  # 21 位掩码，支持 0-2097151
        v = (v | (v << 32)) & 0x1f00000000ffff
        v = (v | (v << 16)) & 0x1f0000ff0000ff
        v = (v | (v << 8)) & 0x100f00f00f00f00f
        v = (v | (v << 4)) & 0x10c30c30c30c30c3
        v = (v | (v << 2)) & 0x1249249249249249
        return v
    
    x = coords[:, 0].long()
    y = coords[:, 1].long()
    z = coords[:, 2].long()
    
    # Clamp to valid range (21-bit)
    x = torch.clamp(x, 0, 0x1fffff)
    y = torch.clamp(y, 0, 0x1fffff)
    z = torch.clamp(z, 0, 0x1fffff)
    
    # Expand bits and interleave
    xx = expand_bits(x.cpu().numpy())
    yy = expand_bits(y.cpu().numpy())
    zz = expand_bits(z.cpu().numpy())
    
    morton_codes = xx | (yy << 1) | (zz << 2)
    
    return torch.from_numpy(morton_codes).to(coords.device)

def morton_decode_3d(morton_codes: torch.Tensor) -> torch.Tensor:
    """
    Decode Morton codes back to 3D coordinates.
    
    Args:
        morton_codes: [N] Morton codes
        
    Returns:
        [N, 3] coordinates
    """
    def compact_bits(v):
        """Compact bits for Morton decoding."""
        v = v & 0x1249249249249249
        v = (v | (v >> 2)) & 0x10c30c30c30c30c3
        v = (v | (v >> 4)) & 0x100f00f00f00f00f
        v = (v | (v >> 8)) & 0x1f0000ff0000ff
        v = (v | (v >> 16)) & 0x1f00000000ffff
        v = (v | (v >> 32)) & 0x1fffff
        return v
    
    morton_np = morton_codes.cpu().numpy()
    
    x = compact_bits(morton_np)
    y = compact_bits(morton_np >> 1)
    z = compact_bits(morton_np >> 2)
    
    coords = np.stack([x, y, z], axis=1)
    
    return torch.from_numpy(coords).to(morton_codes.device)

def hash_function_simple(coords: torch.Tensor, table_size: int) -> torch.Tensor:
    """
    Simple hash function for 3D coordinates.
    
    Args:
        coords: [N, 3] coordinates
        table_size: Size of hash table
        
    Returns:
        [N] hash indices
    """
    # Use prime numbers for better distribution
    primes = torch.tensor([73856093, 19349663, 83492791], device=coords.device, dtype=torch.long)
    
    coords_long = coords.long()
    
    # Compute hash
    hash_val = (coords_long * primes).sum(dim=-1)
    
    # Ensure positive and within table size
    hash_val = hash_val % table_size
    hash_val = torch.abs(hash_val)
    
    return hash_val

def hash_function_coherent(coords: torch.Tensor, table_size: int) -> torch.Tensor:
    """
    Spatial-coherent hash function that preserves locality.
    
    Args:
        coords: [N, 3] coordinates  
        table_size: Size of hash table
        
    Returns:
        [N] hash indices
    """
    # Use bit manipulation for better spatial coherence
    x, y, z = coords[:, 0].long(), coords[:, 1].long(), coords[:, 2].long()
    
    # XOR folding for better distribution
    hash_val = x ^ (y << 1) ^ (z << 2)
    hash_val = hash_val ^ (hash_val >> 16)
    hash_val = hash_val ^ (hash_val >> 8)
    hash_val = hash_val ^ (hash_val >> 4)
    
    return torch.abs(hash_val) % table_size

def compute_hash_grid_size(resolution: int, max_entries: int) -> int:
    """
    Compute optimal hash table size for given resolution.
    
    Args:
        resolution: Grid resolution
        max_entries: Maximum number of hash table entries
        
    Returns:
        Optimal hash table size
    """
    # Total possible grid points
    total_points = resolution ** 3
    
    # Use smaller of max_entries or total_points
    table_size = min(max_entries, total_points)
    
    # Round to next power of 2 for efficient indexing
    table_size = 2 ** int(np.ceil(np.log2(table_size)))
    
    return table_size

def hash_grid_lookup(
    positions: torch.Tensor,
    embeddings: torch.nn.Embedding,
    resolution: int,
    hash_fn: str = 'simple',
) -> torch.Tensor:
    """
    Look up features from hash grid.
    
    Args:
        positions: [N, 3] positions in [0, 1]
        embeddings: Hash table embeddings
        resolution: Grid resolution
        hash_fn: Hash function type ('simple' or 'coherent')
        
    Returns:
        [N, feature_dim] features
    """
    # Scale positions to grid coordinates
    coords = positions * (resolution - 1)
    coords = torch.clamp(coords, 0, resolution - 1)
    
    # Get integer coordinates
    coords_int = coords.long()
    
    # Hash coordinates
    if hash_fn == 'simple':
        hash_indices = hash_function_simple(coords_int, embeddings.num_embeddings)
    elif hash_fn == 'coherent':
        hash_indices = hash_function_coherent(coords_int, embeddings.num_embeddings)
    else:
        raise ValueError(f"Unknown hash function: {hash_fn}")
    
    # Look up features
    features = embeddings(hash_indices)
    
    return features

def compute_hash_collisions(coords_list: list, table_size: int) -> float:
    """
    Compute collision rate for hash function.
    
    Args:
        coords_list: List of coordinate tensors
        table_size: Hash table size
        
    Returns:
        Collision rate (0.0 to 1.0)
    """
    all_coords = torch.cat(coords_list, dim=0)
    hash_indices = hash_function_simple(all_coords, table_size)
    
    # Count unique hash values
    unique_hashes = len(torch.unique(hash_indices))
    total_coords = len(all_coords)
    
    # Collision rate
    collision_rate = 1.0 - (unique_hashes / total_coords)
    
    return collision_rate

def optimize_hash_parameters(coords_list: list, target_collision_rate: float = 0.1) -> dict:
    """
    Optimize hash table parameters for given coordinates.
    
    Args:
        coords_list: List of coordinate tensors
        target_collision_rate: Target collision rate
        
    Returns:
        Dictionary with optimal parameters
    """
    all_coords = torch.cat(coords_list, dim=0)
    total_coords = len(all_coords)
    
    # Try different table sizes
    results = []
    
    for log_size in range(10, 25):  # 2^10 to 2^24
        table_size = 2 ** log_size
        
        if table_size > total_coords * 10:  # Stop if table too large
            break
            
        collision_rate = compute_hash_collisions(coords_list, table_size)
        
        results.append({
            'log_size': log_size, 'table_size': table_size, 'collision_rate': collision_rate, 'memory_mb': table_size * 4 / (
                1024 * 1024,
            )
        })
        
        if collision_rate <= target_collision_rate:
            break
    
    # Find best result
    best_result = min(results, key=lambda x: abs(x['collision_rate'] - target_collision_rate))
    
    return {
        'optimal_log_size': best_result['log_size'], 'optimal_table_size': best_result['table_size'], 'collision_rate': best_result['collision_rate'], 'memory_mb': best_result['memory_mb'], 'all_results': results
    }

def visualize_hash_distribution(
    coords: torch.Tensor,
    table_size: int,
    bins: int = 100,
) -> np.ndarray:
    """
    Visualize hash value distribution.
    
    Args:
        coords: [N, 3] coordinates
        table_size: Hash table size
        bins: Number of histogram bins
        
    Returns:
        Histogram of hash values
    """
    hash_indices = hash_function_simple(coords, table_size)
    
    # Compute histogram
    hist, _ = np.histogram(hash_indices.cpu().numpy(), bins=bins, range=(0, table_size))
    
    return hist

def test_hash_functions() -> None:
    """Test hash function implementations."""
    print("Testing hash functions...")
    
    # Create test coordinates
    coords = torch.randint(0, 64, (1000, 3))
    
    # Test Morton encoding
    morton_codes = morton_encode_3d(coords)
    decoded_coords = morton_decode_3d(morton_codes)
    
    # Check if Morton encoding is reversible
    morton_correct = torch.allclose(coords.float(), decoded_coords.float())
    print(f"Morton encoding test: {'PASS' if morton_correct else 'FAIL'}")
    
    # Test hash functions
    table_size = 1024
    hash_simple = hash_function_simple(coords, table_size)
    hash_coherent = hash_function_coherent(coords, table_size)
    
    # Check hash ranges
    simple_valid = (hash_simple >= 0).all() and (hash_simple < table_size).all()
    coherent_valid = (hash_coherent >= 0).all() and (hash_coherent < table_size).all()
    
    print(f"Simple hash function test: {'PASS' if simple_valid else 'FAIL'}")
    print(f"Coherent hash function test: {'PASS' if coherent_valid else 'FAIL'}")
    
    # Test collision rates
    collision_simple = compute_hash_collisions([coords], table_size)
    print(f"Simple hash collision rate: {collision_simple:.3f}")
    
    print("Hash function tests completed!")

if __name__ == "__main__":
    test_hash_functions() 