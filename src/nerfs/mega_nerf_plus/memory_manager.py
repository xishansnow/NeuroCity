from typing import Any, Optional
"""
Memory management components for Mega-NeRF++

This module implements efficient memory management for large-scale photogrammetric scenes:
- GPU memory monitoring and optimization
- Streaming data loading
- Cache management
- Memory-efficient batch processing
"""

import torch
import torch.nn as nn
import numpy as np
import gc
import psutil
import time
from collections import OrderedDict
import threading
import queue
from pathlib import Path
import pickle

class MemoryManager:
    """
    Centralized memory management for Mega-NeRF++
    """
    
    def __init__(
        self,
        max_memory_gb: float = 16.0,
        cache_size_gb: float = 4.0,
        cleanup_threshold: float = 0.9,
    ) -> None:
        """
        Args:
            max_memory_gb: Maximum GPU memory to use (GB)
            cache_size_gb: Maximum cache size (GB)
            cleanup_threshold: Memory usage threshold to trigger cleanup
        """
        self.max_memory_gb = max_memory_gb
        self.cache_size_gb = cache_size_gb
        self.cleanup_threshold = cleanup_threshold
        
        # Memory tracking
        self.memory_stats = {}
        self.allocation_history = []
        
        # Cache management
        self.tensor_cache = OrderedDict()
        self.cache_access_count = {}
        
        # Monitoring
        self.monitor_thread = None
        self.monitoring = False
        
    def start_monitoring(self):
        """Start memory monitoring thread"""
        if self.monitoring:
            return
            
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_memory, daemon=True)
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop memory monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
    
    def _monitor_memory(self):
        """Background memory monitoring"""
        while self.monitoring:
            self._update_memory_stats()
            
            # Trigger cleanup if memory usage is high
            if self.get_gpu_memory_usage() > self.cleanup_threshold:
                self.cleanup_cache()
            
            time.sleep(1.0)  # Check every second
    
    def _update_memory_stats(self):
        """Update memory usage statistics"""
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / (1024**3)  # GB
            gpu_memory_cached = torch.cuda.memory_reserved() / (1024**3)  # GB
            
            self.memory_stats.update({
                'gpu_allocated': gpu_memory, 'gpu_cached': gpu_memory_cached, 'cpu_percent': psutil.virtual_memory(
                )
            })
    
    def get_memory_stats(self) -> dict[str, float]:
        """Get current memory statistics"""
        self._update_memory_stats()
        return self.memory_stats.copy()
    
    def get_gpu_memory_usage(self) -> float:
        """Get GPU memory usage as fraction of maximum"""
        if not torch.cuda.is_available():
            return 0.0
        
        allocated_gb = torch.cuda.memory_allocated() / (1024**3)
        return allocated_gb / self.max_memory_gb
    
    def cleanup_cache(self):
        """Clean up memory caches"""
        
        # Clear tensor cache if it's getting large
        cache_size_mb = sum(
            tensor.numel() * tensor.element_size() / (1024**2)
            for tensor in self.tensor_cache.values()
        )
        
        if cache_size_mb > self.cache_size_gb * 1024:
            # Remove least recently used items
            items_to_remove = len(self.tensor_cache) // 4  # Remove 25%
            for _ in range(items_to_remove):
                if self.tensor_cache:
                    self.tensor_cache.popitem(last=False)
        
        # Clear PyTorch cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Force garbage collection
        gc.collect()
    
    def cache_tensor(self, key: str, tensor: torch.Tensor):
        """Cache a tensor for reuse"""
        if key in self.tensor_cache:
            # Move to end (most recently used)
            self.tensor_cache.move_to_end(key)
        else:
            self.tensor_cache[key] = tensor.detach().clone()
        
        self.cache_access_count[key] = self.cache_access_count.get(key, 0) + 1
    
    def get_cached_tensor(self, key: str) -> Optional[torch.Tensor]:
        """Retrieve cached tensor"""
        if key in self.tensor_cache:
            # Move to end (most recently used)
            self.tensor_cache.move_to_end(key)
            self.cache_access_count[key] = self.cache_access_count.get(key, 0) + 1
            return self.tensor_cache[key]
        return None
    
    def optimize_batch_size(self, base_batch_size: int, sample_tensor: torch.Tensor) -> int:
        """Optimize batch size based on available memory"""
        
        if not torch.cuda.is_available():
            return base_batch_size
        
        # Estimate memory per sample
        sample_memory = sample_tensor.numel() * sample_tensor.element_size()
        
        # Available memory
        available_memory = (self.max_memory_gb * 1024**3 - 
                          torch.cuda.memory_allocated())
        
        # Estimate optimal batch size (with safety margin)
        safety_factor = 0.7
        optimal_batch_size = int(available_memory * safety_factor / sample_memory)
        
        return min(max(optimal_batch_size, 1), base_batch_size * 2)

class CacheManager:
    """
    Advanced cache management for frequently accessed data
    """
    
    def __init__(self, cache_dir: str, max_cache_size_gb: float = 10.0):
        """
        Args:
            cache_dir: Directory for persistent cache
            max_cache_size_gb: Maximum cache size on disk (GB)
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.max_cache_size_gb = max_cache_size_gb
        
        # In-memory cache
        self.memory_cache = OrderedDict()
        self.cache_metadata = {}
        
        # Load existing cache metadata
        self._load_cache_metadata()
    
    def _load_cache_metadata(self):
        """Load cache metadata from disk"""
        metadata_file = self.cache_dir / 'cache_metadata.pkl'
        
        if metadata_file.exists():
            try:
                with open(metadata_file, 'rb') as f:
                    self.cache_metadata = pickle.load(f)
            except Exception:
                self.cache_metadata = {}
    
    def _save_cache_metadata(self):
        """Save cache metadata to disk"""
        metadata_file = self.cache_dir / 'cache_metadata.pkl'
        
        try:
            with open(metadata_file, 'wb') as f:
                pickle.dump(self.cache_metadata, f)
        except Exception:
            pass  # Fail silently
    
    def cache_data(self, key: str, data: Any, persistent: bool = True):
        """Cache data in memory and optionally on disk"""
        
        # Store in memory cache
        self.memory_cache[key] = data
        
        # Store on disk if persistent
        if persistent:
            cache_file = self.cache_dir / f'{key}.pkl'
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump(data, f)
                
                # Update metadata
                file_size = cache_file.stat().st_size
                self.cache_metadata[key] = {
                    'file_size': file_size,
                    'access_time': time.time(),
                    'access_count': self.cache_metadata.get(key, {}).get('access_count', 0) + 1,
                }
                
                self._cleanup_disk_cache()
                self._save_cache_metadata()
                
            except Exception:
                pass  # Fail silently
    
    def get_cached_data(self, key: str) -> Optional[Any]:
        """Retrieve cached data"""
        
        # Check memory cache first
        if key in self.memory_cache:
            # Move to end (LRU)
            self.memory_cache.move_to_end(key)
            
            # Update access metadata
            if key in self.cache_metadata:
                self.cache_metadata[key]['access_time'] = time.time()
                self.cache_metadata[key]['access_count'] += 1
            
            return self.memory_cache[key]
        
        # Check disk cache
        cache_file = self.cache_dir / f'{key}.pkl'
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    data = pickle.load(f)
                
                # Store in memory cache
                self.memory_cache[key] = data
                
                # Update metadata
                if key in self.cache_metadata:
                    self.cache_metadata[key]['access_time'] = time.time()
                    self.cache_metadata[key]['access_count'] += 1
                
                return data
                
            except Exception:
                # Remove corrupted cache file
                cache_file.unlink(missing_ok=True)
                if key in self.cache_metadata:
                    del self.cache_metadata[key]
        
        return None
    
    def _cleanup_disk_cache(self):
        """Clean up disk cache to stay within size limit"""
        
        total_size = sum(
            metadata['file_size'] 
            for metadata in self.cache_metadata.values()
        )
        
        if total_size > self.max_cache_size_gb * 1024**3:
            # Sort by access time (oldest first)
            sorted_items = sorted(
                self.cache_metadata.items(), key=lambda x: x[1]['access_time']
            )
            
            # Remove oldest items
            for key, metadata in sorted_items:
                cache_file = self.cache_dir / f'{key}.pkl'
                cache_file.unlink(missing_ok=True)
                del self.cache_metadata[key]
                
                total_size -= metadata['file_size']
                
                if total_size <= self.max_cache_size_gb * 1024**3 * 0.8:
                    break
    
    def clear_cache(self):
        """Clear all cached data"""
        
        # Clear memory cache
        self.memory_cache.clear()
        
        # Clear disk cache
        for cache_file in self.cache_dir.glob('*.pkl'):
            if cache_file.name != 'cache_metadata.pkl':
                cache_file.unlink()
        
        self.cache_metadata.clear()
        self._save_cache_metadata()

class StreamingDataLoader:
    """
    Streaming data loader for very large datasets that don't fit in memory
    """
    
    def __init__(self, dataset, batch_size: int = 1, buffer_size: int = 10, num_workers: int = 2):
        """
        Args:
            dataset: Dataset to stream from
            batch_size: Batch size
            buffer_size: Number of batches to buffer
            num_workers: Number of worker threads
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.num_workers = num_workers
        
        # Streaming state
        self.data_queue = queue.Queue(maxsize=buffer_size)
        self.workers = []
        self.stop_event = threading.Event()
        
        # Start workers
        self._start_workers()
    
    def _start_workers(self):
        """Start worker threads for data loading"""
        
        for i in range(self.num_workers):
            worker = threading.Thread(
                target=self._worker_loop, args=(i, ), daemon=True
            )
            worker.start()
            self.workers.append(worker)
    
    def _worker_loop(self, worker_id: int):
        """Worker loop for data loading"""
        
        indices = list(range(len(self.dataset)))
        np.random.shuffle(indices)
        
        batch_indices = []
        
        for idx in indices:
            if self.stop_event.is_set():
                break
            
            batch_indices.append(idx)
            
            if len(batch_indices) >= self.batch_size:
                # Load batch data
                try:
                    batch_data = [self.dataset[i] for i in batch_indices]
                    
                    # Put batch in queue (blocking if full)
                    if not self.stop_event.is_set():
                        self.data_queue.put(batch_data, timeout=1.0)
                    
                    batch_indices = []
                    
                except queue.Full:
                    continue
                except Exception:
                    # Skip problematic batch
                    batch_indices = []
        
        # Handle remaining data
        if batch_indices and not self.stop_event.is_set():
            try:
                batch_data = [self.dataset[i] for i in batch_indices]
                self.data_queue.put(batch_data, timeout=1.0)
            except (queue.Full, Exception):
                pass
    
    def __iter__(self):
        """Iterator interface"""
        return self
    
    def __next__(self):
        """Get next batch"""
        try:
            batch_data = self.data_queue.get(timeout=5.0)
            return self._collate_batch(batch_data)
        except queue.Empty:
            raise StopIteration
    
    def _collate_batch(self, batch_data: list[dict]) -> dict[str, torch.Tensor]:
        """Collate batch data"""
        
        if not batch_data:
            raise StopIteration
        
        # Simple collation - stack tensors
        collated = {}
        
        for key in batch_data[0].keys():
            values = [item[key] for item in batch_data]
            
            if isinstance(values[0], torch.Tensor):
                try:
                    collated[key] = torch.stack(values)
                except Exception:
                    # If stacking fails, keep as list
                    collated[key] = values
            else:
                collated[key] = values
        
        return collated
    
    def stop(self):
        """Stop streaming"""
        self.stop_event.set()
        
        # Clear queue
        while not self.data_queue.empty():
            try:
                self.data_queue.get_nowait()
            except queue.Empty:
                break
    
    def __del__(self):
        """Cleanup"""
        self.stop()

class MemoryOptimizer:
    """
    Memory optimization utilities for efficient training
    """
    
    @staticmethod
    def optimize_model_memory(
        model: nn.Module,
        use_checkpointing: bool = True,
        use_mixed_precision: bool = True,
    ) -> nn.Module:
        """Optimize model for memory efficiency"""
        
        if use_checkpointing:
            # Apply gradient checkpointing to reduce memory usage
            model = MemoryOptimizer._apply_gradient_checkpointing(model)
        
        if use_mixed_precision:
            # Convert to half precision where possible
            model = MemoryOptimizer._optimize_precision(model)
        
        return model
    
    @staticmethod
    def _apply_gradient_checkpointing(model: nn.Module) -> nn.Module:
        """Apply gradient checkpointing to model"""
        
        def make_checkpointed(module):
            if isinstance(module, nn.Sequential) and len(module) > 2:
                # Wrap sequential modules with checkpointing
                return torch.utils.checkpoint.checkpoint_sequential(
                    module, 2, preserve_rng_state=True
                )
            return module
        
        # Apply to all sequential modules
        for name, module in model.named_children():
            if isinstance(module, nn.Sequential):
                setattr(model, name, make_checkpointed(module))
        
        return model
    
    @staticmethod
    def _optimize_precision(model: nn.Module) -> nn.Module:
        """Optimize model precision for memory efficiency"""
        
        # Convert BatchNorm layers to float32 for stability
        for module in model.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                module.float()
        
        return model
    
    @staticmethod
    def chunk_tensor_processing(
        tensor: torch.Tensor,
        processing_fn: callable,
        chunk_size: int = 1024,
        dim: int = 0,
    ) -> torch.Tensor: 
        """Process large tensors in chunks to save memory"""
        
        if tensor.size(dim) <= chunk_size:
            return processing_fn(tensor)
        
        # Split tensor into chunks
        chunks = torch.chunk(tensor, chunks=max(1, tensor.size(dim) // chunk_size), dim=dim)
        
        # Process chunks
        processed_chunks = []
        for chunk in chunks:
            with torch.no_grad():
                processed_chunk = processing_fn(chunk)
            processed_chunks.append(processed_chunk)
        
        # Concatenate results
        return torch.cat(processed_chunks, dim=dim)
    
    @staticmethod
    def estimate_tensor_memory(tensor: torch.Tensor) -> float:
        """Estimate tensor memory usage in MB"""
        return tensor.numel() * tensor.element_size() / (1024**2)
    
    @staticmethod
    def get_model_memory_usage(model: nn.Module) -> dict[str, float]:
        """Get detailed memory usage of model components"""
        
        memory_usage = {}
        total_params = 0
        total_memory = 0
        
        for name, param in model.named_parameters():
            param_memory = MemoryOptimizer.estimate_tensor_memory(param)
            memory_usage[name] = param_memory
            total_params += param.numel()
            total_memory += param_memory
        
        memory_usage['total_parameters'] = total_params
        memory_usage['total_memory_mb'] = total_memory
        
        return memory_usage 