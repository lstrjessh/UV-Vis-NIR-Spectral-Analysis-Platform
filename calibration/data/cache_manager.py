"""
Efficient cache management for data and models.
"""

import hashlib
import pickle
import json
import time
from pathlib import Path
from typing import Any, Optional, Dict, Tuple
from datetime import datetime, timedelta
import numpy as np


class CacheManager:
    """Manages caching for data and model results."""
    
    def __init__(self, 
                 cache_dir: Optional[Path] = None,
                 max_size_mb: float = 500,
                 ttl_hours: float = 24,
                 enable_memory_cache: bool = True,
                 enable_disk_cache: bool = False):
        """
        Initialize cache manager.
        
        Args:
            cache_dir: Directory for disk cache
            max_size_mb: Maximum cache size in MB
            ttl_hours: Time-to-live for cache entries in hours
            enable_memory_cache: Enable in-memory caching
            enable_disk_cache: Enable disk caching
        """
        self.cache_dir = cache_dir or Path.home() / '.calibration_cache'
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.ttl = timedelta(hours=ttl_hours)
        self.enable_memory_cache = enable_memory_cache
        self.enable_disk_cache = enable_disk_cache
        
        # Initialize caches
        self._memory_cache: Dict[str, Tuple[Any, datetime]] = {}
        self._cache_sizes: Dict[str, int] = {}
        self._total_size = 0
        
        # Create cache directory if needed
        if self.enable_disk_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._clean_old_cache()
    
    def get(self, key: str) -> Optional[Any]:
        """
        Retrieve item from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found/expired
        """
        # Check memory cache first
        if self.enable_memory_cache and key in self._memory_cache:
            value, timestamp = self._memory_cache[key]
            if datetime.now() - timestamp < self.ttl:
                return value
            else:
                # Remove expired entry
                del self._memory_cache[key]
                if key in self._cache_sizes:
                    self._total_size -= self._cache_sizes[key]
                    del self._cache_sizes[key]
        
        # Check disk cache
        if self.enable_disk_cache:
            cache_file = self.cache_dir / f"{key}.pkl"
            if cache_file.exists():
                # Check age
                file_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
                if file_age < self.ttl:
                    try:
                        with open(cache_file, 'rb') as f:
                            value = pickle.load(f)
                        
                        # Add to memory cache if enabled
                        if self.enable_memory_cache:
                            self._add_to_memory_cache(key, value)
                        
                        return value
                    except Exception:
                        # Remove corrupted cache file
                        cache_file.unlink(missing_ok=True)
                else:
                    # Remove expired file
                    cache_file.unlink(missing_ok=True)
        
        return None
    
    def put(self, key: str, value: Any) -> bool:
        """
        Store item in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            
        Returns:
            True if successfully cached
        """
        try:
            # Add to memory cache
            if self.enable_memory_cache:
                self._add_to_memory_cache(key, value)
            
            # Add to disk cache
            if self.enable_disk_cache:
                cache_file = self.cache_dir / f"{key}.pkl"
                with open(cache_file, 'wb') as f:
                    pickle.dump(value, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            return True
            
        except Exception:
            return False
    
    def invalidate(self, key: Optional[str] = None, pattern: Optional[str] = None):
        """
        Invalidate cache entries.
        
        Args:
            key: Specific key to invalidate
            pattern: Pattern to match keys for invalidation
        """
        if key:
            # Remove specific key
            if key in self._memory_cache:
                del self._memory_cache[key]
                if key in self._cache_sizes:
                    self._total_size -= self._cache_sizes[key]
                    del self._cache_sizes[key]
            
            if self.enable_disk_cache:
                cache_file = self.cache_dir / f"{key}.pkl"
                cache_file.unlink(missing_ok=True)
        
        elif pattern:
            # Remove keys matching pattern
            keys_to_remove = [k for k in self._memory_cache.keys() if pattern in k]
            for k in keys_to_remove:
                self.invalidate(key=k)
    
    def clear(self):
        """Clear all cache."""
        self._memory_cache.clear()
        self._cache_sizes.clear()
        self._total_size = 0
        
        if self.enable_disk_cache:
            for cache_file in self.cache_dir.glob("*.pkl"):
                cache_file.unlink(missing_ok=True)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        stats = {
            'memory_entries': len(self._memory_cache),
            'memory_size_mb': self._total_size / (1024 * 1024),
            'max_size_mb': self.max_size_bytes / (1024 * 1024),
            'ttl_hours': self.ttl.total_seconds() / 3600
        }
        
        if self.enable_disk_cache:
            cache_files = list(self.cache_dir.glob("*.pkl"))
            disk_size = sum(f.stat().st_size for f in cache_files)
            stats['disk_entries'] = len(cache_files)
            stats['disk_size_mb'] = disk_size / (1024 * 1024)
        
        return stats
    
    @staticmethod
    def create_key(*args, **kwargs) -> str:
        """
        Create cache key from arguments.
        
        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Hash key string
        """
        # Convert arguments to string representation
        key_parts = []
        
        for arg in args:
            if isinstance(arg, np.ndarray):
                key_parts.append(f"array_{arg.shape}_{arg.dtype}_{hash(arg.tobytes())}")
            elif isinstance(arg, (list, tuple)):
                key_parts.append(f"seq_{type(arg).__name__}_{len(arg)}_{hash(tuple(arg))}")
            elif isinstance(arg, dict):
                key_parts.append(f"dict_{len(arg)}_{hash(tuple(sorted(arg.items())))}")
            else:
                key_parts.append(str(arg))
        
        for k, v in sorted(kwargs.items()):
            key_parts.append(f"{k}={v}")
        
        key_string = "_".join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _add_to_memory_cache(self, key: str, value: Any):
        """Add item to memory cache with size management."""
        # Estimate size
        try:
            value_bytes = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
            size = len(value_bytes)
        except:
            size = 1024  # Default size if serialization fails
        
        # Check if we need to evict items
        while self._total_size + size > self.max_size_bytes and self._memory_cache:
            # Evict oldest item (FIFO)
            oldest_key = next(iter(self._memory_cache))
            del self._memory_cache[oldest_key]
            if oldest_key in self._cache_sizes:
                self._total_size -= self._cache_sizes[oldest_key]
                del self._cache_sizes[oldest_key]
        
        # Add new item
        self._memory_cache[key] = (value, datetime.now())
        self._cache_sizes[key] = size
        self._total_size += size
    
    def _clean_old_cache(self):
        """Remove old cache files from disk."""
        if not self.enable_disk_cache:
            return
        
        now = datetime.now()
        for cache_file in self.cache_dir.glob("*.pkl"):
            file_age = now - datetime.fromtimestamp(cache_file.stat().st_mtime)
            if file_age > self.ttl:
                cache_file.unlink(missing_ok=True)
    
    def cache_function(self, func):
        """
        Decorator for caching function results.
        
        Usage:
            @cache_manager.cache_function
            def expensive_function(x, y):
                return complex_computation(x, y)
        """
        def wrapper(*args, **kwargs):
            # Create cache key
            cache_key = f"{func.__name__}_{self.create_key(*args, **kwargs)}"
            
            # Try to get from cache
            result = self.get(cache_key)
            if result is not None:
                return result
            
            # Compute and cache
            result = func(*args, **kwargs)
            self.put(cache_key, result)
            
            return result
        
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        return wrapper


# Global cache instance
_global_cache = None

def get_global_cache() -> CacheManager:
    """Get or create global cache instance."""
    global _global_cache
    if _global_cache is None:
        _global_cache = CacheManager(
            enable_memory_cache=True,
            enable_disk_cache=False,
            max_size_mb=200,
            ttl_hours=24
        )
    return _global_cache
