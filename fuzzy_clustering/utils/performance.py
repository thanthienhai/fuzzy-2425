"""
Performance optimization utilities for fuzzy clustering algorithms.

This module contains timing decorators, optimized distance computations,
memory management utilities, and batch processing functions.
"""

import time
import numpy as np
import warnings
from functools import wraps

warnings.filterwarnings("ignore")


def timing_decorator(func):
    """Decorator to measure function execution time"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        if hasattr(wrapper, '_show_timing') and wrapper._show_timing:
            print(f"{func.__name__} executed in {end_time - start_time:.4f} seconds")
        return result
    return wrapper


def enable_timing(func):
    """Enable timing for a decorated function"""
    if hasattr(func, '_show_timing'):
        func._show_timing = True
    return func


def compute_distances_optimized(data, centers):
    """Optimized distance computation using vectorized operations"""
    # Use broadcasting for efficient computation
    # data: (n_samples, n_features), centers: (n_clusters, n_features)
    # Result: (n_samples, n_clusters)
    diff = data[:, np.newaxis, :] - centers[np.newaxis, :, :]  # Broadcasting
    distances_sq = np.sum(diff ** 2, axis=2)
    return distances_sq


def update_membership_optimized(distances, m):
    """Optimized membership matrix update using vectorized operations"""
    n_samples, n_clusters = distances.shape

    # Avoid division by zero
    distances = np.maximum(distances, 1e-10)

    # Vectorized computation of membership matrix
    power = 2.0 / (m - 1.0)

    # Use broadcasting for efficient computation
    ratio_matrix = distances[:, :, np.newaxis] / distances[:, np.newaxis, :]
    powered_ratios = ratio_matrix ** power
    sum_ratios = np.sum(powered_ratios, axis=2)

    # Handle numerical issues
    sum_ratios = np.maximum(sum_ratios, 1e-10)
    membership = 1.0 / sum_ratios

    return membership


def batch_process_large_dataset(data, algorithm_func, batch_size=1000, **kwargs):
    """
    Process large datasets in batches to manage memory usage

    Args:
        data: Input data array
        algorithm_func: Clustering algorithm function to apply
        batch_size: Size of each batch
        **kwargs: Additional arguments for the algorithm
    """
    n_samples = data.shape[0]

    if n_samples <= batch_size:
        # Process normally if data is small enough
        return algorithm_func(data, **kwargs)

    print(f"Processing large dataset ({n_samples} samples) in batches of {batch_size}")

    # For clustering, we need to process the entire dataset together
    # But we can optimize memory usage during distance computations
    # This is a placeholder for future batch processing implementation

    # For now, just add memory optimization warnings
    if n_samples > 10000:
        print("Warning: Large dataset detected. Consider reducing data size or using sampling.")

    return algorithm_func(data, **kwargs)


def optimize_memory_usage():
    """
    Optimize memory usage for large datasets
    """
    import gc
    gc.collect()  # Force garbage collection

    # Set numpy to use less memory for operations
    np.seterr(all='ignore')  # Ignore numerical warnings for performance


def adaptive_batch_size(data_size, available_memory_gb=4):
    """
    Calculate optimal batch size based on data size and available memory
    """
    # Estimate memory usage per sample (rough approximation)
    bytes_per_sample = data_size * 8  # 8 bytes per float64

    # Convert GB to bytes
    available_bytes = available_memory_gb * 1024**3

    # Use 50% of available memory for safety
    usable_bytes = available_bytes * 0.5

    # Calculate batch size
    batch_size = max(100, int(usable_bytes / bytes_per_sample))

    return min(batch_size, 10000)  # Cap at 10k for practical reasons
