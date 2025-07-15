"""
Utilities module for fuzzy clustering algorithms.

This module contains performance optimization utilities, timing decorators,
and common helper functions used across the clustering algorithms.
"""

from .performance import (
    timing_decorator,
    enable_timing,
    compute_distances_optimized,
    update_membership_optimized,
    batch_process_large_dataset,
    optimize_memory_usage,
    adaptive_batch_size
)

from .learning import (
    AdaptiveLearningRateScheduler,
    EarlyStoppingCriterion,
    adaptive_convergence_detection
)

from .initialization import (
    smart_initialization
)

__all__ = [
    'timing_decorator',
    'enable_timing', 
    'compute_distances_optimized',
    'update_membership_optimized',
    'batch_process_large_dataset',
    'optimize_memory_usage',
    'adaptive_batch_size',
    'AdaptiveLearningRateScheduler',
    'EarlyStoppingCriterion',
    'adaptive_convergence_detection',
    'smart_initialization'
]
