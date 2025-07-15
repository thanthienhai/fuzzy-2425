"""
Data loading and preprocessing module for fuzzy clustering algorithms.

This module contains functions for loading various datasets (USPS, e-commerce, country data)
and preprocessing functions for different clustering algorithms.
"""

from .loaders import (
    load_usps_data,
    load_ecommerce_data,
    load_country_data
)

from .preprocessing import (
    preprocess_usps_for_dekm,
    preprocess_usps_for_pfcm
)

__all__ = [
    'load_usps_data',
    'load_ecommerce_data', 
    'load_country_data',
    'preprocess_usps_for_dekm',
    'preprocess_usps_for_pfcm'
]
