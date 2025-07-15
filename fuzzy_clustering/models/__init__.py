"""
Neural network models module for fuzzy clustering algorithms.

This module contains neural network architectures used in deep clustering
algorithms, particularly autoencoders for dimensionality reduction and
feature learning.
"""

from .autoencoder import (
    AutoEncoder_DEKM,
    train_autoencoder_dekm
)

__all__ = [
    'AutoEncoder_DEKM',
    'train_autoencoder_dekm'
]
