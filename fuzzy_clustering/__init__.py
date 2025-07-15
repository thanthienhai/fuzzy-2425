"""
Fuzzy Clustering Algorithms Package

A comprehensive package for fuzzy clustering algorithms including:
- Deep Embedded K-Means (DEKM)
- Possibilistic Fuzzy C-Means (PFCM)
- Fuzzy C-Means (FCM)
- Collaborative Fuzzy C-Means (CFCM)
- Fuzzy Deep Embedded K-Means (FDEKM)
- Traditional algorithms (K-Means, DBSCAN)

The package is organized into modules for:
- Data loading and preprocessing
- Clustering algorithms
- Neural network models
- Evaluation metrics
- Utilities and performance optimization
"""

__version__ = "1.0.0"
__author__ = "Fuzzy Clustering Team"

# Import main components for easy access
try:
    from . import algorithms
    from . import data
    from . import metrics
    from . import models
    from . import utils
except ImportError:
    # Handle relative import issues
    pass
