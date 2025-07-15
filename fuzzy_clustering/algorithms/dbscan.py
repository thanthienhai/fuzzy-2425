"""
DBSCAN clustering algorithm implementation.

This module contains the implementation of the Density-Based Spatial
Clustering of Applications with Noise (DBSCAN) algorithm using scikit-learn.
"""

import numpy as np
from sklearn.cluster import DBSCAN
from ..metrics.sklearn_metrics import calculate_sklearn_metrics


def run_dbscan(X_data, eps=0.5, min_samples=5, verbose=False):
    """
    Run DBSCAN clustering algorithm.
    
    Args:
        X_data: Input data (n_samples, n_features)
        eps: Maximum distance between two samples for one to be considered
             as in the neighborhood of the other
        min_samples: Number of samples in a neighborhood for a point to be
                    considered as a core point
        verbose: Whether to print progress information
        
    Returns:
        tuple: (labels, metrics) where:
            - labels: Cluster assignments for each sample (-1 for noise)
            - metrics: Dictionary of evaluation metrics
    """
    if X_data is None or X_data.shape[0] == 0:
        print("Error: DBSCAN input data is empty.")
        return np.array([]), {}

    # Initialize and fit DBSCAN
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    try:
        labels = dbscan.fit_predict(X_data)
    except Exception as e:
        print(f"Error during DBSCAN fit_predict: {e}")
        return np.array([]), {"error": str(e)}

    # Calculate standard evaluation metrics
    metrics = calculate_sklearn_metrics(X_data, labels)
    
    # DBSCAN-specific metrics
    n_clusters_found = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    
    metrics['n_clusters_found'] = n_clusters_found
    metrics['n_noise_points'] = n_noise
    metrics['noise_ratio'] = n_noise / len(labels) if len(labels) > 0 else 0
    
    # Fuzzy metrics are not typically applicable to DBSCAN
    metrics['pci'] = np.nan
    metrics['fhv'] = np.nan
    metrics['xbi'] = np.nan
        
    if verbose:
        print(f"DBSCAN completed. Found {n_clusters_found} clusters (excluding noise).")
        print(f"Noise points: {n_noise} ({metrics['noise_ratio']:.2%})")
    
    return labels, metrics
