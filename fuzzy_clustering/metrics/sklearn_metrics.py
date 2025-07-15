"""
Standard sklearn-based evaluation metrics for clustering algorithms.

This module contains functions to calculate standard clustering metrics
like silhouette score, Calinski-Harabasz index, and Davies-Bouldin index.
"""

import numpy as np
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from ..utils.performance import timing_decorator


@timing_decorator
def calculate_sklearn_metrics(X, labels):
    """
    Enhanced sklearn metrics calculation with robust error handling.
    
    Args:
        X: Data matrix (n_samples, n_features)
        labels: Cluster labels for each sample
        
    Returns:
        dict: Dictionary containing calculated metrics
    """
    metrics = {}

    # Handle edge cases
    if X is None or X.shape[0] == 0 or len(labels) == 0:
        return {
            'silhouette': np.nan,
            'calinski_harabasz': np.nan,
            'davies_bouldin': np.nan,
            'n_clusters': 0,
            'n_samples': 0
        }

    # Ensure labels is numpy array
    if not isinstance(labels, np.ndarray):
        labels = np.array(labels)

    # Filter out noise points (label -1 in DBSCAN)
    valid_mask = labels != -1
    if not np.any(valid_mask):
        return {
            'silhouette': np.nan,
            'calinski_harabasz': np.nan,
            'davies_bouldin': np.nan,
            'n_clusters': 0,
            'n_samples': len(labels),
            'n_noise': np.sum(labels == -1)
        }

    X_valid = X[valid_mask]
    labels_valid = labels[valid_mask]

    # Check if we have enough samples and clusters
    unique_labels = np.unique(labels_valid)
    n_clusters = len(unique_labels)
    n_samples = len(labels_valid)

    metrics['n_clusters'] = n_clusters
    metrics['n_samples'] = n_samples
    if -1 in labels:
        metrics['n_noise'] = np.sum(labels == -1)

    # Need at least 2 clusters and 2 samples for meaningful metrics
    if n_clusters < 2 or n_samples < 2:
        metrics.update({
            'silhouette': np.nan,
            'calinski_harabasz': np.nan,
            'davies_bouldin': np.nan
        })
        return metrics

    # Silhouette Score
    try:
        if n_samples > n_clusters and X_valid.shape[1] > 0:
            silhouette = silhouette_score(X_valid, labels_valid)
            metrics['silhouette'] = silhouette
        else:
            metrics['silhouette'] = np.nan
    except Exception as e:
        print(f"Warning: Silhouette score calculation failed: {e}")
        metrics['silhouette'] = np.nan

    # Calinski-Harabasz Index
    try:
        if n_samples > n_clusters and X_valid.shape[1] > 0:
            ch_score = calinski_harabasz_score(X_valid, labels_valid)
            metrics['calinski_harabasz'] = ch_score
        else:
            metrics['calinski_harabasz'] = np.nan
    except Exception as e:
        print(f"Warning: Calinski-Harabasz score calculation failed: {e}")
        metrics['calinski_harabasz'] = np.nan

    # Davies-Bouldin Index
    try:
        if n_samples > n_clusters and X_valid.shape[1] > 0:
            db_score = davies_bouldin_score(X_valid, labels_valid)
            metrics['davies_bouldin'] = db_score
        else:
            metrics['davies_bouldin'] = np.nan
    except Exception as e:
        print(f"Warning: Davies-Bouldin score calculation failed: {e}")
        metrics['davies_bouldin'] = np.nan

    # Additional cluster statistics
    try:
        cluster_sizes = calculate_cluster_sizes(labels_valid)
        metrics['cluster_sizes'] = cluster_sizes
        metrics['min_cluster_size'] = min(cluster_sizes.values()) if cluster_sizes else 0
        metrics['max_cluster_size'] = max(cluster_sizes.values()) if cluster_sizes else 0
        metrics['avg_cluster_size'] = np.mean(list(cluster_sizes.values())) if cluster_sizes else 0
    except Exception:
        pass

    return metrics


def calculate_cluster_sizes(labels):
    """
    Calculate sizes of each cluster.
    
    Args:
        labels: Cluster labels for each sample
        
    Returns:
        dict: Dictionary mapping cluster labels to their sizes
    """
    unique_labels, counts = np.unique(labels, return_counts=True)
    return dict(zip(unique_labels.tolist(), counts.tolist()))
