"""
Smart initialization strategies for clustering algorithms.

This module contains various initialization methods for cluster centers
and membership matrices to improve convergence and clustering quality.
"""

import numpy as np
from sklearn.cluster import KMeans


def smart_initialization(data, n_clusters, method='kmeans++', random_state=42):
    """
    Smart initialization strategies for clustering algorithms

    Args:
        data: Input data
        n_clusters: Number of clusters
        method: Initialization method ('kmeans++', 'random', 'farthest_first')
        random_state: Random seed
    """
    np.random.seed(random_state)
    n_samples, n_features = data.shape

    if method == 'kmeans++':
        # K-means++ initialization
        try:
            kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=1, random_state=random_state)
            kmeans.fit(data)
            return kmeans.cluster_centers_
        except:
            method = 'farthest_first'  # Fallback

    if method == 'farthest_first':
        # Farthest first initialization
        centers = np.zeros((n_clusters, n_features))

        # Choose first center randomly
        centers[0] = data[np.random.randint(n_samples)]

        # Choose remaining centers
        for i in range(1, n_clusters):
            distances = np.min([np.sum((data - center)**2, axis=1) for center in centers[:i]], axis=0)
            probabilities = distances / np.sum(distances)
            cumulative_probs = np.cumsum(probabilities)
            r = np.random.random()
            next_center_idx = np.searchsorted(cumulative_probs, r)
            centers[i] = data[next_center_idx]

        return centers

    elif method == 'random':
        # Random initialization
        indices = np.random.choice(n_samples, n_clusters, replace=False)
        return data[indices]

    else:
        raise ValueError(f"Unknown initialization method: {method}")
