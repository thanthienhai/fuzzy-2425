"""
K-Means clustering algorithm implementation.

This module contains the implementation of the standard K-Means
clustering algorithm using scikit-learn.
"""

import numpy as np
from sklearn.cluster import KMeans
from ..metrics.sklearn_metrics import calculate_sklearn_metrics
from ..metrics.fuzzy_metrics import calculate_custom_metrics


def run_kmeans_standalone(X_data, n_clusters, random_state=42, verbose=False):
    """
    Run K-Means clustering algorithm.
    
    Args:
        X_data: Input data (n_samples, n_features)
        n_clusters: Number of clusters
        random_state: Random seed for reproducibility
        verbose: Whether to print progress information
        
    Returns:
        tuple: (labels, cluster_centers, metrics) where:
            - labels: Cluster assignments for each sample
            - cluster_centers: Coordinates of cluster centers
            - metrics: Dictionary of evaluation metrics
    """
    if X_data is None or X_data.shape[0] == 0:
        print("Error: K-Means input data is empty.")
        return np.array([]), np.array([[]]), {}
    
    if X_data.shape[0] < n_clusters:
        print(f"Warning: K-Means n_samples ({X_data.shape[0]}) < n_clusters ({n_clusters}). Reducing n_clusters.")
        n_clusters = max(1, X_data.shape[0])
    if n_clusters == 0:
        n_clusters = 1

    # Initialize and fit K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init='auto')
    try:
        labels = kmeans.fit_predict(X_data)
    except Exception as e:
        print(f"Error during K-Means fit_predict: {e}")
        return np.array([]), np.array([[]]), {"error": str(e)}

    # Calculate standard evaluation metrics
    metrics = calculate_sklearn_metrics(X_data, labels)
    
    # Prepare membership matrix and centers for custom metrics
    if X_data.shape[0] > 1 and len(np.unique(labels)) > 1:
        num_actual_clusters = len(np.unique(labels))
        if num_actual_clusters > 0 and X_data.shape[0] >= num_actual_clusters:
            # Create hard membership matrix (one-hot encoding)
            U_kmeans = np.zeros((X_data.shape[0], num_actual_clusters))
            
            # Map original labels to 0 to num_actual_clusters-1 for U matrix indexing
            unique_labels_arr_kmeans = np.unique(labels)
            label_to_idx_kmeans = {label: i for i, label in enumerate(unique_labels_arr_kmeans)}
            for i, label in enumerate(labels):
                U_kmeans[i, label_to_idx_kmeans[label]] = 1.0
            
            V_kmeans = kmeans.cluster_centers_
            
            # Ensure cluster centers have matching feature dimensions
            if V_kmeans.shape[1] == X_data.shape[1]:
                metrics.update(calculate_custom_metrics(X_data, U_kmeans, V_kmeans, m=2.0))
            else:
                print(f"KMeans: Centroid feature size {V_kmeans.shape[1]} != Data feature size {X_data.shape[1]}. Skipping custom metrics.")

    # Add K-Means specific metrics
    metrics['inertia'] = kmeans.inertia_
    metrics['n_iter'] = kmeans.n_iter_
    
    if verbose:
        print(f"K-Means completed. Found {len(np.unique(labels))} clusters.")
    
    return labels, kmeans.cluster_centers_, metrics
