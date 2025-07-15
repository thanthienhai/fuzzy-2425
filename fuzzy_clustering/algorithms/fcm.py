"""
Fuzzy C-Means (FCM) clustering algorithm implementation.

This module contains the implementation of the standard Fuzzy C-Means
clustering algorithm using the scikit-fuzzy library.
"""

import numpy as np
import skfuzzy as fuzz
from ..metrics.sklearn_metrics import calculate_sklearn_metrics
from ..metrics.fuzzy_metrics import calculate_custom_metrics


def run_fcm(X_scaled, n_clusters=3, m=2.0, error=0.005, maxiter=100, verbose=False):
    """
    Run Fuzzy C-Means (FCM) clustering algorithm.
    
    Args:
        X_scaled: Scaled input data (n_samples, n_features)
        n_clusters: Number of clusters
        m: Fuzzification parameter (m > 1)
        error: Convergence tolerance
        maxiter: Maximum number of iterations
        verbose: Whether to print progress information
        
    Returns:
        tuple: (labels, U_transposed, cntr, metrics) where:
            - labels: Hard cluster assignments
            - U_transposed: Membership matrix (n_samples, n_clusters)
            - cntr: Cluster centers
            - metrics: Dictionary of evaluation metrics
    """
    if X_scaled is None or X_scaled.shape[0] == 0:
        print("Error: FCM input data is empty.")
        return np.array([]), np.array([[]]), np.array([[]]), {}
    
    if X_scaled.shape[0] < n_clusters:
        print(f"Warning: FCM n_samples ({X_scaled.shape[0]}) < n_clusters ({n_clusters}). Reducing n_clusters.")
        n_clusters = max(1, X_scaled.shape[0])
    if n_clusters == 0:
        n_clusters = 1

    try:
        # skfuzzy.cmeans expects data to be (features, samples)
        cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
            X_scaled.T, c=n_clusters, m=m, error=error, maxiter=maxiter, init=None, seed=42
        )
    except Exception as e:
        print(f"Error during FCM cmeans: {e}")
        return np.array([]), np.array([[]]), np.array([[]]), {"error": str(e)}

    # Get hard cluster assignments
    labels = np.argmax(u, axis=0)
    
    # Transpose membership matrix to match our convention (n_samples, n_clusters)
    U_transposed = u.T 
    
    # Calculate evaluation metrics
    metrics = calculate_sklearn_metrics(X_scaled, labels)
    
    # Calculate custom fuzzy metrics if data is valid
    if X_scaled.shape[0] > 1 and cntr.shape[0] > 0 and U_transposed.shape[1] > 0:
        metrics.update(calculate_custom_metrics(X_scaled, U_transposed, cntr, m=m))
    
    # Add FCM-specific metrics
    metrics['fpc'] = fpc  # Fuzzy Partition Coefficient
    metrics['final_objective'] = jm[-1] if len(jm) > 0 else np.nan
    metrics['n_iterations'] = len(jm)
    
    if verbose:
        print(f"FCM completed. FPC: {fpc:.4f}")
    
    return labels, U_transposed, cntr, metrics
