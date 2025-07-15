"""
Custom fuzzy clustering evaluation metrics.

This module contains specialized metrics for evaluating fuzzy clustering
algorithms, including PCI, FHV, and XBI indices.
"""

import numpy as np
from ..utils.performance import timing_decorator


def pci_index(X, U):
    """
    Partition Coefficient Index (PCI)
    Measures the fuzziness of the clustering
    Range: [1/c, 1] where c is number of clusters
    Higher values indicate better clustering
    """
    if U is None or U.shape[0] == 0 or U.shape[1] == 0:
        return np.nan
    return np.mean(np.sum(U**2, axis=1))


def fhv_index(X, U, V, m):
    """
    Fuzzy Hypervolume (FHV)
    Measures the volume of fuzzy clusters
    Lower values indicate better clustering
    """
    if X is None or U is None or V is None or \
       X.shape[0] == 0 or U.shape[0] == 0 or V.shape[0] == 0:
        return np.nan
    
    n_clusters = V.shape[0]
    fhv = 0
    
    for j in range(n_clusters):
        # Calculate fuzzy covariance matrix for cluster j
        diff = X - V[j]
        weighted_diff = (U[:, j]**m).reshape(-1, 1) * diff
        cov_j = np.dot(weighted_diff.T, diff) / np.sum(U[:, j]**m)
        
        # Calculate determinant of covariance matrix
        try:
            det = np.linalg.det(cov_j)
            if det > 0:  # Only add if determinant is positive
                fhv += np.sqrt(det)
        except np.linalg.LinAlgError:
            continue
    
    return fhv


def xbi_index(X, U, V, m):
    """
    Xie-Beni Index (XBI)
    Measures the ratio of compactness to separation
    Lower values indicate better clustering
    """
    if X is None or U is None or V is None or \
       X.shape[0] == 0 or U.shape[0] == 0 or V.shape[0] == 0:
        return np.nan
    
    n_clusters = V.shape[0]
    
    # Calculate compactness (numerator)
    compactness = 0
    for j in range(n_clusters):
        diff = X - V[j]
        weighted_dist = np.sum((U[:, j]**m).reshape(-1, 1) * (diff**2))
        compactness += weighted_dist
    
    # Calculate minimum separation between cluster centers (denominator)
    min_sep = float('inf')
    for i in range(n_clusters):
        for j in range(i+1, n_clusters):
            sep = np.sum((V[i] - V[j])**2)
            min_sep = min(min_sep, sep)
    
    if min_sep == 0 or min_sep == float('inf'):
        return np.nan
        
    return compactness / (X.shape[0] * min_sep)


@timing_decorator
def calculate_custom_metrics(X, U, V, m):
    """
    Enhanced custom fuzzy clustering metrics with robust error handling.
    
    Args:
        X: Data matrix (n_samples, n_features)
        U: Membership matrix (n_samples, n_clusters)
        V: Cluster centers (n_clusters, n_features)
        m: Fuzzification parameter
        
    Returns:
        dict: Dictionary containing calculated custom metrics
    """
    metrics = {}

    # Handle edge cases
    if X is None or U is None or V is None:
        return {
            'pci': np.nan,
            'fhv': np.nan,
            'xbi': np.nan
        }

    if X.shape[0] == 0 or U.shape[0] == 0 or V.shape[0] == 0:
        return {
            'pci': np.nan,
            'fhv': np.nan,
            'xbi': np.nan
        }

    # Ensure arrays are numpy arrays
    if not isinstance(X, np.ndarray):
        X = np.array(X)
    if not isinstance(U, np.ndarray):
        U = np.array(U)
    if not isinstance(V, np.ndarray):
        V = np.array(V)

    # Check dimensions compatibility
    if X.shape[0] != U.shape[0]:
        print(f"Warning: X samples ({X.shape[0]}) != U samples ({U.shape[0]})")
        return {
            'pci': np.nan,
            'fhv': np.nan,
            'xbi': np.nan
        }

    if U.shape[1] != V.shape[0]:
        print(f"Warning: U clusters ({U.shape[1]}) != V clusters ({V.shape[0]})")
        return {
            'pci': np.nan,
            'fhv': np.nan,
            'xbi': np.nan
        }

    if X.shape[1] != V.shape[1]:
        print(f"Warning: X features ({X.shape[1]}) != V features ({V.shape[1]})")
        return {
            'pci': np.nan,
            'fhv': np.nan,
            'xbi': np.nan
        }

    # Calculate PCI (Partition Coefficient Index)
    try:
        metrics['pci'] = pci_index(X, U)
    except Exception as e:
        print(f"Warning: PCI calculation failed: {e}")
        metrics['pci'] = np.nan

    # Calculate FHV (Fuzzy Hypervolume)
    try:
        metrics['fhv'] = fhv_index(X, U, V, m)
    except Exception as e:
        print(f"Warning: FHV calculation failed: {e}")
        metrics['fhv'] = np.nan

    # Calculate XBI (Xie-Beni Index)
    try:
        metrics['xbi'] = xbi_index(X, U, V, m)
    except Exception as e:
        print(f"Warning: XBI calculation failed: {e}")
        metrics['xbi'] = np.nan

    return metrics
