"""
Possibilistic Fuzzy C-Means (PFCM) clustering algorithm implementation.

This module contains the implementation of the Possibilistic Fuzzy C-Means
algorithm which combines both fuzzy and possibilistic clustering approaches.
"""

import numpy as np
from sklearn.cluster import KMeans
from ..utils.performance import timing_decorator, compute_distances_optimized, update_membership_optimized
from ..metrics.sklearn_metrics import calculate_sklearn_metrics
from ..metrics.fuzzy_metrics import calculate_custom_metrics


def initialize_membership_matrix_pfcm(n_samples, n_clusters):
    """
    Initialize membership matrix for PFCM with random values.
    
    Args:
        n_samples: Number of data samples
        n_clusters: Number of clusters
        
    Returns:
        numpy.ndarray: Normalized membership matrix (n_samples, n_clusters)
    """
    U = np.random.rand(n_samples, n_clusters)
    # Normalize, handle sum U is zero for a row if all random numbers were 0 (highly unlikely)
    row_sums = np.sum(U, axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    return U / row_sums


def calculate_centroids_pfcm(X, U, T, m, eta):
    """
    Calculate cluster centroids for PFCM algorithm.
    
    Args:
        X: Data matrix (n_samples, n_features)
        U: Membership matrix (n_samples, n_clusters)
        T: Typicality matrix (n_samples, n_clusters)
        m: Fuzzification parameter for membership
        eta: Fuzzification parameter for typicality
        
    Returns:
        numpy.ndarray: Cluster centroids (n_clusters, n_features)
    """
    if X.shape[0] == 0 or U.shape[0] == 0 or T.shape[0] == 0:
        return np.array([])
    
    # Add epsilon to U**m and T**m to avoid issues if U or T are exactly 0 for all samples in a cluster
    # This can happen if a cluster becomes empty.
    Um_plus_eta_Tm = (U**m + 1e-9) + eta * (T**m + 1e-9)
    num = Um_plus_eta_Tm.T @ X
    denom = np.sum(Um_plus_eta_Tm, axis=0).reshape(-1, 1)
    denom[denom == 0] = 1e-9  # Avoid division by zero
    return num / denom


@timing_decorator
def update_U_T_pfcm(X, C, m, eta, gamma=None):
    """
    Optimized PFCM membership and typicality matrix updates using vectorized operations.

    Args:
        X: Data points (n_samples, n_features)
        C: Cluster centers (n_clusters, n_features)
        m: Fuzzifier for membership (m > 1)
        eta: Fuzzifier for typicality (eta > 1)
        gamma: Typicality parameter (auto-computed if None)
        
    Returns:
        tuple: (U_new, T_new) updated membership and typicality matrices
    """
    if X.shape[0] == 0 or C.shape[0] == 0:
        return np.array([]), np.array([])

    n_samples, n_features = X.shape
    n_clusters = C.shape[0]

    # Use optimized distance computation
    dist_sq = compute_distances_optimized(X, C)
    dist_sq = np.maximum(dist_sq, 1e-10)

    # Optimized membership matrix update using vectorized operations
    U_new = update_membership_optimized(dist_sq, m)

    # Compute gamma parameter for typicality (vectorized)
    if gamma is None:
        U_m = U_new ** m
        numerator = np.sum(U_m * dist_sq, axis=0)
        denominator = np.sum(U_m, axis=0)
        gamma = numerator / np.maximum(denominator, 1e-10)
        gamma = np.maximum(gamma, 1e-10)
    elif np.isscalar(gamma):
        gamma = np.full(n_clusters, gamma)

    # Optimized typicality matrix update using broadcasting
    # T[i,j] = 1 / (1 + (dist_sq[i,j] / gamma[j]) ** (1/(eta-1)))
    power_term = 1.0 / (eta - 1.0)
    ratio_matrix = dist_sq / gamma[np.newaxis, :]  # Broadcasting
    powered_ratios = ratio_matrix ** power_term
    T_new = 1.0 / (1.0 + powered_ratios)

    return U_new, T_new


def compute_pfcm_objective(X, U, T, C, m, eta):
    """
    Compute PFCM objective function.
    
    Args:
        X: Data matrix (n_samples, n_features)
        U: Membership matrix (n_samples, n_clusters)
        T: Typicality matrix (n_samples, n_clusters)
        C: Cluster centers (n_clusters, n_features)
        m: Fuzzification parameter for membership
        eta: Fuzzification parameter for typicality
        
    Returns:
        float: PFCM objective function value
    """
    n_samples, n_clusters = U.shape
    objective = 0.0

    # Membership term
    for i in range(n_samples):
        for j in range(n_clusters):
            distance_sq = np.sum((X[i] - C[j]) ** 2)
            objective += (U[i, j] ** m) * distance_sq

    # Typicality term
    for i in range(n_samples):
        for j in range(n_clusters):
            distance_sq = np.sum((X[i] - C[j]) ** 2)
            objective += (T[i, j] ** eta) * distance_sq

    # Regularization terms
    for i in range(n_samples):
        for j in range(n_clusters):
            if T[i, j] > 0:
                objective += T[i, j] * np.log(T[i, j])

    return objective


def run_pfcm(X_data, n_clusters=10, m=2.0, eta=2.0, max_iter=100, tol=1e-6, verbose=False):
    """
    Run Possibilistic Fuzzy C-Means (PFCM) clustering algorithm.

    Args:
        X_data: Input data (n_samples, n_features)
        n_clusters: Number of clusters
        m: Fuzzifier for membership (m > 1)
        eta: Fuzzifier for typicality (eta > 1)
        max_iter: Maximum iterations
        tol: Convergence tolerance
        verbose: Print progress

    Returns:
        tuple: (labels, U, T, C, metrics) where:
            - labels: Hard cluster assignments
            - U: Membership matrix (n_samples, n_clusters)
            - T: Typicality matrix (n_samples, n_clusters)
            - C: Cluster centers (n_clusters, n_features)
            - metrics: Dictionary of evaluation metrics
    """
    if X_data is None or X_data.shape[0] == 0:
        print("Error: PFCM input data is empty.")
        return np.array([]), np.array([[]]), np.array([[]]), np.array([[]]), {}

    if X_data.shape[0] < n_clusters:
        print(f"Warning: PFCM n_samples ({X_data.shape[0]}) < n_clusters ({n_clusters}). Reducing n_clusters.")
        n_clusters = max(1, X_data.shape[0])
    if n_clusters == 0:
        n_clusters = 1

    n_samples, n_features = X_data.shape

    # Better initialization using K-means++
    try:
        kmeans_init = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10, random_state=42)
        kmeans_init.fit(X_data)
        C = kmeans_init.cluster_centers_

        # Initialize membership matrix based on distances to initial centroids
        U = initialize_membership_matrix_pfcm(n_samples, n_clusters)
        distances = np.zeros((n_samples, n_clusters))
        for j in range(n_clusters):
            distances[:, j] = np.sum((X_data - C[j]) ** 2, axis=1)
        distances = np.maximum(distances, 1e-10)

        # Soft initialization based on distances
        for i in range(n_samples):
            for j in range(n_clusters):
                sum_term = np.sum((distances[i, j] / distances[i, :]) ** (1 / (m - 1)))
                U[i, j] = 1.0 / sum_term

        # Initialize typicality matrix
        T = U.copy()

    except Exception as e:
        print(f"Warning: K-means++ initialization failed: {e}. Using random initialization.")
        U = initialize_membership_matrix_pfcm(n_samples, n_clusters)
        T = U.copy()
        C = np.random.rand(n_clusters, n_features)

    # Main PFCM iteration loop
    prev_objective = float('inf')

    for i in range(max_iter):
        C_old = C.copy()
        U_old = U.copy()
        T_old = T.copy()

        # Update centroids
        C = calculate_centroids_pfcm(X_data, U, T, m, eta)
        if C.size == 0 or np.isnan(C).any() or np.isinf(C).any():
            print(f"Error: PFCM centroids calculation failed at iteration {i}. Stopping.")
            labels = np.argmax(U_old, axis=1) if U_old.size > 0 else np.array([])
            return labels, U_old, T_old, C_old, {"error": "Centroid calculation failed"}

        # Update membership and typicality matrices
        U_new, T_new = update_U_T_pfcm(X_data, C, m, eta)
        if U_new.size == 0 or T_new.size == 0 or np.isnan(U_new).any() or np.isinf(U_new).any() or np.isnan(T_new).any() or np.isinf(T_new).any():
            print(f"Error: PFCM U/T update failed at iteration {i}. Stopping.")
            labels = np.argmax(U_old, axis=1) if U_old.size > 0 else np.array([])
            return labels, U_old, T_old, C, {"error": "U/T update failed"}

        # Calculate objective function for convergence check
        current_objective = compute_pfcm_objective(X_data, U_new, T_new, C, m, eta)

        # Check convergence
        if i > 0:
            if abs(prev_objective - current_objective) < tol:
                if verbose:
                    print(f"PFCM converged (objective) at iteration {i+1}/{max_iter}")
                U, T = U_new, T_new
                break

            if (U.shape == U_new.shape and T.shape == T_new.shape and
                np.linalg.norm(U - U_new) < tol and np.linalg.norm(T - T_new) < tol):
                if verbose:
                    print(f"PFCM converged (parameters) at iteration {i+1}/{max_iter}")
                U, T = U_new, T_new
                break

        U, T = U_new, T_new
        prev_objective = current_objective

        if verbose and (i+1) % 10 == 0:
            print(f"PFCM iteration {i+1}/{max_iter}, Objective: {current_objective:.6f}")

    # Final evaluation
    labels = np.argmax(U, axis=1) if U.size > 0 else np.array([])
    metrics = calculate_sklearn_metrics(X_data, labels)

    if X_data.shape[0] > 1 and C.shape[0] > 0 and U.shape[1] > 0:
        metrics.update(calculate_custom_metrics(X_data, U, C, m=m))

        # Add PFCM-specific metrics
        metrics['pfcm_objective'] = compute_pfcm_objective(X_data, U, T, C, m, eta)
        metrics['membership_entropy'] = -np.mean(np.sum(U * np.log(U + 1e-10), axis=1))
        metrics['typicality_sum'] = np.mean(np.sum(T, axis=1))

    return labels, U, T, C, metrics
