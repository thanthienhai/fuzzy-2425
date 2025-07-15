"""
Deep Embedded K-Means (DEKM) clustering algorithm implementation.

This module contains the implementation of the Deep Embedded K-Means
algorithm which combines autoencoder-based dimensionality reduction
with K-means clustering in the latent space.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.cluster import KMeans
from scipy.linalg import eigh

from ..models.autoencoder import AutoEncoder_DEKM, train_autoencoder_dekm
from ..utils.initialization import smart_initialization
from ..metrics.sklearn_metrics import calculate_sklearn_metrics
from ..metrics.fuzzy_metrics import calculate_custom_metrics


def compute_within_cluster_scatter(embeddings, labels, n_clusters):
    """
    Compute within-cluster scatter matrix efficiently.
    
    Args:
        embeddings: Embedded data points (n_samples, n_features)
        labels: Cluster labels for each sample
        n_clusters: Number of clusters
        
    Returns:
        numpy.ndarray: Within-cluster scatter matrix (n_features, n_features)
    """
    n_samples, n_features = embeddings.shape
    Sw = np.zeros((n_features, n_features))

    for i in range(n_clusters):
        cluster_mask = (labels == i)
        if not np.any(cluster_mask):
            continue

        cluster_points = embeddings[cluster_mask]
        cluster_mean = np.mean(cluster_points, axis=0)

        # Vectorized computation of scatter matrix
        centered_points = cluster_points - cluster_mean
        Sw += centered_points.T @ centered_points

    return Sw


def compute_clustering_loss(embeddings, labels, centroids):
    """
    Compute clustering loss: sum of squared distances to assigned centroids.
    
    Args:
        embeddings: Embedded data points tensor
        labels: Cluster labels for each sample
        centroids: Cluster centroids tensor
        
    Returns:
        torch.Tensor: Mean clustering loss
    """
    device = embeddings.device
    labels_tensor = torch.tensor(labels, dtype=torch.long, device=device)

    # Get assigned centroids for each point
    assigned_centroids = centroids[labels_tensor]

    # Compute squared distances
    distances = torch.sum((embeddings - assigned_centroids) ** 2, dim=1)

    return torch.mean(distances)


def initialize_fuzzy_membership(data, centroids, m):
    """
    Initialize fuzzy membership matrix with soft assignments.
    
    Args:
        data: Data points (n_samples, n_features)
        centroids: Cluster centroids (n_clusters, n_features)
        m: Fuzzification parameter
        
    Returns:
        numpy.ndarray: Membership matrix (n_samples, n_clusters)
    """
    n_samples, n_features = data.shape
    n_clusters = centroids.shape[0]

    # Compute distances
    distances = np.zeros((n_samples, n_clusters))
    for j in range(n_clusters):
        distances[:, j] = np.sum((data - centroids[j]) ** 2, axis=1)

    # Avoid division by zero
    distances = np.maximum(distances, 1e-10)

    # Initialize with soft membership based on distances
    U = np.zeros((n_samples, n_clusters))
    for i in range(n_samples):
        for j in range(n_clusters):
            sum_term = np.sum((distances[i, j] / distances[i, :]) ** (2 / (m - 1)))
            U[i, j] = 1.0 / sum_term

    return U


def compute_structure_preservation_loss(embeddings, original_embeddings):
    """
    Compute structure preservation loss to maintain local structure.
    
    Args:
        embeddings: Current embeddings tensor
        original_embeddings: Original embeddings tensor or numpy array
        
    Returns:
        torch.Tensor: Structure preservation loss
    """
    # Convert to tensor if needed
    if isinstance(original_embeddings, np.ndarray):
        original_tensor = torch.tensor(original_embeddings, dtype=torch.float32, device=embeddings.device)
    else:
        original_tensor = original_embeddings

    # Simple L2 loss to preserve structure
    return torch.norm(embeddings - original_tensor)


def run_dekm(X_tensor, X_for_metrics, k=10, Iter=15, input_dim=None, hidden_dim_ae=10,
             alpha=1.0, beta=0.1, verbose=False):
    """
    Run Deep Embedded K-Means (DEKM) clustering algorithm.

    Args:
        X_tensor: Input data tensor (N, D)
        X_for_metrics: Data for metric calculation (can be same as X_tensor)
        k: Number of clusters
        Iter: Number of DEKM iterations
        input_dim: Input dimension (auto-detected if None)
        hidden_dim_ae: Latent dimension
        alpha: Weight for reconstruction loss
        beta: Weight for clustering loss
        verbose: Print progress

    Returns:
        tuple: (labels_final, H_np_final, metrics) where:
            - labels_final: Final cluster assignments
            - H_np_final: Final embeddings in latent space
            - metrics: Dictionary of evaluation metrics
    """
    if X_tensor.shape[0] == 0:
        print("Error: DEKM input tensor is empty.")
        return np.array([]), np.array([]), {}

    if input_dim is None:
        input_dim = X_tensor.shape[1]

    if X_tensor.shape[0] < k:
        print(f"Warning: DEKM num_samples ({X_tensor.shape[0]}) < k ({k}). Reducing k.")
        k = max(1, X_tensor.shape[0])

    # Initialize autoencoder with improved architecture
    model = AutoEncoder_DEKM(input_dim=input_dim, hidden_dim_ae=hidden_dim_ae)

    # Enhanced pretraining with adaptive features
    if verbose:
        print("Starting DEKM autoencoder pretraining with adaptive learning...")
    train_autoencoder_dekm(model, X_tensor, epochs=100, verbose=verbose, use_adaptive_lr=True)

    # Initialize variables
    H_np_final = None
    labels_final = None
    best_loss = float('inf')

    # Main DEKM optimization loop
    optimizer_joint = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

    for it in range(Iter):
        if verbose:
            print(f"\n--- DEKM Iteration {it+1}/{Iter} ---")

        # Step 1: Get current embeddings
        model.eval()
        with torch.no_grad():
            _, H = model(X_tensor)
            H_np = H.cpu().numpy()

        if H_np.shape[0] == 0 or H_np.shape[1] == 0:
            print(f"DEKM Iter {it+1}: Invalid embeddings. Skipping iteration.")
            continue

        # Step 2: K-means clustering in latent space
        actual_k = min(k, H_np.shape[0])
        if actual_k < 1:
            print(f"DEKM Iter {it+1}: Invalid cluster count. Skipping iteration.")
            continue

        try:
            # Use smart initialization for better clustering
            if it == 0:
                # First iteration: use smart initialization
                initial_centers = smart_initialization(H_np, actual_k, method='kmeans++', random_state=42)
                kmeans = KMeans(n_clusters=actual_k, init=initial_centers, n_init=1,
                              max_iter=300, random_state=42)
            else:
                # Subsequent iterations: use previous centroids as initialization
                kmeans = KMeans(n_clusters=actual_k, init='k-means++', n_init=3,
                              max_iter=300, random_state=42)

            labels = kmeans.fit_predict(H_np)
            centroids = kmeans.cluster_centers_
        except Exception as e:
            print(f"DEKM Iter {it+1}: K-Means failed: {e}. Skipping iteration.")
            continue

        H_np_final = H_np
        labels_final = labels

        # Step 3: Compute within-cluster scatter matrix (improved)
        Sw = compute_within_cluster_scatter(H_np, labels, actual_k)

        # Step 4: Eigendecomposition for structure preservation
        try:
            # Get the smallest eigenvalues/eigenvectors (most compact directions)
            eigvals, eigvecs = eigh(Sw)

            # Select eigenvectors corresponding to smallest eigenvalues
            # These represent the most compact clustering directions
            n_components = min(hidden_dim_ae, eigvecs.shape[1])
            V_np = eigvecs[:, :n_components]  # Take first n_components eigenvectors

        except np.linalg.LinAlgError:
            print(f"Warning: DEKM eigendecomposition failed in iter {it+1}. Using identity.")
            V_np = np.eye(H_np.shape[1], min(H_np.shape[1], hidden_dim_ae))
        except Exception as e:
            print(f"Error in DEKM eigendecomposition iter {it+1}: {e}. Using identity.")
            V_np = np.eye(H_np.shape[1], min(H_np.shape[1], hidden_dim_ae))

        # Step 5: Joint optimization of autoencoder and clustering
        if V_np.size > 0:
            V_tensor = torch.tensor(V_np, dtype=torch.float32, device=X_tensor.device)
            centroids_tensor = torch.tensor(centroids, dtype=torch.float32, device=X_tensor.device)

            # Joint training for several epochs
            model.train()
            for epoch in range(5):  # Reduced for efficiency
                optimizer_joint.zero_grad()

                # Forward pass
                reconstructed, embeddings = model(X_tensor)

                # Reconstruction loss
                loss_recon = nn.MSELoss()(reconstructed, X_tensor)

                # Clustering loss: minimize distance to assigned centroids
                loss_cluster = compute_clustering_loss(embeddings, labels_final, centroids_tensor)

                # Structure preservation loss
                transformed_embeddings = embeddings @ V_tensor
                loss_structure = torch.norm(embeddings - transformed_embeddings @ V_tensor.T)

                # Combined loss
                total_loss = alpha * loss_recon + beta * loss_cluster + 0.01 * loss_structure

                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer_joint.step()

                if epoch == 4:  # Store final loss
                    final_loss = total_loss.item()

        if verbose:
            print(f"DEKM Iteration {it+1} completed. Loss: {final_loss:.6f}")

    # Final evaluation and metrics calculation
    metrics = {}
    if labels_final is not None and H_np_final is not None and H_np_final.shape[0] > 0:
        # Calculate metrics on latent space (more meaningful for DEKM)
        metrics = calculate_sklearn_metrics(H_np_final, labels_final)

        if H_np_final.shape[0] > 1 and len(np.unique(labels_final)) > 1:
            num_clusters_dekm = len(np.unique(labels_final))
            if num_clusters_dekm > 0 and H_np_final.shape[0] >= num_clusters_dekm:
                # Create hard membership matrix for custom metrics
                U_dekm_hard = np.zeros((H_np_final.shape[0], num_clusters_dekm))
                unique_labels_arr = np.unique(labels_final)
                label_to_idx = {label: i for i, label in enumerate(unique_labels_arr)}
                for i, label in enumerate(labels_final):
                    U_dekm_hard[i, label_to_idx[label]] = 1.0

                # Get final centroids
                kmeans_final = KMeans(n_clusters=num_clusters_dekm, init='k-means++',
                                    n_init=10, random_state=42).fit(H_np_final)
                V_dekm_latent = kmeans_final.cluster_centers_

                # Calculate custom fuzzy metrics
                metrics.update(calculate_custom_metrics(H_np_final, U_dekm_hard, V_dekm_latent, m=2.0))

                # Add DEKM-specific metrics
                metrics['inertia'] = kmeans_final.inertia_
                metrics['n_iter'] = kmeans_final.n_iter_

    else:
        # Fallback for failed cases
        labels_final = np.array([])
        H_np_final = np.array([])
        metrics = {
            'silhouette': np.nan, 'calinski_harabasz': np.nan, 'davies_bouldin': np.nan,
            'pci': np.nan, 'fhv': np.nan, 'xbi': np.nan,
            'inertia': np.nan, 'n_iter': np.nan
        }

    if verbose:
        print(f"DEKM completed. Final metrics: {metrics}")

    return labels_final, H_np_final, metrics
