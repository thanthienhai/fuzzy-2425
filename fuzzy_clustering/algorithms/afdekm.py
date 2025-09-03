"""
Attentive Fuzzy Deep Embedded K-Means (A-FDEKM) clustering algorithm implementation.

This module contains the implementation of the A-FDEKM algorithm which enhances FDEKM
with attention mechanisms, dual loss function, and intelligent initialization.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.cluster import KMeans
from ..models.autoencoder import AutoEncoder_DEKM, train_autoencoder_dekm
from ..metrics.sklearn_metrics import calculate_sklearn_metrics
from ..metrics.fuzzy_metrics import calculate_custom_metrics


class AttentionEncoder(nn.Module):
    """
    Attention-based encoder for A-FDEKM algorithm.
    
    This encoder incorporates an attention mechanism to automatically learn
    the importance of each feature (L, R, F, M indices) and create a 
    context-aware representation space.
    """
    
    def __init__(self, input_dim, hidden_dim, dropout_rate=0.2):
        super(AttentionEncoder, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Attention layer
        self.attention_layer = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, input_dim),
            nn.Softmax(dim=1)  # Ensure attention weights sum to 1
        )
        
        # Deep encoder layers
        if input_dim <= hidden_dim:
            # Simple case: direct mapping
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.Tanh()
            )
        else:
            # Multi-layer encoder
            intermediate_dim = max(hidden_dim * 2, min(input_dim // 2, 256))
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, intermediate_dim),
                nn.BatchNorm1d(intermediate_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(intermediate_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.Tanh()
            )
    
    def forward(self, x):
        """
        Forward pass through attention encoder.
        
        Args:
            x: Input tensor (batch_size, input_dim)
            
        Returns:
            tuple: (attention_weights, weighted_features, encoded_features)
        """
        # Generate attention weights
        attention_weights = self.attention_layer(x)
        
        # Apply attention weights to input features
        weighted_features = x * attention_weights
        
        # Encode weighted features
        encoded_features = self.encoder(weighted_features)
        
        return attention_weights, weighted_features, encoded_features


class AFDEKM_Model(nn.Module):
    """
    Complete A-FDEKM model combining attention encoder with fuzzy clustering.
    """
    
    def __init__(self, input_dim, hidden_dim, n_clusters, m=2.0, dropout_rate=0.2):
        super(AFDEKM_Model, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_clusters = n_clusters
        self.m = m  # Fuzzification parameter
        
        # Attention encoder
        self.attention_encoder = AttentionEncoder(input_dim, hidden_dim, dropout_rate)
        
        # Fuzzy cluster centers (learnable parameters)
        self.cluster_centers = nn.Parameter(torch.randn(n_clusters, hidden_dim))
        
        # Initialize cluster centers properly
        nn.init.xavier_uniform_(self.cluster_centers)
    
    def forward(self, x):
        """
        Forward pass through A-FDEKM model.
        
        Args:
            x: Input tensor (batch_size, input_dim)
            
        Returns:
            tuple: (attention_weights, encoded_features, fuzzy_membership)
        """
        # Get attention weights and encoded features
        attention_weights, weighted_features, encoded_features = self.attention_encoder(x)
        
        # Compute fuzzy membership matrix
        fuzzy_membership = self._compute_fuzzy_membership(encoded_features)
        
        return attention_weights, encoded_features, fuzzy_membership
    
    def _compute_fuzzy_membership(self, encoded_features):
        """
        Compute fuzzy membership matrix based on distances to cluster centers.
        
        Args:
            encoded_features: Encoded feature tensor (batch_size, hidden_dim)
            
        Returns:
            torch.Tensor: Fuzzy membership matrix (batch_size, n_clusters)
        """
        # Compute distances from each point to each cluster center
        distances = torch.cdist(encoded_features, self.cluster_centers, p=2)
        
        # Avoid division by zero
        distances = torch.clamp(distances, min=1e-10)
        
        # Compute fuzzy membership using standard FCM formula
        if self.m == 1.0:
            # Hard clustering case
            min_distances = torch.argmin(distances, dim=1)
            membership = torch.zeros_like(distances)
            membership.scatter_(1, min_distances.unsqueeze(1), 1.0)
        else:
            # Fuzzy clustering case
            power = 2.0 / (self.m - 1.0)
            membership = 1.0 / torch.sum((distances.unsqueeze(2) / distances.unsqueeze(1)) ** power, dim=2)
            # Normalize to ensure sum equals 1
            membership = membership / torch.sum(membership, dim=1, keepdim=True)
        
        return membership


def compute_dual_loss(encoded_features, fuzzy_membership, cluster_centers, 
                     attention_weights=None, gamma=0.1, alpha=0.01):
    """
    Compute the dual loss function for A-FDEKM.
    
    Args:
        encoded_features: Encoded feature tensor (batch_size, hidden_dim)
        fuzzy_membership: Fuzzy membership matrix (batch_size, n_clusters)
        cluster_centers: Cluster center tensor (n_clusters, hidden_dim)
        attention_weights: Attention weights (batch_size, input_dim)
        gamma: Weight for separation loss
        alpha: Weight for attention regularization
        
    Returns:
        tuple: (total_loss, fuzzy_loss, separation_loss, attention_reg)
    """
    # Fuzzy clustering loss (intra-cluster compactness)
    distances = torch.cdist(encoded_features, cluster_centers, p=2) ** 2
    fuzzy_loss = torch.sum(fuzzy_membership * distances)
    
    # Inter-cluster separation loss
    n_clusters = cluster_centers.shape[0]
    separation_loss = 0.0
    
    for i in range(n_clusters):
        for j in range(i + 1, n_clusters):
            center_distance = torch.norm(cluster_centers[i] - cluster_centers[j], p=2)
            # Add small epsilon to avoid division by zero
            separation_loss += 1.0 / (center_distance + 1e-6)
    
    # Attention regularization (encourage diversity in attention weights)
    attention_reg = 0.0
    if attention_weights is not None:
        # Encourage attention weights to be diverse (not all equal)
        attention_variance = torch.var(attention_weights, dim=1).mean()
        attention_reg = -attention_variance  # Negative because we want to maximize variance
    
    # Total loss
    total_loss = fuzzy_loss + gamma * separation_loss + alpha * attention_reg
    
    return total_loss, fuzzy_loss, separation_loss, attention_reg


def intelligent_initialization(X, n_clusters, hidden_dim, pretrain_epochs=50, verbose=False):
    """
    Intelligent initialization using K-Means++ on pretrained embedding space.
    
    Args:
        X: Input data tensor
        n_clusters: Number of clusters
        hidden_dim: Hidden dimension for autoencoder
        pretrain_epochs: Number of pretraining epochs
        verbose: Whether to print progress
        
    Returns:
        tuple: (initial_cluster_centers, pretrained_encoder)
    """
    input_dim = X.shape[1]
    
    # Step 1: Pretrain standard autoencoder
    pretrained_ae = AutoEncoder_DEKM(input_dim=input_dim, hidden_dim_ae=hidden_dim)
    
    if verbose:
        print("Pretraining autoencoder for intelligent initialization...")
    
    train_autoencoder_dekm(pretrained_ae, X, epochs=pretrain_epochs, verbose=verbose)
    
    # Step 2: Get initial embeddings
    pretrained_ae.eval()
    with torch.no_grad():
        _, initial_embeddings = pretrained_ae(X)
        initial_embeddings_np = initial_embeddings.cpu().numpy()
    
    # Step 3: Run K-Means++ on embeddings
    if verbose:
        print("Running K-Means++ for cluster center initialization...")
    
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10, random_state=42)
    kmeans.fit(initial_embeddings_np)
    
    initial_cluster_centers = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)
    
    return initial_cluster_centers, pretrained_ae


def run_afdekm(X_tensor, X_for_metrics, k=10, Iter=50, input_dim=None, hidden_dim_ae=10,
               m=2.0, gamma=0.1, alpha=0.01, lr=1e-3, pretrain_epochs=50, verbose=False):
    """
    Run Attentive Fuzzy Deep Embedded K-Means (A-FDEKM) clustering algorithm.
    
    Args:
        X_tensor: Input data tensor (N, D)
        X_for_metrics: Data for metric calculation (can be same as X_tensor)
        k: Number of clusters
        Iter: Number of A-FDEKM iterations
        input_dim: Input dimension (auto-detected if None)
        hidden_dim_ae: Latent dimension
        m: Fuzzification parameter
        gamma: Weight for separation loss
        alpha: Weight for attention regularization
        lr: Learning rate
        pretrain_epochs: Number of pretraining epochs
        verbose: Print progress
        
    Returns:
        tuple: (labels, H_np_final, U_final, V_final, attention_weights_final, metrics) where:
            - labels: Hard cluster assignments
            - H_np_final: Final embeddings in latent space
            - U_final: Final fuzzy membership matrix
            - V_final: Final cluster centers
            - attention_weights_final: Final attention weights
            - metrics: Dictionary of evaluation metrics
    """
    if X_tensor.shape[0] == 0:
        print("Error: A-FDEKM input tensor is empty.")
        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), {}

    if input_dim is None:
        input_dim = X_tensor.shape[1]

    if X_tensor.shape[0] < k:
        print(f"Warning: A-FDEKM num_samples ({X_tensor.shape[0]}) < k ({k}). Reducing k.")
        k = max(1, X_tensor.shape[0])

    try:
        # Step 0: Intelligent Initialization
        if verbose:
            print("Starting A-FDEKM intelligent initialization...")
        
        initial_cluster_centers, pretrained_encoder = intelligent_initialization(
            X_tensor, k, hidden_dim_ae, pretrain_epochs, verbose
        )
        
        # Initialize A-FDEKM model
        model = AFDEKM_Model(input_dim, hidden_dim_ae, k, m)
        
        # Initialize cluster centers with intelligent initialization
        model.cluster_centers.data = initial_cluster_centers
        
        # Initialize attention weights to be equal initially
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        # Step 1: Concurrent Optimization Loop
        if verbose:
            print("Starting A-FDEKM concurrent optimization...")
        
        best_loss = float('inf')
        patience_counter = 0
        patience = 10
        
        for iteration in range(Iter):
            model.train()
            optimizer.zero_grad()
            
            # Forward pass
            attention_weights, encoded_features, fuzzy_membership = model(X_tensor)
            
            # Compute dual loss
            total_loss, fuzzy_loss, separation_loss, attention_reg = compute_dual_loss(
                encoded_features, fuzzy_membership, model.cluster_centers.data,
                attention_weights, gamma, alpha
            )
            
            # Backward pass
            total_loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Early stopping check
            if total_loss.item() < best_loss:
                best_loss = total_loss.item()
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                if verbose:
                    print(f"Early stopping at iteration {iteration + 1}")
                break
            
            if verbose and (iteration + 1) % 10 == 0:
                print(f"A-FDEKM Iteration {iteration + 1}/{Iter}, "
                      f"Total Loss: {total_loss.item():.6f}, "
                      f"Fuzzy Loss: {fuzzy_loss.item():.6f}, "
                      f"Separation Loss: {separation_loss.item():.6f}, "
                      f"Attention Reg: {attention_reg.item():.6f}")
        
        # Step 2: Final Output
        model.eval()
        with torch.no_grad():
            attention_weights_final, H_final, U_final = model(X_tensor)
            
            # Convert to numpy
            H_np_final = H_final.cpu().numpy()
            U_final_np = U_final.cpu().numpy()
            V_final = model.cluster_centers.data.cpu().numpy()
            attention_weights_final_np = attention_weights_final.cpu().numpy()
            
            # Get hard labels
            labels = np.argmax(U_final_np, axis=1)
        
        # Calculate evaluation metrics
        metrics = calculate_sklearn_metrics(H_np_final, labels)
        
        if H_np_final.shape[0] > 1 and len(np.unique(labels)) > 1:
            metrics.update(calculate_custom_metrics(H_np_final, U_final_np, V_final, m=m))
        
        # Add A-FDEKM-specific metrics
        metrics['fuzzy_partition_coefficient'] = np.mean(np.sum(U_final_np ** 2, axis=1))
        metrics['fuzzy_partition_entropy'] = -np.mean(np.sum(U_final_np * np.log(U_final_np + 1e-10), axis=1))
        metrics['final_objective'] = best_loss
        metrics['attention_weights_mean'] = np.mean(attention_weights_final_np, axis=0).tolist()
        metrics['attention_weights_std'] = np.std(attention_weights_final_np, axis=0).tolist()
        
        # Calculate inter-cluster separation
        inter_cluster_distances = []
        for i in range(k):
            for j in range(i + 1, k):
                distance = np.linalg.norm(V_final[i] - V_final[j])
                inter_cluster_distances.append(distance)
        
        metrics['mean_inter_cluster_distance'] = np.mean(inter_cluster_distances) if inter_cluster_distances else 0.0
        metrics['min_inter_cluster_distance'] = np.min(inter_cluster_distances) if inter_cluster_distances else 0.0
        
        if verbose:
            print(f"A-FDEKM completed. Final metrics: {metrics}")
            print(f"Average attention weights: {metrics['attention_weights_mean']}")
        
        return labels, H_np_final, U_final_np, V_final, attention_weights_final_np, metrics
    
    except Exception as e:
        print(f"Error in A-FDEKM: {e}")
        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), {"error": str(e)}
