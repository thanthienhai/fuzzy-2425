import pandas as pd
import numpy as np
import h5py
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.cluster import KMeans
from scipy.linalg import eigh
import skfuzzy as fuzz
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import os # For path joining
from sklearn.cluster import DBSCAN
import time
import warnings
from functools import wraps

warnings.filterwarnings("ignore")

# Performance optimization utilities
def timing_decorator(func):
    """Decorator to measure function execution time"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        if hasattr(wrapper, '_show_timing') and wrapper._show_timing:
            print(f"{func.__name__} executed in {end_time - start_time:.4f} seconds")
        return result
    return wrapper

def enable_timing(func):
    """Enable timing for a decorated function"""
    if hasattr(func, '_show_timing'):
        func._show_timing = True
    return func

def compute_distances_optimized(data, centers):
    """Optimized distance computation using vectorized operations"""
    # Use broadcasting for efficient computation
    # data: (n_samples, n_features), centers: (n_clusters, n_features)
    # Result: (n_samples, n_clusters)
    diff = data[:, np.newaxis, :] - centers[np.newaxis, :, :]  # Broadcasting
    distances_sq = np.sum(diff ** 2, axis=2)
    return distances_sq

def update_membership_optimized(distances, m):
    """Optimized membership matrix update using vectorized operations"""
    n_samples, n_clusters = distances.shape

    # Avoid division by zero
    distances = np.maximum(distances, 1e-10)

    # Vectorized computation of membership matrix
    power = 2.0 / (m - 1.0)

    # Use broadcasting for efficient computation
    ratio_matrix = distances[:, :, np.newaxis] / distances[:, np.newaxis, :]
    powered_ratios = ratio_matrix ** power
    sum_ratios = np.sum(powered_ratios, axis=2)

    # Handle numerical issues
    sum_ratios = np.maximum(sum_ratios, 1e-10)
    membership = 1.0 / sum_ratios

    return membership

def batch_process_large_dataset(data, algorithm_func, batch_size=1000, **kwargs):
    """
    Process large datasets in batches to manage memory usage

    Args:
        data: Input data array
        algorithm_func: Clustering algorithm function to apply
        batch_size: Size of each batch
        **kwargs: Additional arguments for the algorithm
    """
    n_samples = data.shape[0]

    if n_samples <= batch_size:
        # Process normally if data is small enough
        return algorithm_func(data, **kwargs)

    print(f"Processing large dataset ({n_samples} samples) in batches of {batch_size}")

    # For clustering, we need to process the entire dataset together
    # But we can optimize memory usage during distance computations
    # This is a placeholder for future batch processing implementation

    # For now, just add memory optimization warnings
    if n_samples > 10000:
        print("Warning: Large dataset detected. Consider reducing data size or using sampling.")

    return algorithm_func(data, **kwargs)

def optimize_memory_usage():
    """
    Optimize memory usage for large datasets
    """
    import gc
    gc.collect()  # Force garbage collection

    # Set numpy to use less memory for operations
    np.seterr(all='ignore')  # Ignore numerical warnings for performance

def adaptive_batch_size(data_size, available_memory_gb=4):
    """
    Calculate optimal batch size based on data size and available memory
    """
    # Estimate memory usage per sample (rough approximation)
    bytes_per_sample = data_size * 8  # 8 bytes per float64

    # Convert GB to bytes
    available_bytes = available_memory_gb * 1024**3

    # Use 50% of available memory for safety
    usable_bytes = available_bytes * 0.5

    # Calculate batch size
    batch_size = max(100, int(usable_bytes / bytes_per_sample))

    return min(batch_size, 10000)  # Cap at 10k for practical reasons

# Advanced Features: Adaptive Learning and Initialization
class AdaptiveLearningRateScheduler:
    """
    Adaptive learning rate scheduler for neural network training
    """
    def __init__(self, initial_lr=1e-3, patience=5, factor=0.5, min_lr=1e-6):
        self.initial_lr = initial_lr
        self.current_lr = initial_lr
        self.patience = patience
        self.factor = factor
        self.min_lr = min_lr
        self.wait = 0
        self.best_loss = float('inf')

    def step(self, current_loss):
        """Update learning rate based on current loss"""
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.wait = 0
        else:
            self.wait += 1

        if self.wait >= self.patience:
            self.current_lr = max(self.current_lr * self.factor, self.min_lr)
            self.wait = 0
            return True  # Learning rate was updated
        return False

    def get_lr(self):
        return self.current_lr

class EarlyStoppingCriterion:
    """
    Early stopping criterion to prevent overfitting
    """
    def __init__(self, patience=10, min_delta=1e-6, restore_best=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best = restore_best
        self.wait = 0
        self.best_loss = float('inf')
        self.best_params = None

    def should_stop(self, current_loss, current_params=None):
        """Check if training should stop"""
        if current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.wait = 0
            if current_params is not None and self.restore_best:
                self.best_params = current_params.copy() if hasattr(current_params, 'copy') else current_params
        else:
            self.wait += 1

        return self.wait >= self.patience

    def get_best_params(self):
        return self.best_params

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

def adaptive_convergence_detection(history, window_size=5, threshold=1e-6):
    """
    Adaptive convergence detection based on loss history

    Args:
        history: List of loss values
        window_size: Size of the sliding window
        threshold: Convergence threshold
    """
    if len(history) < window_size + 1:
        return False

    # Check if the improvement in the last window_size iterations is below threshold
    recent_losses = history[-window_size-1:]
    improvements = [recent_losses[i] - recent_losses[i+1] for i in range(window_size)]
    avg_improvement = np.mean(improvements)

    return avg_improvement < threshold

# Global placeholders for custom metric functions
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

# --- Data Loading and Preprocessing ---
def load_usps_data(base_path='.'):
    file_path = os.path.join(base_path, 'usps.h5')
    try:
        with h5py.File(file_path, 'r') as f:
            train_X = np.array(f['train']['data'])
            train_y = np.array(f['train']['target'])
            test_X = np.array(f['test']['data'])
            test_y = np.array(f['test']['target'])
        X = np.concatenate([train_X, test_X], axis=0)
        y = np.concatenate([train_y, test_y], axis=0)
        X_flat = X.reshape((X.shape[0], -1))
        print(f"USPS data loaded successfully from {file_path}. Shape: {X_flat.shape}")
        return X_flat, y
    except FileNotFoundError:
        print(f"Error: USPS data file '{file_path}' not found.")
        return None, None
    except Exception as e:
        print(f"Error loading USPS data from '{file_path}': {e}")
        return None, None


def preprocess_usps_for_dekm(X_flat):
    if X_flat is None: return None, None
    X_proc = X_flat.astype('float32') / 255.0
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_proc)
    return torch.tensor(X_scaled, dtype=torch.float32), X_scaled

def preprocess_usps_for_pfcm(X_flat):
    if X_flat is None: return None
    X_proc = X_flat.astype('float32') / 255.0
    # Ensure n_components is not greater than number of samples or features
    n_components = min(50, X_proc.shape[0], X_proc.shape[1])
    if n_components < 1:
        print(f"Warning: Not enough features/samples for PCA ({X_proc.shape}). Returning raw scaled data for PFCM.")
        return X_proc # Or handle as an error
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_proc)
    return X_pca

def load_ecommerce_data(base_path='.'):
    orders_path = os.path.join(base_path, 'FCM_NA', 'List of Orders.csv')
    details_path = os.path.join(base_path, 'FCM_NA', 'Order Details.csv')
    try:
        orders = pd.read_csv(orders_path)
        order_details = pd.read_csv(details_path)
        print(f"E-commerce data loaded from {orders_path} and {details_path}")
    except FileNotFoundError:
        print(f"Error: E-commerce CSV files not found. Checked: '{orders_path}' and '{details_path}'.")
        return None
    except Exception as e:
        print(f"Error loading E-commerce data: {e}")
        return None
        
    orders.dropna(inplace=True)
    order_details.dropna(inplace=True)
    df = pd.merge(order_details, orders, on='Order ID')
    features = ['Amount', 'Profit', 'Quantity']
    if not all(feature in df.columns for feature in features):
        print(f"Error: Required features {features} not all found in merged e-commerce data.")
        return None
    data = df[features].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data)
    return X_scaled

def load_country_data(base_path='.'):
    file_path = os.path.join(base_path, 'Country-data.csv')
    try:
        df = pd.read_csv(file_path)
        print(f"Country data loaded successfully from {file_path}. Shape: {df.shape}")
        
        # Drop the 'country' column as it's not a feature
        features = df.drop('country', axis=1)
        
        # Scale the features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(features)
        return X_scaled, df['country']
    except FileNotFoundError:
        print(f"Error: Country data file '{file_path}' not found.")
        return None, None
    except Exception as e:
        print(f"Error loading Country data from '{file_path}': {e}")
        return None, None

# --- Algorithm Implementations ---

class AutoEncoder_DEKM(nn.Module):
    def __init__(self, input_dim, hidden_dim_ae, dropout_rate=0.2):
        super(AutoEncoder_DEKM, self).__init__()

        # Ensure valid dimensions
        if hidden_dim_ae <= 0:
            hidden_dim_ae = max(1, input_dim // 4)
        if input_dim <= 0:
            raise ValueError("Input dimension must be positive")

        # Adaptive intermediate dimension based on input size
        if input_dim <= 64:
            intermediate_dim = max(hidden_dim_ae * 2, 32)
        elif input_dim <= 256:
            intermediate_dim = max(hidden_dim_ae * 3, 128)
        elif input_dim <= 1024:
            intermediate_dim = max(hidden_dim_ae * 4, 256)
        else:
            intermediate_dim = max(hidden_dim_ae * 4, 512)

        # Ensure intermediate dimension is reasonable
        intermediate_dim = min(intermediate_dim, input_dim * 2)
        intermediate_dim = max(intermediate_dim, hidden_dim_ae)

        # Build encoder with batch normalization and dropout
        if input_dim <= hidden_dim_ae:
            # Simple case: direct mapping
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim_ae),
                nn.BatchNorm1d(hidden_dim_ae),
                nn.Tanh()  # Bounded activation for better stability
            )
            self.decoder = nn.Sequential(
                nn.Linear(hidden_dim_ae, input_dim),
                nn.Sigmoid()  # Assuming normalized input data
            )
        else:
            # Multi-layer encoder with proper regularization
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, intermediate_dim),
                nn.BatchNorm1d(intermediate_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(intermediate_dim, hidden_dim_ae),
                nn.BatchNorm1d(hidden_dim_ae),
                nn.Tanh()  # Bounded activation for latent space
            )
            self.decoder = nn.Sequential(
                nn.Linear(hidden_dim_ae, intermediate_dim),
                nn.BatchNorm1d(intermediate_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(intermediate_dim, input_dim),
                nn.Sigmoid()  # Assuming normalized input data
            )

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim_ae

    def forward(self, x):
        h = self.encoder(x)
        out = self.decoder(h)
        return out, h

    def encode(self, x):
        """Get only the encoded representation"""
        return self.encoder(x)

@timing_decorator
def train_autoencoder_dekm(model, data, epochs=100, lr=1e-3, weight_decay=1e-5,
                          patience=10, min_delta=1e-6, verbose=False, use_adaptive_lr=True):
    """
    Enhanced autoencoder training with adaptive learning rates and advanced features
    """
    if data.shape[0] == 0:
        print("Error: DEKM Autoencoder training data is empty.")
        return

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    # Initialize advanced features
    if use_adaptive_lr:
        lr_scheduler = AdaptiveLearningRateScheduler(initial_lr=lr, patience=patience//3)
    early_stopping = EarlyStoppingCriterion(patience=patience, min_delta=min_delta)

    loss_history = []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output, latent = model(data)

        # Reconstruction loss
        recon_loss = criterion(output, data)

        # Add regularization terms
        l2_reg = torch.norm(latent, p=2, dim=1).mean()

        # Sparsity regularization on latent representations
        sparsity_reg = torch.mean(torch.abs(latent))

        # Total loss with adaptive weighting
        total_loss = recon_loss + 0.001 * l2_reg + 0.0001 * sparsity_reg

        total_loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # Update learning rate adaptively
        if use_adaptive_lr:
            lr_updated = lr_scheduler.step(total_loss.item())
            if lr_updated:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_scheduler.get_lr()
                if verbose:
                    print(f"Learning rate updated to {lr_scheduler.get_lr():.6f}")

        # Track loss history
        loss_history.append(total_loss.item())

        # Check early stopping
        if early_stopping.should_stop(total_loss.item()):
            if verbose:
                print(f"Early stopping at epoch {epoch+1}")
            break

        # Adaptive convergence detection
        if epoch > 10 and adaptive_convergence_detection(loss_history, window_size=5):
            if verbose:
                print(f"Adaptive convergence detected at epoch {epoch+1}")
            break

        if verbose and (epoch+1) % 20 == 0:
            print(f"AE Epoch {epoch+1}/{epochs}, Loss: {recon_loss.item():.6f}, "
                  f"L2: {l2_reg.item():.6f}, Sparsity: {sparsity_reg.item():.6f}, "
                  f"LR: {optimizer.param_groups[0]['lr']:.6f}")

def compute_within_cluster_scatter(embeddings, labels, n_clusters):
    """
    Compute within-cluster scatter matrix efficiently
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
    Compute clustering loss: sum of squared distances to assigned centroids
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
    Initialize fuzzy membership matrix with soft assignments
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

@timing_decorator
def update_fuzzy_membership(data, centroids, m):
    """
    Optimized fuzzy membership matrix update using vectorized operations
    """
    # Use optimized distance computation
    distances = compute_distances_optimized(data, centroids)

    # Avoid division by zero
    distances = np.maximum(distances, 1e-10)

    # Use optimized membership update
    U = update_membership_optimized(distances, m)

    return U

def update_fuzzy_centroids(data, membership, m):
    """
    Update cluster centroids using weighted average
    """
    n_samples, n_features = data.shape
    n_clusters = membership.shape[1]

    centroids = np.zeros((n_clusters, n_features))
    for j in range(n_clusters):
        weights = membership[:, j] ** m
        centroids[j] = np.sum(weights.reshape(-1, 1) * data, axis=0) / np.sum(weights)

    return centroids

def compute_fcm_objective(data, membership, centroids, m):
    """
    Compute FCM objective function
    """
    n_samples, n_clusters = membership.shape
    objective = 0.0

    for i in range(n_samples):
        for j in range(n_clusters):
            distance = np.sum((data[i] - centroids[j]) ** 2)
            objective += (membership[i, j] ** m) * distance

    return objective

def compute_fuzzy_clustering_loss(embeddings, membership, centroids, m):
    """
    Compute fuzzy clustering loss for backpropagation
    """
    device = embeddings.device
    n_samples = embeddings.shape[0]
    n_clusters = centroids.shape[0]

    loss = 0.0
    for j in range(n_clusters):
        # Compute distances to centroid j
        distances = torch.sum((embeddings - centroids[j]) ** 2, dim=1)
        # Weight by membership values
        weights = membership[:, j] ** m
        loss += torch.sum(weights * distances)

    return loss / n_samples

def compute_structure_preservation_loss(embeddings, original_embeddings):
    """
    Compute structure preservation loss to maintain local structure
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
    Improved Deep Embedded K-Means (DEKM) Algorithm

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


# PFCM
def initialize_membership_matrix_pfcm(n_samples, n_clusters):
    U = np.random.rand(n_samples, n_clusters)
    # Normalize, handle sum U is zero for a row if all random numbers were 0 (highly unlikely)
    row_sums = np.sum(U, axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1 # Avoid division by zero
    return U / row_sums

def calculate_centroids_pfcm(X, U, T, m, eta):
    if X.shape[0] == 0 or U.shape[0] == 0 or T.shape[0] == 0: return np.array([])
    # Add epsilon to U**m and T**m to avoid issues if U or T are exactly 0 for all samples in a cluster
    # This can happen if a cluster becomes empty.
    Um_plus_eta_Tm = (U**m + 1e-9) + eta * (T**m + 1e-9)
    num = Um_plus_eta_Tm.T @ X
    denom = np.sum(Um_plus_eta_Tm, axis=0).reshape(-1, 1)
    denom[denom == 0] = 1e-9 # Avoid division by zero
    return num / denom

@timing_decorator
def update_U_T_pfcm(X, C, m, eta, gamma=None):
    """
    Optimized PFCM membership and typicality matrix updates using vectorized operations

    Args:
        X: Data points (n_samples, n_features)
        C: Cluster centers (n_clusters, n_features)
        m: Fuzzifier for membership (m > 1)
        eta: Fuzzifier for typicality (eta > 1)
        gamma: Typicality parameter (auto-computed if None)
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
    Compute PFCM objective function
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
    Improved Possibilistic Fuzzy C-Means (PFCM) Algorithm

    Args:
        X_data: Input data (n_samples, n_features)
        n_clusters: Number of clusters
        m: Fuzzifier for membership (m > 1)
        eta: Fuzzifier for typicality (eta > 1)
        max_iter: Maximum iterations
        tol: Convergence tolerance
        verbose: Print progress
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
        from sklearn.cluster import KMeans
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

# FCM
def run_fcm(X_scaled, n_clusters=3, m=2.0, error=0.005, maxiter=100, verbose=False): # Reduced maxiter
    if X_scaled is None or X_scaled.shape[0] == 0:
        print("Error: FCM input data is empty.")
        return np.array([]), np.array([[]]), np.array([[]]), {}
    
    if X_scaled.shape[0] < n_clusters:
        print(f"Warning: FCM n_samples ({X_scaled.shape[0]}) < n_clusters ({n_clusters}). Reducing n_clusters.")
        n_clusters = max(1, X_scaled.shape[0])
    if n_clusters == 0 : n_clusters = 1


    try:
        # skfuzzy.cmeans expects data to be (features, samples)
        cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
            X_scaled.T, c=n_clusters, m=m, error=error, maxiter=maxiter, init=None, seed=42
        )
    except Exception as e:
        print(f"Error during FCM cmeans: {e}")
        return np.array([]), np.array([[]]), np.array([[]]), {"error": str(e)}

    labels = np.argmax(u, axis=0)
    U_transposed = u.T 
    metrics = calculate_sklearn_metrics(X_scaled, labels)
    if X_scaled.shape[0] > 1 and cntr.shape[0] > 0 and U_transposed.shape[1] > 0:
        metrics.update(calculate_custom_metrics(X_scaled, U_transposed, cntr, m=m))
    
    if verbose: print(f"FCM completed. FPC: {fpc:.4f}")
    return labels, U_transposed, cntr, metrics

# K-Means (Standalone)
def run_kmeans_standalone(X_data, n_clusters, random_state=42, verbose=False):
    if X_data is None or X_data.shape[0] == 0:
        print("Error: K-Means input data is empty.")
        return np.array([]), np.array([[]]), {}
    
    if X_data.shape[0] < n_clusters:
        print(f"Warning: K-Means n_samples ({X_data.shape[0]}) < n_clusters ({n_clusters}). Reducing n_clusters.")
        n_clusters = max(1, X_data.shape[0])
    if n_clusters == 0: n_clusters = 1

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init='auto')
    try:
        labels = kmeans.fit_predict(X_data)
    except Exception as e:
        print(f"Error during K-Means fit_predict: {e}")
        return np.array([]), np.array([[]]), {"error": str(e)}

    metrics = calculate_sklearn_metrics(X_data, labels)
    
    # Prepare U and V for custom metrics
    if X_data.shape[0] > 1 and len(np.unique(labels)) > 1:
        num_actual_clusters = len(np.unique(labels))
        if num_actual_clusters > 0 and X_data.shape[0] >= num_actual_clusters:
            U_kmeans = np.zeros((X_data.shape[0], num_actual_clusters))
            # Map original labels (which might not be 0 to k-1 if k-means failed to find k clusters)
            # to 0 to num_actual_clusters-1 for U matrix indexing
            unique_labels_arr_kmeans = np.unique(labels)
            label_to_idx_kmeans = {label: i for i, label in enumerate(unique_labels_arr_kmeans)}
            for i, label in enumerate(labels):
                U_kmeans[i, label_to_idx_kmeans[label]] = 1.0
            
            V_kmeans = kmeans.cluster_centers_
            # Ensure V_kmeans has features matching X_data for custom metrics
            if V_kmeans.shape[1] == X_data.shape[1]:
                 metrics.update(calculate_custom_metrics(X_data, U_kmeans, V_kmeans, m=2.0)) # m=2.0 is a common default
            else:
                print(f"KMeans: Centroid feature size {V_kmeans.shape[1]} != Data feature size {X_data.shape[1]}. Skipping custom metrics.")


    if verbose: print(f"K-Means completed. Found {len(np.unique(labels))} clusters.")
    return labels, kmeans.cluster_centers_, metrics

# DBSCAN
def run_dbscan(X_data, eps=0.5, min_samples=5, verbose=False):
    if X_data is None or X_data.shape[0] == 0:
        print("Error: DBSCAN input data is empty.")
        return np.array([]), {}

    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    try:
        labels = dbscan.fit_predict(X_data)
    except Exception as e:
        print(f"Error during DBSCAN fit_predict: {e}")
        return np.array([]), {"error": str(e)}

    metrics = calculate_sklearn_metrics(X_data, labels)
    
    # Fuzzy metrics are not typically applicable to DBSCAN
    metrics['pci'] = np.nan
    metrics['fhv'] = np.nan
    metrics['xbi'] = np.nan
        
    n_clusters_found = len(set(labels)) - (1 if -1 in labels else 0)
    if verbose: print(f"DBSCAN completed. Found {n_clusters_found} clusters (excluding noise).")
    return labels, metrics

# CFCM (from Collaborative_FCM/CFCM.py)
@timing_decorator
def compute_distances_cfcm(data, centers):
    """Optimized computation of squared Euclidean distances using vectorized operations."""
    # Use the optimized distance computation function
    return compute_distances_optimized(data, centers)

def initialize_feature_weights(data, n_features):
    """
    Initialize adaptive feature weights based on data variance
    """
    # Calculate feature variances
    feature_variances = np.var(data, axis=0)

    # Normalize variances to get weights (higher variance = higher weight)
    feature_weights = feature_variances / (np.sum(feature_variances) + 1e-10)

    # Ensure minimum weight to avoid zero weights
    feature_weights = np.maximum(feature_weights, 0.1 / n_features)

    # Normalize to sum to 1
    feature_weights = feature_weights / np.sum(feature_weights)

    return feature_weights.reshape(1, -1)

def update_cfcm_membership(data, centers, feature_weights, beta, u_old):
    """
    Improved CFCM membership update with better numerical stability
    """
    n_samples, n_features = data.shape
    n_clusters = centers.shape[0]

    distances = compute_distances_cfcm(data, centers)
    distances = np.maximum(distances, 1e-10)

    u_new = np.zeros((n_samples, n_clusters))

    for i in range(n_samples):
        for j in range(n_clusters):
            # Standard FCM term
            sum_term = np.sum((distances[i, j] / distances[i, :]))
            if sum_term == 0:
                sum_term = 1e-10
            fcm_term = 1.0 / sum_term

            # Collaborative term
            collab_numerator = beta * np.sum(feature_weights * (u_old[i, j] ** 2))
            collab_denominator = 1 + beta * (n_features - 1)
            collab_term = collab_numerator / max(collab_denominator, 1e-10)

            u_new[i, j] = fcm_term * (1 - collab_term) + collab_term

    # Normalize to ensure sum = 1 for each sample
    row_sums = np.sum(u_new, axis=1, keepdims=True)
    row_sums = np.maximum(row_sums, 1e-10)
    u_new = u_new / row_sums

    return u_new

def update_cfcm_centers(data, u, feature_weights, beta):
    """
    Update CFCM cluster centers with collaborative terms
    """
    n_samples, n_features = data.shape
    n_clusters = u.shape[1]

    centers_new = np.zeros((n_clusters, n_features))

    for j in range(n_clusters):
        numerator = np.sum(u[:, j].reshape(-1, 1) * data, axis=0)
        denominator = np.sum(u[:, j]) + beta * np.sum(feature_weights)
        centers_new[j] = numerator / max(denominator, 1e-10)

    return centers_new

def update_feature_weights(data, u, centers, current_weights):
    """
    Update feature weights based on clustering quality
    """
    n_samples, n_features = data.shape
    n_clusters = u.shape[1]

    # Calculate feature importance based on within-cluster variance
    feature_importance = np.zeros(n_features)

    for j in range(n_clusters):
        cluster_weights = u[:, j]
        weighted_mean = np.sum(cluster_weights.reshape(-1, 1) * data, axis=0) / np.sum(cluster_weights)

        for f in range(n_features):
            variance = np.sum(cluster_weights * (data[:, f] - weighted_mean[f]) ** 2) / np.sum(cluster_weights)
            feature_importance[f] += 1.0 / (variance + 1e-10)  # Inverse variance

    # Normalize and smooth with current weights
    feature_importance = feature_importance / np.sum(feature_importance)
    alpha = 0.7  # Smoothing factor
    new_weights = alpha * current_weights.flatten() + (1 - alpha) * feature_importance

    # Ensure minimum weight
    new_weights = np.maximum(new_weights, 0.01 / n_features)
    new_weights = new_weights / np.sum(new_weights)

    return new_weights.reshape(1, -1)

def compute_cfcm_objective(data, u, centers, feature_weights, beta):
    """
    Compute CFCM objective function
    """
    n_samples, n_features = data.shape
    n_clusters = u.shape[1]

    objective = 0.0

    # Standard FCM term
    for i in range(n_samples):
        for j in range(n_clusters):
            distance_sq = np.sum((data[i] - centers[j]) ** 2)
            objective += (u[i, j] ** 2) * distance_sq

    # Collaborative regularization term
    for i in range(n_samples):
        for j in range(n_clusters):
            collab_term = beta * np.sum(feature_weights * (u[i, j] ** 2))
            objective += collab_term

    return objective

def run_cfcm(data_input, n_clusters, beta=1.0, max_iter=100, tol=1e-6, random_state=42, verbose=False):
    """
    Improved Collaborative Fuzzy C-Means (CFCM) Algorithm

    Args:
        data_input: Input data (n_samples, n_features)
        n_clusters: Number of clusters
        beta: Collaboration parameter (controls feature interaction)
        max_iter: Maximum iterations
        tol: Convergence tolerance
        random_state: Random seed
        verbose: Print progress
    """
    if data_input is None or data_input.shape[0] == 0:
        print("Error: CFCM input data is empty.")
        return np.array([]), np.array([[]]), np.array([[]]), {}

    if data_input.shape[0] < n_clusters:
        print(f"Warning: CFCM n_samples ({data_input.shape[0]}) < n_clusters ({n_clusters}). Reducing n_clusters.")
        n_clusters = max(1, data_input.shape[0])
    if n_clusters == 0:
        n_clusters = 1

    np.random.seed(random_state)
    n_samples, n_features = data_input.shape

    # Better initialization using K-means++
    try:
        kmeans_init = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10, random_state=random_state)
        kmeans_init.fit(data_input)
        centers = kmeans_init.cluster_centers_

        # Initialize membership matrix based on distances to initial centroids
        distances = compute_distances_cfcm(data_input, centers)
        distances = np.maximum(distances, 1e-10)

        u = np.zeros((n_samples, n_clusters))
        for i in range(n_samples):
            for j in range(n_clusters):
                sum_term = np.sum((distances[i, j] / distances[i, :]) ** 1.0)  # m=2 equivalent
                u[i, j] = 1.0 / sum_term

    except Exception as e:
        print(f"Warning: K-means++ initialization failed: {e}. Using random initialization.")
        # Fallback to random initialization
        u = np.random.rand(n_samples, n_clusters)
        row_sums_u = np.sum(u, axis=1, keepdims=True)
        row_sums_u[row_sums_u == 0] = 1
        u /= row_sums_u

        # Random center initialization
        initial_center_indices = np.random.choice(n_samples, n_clusters, replace=False)
        centers = data_input[initial_center_indices]

    # Initialize adaptive feature weights
    feature_weights = initialize_feature_weights(data_input, n_features)
    
    # Main CFCM iteration loop
    prev_objective = float('inf')

    for iteration in range(max_iter):
        u_old = u.copy()
        centers_old = centers.copy()

        # Update membership matrix with improved collaborative terms
        u = update_cfcm_membership(data_input, centers, feature_weights, beta, u_old)

        # Update cluster centers
        centers = update_cfcm_centers(data_input, u, feature_weights, beta)

        # Update feature weights adaptively (optional enhancement)
        if iteration % 10 == 0:  # Update weights every 10 iterations
            feature_weights = update_feature_weights(data_input, u, centers, feature_weights)

        # Calculate objective function for convergence check
        current_objective = compute_cfcm_objective(data_input, u, centers, feature_weights, beta)

        # Check convergence
        if iteration > 0:
            if abs(prev_objective - current_objective) < tol:
                if verbose:
                    print(f"CFCM converged (objective) at iteration {iteration+1}/{max_iter}")
                break

            if (np.max(np.abs(u - u_old)) < tol and
                np.max(np.abs(centers - centers_old)) < tol):
                if verbose:
                    print(f"CFCM converged (parameters) at iteration {iteration+1}/{max_iter}")
                break

        prev_objective = current_objective

        if verbose and (iteration + 1) % 20 == 0:
            print(f"CFCM Iteration {iteration+1}/{max_iter}, Objective: {current_objective:.6f}")
    
    # Final evaluation and metrics calculation
    labels = np.argmax(u, axis=1)
    metrics = calculate_sklearn_metrics(data_input, labels)

    if data_input.shape[0] > 1 and centers.shape[0] > 0 and u.shape[1] > 0:
        # Calculate fuzzy clustering metrics (using m=2 as standard)
        metrics.update(calculate_custom_metrics(data_input, u, centers, m=2.0))

        # Add CFCM-specific metrics
        metrics['cfcm_objective'] = compute_cfcm_objective(data_input, u, centers, feature_weights, beta)
        metrics['collaboration_strength'] = beta
        metrics['feature_weight_entropy'] = -np.sum(feature_weights * np.log(feature_weights + 1e-10))
        metrics['membership_concentration'] = np.mean(np.max(u, axis=1))

    if verbose:
        print(f"CFCM processing complete. Final metrics: {metrics}")

    return labels, u, centers, metrics

# --- Evaluation Metrics ---
@timing_decorator
def calculate_sklearn_metrics(X, labels):
    """
    Enhanced sklearn metrics calculation with robust error handling
    """
    metrics = {}

    # Input validation
    if X is None or labels is None:
        return _get_nan_metrics()

    if len(labels) == 0 or X.shape[0] < 2:
        return _get_nan_metrics()

    # Check for valid clustering
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)

    if n_clusters < 2 or n_clusters > X.shape[0] - 1:
        return _get_nan_metrics()

    # Remove noise points (label -1) if present
    if -1 in unique_labels:
        mask = labels != -1
        if np.sum(mask) < 2:
            return _get_nan_metrics()
        X_clean = X[mask]
        labels_clean = labels[mask]
        unique_clean = np.unique(labels_clean)
        if len(unique_clean) < 2:
            return _get_nan_metrics()
    else:
        X_clean = X
        labels_clean = labels

    # Calculate metrics with enhanced error handling
    try:
        metrics['silhouette'] = silhouette_score(X_clean, labels_clean)
    except (ValueError, RuntimeError) as e:
        metrics['silhouette'] = np.nan
        if len(X_clean) > 1000:  # Only warn for large datasets
            print(f"Warning: Silhouette score calculation failed: {e}")

    try:
        metrics['calinski_harabasz'] = calinski_harabasz_score(X_clean, labels_clean)
    except (ValueError, RuntimeError) as e:
        metrics['calinski_harabasz'] = np.nan
        if len(X_clean) > 1000:
            print(f"Warning: Calinski-Harabasz score calculation failed: {e}")

    try:
        metrics['davies_bouldin'] = davies_bouldin_score(X_clean, labels_clean)
    except (ValueError, RuntimeError) as e:
        metrics['davies_bouldin'] = np.nan
        if len(X_clean) > 1000:
            print(f"Warning: Davies-Bouldin score calculation failed: {e}")

    # Add additional robust metrics
    try:
        metrics['inertia'] = _calculate_inertia(X_clean, labels_clean)
        metrics['n_clusters'] = len(np.unique(labels_clean))
        metrics['n_noise'] = np.sum(labels == -1) if -1 in labels else 0
        metrics['cluster_sizes'] = _calculate_cluster_sizes(labels_clean)
    except Exception as e:
        print(f"Warning: Additional metrics calculation failed: {e}")

    return metrics

def _get_nan_metrics():
    """Return dictionary with NaN values for all metrics"""
    return {
        'silhouette': np.nan,
        'calinski_harabasz': np.nan,
        'davies_bouldin': np.nan,
        'inertia': np.nan,
        'n_clusters': 0,
        'n_noise': 0,
        'cluster_sizes': []
    }

def _calculate_inertia(X, labels):
    """Calculate within-cluster sum of squares (inertia)"""
    inertia = 0.0
    unique_labels = np.unique(labels)

    for label in unique_labels:
        cluster_points = X[labels == label]
        if len(cluster_points) > 0:
            centroid = np.mean(cluster_points, axis=0)
            inertia += np.sum((cluster_points - centroid) ** 2)

    return inertia

def _calculate_cluster_sizes(labels):
    """Calculate sizes of each cluster"""
    unique_labels, counts = np.unique(labels, return_counts=True)
    return dict(zip(unique_labels.tolist(), counts.tolist()))

@timing_decorator
def calculate_custom_metrics(X, U, V, m):
    """
    Enhanced custom fuzzy clustering metrics with robust error handling
    """
    metrics = {}

    # Enhanced validation for custom metrics inputs
    if X is None or U is None or V is None:
        return _get_nan_custom_metrics()

    if (X.shape[0] == 0 or U.shape[0] == 0 or V.shape[0] == 0 or
        U.shape[0] != X.shape[0] or U.shape[1] != V.shape[0] or
        V.shape[1] != X.shape[1] or U.shape[1] == 0):
        return _get_nan_custom_metrics()

    # Calculate PCI with enhanced error handling
    try:
        pci_val = pci_index(X, U)
        metrics['pci'] = pci_val if not (np.isnan(pci_val) or np.isinf(pci_val)) else np.nan
    except Exception as e:
        metrics['pci'] = np.nan
        print(f"Warning: PCI calculation failed: {e}")

    # Calculate FHV with enhanced error handling
    try:
        fhv_val = fhv_index(X, U, V, m)
        metrics['fhv'] = fhv_val if not (np.isnan(fhv_val) or np.isinf(fhv_val)) else np.nan
    except Exception as e:
        metrics['fhv'] = np.nan
        print(f"Warning: FHV calculation failed: {e}")

    # Calculate XBI with enhanced error handling
    try:
        xbi_val = xbi_index(X, U, V, m)
        metrics['xbi'] = xbi_val if not (np.isnan(xbi_val) or np.isinf(xbi_val)) else np.nan
    except Exception as e:
        metrics['xbi'] = np.nan
        print(f"Warning: XBI calculation failed: {e}")

    # Add additional robust metrics
    try:
        metrics['partition_entropy'] = _calculate_partition_entropy(U)
        metrics['fuzzy_silhouette'] = _calculate_fuzzy_silhouette(X, U)
        metrics['normalized_pci'] = metrics['pci'] / U.shape[1] if not np.isnan(metrics['pci']) else np.nan
    except Exception as e:
        print(f"Warning: Additional metrics calculation failed: {e}")
        metrics.update({
            'partition_entropy': np.nan,
            'fuzzy_silhouette': np.nan,
            'normalized_pci': np.nan
        })

    return metrics

def _get_nan_custom_metrics():
    """Return dictionary with NaN values for all custom metrics"""
    return {
        'pci': np.nan,
        'fhv': np.nan,
        'xbi': np.nan,
        'partition_entropy': np.nan,
        'fuzzy_silhouette': np.nan,
        'normalized_pci': np.nan
    }

def _calculate_partition_entropy(U):
    """Calculate partition entropy"""
    U_safe = np.maximum(U, 1e-10)  # Avoid log(0)
    return -np.mean(np.sum(U_safe * np.log(U_safe), axis=1))

def _calculate_fuzzy_silhouette(X, U):
    """Calculate fuzzy silhouette coefficient"""
    try:
        n_samples, n_clusters = U.shape
        if n_samples < 2 or n_clusters < 2:
            return np.nan

        # Convert to hard clustering for silhouette calculation
        hard_labels = np.argmax(U, axis=1)

        # Weight by membership strength
        membership_strength = np.max(U, axis=1)

        # Calculate standard silhouette
        from sklearn.metrics import silhouette_samples
        silhouette_vals = silhouette_samples(X, hard_labels)

        # Weight by fuzzy membership strength
        fuzzy_silhouette = np.mean(silhouette_vals * membership_strength)

        return fuzzy_silhouette
    except Exception:
        return np.nan

def comprehensive_evaluation(data, labels, membership_matrix=None, centroids=None,
                           algorithm_name="Unknown", verbose=True):
    """
    Comprehensive evaluation framework for clustering results

    Args:
        data: Input data
        labels: Cluster labels
        membership_matrix: Fuzzy membership matrix (optional)
        centroids: Cluster centroids (optional)
        algorithm_name: Name of the algorithm
        verbose: Print detailed results
    """
    evaluation_results = {
        'algorithm': algorithm_name,
        'n_samples': len(data) if data is not None else 0,
        'n_features': data.shape[1] if data is not None and len(data.shape) > 1 else 0,
        'n_clusters_found': len(np.unique(labels)) if labels is not None and len(labels) > 0 else 0
    }

    # Calculate standard metrics
    try:
        sklearn_metrics = calculate_sklearn_metrics(data, labels)
        evaluation_results.update(sklearn_metrics)
    except Exception as e:
        if verbose:
            print(f"Warning: Standard metrics calculation failed for {algorithm_name}: {e}")
        evaluation_results.update(_get_nan_metrics())

    # Calculate fuzzy metrics if membership matrix is available
    if membership_matrix is not None and centroids is not None:
        try:
            custom_metrics = calculate_custom_metrics(data, membership_matrix, centroids, m=2.0)
            evaluation_results.update(custom_metrics)
        except Exception as e:
            if verbose:
                print(f"Warning: Custom metrics calculation failed for {algorithm_name}: {e}")
            evaluation_results.update(_get_nan_custom_metrics())

    # Add algorithm-specific quality indicators
    try:
        evaluation_results['quality_score'] = _calculate_overall_quality_score(evaluation_results)
        evaluation_results['convergence_quality'] = _assess_convergence_quality(evaluation_results)
        evaluation_results['stability_score'] = _assess_stability(data, labels)
    except Exception as e:
        if verbose:
            print(f"Warning: Quality assessment failed for {algorithm_name}: {e}")
        evaluation_results.update({
            'quality_score': np.nan,
            'convergence_quality': 'unknown',
            'stability_score': np.nan
        })

    if verbose:
        print(f"\n=== {algorithm_name} Evaluation Results ===")
        print(f"Samples: {evaluation_results['n_samples']}, Features: {evaluation_results['n_features']}")
        print(f"Clusters found: {evaluation_results['n_clusters_found']}")
        print(f"Silhouette Score: {evaluation_results.get('silhouette', 'N/A'):.4f}" if not np.isnan(evaluation_results.get('silhouette', np.nan)) else "Silhouette Score: N/A")
        print(f"Calinski-Harabasz: {evaluation_results.get('calinski_harabasz', 'N/A'):.4f}" if not np.isnan(evaluation_results.get('calinski_harabasz', np.nan)) else "Calinski-Harabasz: N/A")
        print(f"Davies-Bouldin: {evaluation_results.get('davies_bouldin', 'N/A'):.4f}" if not np.isnan(evaluation_results.get('davies_bouldin', np.nan)) else "Davies-Bouldin: N/A")
        if 'pci' in evaluation_results:
            print(f"PCI: {evaluation_results.get('pci', 'N/A'):.4f}" if not np.isnan(evaluation_results.get('pci', np.nan)) else "PCI: N/A")
        print(f"Overall Quality: {evaluation_results.get('quality_score', 'N/A'):.4f}" if not np.isnan(evaluation_results.get('quality_score', np.nan)) else "Overall Quality: N/A")

    return evaluation_results

def _calculate_overall_quality_score(metrics):
    """Calculate an overall quality score from multiple metrics"""
    scores = []
    weights = []

    # Silhouette score (higher is better, range [-1, 1])
    if not np.isnan(metrics.get('silhouette', np.nan)):
        scores.append((metrics['silhouette'] + 1) / 2)  # Normalize to [0, 1]
        weights.append(0.3)

    # Calinski-Harabasz (higher is better, normalize by log)
    if not np.isnan(metrics.get('calinski_harabasz', np.nan)) and metrics['calinski_harabasz'] > 0:
        scores.append(min(1.0, np.log(metrics['calinski_harabasz']) / 10))  # Rough normalization
        weights.append(0.2)

    # Davies-Bouldin (lower is better)
    if not np.isnan(metrics.get('davies_bouldin', np.nan)) and metrics['davies_bouldin'] > 0:
        scores.append(1.0 / (1.0 + metrics['davies_bouldin']))  # Inverse normalization
        weights.append(0.2)

    # PCI (higher is better, range [0, 1])
    if not np.isnan(metrics.get('pci', np.nan)):
        scores.append(metrics['pci'])
        weights.append(0.3)

    if len(scores) == 0:
        return np.nan

    # Weighted average
    weights = np.array(weights)
    weights = weights / np.sum(weights)  # Normalize weights

    return np.average(scores, weights=weights)

def _assess_convergence_quality(metrics):
    """Assess the quality of convergence based on metrics"""
    if np.isnan(metrics.get('silhouette', np.nan)):
        return 'poor'

    silhouette = metrics['silhouette']

    if silhouette > 0.7:
        return 'excellent'
    elif silhouette > 0.5:
        return 'good'
    elif silhouette > 0.25:
        return 'fair'
    else:
        return 'poor'

def _assess_stability(data, labels):
    """Assess clustering stability using bootstrap sampling"""
    try:
        if data is None or labels is None or len(labels) < 10:
            return np.nan

        n_samples = len(labels)
        n_bootstrap = min(5, n_samples // 10)  # Limit bootstrap samples

        stability_scores = []

        for _ in range(n_bootstrap):
            # Bootstrap sample
            indices = np.random.choice(n_samples, size=n_samples//2, replace=True)
            sample_data = data[indices]
            sample_labels = labels[indices]

            # Calculate silhouette for bootstrap sample
            try:
                if len(np.unique(sample_labels)) > 1:
                    from sklearn.metrics import silhouette_score
                    score = silhouette_score(sample_data, sample_labels)
                    stability_scores.append(score)
            except:
                continue

        if len(stability_scores) > 0:
            return np.std(stability_scores)  # Lower std = more stable
        else:
            return np.nan

    except Exception:
        return np.nan

# --- Main Execution ---
def main():
    results = []
    verbose_run = True 
    base_path = '.' # Assuming run.py is in the project root

    # Load custom evaluation metrics from 'tools' directory
    global pci_index, fhv_index, xbi_index
    tools_dir = os.path.join(base_path, 'tools')
    try:
        with open(os.path.join(tools_dir, 'pci.py'), 'r') as f_pci: exec(f_pci.read(), globals())
        with open(os.path.join(tools_dir, 'fhv.py'), 'r') as f_fhv: exec(f_fhv.read(), globals())
        with open(os.path.join(tools_dir, 'xbi.py'), 'r') as f_xbi: exec(f_xbi.read(), globals())
        print(f"Successfully loaded custom evaluation metrics from '{tools_dir}'.")
    except FileNotFoundError:
        print(f"Warning: One or more evaluation tool files (pci.py, fhv.py, xbi.py) not found in '{tools_dir}'.")
        print("Custom metrics (PCI, FHV, XBI) will default to NaN.")
    except Exception as e:
        print(f"Error loading custom metrics from '{tools_dir}': {e}. Custom metrics will default to NaN.")


    # --- USPS Dataset ---
    print("\nProcessing USPS Dataset...")
    X_usps_flat, y_usps = load_usps_data(base_path)
    
    if X_usps_flat is not None:
        usps_n_clusters = 10 
        usps_input_dim = X_usps_flat.shape[1]

        print("\nRunning DEKM on USPS...")
        try:
            X_usps_tensor, X_usps_scaled_np = preprocess_usps_for_dekm(X_usps_flat.copy())
            if X_usps_tensor is not None:
                dekm_labels, _, dekm_metrics = run_dekm(
                    X_usps_tensor, X_usps_scaled_np, k=usps_n_clusters, Iter=3, 
                    input_dim=usps_input_dim, hidden_dim_ae=usps_n_clusters, verbose=verbose_run
                )
                results.append({'dataset': 'USPS', 'algorithm': 'DEKM', **dekm_metrics})
                print("DEKM on USPS results:", dekm_metrics)
            else: results.append({'dataset': 'USPS', 'algorithm': 'DEKM', 'error': 'Preprocessing failed'})
        except Exception as e:
            print(f"Error running DEKM on USPS: {e}")
            results.append({'dataset': 'USPS', 'algorithm': 'DEKM', 'error': str(e)})

        print("\nRunning PFCM on USPS...")
        try:
            X_usps_pca = preprocess_usps_for_pfcm(X_usps_flat.copy())
            if X_usps_pca is not None:
                _, _, _, _, pfcm_metrics = run_pfcm(
                    X_usps_pca, n_clusters=usps_n_clusters, verbose=verbose_run
                )
                results.append({'dataset': 'USPS', 'algorithm': 'PFCM', **pfcm_metrics})
                print("PFCM on USPS results:", pfcm_metrics)
            else: results.append({'dataset': 'USPS', 'algorithm': 'PFCM', 'error': 'Preprocessing failed'})
        except Exception as e:
            print(f"Error running PFCM on USPS: {e}")
            results.append({'dataset': 'USPS', 'algorithm': 'PFCM', 'error': str(e)})

        print("\nRunning FCM on USPS...")
        try:
            _, X_usps_fcm_input = preprocess_usps_for_dekm(X_usps_flat.copy()) # Use standard scaled data
            if X_usps_fcm_input is not None:
                _, _, _, fcm_metrics_usps = run_fcm(
                    X_usps_fcm_input, n_clusters=usps_n_clusters, verbose=verbose_run
                )
                results.append({'dataset': 'USPS', 'algorithm': 'FCM', **fcm_metrics_usps})
                print("FCM on USPS results:", fcm_metrics_usps)
            else: results.append({'dataset': 'USPS', 'algorithm': 'FCM', 'error': 'Preprocessing failed'})
        except Exception as e:
            print(f"Error running FCM on USPS: {e}")
            results.append({'dataset': 'USPS', 'algorithm': 'FCM', 'error': str(e)})

        print("\nRunning K-Means (Standalone) on USPS...")
        try:
            # X_usps_scaled_np is available from DEKM preprocessing, or could be X_usps_fcm_input
            if X_usps_scaled_np is not None:
                _, _, kmeans_metrics_usps = run_kmeans_standalone(
                    X_usps_scaled_np, n_clusters=usps_n_clusters, verbose=verbose_run
                )
                results.append({'dataset': 'USPS', 'algorithm': 'KMeans', **kmeans_metrics_usps})
                print("K-Means on USPS results:", kmeans_metrics_usps)
            else: results.append({'dataset': 'USPS', 'algorithm': 'KMeans', 'error': 'Preprocessing failed or input data missing'})
        except Exception as e:
            print(f"Error running K-Means on USPS: {e}")
            results.append({'dataset': 'USPS', 'algorithm': 'KMeans', 'error': str(e)})

        print("\nRunning DBSCAN on USPS...")
        try:
            if X_usps_scaled_np is not None:
                _, dbscan_metrics_usps = run_dbscan(
                    X_usps_scaled_np, eps=2.0, min_samples=10, verbose=verbose_run # Params might need tuning
                )
                results.append({'dataset': 'USPS', 'algorithm': 'DBSCAN', **dbscan_metrics_usps})
                print("DBSCAN on USPS results:", dbscan_metrics_usps)
            else: results.append({'dataset': 'USPS', 'algorithm': 'DBSCAN', 'error': 'Preprocessing failed or input data missing'})
        except Exception as e:
            print(f"Error running DBSCAN on USPS: {e}")
            results.append({'dataset': 'USPS', 'algorithm': 'DBSCAN', 'error': str(e)})

        print("\nRunning CFCM on USPS...")
        try:
            if X_usps_scaled_np is not None:
                _, _, _, cfcm_metrics_usps = run_cfcm(
                    X_usps_scaled_np, n_clusters=usps_n_clusters, verbose=verbose_run
                )
                results.append({'dataset': 'USPS', 'algorithm': 'CFCM', **cfcm_metrics_usps})
                print("CFCM on USPS results:", cfcm_metrics_usps)
            else: results.append({'dataset': 'USPS', 'algorithm': 'CFCM', 'error': 'Preprocessing failed or input data missing'})
        except Exception as e:
            print(f"Error running CFCM on USPS: {e}")
            results.append({'dataset': 'USPS', 'algorithm': 'CFCM', 'error': str(e)})
    else:
        print("Skipping USPS dataset processing due to loading error.")

    # --- E-Commerce Dataset ---
    print("\nProcessing E-Commerce Dataset...")
    X_ecommerce = load_ecommerce_data(base_path)

    if X_ecommerce is not None:
        ecommerce_n_clusters = 3
        ecom_input_dim = X_ecommerce.shape[1]

        print("\nRunning FCM on E-Commerce...")
        try:
            _, _, _, fcm_metrics_ecom = run_fcm(
                X_ecommerce.copy(), n_clusters=ecommerce_n_clusters, verbose=verbose_run
            )
            results.append({'dataset': 'E-Commerce', 'algorithm': 'FCM', **fcm_metrics_ecom})
            print("FCM on E-Commerce results:", fcm_metrics_ecom)
        except Exception as e:
            print(f"Error running FCM on E-Commerce: {e}")
            results.append({'dataset': 'E-Commerce', 'algorithm': 'FCM', 'error': str(e)})

        print("\nRunning DEKM on E-Commerce...")
        try:
            X_ecom_tensor = torch.tensor(X_ecommerce.copy(), dtype=torch.float32)
            dekm_labels_ecom, _, dekm_metrics_ecom = run_dekm(
                X_ecom_tensor, X_ecommerce.copy(), k=ecommerce_n_clusters, Iter=3,
                input_dim=ecom_input_dim, hidden_dim_ae=ecommerce_n_clusters, verbose=verbose_run
            )
            results.append({'dataset': 'E-Commerce', 'algorithm': 'DEKM', **dekm_metrics_ecom})
            print("DEKM on E-Commerce results:", dekm_metrics_ecom)
        except Exception as e:
            print(f"Error running DEKM on E-Commerce: {e}")
            results.append({'dataset': 'E-Commerce', 'algorithm': 'DEKM', 'error': str(e)})
            
        print("\nRunning PFCM on E-Commerce...")
        try:
            _, _, _, _, pfcm_metrics_ecom = run_pfcm(
                X_ecommerce.copy(), n_clusters=ecommerce_n_clusters, verbose=verbose_run
            )
            results.append({'dataset': 'E-Commerce', 'algorithm': 'PFCM', **pfcm_metrics_ecom})
            print("PFCM on E-Commerce results:", pfcm_metrics_ecom)
        except Exception as e:
            print(f"Error running PFCM on E-Commerce: {e}")
            results.append({'dataset': 'E-Commerce', 'algorithm': 'PFCM', 'error': str(e)})

        print("\nRunning K-Means (Standalone) on E-Commerce...")
        try:
            _, _, kmeans_metrics_ecom = run_kmeans_standalone(
                X_ecommerce.copy(), n_clusters=ecommerce_n_clusters, verbose=verbose_run
            )
            results.append({'dataset': 'E-Commerce', 'algorithm': 'KMeans', **kmeans_metrics_ecom})
            print("K-Means on E-Commerce results:", kmeans_metrics_ecom)
        except Exception as e:
            print(f"Error running K-Means on E-Commerce: {e}")
            results.append({'dataset': 'E-Commerce', 'algorithm': 'KMeans', 'error': str(e)})

        print("\nRunning DBSCAN on E-Commerce...")
        try:
            _, dbscan_metrics_ecom = run_dbscan(
                X_ecommerce.copy(), eps=0.5, min_samples=5, verbose=verbose_run # Params might need tuning
            )
            results.append({'dataset': 'E-Commerce', 'algorithm': 'DBSCAN', **dbscan_metrics_ecom})
            print("DBSCAN on E-Commerce results:", dbscan_metrics_ecom)
        except Exception as e:
            print(f"Error running DBSCAN on E-Commerce: {e}")
            results.append({'dataset': 'E-Commerce', 'algorithm': 'DBSCAN', 'error': str(e)})

        print("\nRunning CFCM on E-Commerce...")
        try:
            if X_ecommerce is not None:
                _, _, _, cfcm_metrics_ecom = run_cfcm(
                    X_ecommerce.copy(), n_clusters=ecommerce_n_clusters, verbose=verbose_run
                )
                results.append({'dataset': 'E-Commerce', 'algorithm': 'CFCM', **cfcm_metrics_ecom})
                print("CFCM on E-Commerce results:", cfcm_metrics_ecom)
            else: results.append({'dataset': 'E-Commerce', 'algorithm': 'CFCM', 'error': 'Preprocessing failed or input data missing'})
        except Exception as e:
            print(f"Error running CFCM on E-Commerce: {e}")
            results.append({'dataset': 'E-Commerce', 'algorithm': 'CFCM', 'error': str(e)})
    else:
        print("Skipping E-Commerce dataset processing due to loading error.")

    # --- Country Data ---
    print("\nProcessing Country Data...")
    X_country, y_country = load_country_data(base_path)
    
    if X_country is not None:
        country_n_clusters = 3
        country_input_dim = X_country.shape[1]

        print("\nRunning FDEKM on Country Data...")
        try:
            X_country_tensor = torch.tensor(X_country.copy(), dtype=torch.float32)
            fdekm_labels_country, _, fdekm_metrics_country = run_fdekm(
                X_country_tensor, X_country.copy(), k=country_n_clusters, Iter=3,
                input_dim=country_input_dim, hidden_dim_ae=country_n_clusters, m=2.0, verbose=verbose_run
            )
            results.append({'dataset': 'Country', 'algorithm': 'FDEKM', **fdekm_metrics_country})
            print("FDEKM on Country Data results:", fdekm_metrics_country)
        except Exception as e:
            print(f"Error running FDEKM on Country Data: {e}")
            results.append({'dataset': 'Country', 'algorithm': 'FDEKM', 'error': str(e)})

        print("\nRunning FCM on Country Data...")
        try:
            _, _, _, fcm_metrics_country = run_fcm(
                X_country.copy(), n_clusters=country_n_clusters, verbose=verbose_run
            )
            results.append({'dataset': 'Country', 'algorithm': 'FCM', **fcm_metrics_country})
            print("FCM on Country Data results:", fcm_metrics_country)
        except Exception as e:
            print(f"Error running FCM on Country Data: {e}")
            results.append({'dataset': 'Country', 'algorithm': 'FCM', 'error': str(e)})

        print("\nRunning DEKM on Country Data...")
        try:
            X_country_tensor = torch.tensor(X_country.copy(), dtype=torch.float32)
            dekm_labels_country, _, dekm_metrics_country = run_dekm(
                X_country_tensor, X_country.copy(), k=country_n_clusters, Iter=3,
                input_dim=country_input_dim, hidden_dim_ae=country_n_clusters, verbose=verbose_run
            )
            results.append({'dataset': 'Country', 'algorithm': 'DEKM', **dekm_metrics_country})
            print("DEKM on Country Data results:", dekm_metrics_country)
        except Exception as e:
            print(f"Error running DEKM on Country Data: {e}")
            results.append({'dataset': 'Country', 'algorithm': 'DEKM', 'error': str(e)})
            
        print("\nRunning PFCM on Country Data...")
        try:
            _, _, _, _, pfcm_metrics_country = run_pfcm(
                X_country.copy(), n_clusters=country_n_clusters, verbose=verbose_run
            )
            results.append({'dataset': 'Country', 'algorithm': 'PFCM', **pfcm_metrics_country})
            print("PFCM on Country Data results:", pfcm_metrics_country)
        except Exception as e:
            print(f"Error running PFCM on Country Data: {e}")
            results.append({'dataset': 'Country', 'algorithm': 'PFCM', 'error': str(e)})

        print("\nRunning K-Means (Standalone) on Country Data...")
        try:
            _, _, kmeans_metrics_country = run_kmeans_standalone(
                X_country.copy(), n_clusters=country_n_clusters, verbose=verbose_run
            )
            results.append({'dataset': 'Country', 'algorithm': 'KMeans', **kmeans_metrics_country})
            print("K-Means on Country Data results:", kmeans_metrics_country)
        except Exception as e:
            print(f"Error running K-Means on Country Data: {e}")
            results.append({'dataset': 'Country', 'algorithm': 'KMeans', 'error': str(e)})

        print("\nRunning DBSCAN on Country Data...")
        try:
            # Adjust DBSCAN parameters for better clustering
            _, dbscan_metrics_country = run_dbscan(
                X_country.copy(), eps=2.0, min_samples=3, verbose=verbose_run  # Adjusted parameters
            )
            results.append({'dataset': 'Country', 'algorithm': 'DBSCAN', **dbscan_metrics_country})
            print("DBSCAN on Country Data results:", dbscan_metrics_country)
        except Exception as e:
            print(f"Error running DBSCAN on Country Data: {e}")
            results.append({'dataset': 'Country', 'algorithm': 'DBSCAN', 'error': str(e)})

        print("\nRunning CFCM on Country Data...")
        try:
            if X_country is not None:
                # Ensure data dimensions match for CFCM
                X_country_cfcm = X_country.copy()
                _, _, _, cfcm_metrics_country = run_cfcm(
                    X_country_cfcm, n_clusters=country_n_clusters, beta=0.5, max_iter=50, verbose=verbose_run
                )
                results.append({'dataset': 'Country', 'algorithm': 'CFCM', **cfcm_metrics_country})
                print("CFCM on Country Data results:", cfcm_metrics_country)
            else: results.append({'dataset': 'Country', 'algorithm': 'CFCM', 'error': 'Preprocessing failed or input data missing'})
        except Exception as e:
            print(f"Error running CFCM on Country Data: {e}")
            results.append({'dataset': 'Country', 'algorithm': 'CFCM', 'error': str(e)})
    else:
        print("Skipping Country dataset processing due to loading error.")

    # --- Save results to CSV ---
    if results:
        df_results = pd.DataFrame(results)
        # Define column order for clarity
        cols = ['dataset', 'algorithm', 'silhouette', 'calinski_harabasz', 'davies_bouldin', 'pci', 'fhv', 'xbi', 'error']
        # Filter df_results to only include existing columns from cols
        df_results = df_results.reindex(columns=[col for col in cols if col in df_results.columns])
        
        output_file = os.path.join(base_path, 'output.csv')
        try:
            df_results.to_csv(output_file, index=False)
            print(f"\nResults saved to {output_file}")
        except Exception as e:
            print(f"\nError saving results to CSV '{output_file}': {e}")
            print("Printing results to console instead:")
            print(df_results.to_string())
    else:
        print("\nNo results to save.")

def run_fdekm(X_tensor, X_for_metrics, k=10, Iter=15, input_dim=None, hidden_dim_ae=10,
              m=2.0, alpha=1.0, beta=0.5, gamma=0.1, verbose=False):
    """
    Improved Fuzzy Deep Embedded K-Means (FDEKM) Algorithm

    Args:
        X_tensor: Input data tensor (N, D)
        X_for_metrics: Data for metric calculation
        k: Number of clusters
        Iter: Number of FDEKM iterations
        input_dim: Input dimension (auto-detected if None)
        hidden_dim_ae: Latent dimension
        m: Fuzzifier parameter (m > 1)
        alpha: Weight for reconstruction loss
        beta: Weight for fuzzy clustering loss
        gamma: Weight for structure preservation loss
        verbose: Print progress
    """
    if X_tensor.shape[0] == 0:
        print("Error: FDEKM input tensor is empty.")
        return np.array([]), np.array([]), np.array([]), np.array([]), {}

    if input_dim is None:
        input_dim = X_tensor.shape[1]

    if X_tensor.shape[0] < k:
        print(f"Warning: FDEKM num_samples ({X_tensor.shape[0]}) < k ({k}). Reducing k.")
        k = max(1, X_tensor.shape[0])

    # Initialize improved autoencoder
    model = AutoEncoder_DEKM(input_dim=input_dim, hidden_dim_ae=hidden_dim_ae)

    # Enhanced pretraining
    if verbose:
        print("Starting FDEKM autoencoder pretraining...")
    train_autoencoder_dekm(model, X_tensor, epochs=100, verbose=verbose)

    # Initialize variables
    H_np_final = None
    U_final = None
    V_final = None
    best_loss = float('inf')

    for it in range(Iter):
        model.eval()
        with torch.no_grad():
            _, H = model(X_tensor)
            H_np = H.numpy()
        
        if H_np.shape[0] == 0:
            print(f"FDEKM Iter {it+1}: H_np is empty. Skipping iteration.")
            continue
        
        actual_k = min(k, H_np.shape[0])
        if actual_k < 1:
            print(f"FDEKM Iter {it+1}: actual_k is {actual_k}. Skipping iteration.")
            continue

        # Initialize cluster centers using K-means
        kmeans = KMeans(n_clusters=actual_k, n_init='auto', random_state=42)
        try:
            kmeans.fit(H_np)
            V = kmeans.cluster_centers_
        except Exception as e:
            print(f"FDEKM Iter {it+1}: K-means initialization failed: {e}. Skipping iteration.")
            continue

        # Initialize membership matrix U
        U = np.random.rand(H_np.shape[0], actual_k)
        U = U / np.sum(U, axis=1, keepdims=True)

        # Enhanced Fuzzy C-Means clustering
        try:
            # Initialize fuzzy membership matrix with soft assignments
            U = initialize_fuzzy_membership(H_np, V, m)

            # Enhanced FCM iterations with better convergence
            prev_objective = float('inf')
            max_fuzzy_iter = 50
            for fuzzy_iter in range(max_fuzzy_iter):
                U_old = U.copy()

                # Update membership matrix with numerical stability
                U = update_fuzzy_membership(H_np, V, m)

                # Update centroids with weighted average
                V = update_fuzzy_centroids(H_np, U, m)

                # Calculate objective function for convergence check
                current_objective = compute_fcm_objective(H_np, U, V, m)

                # Check convergence
                if abs(prev_objective - current_objective) < 1e-6:
                    if verbose:
                        print(f"  FDEKM FCM converged at iteration {fuzzy_iter+1}")
                    break

                if np.allclose(U, U_old, atol=1e-6):
                    if verbose:
                        print(f"  FDEKM FCM converged (parameters) at iteration {fuzzy_iter+1}")
                    break

                prev_objective = current_objective

        except Exception as e:
            print(f"FDEKM Iter {it+1}: Enhanced FCM failed: {e}. Using basic FCM.")
            # Fallback to basic FCM if enhanced version fails
            U = np.random.rand(H_np.shape[0], actual_k)
            U = U / np.sum(U, axis=1, keepdims=True)

        # Joint optimization with improved fuzzy constraints
        V_tensor = torch.tensor(V, dtype=torch.float32, device=X_tensor.device)
        U_tensor = torch.tensor(U, dtype=torch.float32, device=X_tensor.device)

        optimizer_joint = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

        model.train()
        for epoch in range(5):  # Joint training epochs
            optimizer_joint.zero_grad()

            # Forward pass
            reconstructed, embeddings = model(X_tensor)

            # Reconstruction loss
            loss_recon = nn.MSELoss()(reconstructed, X_tensor)

            # Fuzzy clustering loss (improved)
            loss_fuzzy = compute_fuzzy_clustering_loss(embeddings, U_tensor, V_tensor, m)

            # Structure preservation loss
            loss_structure = compute_structure_preservation_loss(embeddings, H_np)

            # Combined loss with proper weighting
            total_loss = alpha * loss_recon + beta * loss_fuzzy + gamma * loss_structure

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer_joint.step()

            if epoch == 4:  # Store final loss
                final_loss = total_loss.item()

        H_np_final = H_np
        U_final = U
        V_final = V

        if verbose:
            print(f"FDEKM Iteration {it+1}/{Iter} completed. Loss: {final_loss:.6f}")

    # Final evaluation and metrics calculation
    metrics = {}
    if U_final is not None and H_np_final is not None and H_np_final.shape[0] > 0:
        # Get hard labels from fuzzy membership
        labels = np.argmax(U_final, axis=1)

        # Calculate metrics on latent space
        metrics = calculate_sklearn_metrics(H_np_final, labels)

        if H_np_final.shape[0] > 1 and len(np.unique(labels)) > 1:
            # Calculate fuzzy clustering specific metrics
            metrics.update(calculate_custom_metrics(H_np_final, U_final, V_final, m=m))

            # Add FDEKM-specific metrics
            metrics['fuzzy_partition_coefficient'] = np.mean(np.sum(U_final ** 2, axis=1))
            metrics['fuzzy_partition_entropy'] = -np.mean(np.sum(U_final * np.log(U_final + 1e-10), axis=1))
            metrics['final_objective'] = compute_fcm_objective(H_np_final, U_final, V_final, m)

    else:
        # Fallback for failed cases
        labels = np.array([])
        H_np_final = np.array([])
        U_final = np.array([])
        V_final = np.array([])
        metrics = {
            'silhouette': np.nan, 'calinski_harabasz': np.nan, 'davies_bouldin': np.nan,
            'pci': np.nan, 'fhv': np.nan, 'xbi': np.nan,
            'fuzzy_partition_coefficient': np.nan, 'fuzzy_partition_entropy': np.nan,
            'final_objective': np.nan
        }

    if verbose:
        print(f"FDEKM completed. Final metrics: {metrics}")

    return labels, H_np_final, U_final, V_final, metrics

if __name__ == '__main__':
    main() 