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
    def __init__(self, input_dim, hidden_dim_ae):
        super(AutoEncoder_DEKM, self).__init__()
        intermediate_dim1 = 128
        if input_dim < intermediate_dim1 / 2 and input_dim > 0 :
            intermediate_dim1 = max(hidden_dim_ae * 2, input_dim * 2, 1) # Ensure > 0
        elif input_dim == 0: # Should not happen with valid data
            intermediate_dim1 = max(hidden_dim_ae *2, 1)


        if hidden_dim_ae <=0: hidden_dim_ae=1 # Ensure hidden_dim is at least 1

        # Ensure intermediate_dim1 is not smaller than hidden_dim_ae
        if intermediate_dim1 < hidden_dim_ae and input_dim > hidden_dim_ae :
            intermediate_dim1 = hidden_dim_ae * 2
            
        if input_dim <= hidden_dim_ae : # Simplified AE if input_dim is too small or not for reduction
             print(f"DEKM AutoEncoder: Input dim ({input_dim}) <= Hidden dim ({hidden_dim_ae}). Using very simplified AE.")
             self.encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim_ae))
             self.decoder = nn.Sequential(nn.Linear(hidden_dim_ae, input_dim))
        else:
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, intermediate_dim1),
                nn.ReLU(),
                nn.Linear(intermediate_dim1, hidden_dim_ae)
            )
            self.decoder = nn.Sequential(
                nn.Linear(hidden_dim_ae, intermediate_dim1),
                nn.ReLU(),
                nn.Linear(intermediate_dim1, input_dim)
            )

    def forward(self, x):
        h = self.encoder(x)
        out = self.decoder(h)
        return out, h

def train_autoencoder_dekm(model, data, epochs=50, lr=1e-3, verbose=False):
    if data.shape[0] == 0:
        print("Error: DEKM Autoencoder training data is empty.")
        return
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output, _ = model(data)
        loss = criterion(output, data)
        loss.backward()
        optimizer.step()
        if verbose and (epoch+1) % 10 == 0:
            print(f"AE Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

def run_dekm(X_tensor, X_for_metrics, k=10, Iter=10, input_dim=256, hidden_dim_ae=10, verbose=False):
    if X_tensor.shape[0] == 0:
        print("Error: DEKM input tensor is empty.")
        return np.array([]), np.array([]), {}
    if X_tensor.shape[0] < k :
        print(f"Warning: DEKM num_samples ({X_tensor.shape[0]}) < k ({k}). Reducing k for K-Means.")
        k = max(1,X_tensor.shape[0]) # k must be at least 1

    model = AutoEncoder_DEKM(input_dim=input_dim, hidden_dim_ae=hidden_dim_ae)
    train_autoencoder_dekm(model, X_tensor, epochs=20, verbose=verbose) # Reduced epochs for speed

    H_np_final = None
    labels_final = None
    total_loss_final = torch.tensor(float('nan'))


    for it in range(Iter):
        model.eval()
        with torch.no_grad():
            _, H = model(X_tensor)
            H_np = H.numpy()
        
        if H_np.shape[0] == 0: 
            print(f"DEKM Iter {it+1}: H_np is empty, cannot run K-Means. Skipping iteration.")
            continue
        
        actual_k_kmeans = min(k, H_np.shape[0])
        if actual_k_kmeans < 1:
            print(f"DEKM Iter {it+1}: actual_k_kmeans is {actual_k_kmeans}, cannot run K-Means. Skipping iteration.")
            continue
            
        kmeans = KMeans(n_clusters=actual_k_kmeans, n_init='auto', random_state=42)
        try:
            labels = kmeans.fit_predict(H_np)
        except ValueError as e:
            print(f"DEKM Iter {it+1}: K-Means failed: {e}. Skipping iteration.")
            continue # Skip if k-means fails

        H_np_final = H_np
        labels_final = labels
        
        # Sw calculation
        Sw = np.zeros((H_np.shape[1], H_np.shape[1]))
        if H_np.shape[1] == 0: # Latent dim is 0
            print(f"DEKM Iter {it+1}: Latent dimension is 0. Skipping Sw calculation.")
            V_np = np.array([]) # No V if latent dim is 0
        else:
            for i in range(actual_k_kmeans):
                cluster_points = H_np[labels == i]
                if len(cluster_points) == 0: continue
                mu_i = np.mean(cluster_points, axis=0, keepdims=True)
                for h_val in cluster_points:
                    diff = (h_val - mu_i).reshape(-1, 1)
                    Sw += diff @ diff.T
            
            try:
                # Target rank for V is hidden_dim_ae (model.encoder[-1].out_features)
                target_v_cols = model.encoder[-1][2].out_features if isinstance(model.encoder[-1], nn.Sequential) else model.encoder[-1].out_features

                eigvals, eigvecs = eigh(Sw)
                V_np = eigvecs[:, :target_v_cols]
            except np.linalg.LinAlgError:
                print(f"Warning: DEKM eigh(Sw) failed in iter {it+1}. Using identity if possible.")
                if H_np.shape[1] > 0:
                     V_np = np.eye(H_np.shape[1], min(H_np.shape[1], target_v_cols))
                else: V_np = np.array([]) # Cannot form identity
            except Exception as e:
                 print(f"Error in DEKM Sw/V computation iter {it+1}: {e}. Using identity if possible.")
                 if H_np.shape[1] > 0:
                     V_np = np.eye(H_np.shape[1], min(H_np.shape[1], target_v_cols))
                 else: V_np = np.array([])


        if V_np.size == 0 : # If V could not be computed
            print(f"DEKM Iter {it+1}: V matrix is empty. Skipping model refinement.")
        else:
            V = torch.tensor(V_np, dtype=torch.float32)
            optimizer = optim.Adam(model.parameters(), lr=1e-3)
            lambda_reg = 0.1

            for epoch_refine in range(10): # Reduced epochs for speed
                model.train()
                optimizer.zero_grad()
                out, H_refined = model(X_tensor)
                projection = H_refined @ V @ V.T
                loss_recon = nn.MSELoss()(out, X_tensor)
                # Ensure H_refined and projection have compatible shapes for norm calculation
                if H_refined.shape == projection.shape:
                    loss_constraint = torch.norm(H_refined - projection)
                    total_loss_final = loss_recon + lambda_reg * loss_constraint
                    total_loss_final.backward()
                    optimizer.step()
                else:
                    # This case means V was problematic or H_refined shape is unexpected
                    print(f"DEKM Refinement Warning: Shape mismatch H_refined {H_refined.shape} vs projection {projection.shape}. Skipping constraint.")
                    total_loss_final = loss_recon
                    total_loss_final.backward()
                    optimizer.step()


        if verbose:
            print(f"DEKM Iteration {it+1}/{Iter}, Loss: {total_loss_final.item():.4f}")
    
    metrics = {}
    if labels_final is not None and H_np_final is not None and H_np_final.shape[0] > 0:
        metrics = calculate_sklearn_metrics(H_np_final, labels_final) # Metrics on latent space
        if H_np_final.shape[0] > 1 and len(np.unique(labels_final)) > 1:
            num_clusters_dekm = len(np.unique(labels_final))
            if num_clusters_dekm > 0 and H_np_final.shape[0] >= num_clusters_dekm:
                U_dekm_full = np.zeros((H_np_final.shape[0], num_clusters_dekm))
                unique_labels_arr = np.unique(labels_final)
                label_to_idx = {label: i for i, label in enumerate(unique_labels_arr)}
                for i, label in enumerate(labels_final):
                    U_dekm_full[i, label_to_idx[label]] = 1.0
                
                kmeans_final = KMeans(n_clusters=num_clusters_dekm, n_init='auto', random_state=42).fit(H_np_final)
                V_dekm_latent = kmeans_final.cluster_centers_
                metrics.update(calculate_custom_metrics(H_np_final, U_dekm_full, V_dekm_latent, m=2.0))
    else: # No valid labels or H_np
        labels_final = np.array([])
        H_np_final = np.array([])
        
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

def update_U_T_pfcm(X, C, m, eta):
    if X.shape[0] == 0 or C.shape[0] == 0 : return np.array([]), np.array([])
    dist = np.linalg.norm(X[:, np.newaxis] - C, axis=2) + 1e-9 # Avoid division by zero in dist
    
    # U_new
    u_power = 2.0 / (m - 1)
    U_new_inv = dist ** u_power
    U_new = 1.0 / U_new_inv
    row_sums_U = np.sum(U_new, axis=1, keepdims=True)
    row_sums_U[row_sums_U == 0] = 1e-9 # Avoid division by zero
    U_new = U_new / row_sums_U
    
    # T_new
    dist_sq = dist**2
    mean_dist_sq = np.mean(dist_sq)
    if mean_dist_sq == 0: mean_dist_sq = 1e-9
    T_new = np.exp(-dist_sq / mean_dist_sq)
    row_sums_T = np.sum(T_new, axis=1, keepdims=True)
    row_sums_T[row_sums_T == 0] = 1e-9 # Avoid division by zero
    T_new = T_new / row_sums_T
    return U_new, T_new

def run_pfcm(X_data, n_clusters=10, m=2.0, eta=2.0, max_iter=30, verbose=False): # Reduced max_iter
    if X_data is None or X_data.shape[0] == 0:
        print("Error: PFCM input data is empty.")
        return np.array([]), np.array([[]]), np.array([[]]), np.array([[]]), {}
    
    if X_data.shape[0] < n_clusters :
        print(f"Warning: PFCM n_samples ({X_data.shape[0]}) < n_clusters ({n_clusters}). Reducing n_clusters.")
        n_clusters = max(1, X_data.shape[0])
    if n_clusters == 0 : n_clusters = 1

    n_samples = X_data.shape[0]
    U = initialize_membership_matrix_pfcm(n_samples, n_clusters)
    T = U.copy() 
    C = np.array([[]])

    for i in range(max_iter):
        C_old = C.copy()
        C = calculate_centroids_pfcm(X_data, U, T, m, eta)
        if C.size == 0 or np.isnan(C).any() or np.isinf(C).any():
            print(f"Error: PFCM centroids calculation failed at iteration {i}. Stopping.")
            labels = np.argmax(U, axis=1) if U.size > 0 else np.array([])
            return labels, U, T, C_old if C_old.size > 0 else C, {"error": "Centroid calculation failed"}

        U_new, T_new = update_U_T_pfcm(X_data, C, m, eta)
        if U_new.size == 0 or T_new.size == 0 or np.isnan(U_new).any() or np.isinf(U_new).any() or np.isnan(T_new).any() or np.isinf(T_new).any():
            print(f"Error: PFCM U/T update failed at iteration {i}. Stopping.")
            labels = np.argmax(U, axis=1) if U.size > 0 else np.array([])
            return labels, U, T, C, {"error": "U/T update failed"}
        
        if i > 0 and U.shape == U_new.shape and np.linalg.norm(U - U_new) < 1e-4: # Convergence
            if verbose: print(f"PFCM converged at iteration {i+1}/{max_iter}")
            U, T = U_new, T_new # Ensure last update is used
            break
        U, T = U_new, T_new
        if verbose and (i+1)%5 == 0: print(f"PFCM iteration {i+1}/{max_iter}")
    
    labels = np.argmax(U, axis=1) if U.size > 0 else np.array([])
    metrics = calculate_sklearn_metrics(X_data, labels)
    if X_data.shape[0] > 1 and C.shape[0] > 0 and U.shape[1] > 0 :
        metrics.update(calculate_custom_metrics(X_data, U, C, m=m))
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
def compute_distances_cfcm(data, centers):
    """Tính khoảng cách Euclidean bình phương giữa các điểm dữ liệu và tâm cụm."""
    n_samples = data.shape[0]
    n_clusters = centers.shape[0]
    distances = np.zeros((n_samples, n_clusters))
    for k_idx in range(n_clusters):
        distances[:, k_idx] = np.sum((data - centers[k_idx]) ** 2, axis=1)
    return distances

def run_cfcm(data_input, n_clusters, beta=1.0, max_iter=30, tol=1e-4, random_state=42, verbose=False):
    """
    Thuật toán phân cụm mờ dựa trên công thức từ bài báo (Collaborative FCM).
    Adapted from Collaborative_FCM/CFCM.py for integration.
    """
    if data_input is None or data_input.shape[0] == 0:
        print("Error: CFCM input data is empty.")
        return np.array([]), np.array([[]]), np.array([[]]), {}

    if data_input.shape[0] < n_clusters:
        print(f"Warning: CFCM n_samples ({data_input.shape[0]}) < n_clusters ({n_clusters}). Reducing n_clusters.")
        n_clusters = max(1, data_input.shape[0])
    if n_clusters == 0: n_clusters = 1

    np.random.seed(random_state)
    n_samples, n_features = data_input.shape
    
    # Khởi tạo ngẫu nhiên giá trị thành viên
    u = np.random.rand(n_samples, n_clusters)
    row_sums_u = np.sum(u, axis=1, keepdims=True)
    row_sums_u[row_sums_u == 0] = 1 # Avoid division by zero
    u /= row_sums_u
    
    # Khởi tạo ngẫu nhiên tâm cụm
    initial_center_indices = np.random.choice(n_samples, n_clusters, replace=False)
    centers = data_input[initial_center_indices]
        
    # Reshape feature_weights to match data dimensions
    feature_weights = np.ones(n_features).reshape(1, -1)  # Shape: (1, n_features)
    
    for iteration in range(max_iter):
        u_old = u.copy()
        
        distances = compute_distances_cfcm(data_input, centers)
        distances = np.maximum(distances, 1e-10)  # Tránh chia cho 0
        
        u_new_iter = np.zeros((n_samples, n_clusters))
        P_features = n_features # Number of features
        denom_beta_P = 1 + beta * (P_features - 1)
        if denom_beta_P == 0: denom_beta_P = 1e-10 # Avoid division by zero

        for r_sample_idx in range(n_samples):
            for k_cluster_idx in range(n_clusters):
                # Phần 1: (1 / sum_s (d_rk^2 / d_rs^2))
                ratio_sum_dist = np.sum(distances[r_sample_idx, k_cluster_idx] / distances[r_sample_idx, :]) 
                if ratio_sum_dist == 0: ratio_sum_dist = 1e-10
                term1_coeff = 1.0 / ratio_sum_dist

                # Phần 2: 1 - beta * sum_j w_j^2 (u_rj^CFCM)^2 / (1+beta(P-1))
                # Reshape u_old[r_sample_idx, :] to match feature_weights dimensions
                u_old_row = u_old[r_sample_idx, :].reshape(1, -1)  # Shape: (1, n_clusters)
                reg_term_num = beta * np.sum(feature_weights * (u_old_row ** 2))
                reg_term = reg_term_num / denom_beta_P
                term2_factor = 1.0 - reg_term

                # Phần 3: beta * sum_j w_j^2 (u_rk^CFCM)^2 / (1+beta(P-1))
                term3_num = beta * np.sum(feature_weights) * (u_old[r_sample_idx, k_cluster_idx] ** 2)
                term3 = term3_num / denom_beta_P
                
                u_new_iter[r_sample_idx, k_cluster_idx] = term1_coeff * term2_factor + term3
        
        u = u_new_iter
        # Chuẩn hóa u để tổng = 1
        u_sum_iter = np.sum(u, axis=1, keepdims=True)
        u = np.divide(u, u_sum_iter, where=u_sum_iter != 0, out=np.full_like(u, 1.0/n_clusters))
        u = np.maximum(u, 1e-9) # ensure positivity and avoid log(0) if used later
        u_sum_check = np.sum(u, axis=1, keepdims=True)
        u = np.divide(u, u_sum_check, where=u_sum_check !=0, out=np.full_like(u, 1.0/n_clusters)) # re-normalize robustly
        
        # Cập nhật tâm cụm v_rt
        centers_new = np.zeros((n_clusters, n_features))
        for k_cluster_idx_c in range(n_clusters):
            u_k_col = u[:, k_cluster_idx_c].reshape(-1, 1)  # Shape: (n_samples, 1)
            numerator_c = np.sum(u_k_col * data_input, axis=0)  # Shape: (n_features,)
            # Denominator from paper for v_rt: sum_r u_rk + beta * sum_j w_j^2
            denominator_c = np.sum(u_k_col) + beta * np.sum(feature_weights)
            centers_new[k_cluster_idx_c] = numerator_c / max(denominator_c, 1e-10)
        centers = centers_new
        
        if verbose and (iteration + 1) % 10 == 0:
            print(f"CFCM Iteration {iteration+1}/{max_iter}")

        if np.max(np.abs(u - u_old)) < tol:
            if verbose: print(f"CFCM converged at iteration {iteration+1}.")
            break
    
    labels = np.argmax(u, axis=1)
    metrics = calculate_sklearn_metrics(data_input, labels)
    if data_input.shape[0] > 1 and centers.shape[0] > 0 and u.shape[1] > 0:
         # For custom metrics, m is the fuzzifier, CFCM uses beta. We'll use m=2 as a common default for metric calculation.
        metrics.update(calculate_custom_metrics(data_input, u, centers, m=2.0))

    if verbose: print(f"CFCM processing complete.")
    return labels, u, centers, metrics

# --- Evaluation Metrics ---
def calculate_sklearn_metrics(X, labels):
    metrics = {}
    if X is None or labels is None or len(labels) == 0 or X.shape[0] < 2 or len(np.unique(labels)) < 2 or len(np.unique(labels)) > X.shape[0]-1:
        # print("Warning: Cannot calculate sklearn metrics due to insufficient samples or unique labels for meaningful clustering.")
        metrics['silhouette'] = np.nan
        metrics['calinski_harabasz'] = np.nan
        metrics['davies_bouldin'] = np.nan
        return metrics
    try:
        metrics['silhouette'] = silhouette_score(X, labels)
    except ValueError: metrics['silhouette'] = np.nan
    try:
        metrics['calinski_harabasz'] = calinski_harabasz_score(X, labels)
    except ValueError: metrics['calinski_harabasz'] = np.nan
    try:
        metrics['davies_bouldin'] = davies_bouldin_score(X, labels)
    except ValueError: metrics['davies_bouldin'] = np.nan
    return metrics

def calculate_custom_metrics(X, U, V, m):
    metrics = {}
    # Basic validation for custom metrics inputs
    if X is None or U is None or V is None or \
       X.shape[0] == 0 or U.shape[0] == 0 or V.shape[0] == 0 or \
       U.shape[0] != X.shape[0] or U.shape[1] != V.shape[0] or V.shape[1] != X.shape[1] or U.shape[1] == 0:
        # print(f"Warning: Cannot calculate custom metrics due to shape mismatch or invalid data. X:{X.shape if X is not None else 'None'}, U:{U.shape if U is not None else 'None'}, V:{V.shape if V is not None else 'None'}")
        metrics['pci'] = np.nan; metrics['fhv'] = np.nan; metrics['xbi'] = np.nan
        return metrics

    try: metrics['pci'] = pci_index(X, U)
    except Exception: metrics['pci'] = np.nan
    try: metrics['fhv'] = fhv_index(X, U, V, m)
    except Exception: metrics['fhv'] = np.nan
    try: metrics['xbi'] = xbi_index(X, U, V, m)
    except Exception: metrics['xbi'] = np.nan
    return metrics

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

def run_fdekm(X_tensor, X_for_metrics, k=10, Iter=10, input_dim=256, hidden_dim_ae=10, m=2.0, verbose=False):
    """
    Fuzzy Deep Embedded K-Means (FDEKM) - Kết hợp DEKM với fuzzy clustering
    Parameters:
    - X_tensor: Input data tensor
    - X_for_metrics: Input data for metrics calculation
    - k: Number of clusters
    - Iter: Number of iterations
    - input_dim: Input dimension
    - hidden_dim_ae: Hidden dimension of autoencoder
    - m: Fuzzifier parameter (m > 1)
    - verbose: Whether to print progress
    """
    if X_tensor.shape[0] == 0:
        print("Error: FDEKM input tensor is empty.")
        return np.array([]), np.array([]), {}
    if X_tensor.shape[0] < k:
        print(f"Warning: FDEKM num_samples ({X_tensor.shape[0]}) < k ({k}). Reducing k.")
        k = max(1, X_tensor.shape[0])

    model = AutoEncoder_DEKM(input_dim=input_dim, hidden_dim_ae=hidden_dim_ae)
    train_autoencoder_dekm(model, X_tensor, epochs=20, verbose=verbose)

    H_np_final = None
    U_final = None
    V_final = None
    total_loss_final = torch.tensor(float('nan'))

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

        # Fuzzy clustering iterations
        max_fuzzy_iter = 50
        for fuzzy_iter in range(max_fuzzy_iter):
            U_old = U.copy()
            
            # Update membership matrix U
            dist = np.zeros((H_np.shape[0], actual_k))
            for j in range(actual_k):
                # Reshape V[j] to match H_np dimensions for broadcasting
                V_j = V[j].reshape(1, -1)  # Shape: (1, n_features)
                dist[:, j] = np.sum((H_np - V_j)**2, axis=1)
            
            # Avoid division by zero
            dist = np.maximum(dist, 1e-10)
            
            # Update U using fuzzy membership formula
            # Reshape dist for proper broadcasting
            dist_reshaped = dist.reshape(H_np.shape[0], actual_k, 1)  # Shape: (n_samples, n_clusters, 1)
            dist_reshaped_t = dist.reshape(H_np.shape[0], 1, actual_k)  # Shape: (n_samples, 1, n_clusters)
            U = 1.0 / np.sum((dist_reshaped / dist_reshaped_t)**(2/(m-1)), axis=2)
            
            # Update cluster centers V
            for j in range(actual_k):
                # Reshape U[:, j] for proper broadcasting
                U_j = U[:, j].reshape(-1, 1)  # Shape: (n_samples, 1)
                V[j] = np.sum(U_j**m * H_np, axis=0) / np.sum(U_j**m)

            # Check convergence
            if np.max(np.abs(U - U_old)) < 1e-4:
                if verbose:
                    print(f"FDEKM fuzzy clustering converged at iteration {fuzzy_iter+1}")
                break

        # Update autoencoder with fuzzy constraints
        V_tensor = torch.tensor(V, dtype=torch.float32)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        lambda_reg = 0.1

        for epoch_refine in range(10):
            model.train()
            optimizer.zero_grad()
            out, H_refined = model(X_tensor)
            
            # Reconstruction loss
            loss_recon = nn.MSELoss()(out, X_tensor)
            
            # Fuzzy clustering loss
            H_np_refined = H_refined.detach().numpy()
            dist_refined = np.zeros((H_np_refined.shape[0], actual_k))
            for j in range(actual_k):
                # Reshape V[j] for proper broadcasting
                V_j = V[j].reshape(1, -1)  # Shape: (1, n_features)
                dist_refined[:, j] = np.sum((H_np_refined - V_j)**2, axis=1)
            
            # Reshape dist_refined for proper broadcasting
            dist_refined_reshaped = dist_refined.reshape(H_np_refined.shape[0], actual_k, 1)
            dist_refined_reshaped_t = dist_refined.reshape(H_np_refined.shape[0], 1, actual_k)
            U_refined = 1.0 / np.sum((dist_refined_reshaped / dist_refined_reshaped_t)**(2/(m-1)), axis=2)
            
            loss_fuzzy = torch.tensor(np.sum(U_refined**m * dist_refined))
            
            # Total loss
            total_loss = loss_recon + lambda_reg * loss_fuzzy
            total_loss.backward()
            optimizer.step()

        H_np_final = H_np
        U_final = U
        V_final = V
        total_loss_final = total_loss

        if verbose:
            print(f"FDEKM Iteration {it+1}/{Iter}, Loss: {total_loss_final.item():.4f}")

    metrics = {}
    if U_final is not None and H_np_final is not None and H_np_final.shape[0] > 0:
        labels = np.argmax(U_final, axis=1)
        metrics = calculate_sklearn_metrics(H_np_final, labels)
        if H_np_final.shape[0] > 1 and len(np.unique(labels)) > 1:
            metrics.update(calculate_custom_metrics(H_np_final, U_final, V_final, m=m))
    else:
        labels = np.array([])
        H_np_final = np.array([])
        
    return labels, H_np_final, metrics

if __name__ == '__main__':
    main() 