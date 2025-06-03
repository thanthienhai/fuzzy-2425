import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from scipy.linalg import eigh
import skfuzzy as fuzz
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

# --- Metric Placeholders (similar to run.py) ---
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
        diff = X - V[j]
        weighted_diff = (U[:, j]**m).reshape(-1, 1) * diff
        if np.sum(U[:, j]**m) == 0:
            cov_j = np.zeros((X.shape[1], X.shape[1]))
        else:
            cov_j = np.dot(weighted_diff.T, diff) / np.sum(U[:, j]**m)
        
        try:
            det = np.linalg.det(cov_j)
            if det > 0: 
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
    compactness = 0
    for j in range(n_clusters):
        diff = X - V[j]
        weighted_dist = np.sum((U[:, j]**m).reshape(-1, 1) * (diff**2))
        compactness += weighted_dist
    
    min_sep = float('inf')
    for i in range(n_clusters):
        for j in range(i+1, n_clusters):
            sep = np.sum((V[i] - V[j])**2)
            min_sep = min(min_sep, sep)
    
    if min_sep == 0 or min_sep == float('inf') or X.shape[0] == 0:
        return np.nan
        
    return compactness / (X.shape[0] * min_sep)

def calculate_sklearn_metrics(X, labels):
    if X is None or labels is None or X.shape[0] == 0 or labels.shape[0] == 0 or X.shape[0] != labels.shape[0]:
        return {"silhouette": np.nan, "calinski_harabasz": np.nan, "davies_bouldin": np.nan}
    if len(np.unique(labels)) < 2 or len(np.unique(labels)) > X.shape[0] - 1:
        return {"silhouette": np.nan, "calinski_harabasz": np.nan, "davies_bouldin": np.nan}
    try:
        silhouette = silhouette_score(X, labels)
        calinski = calinski_harabasz_score(X, labels)
        davies = davies_bouldin_score(X, labels)
        return {"silhouette": silhouette, "calinski_harabasz": calinski, "davies_bouldin": davies}
    except ValueError as e:
        # print(f"Metric calculation error: {e}")
        return {"silhouette": np.nan, "calinski_harabasz": np.nan, "davies_bouldin": np.nan}

def calculate_custom_metrics(X, U, V, m):
    if X is None or U is None or V is None or X.shape[0] == 0 or U.shape[0] == 0 or V.shape[0] == 0:
        return {"pci": np.nan, "fhv": np.nan, "xbi": np.nan}
    return {
        "pci": pci_index(X, U),
        "fhv": fhv_index(X, U, V, m),
        "xbi": xbi_index(X, U, V, m)
    }

# --- Autoencoder Definition (similar to AutoEncoder_DEKM) ---
class AutoEncoder_FDEKM(nn.Module):
    def __init__(self, input_dim, hidden_dim_ae):
        super(AutoEncoder_FDEKM, self).__init__()
        intermediate_dim1 = 128
        if input_dim > 0 and input_dim < intermediate_dim1 / 2:
            intermediate_dim1 = max(hidden_dim_ae * 2, input_dim * 2, 1)
        elif input_dim == 0:
            intermediate_dim1 = max(hidden_dim_ae * 2, 1)

        if hidden_dim_ae <= 0: hidden_dim_ae = 1

        if intermediate_dim1 < hidden_dim_ae and input_dim > hidden_dim_ae:
            intermediate_dim1 = hidden_dim_ae * 2
            
        if input_dim <= hidden_dim_ae or input_dim == 0: 
             # print(f"FDEKM AutoEncoder: Input dim ({input_dim}) <= Hidden dim ({hidden_dim_ae}). Using simplified AE.")
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

def train_autoencoder_fdekm(model, data, epochs=50, lr=1e-3, verbose=False):
    if data is None or data.shape[0] == 0:
        print("Error: FDEKM Autoencoder training data is empty.")
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
        if verbose and (epoch + 1) % 10 == 0:
            print(f"AE Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")

# --- FDEKM Algorithm ---
def run_fdekm(X_tensor, X_for_metrics, k=10, m_fcm=2.0, Iter_fdekm=10, Iter_refine_ae=5, input_dim=None, hidden_dim_ae=10, lr_ae_pretrain=1e-3, lr_ae_refine=1e-4, verbose=False):
    """
    Fuzzy Deep Embedded K-Means (FDEKM)
    X_tensor: Input data as a PyTorch tensor (N x d_input)
    X_for_metrics: Original data as NumPy array for metric calculation (N x d_input or N x d_latent)
    k: Number of clusters
    m_fcm: Fuzzifier for FCM (typically 2.0)
    Iter_fdekm: Number of FDEKM iterations (outer loop)
    Iter_refine_ae: Number of epochs to refine AE in each FDEKM iteration
    input_dim: Dimensionality of the input data (X_tensor.shape[1])
    hidden_dim_ae: Dimensionality of the autoencoder's latent space
    lr_ae_pretrain: Learning rate for AE pretraining
    lr_ae_refine: Learning rate for AE refinement during FDEKM iterations
    verbose: Print progress messages
    """
    if X_tensor is None or X_tensor.shape[0] == 0:
        print("Error: FDEKM input tensor is empty.")
        return None, np.array([]), np.array([[]]), np.array([[]]), {}
    
    if input_dim is None:
        input_dim = X_tensor.shape[1]
    
    if X_tensor.shape[0] < k:
        print(f"Warning: FDEKM num_samples ({X_tensor.shape[0]}) < k ({k}). Reducing k.")
        k = max(1, X_tensor.shape[0])
    if k == 0: k = 1

    # 1. Pretraining Phase
    autoencoder = AutoEncoder_FDEKM(input_dim=input_dim, hidden_dim_ae=hidden_dim_ae)
    if verbose: print("Starting FDEKM Autoencoder Pretraining...")
    train_autoencoder_fdekm(autoencoder, X_tensor, epochs=20, lr=lr_ae_pretrain, verbose=verbose) # Reduced epochs for speed
    if verbose: print("FDEKM Autoencoder Pretraining Finished.")

    U_final = np.array([[]])
    centroids_final = np.array([[]])
    labels_final = np.array([])
    H_np_final = np.array([])

    optimizer_ae_refine = optim.Adam(autoencoder.parameters(), lr=lr_ae_refine)

    # 2. Iterative Optimization Phase
    for it_fdekm in range(Iter_fdekm):
        if verbose: print(f"--- FDEKM Iteration {it_fdekm + 1}/{Iter_fdekm} ---")
        autoencoder.eval() # Encoder is used for embedding
        with torch.no_grad():
            _, H_tensor = autoencoder(X_tensor) # H_tensor: N x hidden_dim_ae
            H_np = H_tensor.cpu().numpy()
        
        if H_np.shape[0] == 0 or H_np.shape[1] == 0:
            print(f"FDEKM Iter {it_fdekm + 1}: Embeddings H_np are empty or have zero dimension. Skipping iteration.")
            continue
        H_np_final = H_np # Store last valid embeddings

        # 2.2 Fuzzy C-Means (FCM) Clustering on H_np
        try:
            # skfuzzy.cmeans expects data to be (features, samples)
            # H_np is (samples, features), so H_np.T is (features, samples)
            # cntr_h: (k, hidden_dim_ae), U_fcm: (k, N)
            cntr_h, U_fcm_transposed, _, _, _, _, _ = fuzz.cluster.cmeans(
                H_np.T, c=k, m=m_fcm, error=0.005, maxiter=100, init=None, seed=42
            )
            U_fcm = U_fcm_transposed.T # U_fcm: (N, k)
            U_final = U_fcm # Store last valid U
            centroids_final = cntr_h # Store last valid centroids in latent space
        except Exception as e:
            print(f"FDEKM Iter {it_fdekm + 1}: FCM failed: {e}. Using previous U/centroids or random if first iter.")
            if it_fdekm == 0: # First iteration and FCM failed
                U_fcm = np.random.rand(H_np.shape[0], k)
                U_fcm = U_fcm / np.sum(U_fcm, axis=1, keepdims=True)
                # Initialize centroids randomly if FCM fails on first try
                random_indices = np.random.choice(H_np.shape[0], size=k, replace=False if H_np.shape[0] >= k else True)
                cntr_h = H_np[random_indices, :]
                U_final = U_fcm
                centroids_final = cntr_h
            # else: use U_fcm, cntr_h from previous iteration (already stored in U_final, centroids_final)
            # This means if FCM fails, we reuse the last successful U and centroids for S_W calc.

        if U_final.size == 0 or centroids_final.size == 0:
            print(f"FDEKM Iter {it_fdekm + 1}: U or centroids are empty after FCM. Skipping iteration.")
            continue

        # 2.3 Compute Fuzzy Scatter Matrix S_W
        # S_W = sum_j sum_i (u_ij^m * (h_i - mu_j)(h_i - mu_j)^T)
        # h_i: (hidden_dim_ae,), mu_j: (hidden_dim_ae,)
        # (h_i - mu_j): (hidden_dim_ae,)
        # (h_i - mu_j)(h_i - mu_j)^T: (hidden_dim_ae, hidden_dim_ae)
        S_W = np.zeros((hidden_dim_ae, hidden_dim_ae))
        for j_cluster in range(k):
            mu_j = centroids_final[j_cluster, :] # (hidden_dim_ae,)
            for i_sample in range(H_np.shape[0]):
                h_i = H_np[i_sample, :] # (hidden_dim_ae,)
                u_ij_m = U_final[i_sample, j_cluster] ** m_fcm
                diff = (h_i - mu_j).reshape(hidden_dim_ae, 1)
                S_W += u_ij_m * (diff @ diff.T)

        # 2.4 Structure-Preserving Transformation
        try:
            eigvals, eigvecs = eigh(S_W) # eigvals sorted ascending by default by eigh
            V_transform = eigvecs # V_transform: (hidden_dim_ae, hidden_dim_ae)
        except np.linalg.LinAlgError:
            print(f"FDEKM Iter {it_fdekm + 1}: Eigendecomposition of S_W failed. Using identity matrix.")
            V_transform = np.eye(hidden_dim_ae)
        
        # Y_transformed_np = H_np @ V_transform.T # (N, hidden_dim_ae) - Note: Algorithm says Vh, so H @ V if V columns are eigenvectors
                                                # If V rows are eigenvectors (eigh default), then H @ V.T
                                                # Let's assume eigvecs columns are eigenvectors as per standard V^T Lambda V
                                                # So S_W = V @ Lambda @ V.T. We need V.
                                                # y_i = V h_i. If h_i is row vector, then h_i @ V^T. If h_i is col vector, V @ h_i.
                                                # H_np is (N, hidden_dim_ae), h_i are rows.
                                                # To transform h_i (row) by V (cols are evecs): h_i_transformed_row = h_i @ V
        Y_transformed_np = H_np @ V_transform # (N, hidden_dim_ae)
        # m_centroids_transformed_np = centroids_final @ V_transform.T # (k, hidden_dim_ae)
        m_centroids_transformed_np = centroids_final @ V_transform # (k, hidden_dim_ae)

        # 2.5 Fuzzy Target Construction
        Y_prime_np = np.copy(Y_transformed_np)
        # Compute fuzzy-weighted average for the last dimension of Y_prime_np
        # y_i'[-1] = sum_j (u_ij^m * m_j_transformed[-1]) / sum_j (u_ij^m)
        for i_sample in range(H_np.shape[0]):
            numerator = 0.0
            denominator = 0.0
            for j_cluster in range(k):
                u_ij_m = U_final[i_sample, j_cluster] ** m_fcm
                numerator += u_ij_m * m_centroids_transformed_np[j_cluster, -1]
                denominator += u_ij_m
            if denominator == 0: # Avoid division by zero if all memberships are zero for a sample (unlikely with FCM)
                Y_prime_np[i_sample, -1] = Y_transformed_np[i_sample, -1] # Keep original if denominator is zero
            else:
                Y_prime_np[i_sample, -1] = numerator / denominator
        
        Y_prime_tensor = torch.tensor(Y_prime_np, dtype=X_tensor.dtype, device=X_tensor.device)
        V_transform_tensor = torch.tensor(V_transform, dtype=X_tensor.dtype, device=X_tensor.device)

        # 2.6 Representation Loss and Encoder Update
        # L_fdekm = sum_i || V f(x_i) - y_i' ||^2
        # V is V_transform_tensor, f(x_i) is H_tensor (current embeddings)
        # V f(x_i) means H_tensor @ V_transform_tensor (if H_tensor rows are samples)
        autoencoder.train() # Switch to training mode for encoder update
        for epoch_refine in range(Iter_refine_ae):
            optimizer_ae_refine.zero_grad()
            # Get current embeddings f(x_i) = H_current_tensor
            _, H_current_tensor = autoencoder(X_tensor)
            
            # Transformed embeddings: V * f(x_i)
            # H_current_tensor: (N, hidden_dim_ae), V_transform_tensor: (hidden_dim_ae, hidden_dim_ae)
            # Transformed_H = H_current_tensor @ V_transform_tensor.T (if V_transform_tensor rows are eigenvectors)
            # Or H_current_tensor @ V_transform_tensor (if V_transform_tensor columns are eigenvectors)
            # Based on y_i = V h_i, and h_i being a row vector, this should be H_current_tensor @ V_transform_tensor
            Transformed_H_pred = H_current_tensor @ V_transform_tensor
            
            loss_fdekm = nn.MSELoss()(Transformed_H_pred, Y_prime_tensor)
            loss_fdekm.backward()
            optimizer_ae_refine.step()
            if verbose and (epoch_refine + 1) % 5 == 0:
                print(f"  AE Refine Epoch {epoch_refine + 1}/{Iter_refine_ae}, Loss_FDEKM: {loss_fdekm.item():.4f}")
        
        if verbose:
            print(f"FDEKM Iteration {it_fdekm + 1} finished. Last FDEKM Loss: {loss_fdekm.item():.4f}")

    # 3. Output
    # Final embeddings from the trained encoder
    autoencoder.eval()
    with torch.no_grad():
        _, H_final_tensor = autoencoder(X_tensor)
        H_np_final = H_final_tensor.cpu().numpy()

    # Recalculate U and centroids with final H_np_final for consistency
    if H_np_final.shape[0] > 0 and H_np_final.shape[1] > 0 and k > 0:
        try:
            final_cntr_h, final_U_transposed, _, _, _, _, _ = fuzz.cluster.cmeans(
                H_np_final.T, c=k, m=m_fcm, error=0.005, maxiter=100, init=None, seed=42
            )
            U_final = final_U_transposed.T
            centroids_final = final_cntr_h
            labels_final = np.argmax(U_final, axis=1)
        except Exception as e:
            print(f"Final FCM for output failed: {e}. Returning last known U/centroids.")
            if U_final.size > 0: labels_final = np.argmax(U_final, axis=1)
            else: labels_final = np.array([]) # Ensure labels_final is an array
    else:
        labels_final = np.array([])
        # U_final and centroids_final would be from the last successful iteration or empty

    metrics = {}
    if X_for_metrics is not None and H_np_final.shape[0] > 0 and labels_final.shape[0] == H_np_final.shape[0] and len(np.unique(labels_final)) > 1:
        # Decide whether to calculate metrics on original X_for_metrics or latent H_np_final
        # The description implies clustering is on H, so metrics on H are more direct.
        metrics.update(calculate_sklearn_metrics(H_np_final, labels_final))
    
    if X_for_metrics is not None and H_np_final.shape[0] > 0 and U_final.shape[0] == H_np_final.shape[0] and centroids_final.shape[0] > 0 and U_final.shape[1] == k:
        metrics.update(calculate_custom_metrics(H_np_final, U_final, centroids_final, m=m_fcm))
    else:
        # Ensure custom metrics have NaN placeholders if calculation is skipped
        metrics.update({"pci": np.nan, "fhv": np.nan, "xbi": np.nan})
        if "silhouette" not in metrics: # If sklearn metrics also failed/skipped
             metrics.update({"silhouette": np.nan, "calinski_harabasz": np.nan, "davies_bouldin": np.nan})


    return autoencoder, labels_final, U_final, centroids_final, H_np_final, metrics

if __name__ == '__main__':
    # Example Usage (requires data loading and preprocessing similar to run.py)
    print("FDEKM module loaded. Example usage would require data.")

    # Dummy data for basic testing of the structure
    N_samples = 100
    d_input_dim = 784 # e.g., flattened MNIST
    d_hidden_dim = 30
    n_clusters_k = 5

    # Create dummy torch tensor
    dummy_X_tensor = torch.randn(N_samples, d_input_dim)
    dummy_X_np = dummy_X_tensor.numpy()

    print(f"Running FDEKM with dummy data: {N_samples} samples, {d_input_dim} features, {d_hidden_dim} latent dim, {n_clusters_k} clusters.")

    try:
        trained_encoder, fdekm_labels, fdekm_U, fdekm_centroids, fdekm_H_latent, fdekm_metrics = run_fdekm(
            X_tensor=dummy_X_tensor,
            X_for_metrics=dummy_X_np, # or dummy_X_tensor.numpy() if metrics on original space
            k=n_clusters_k,
            m_fcm=2.0,
            Iter_fdekm=3, # Short iterations for testing
            Iter_refine_ae=2, # Short refinement for testing
            input_dim=d_input_dim,
            hidden_dim_ae=d_hidden_dim,
            verbose=True
        )

        print("\nFDEKM execution finished.")
        if trained_encoder is not None:
            print(f"  Trained Encoder: {trained_encoder}")
            print(f"  Labels shape: {fdekm_labels.shape if fdekm_labels is not None else 'None'}")
            print(f"  U matrix shape: {fdekm_U.shape if fdekm_U is not None else 'None'}")
            print(f"  Centroids shape: {fdekm_centroids.shape if fdekm_centroids is not None else 'None'}")
            print(f"  Latent Embeddings H shape: {fdekm_H_latent.shape if fdekm_H_latent is not None else 'None'}")
            print(f"  Metrics: {fdekm_metrics}")

            # Example: Get hard labels
            if fdekm_U is not None and fdekm_U.size > 0:
                hard_labels = np.argmax(fdekm_U, axis=1)
                print(f"  Hard labels from U: {hard_labels[:10]}...") # Print first 10

    except Exception as e:
        print(f"An error occurred during the FDEKM dummy run: {e}")
        import traceback
        traceback.print_exc()