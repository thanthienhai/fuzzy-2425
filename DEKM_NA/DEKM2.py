import h5py
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch

'''
    Du lieu
'''
# Đọc dữ liệu từ file HDF5
with h5py.File('usps.h5', 'r') as f:
    train_X = np.array(f['train']['data'])
    train_y = np.array(f['train']['target'])
    test_X = np.array(f['test']['data'])
    test_y = np.array(f['test']['target'])

# Gộp train và test
X = np.concatenate([train_X, test_X], axis=0)
y = np.concatenate([train_y, test_y], axis=0)

# Tiền xử lý
X = X.astype('float32') / 255.0
X = X.reshape((X.shape[0], -1))  # (num_samples, 256)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Chuyển thành Tensor
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)


'''
    Mo hinh
'''
import torch.nn as nn

class AutoEncoder(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=10):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, hidden_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )

    def forward(self, x):
        h = self.encoder(x)
        out = self.decoder(h)
        return out, h

model = AutoEncoder()

import torch.optim as optim

def train_autoencoder(model, data, epochs=50, lr=1e-3):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output, _ = model(data)
        loss = criterion(output, data)
        loss.backward()
        optimizer.step()
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

train_autoencoder(model, X_tensor)

from sklearn.cluster import KMeans
from scipy.linalg import eigh
import numpy as np

def DEKM(model, X, k=10, Iter=10):
    train_autoencoder(model, X)

    for it in range(Iter):
        model.eval()
        with torch.no_grad():
            _, H = model(X)
            H_np = H.numpy()

        kmeans = KMeans(n_clusters=k, n_init='auto', random_state=42)
        labels = kmeans.fit_predict(H_np)

        Sw = np.zeros((H_np.shape[1], H_np.shape[1]))
        for i in range(k):
            cluster_points = H_np[labels == i]
            mu_i = np.mean(cluster_points, axis=0, keepdims=True)
            for h in cluster_points:
                diff = (h - mu_i).reshape(-1, 1)
                Sw += diff @ diff.T

        eigvals, eigvecs = eigh(Sw)
        V = eigvecs[:, :model.encoder[-1].out_features]

        V = torch.tensor(V, dtype=torch.float32)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        lambda_reg = 0.1

        for epoch in range(30):
            model.train()
            optimizer.zero_grad()
            out, H = model(X)
            projection = H @ V @ V.T
            loss_recon = nn.MSELoss()(out, X)
            loss_constraint = torch.norm(H - projection)
            total_loss = loss_recon + lambda_reg * loss_constraint
            total_loss.backward()
            optimizer.step()

        print(f"Iteration {it+1}, Loss: {total_loss.item():.4f}")

    return labels, H_np

cluster_labels, H_np = DEKM(model, X_tensor, k=10)

# Chi so danh gia
from sklearn.metrics import silhouette_score,calinski_harabasz_score,davies_bouldin_score

score = silhouette_score(H_np, cluster_labels)
ch_score = calinski_harabasz_score(H_np, cluster_labels)
db_score = davies_bouldin_score(H_np, cluster_labels)
print(f"Silhouette Score: {score:.4f}")
print(f"Calinski-Harabasz Index: {ch_score:.4f}")
print(f"Davies-Bouldin Index: {db_score:.4f}")


