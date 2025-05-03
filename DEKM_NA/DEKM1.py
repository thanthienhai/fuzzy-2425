import pandas as pd
'''
    Du lieu
'''
# Đọc dữ liệu
orders = pd.read_csv("List of Orders.csv", encoding='ISO-8859-1')
details = pd.read_csv("Order Details.csv", encoding='ISO-8859-1')

# Merge 2 bảng theo Order ID
df = pd.merge(details, orders, on='Order ID')

# Xoá các giá trị thiếu
df.dropna(inplace=True)

# Chọn các đặc trưng số
features = ['Amount', 'Profit', 'Quantity']
data = df[features].copy()

# Chuẩn hóa dữ liệu
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(data)

X_scaled

'''
    Mo hinh, thuat tuan va chi so danh gia
'''
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from numpy.linalg import eigh

# Dữ liệu: X_scaled từ bước tiền xử lý
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

# AutoEncoder
class AutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, latent_dim=10):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x):
        latent = self.encoder(x)
        out = self.decoder(latent)
        return out, latent
    def train_autoencoder(model, X, epochs=100, lr=1e-3):
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            output, _ = model(X)
            loss = criterion(output, X)
            loss.backward()
            optimizer.step()
            if (epoch+1) % 10 == 0:
                print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
    def DEKM(model, X, k=4, Iter=10):
        AutoEncoder.train_autoencoder(model, X)  # bước 1: huấn luyện AE ban đầu

        for it in range(Iter):
            model.eval()
            with torch.no_grad():
                _, H = model(X)  # bước 2: lấy đặc trưng ẩn
                H_np = H.numpy()

            # bước 3: KMeans trên đặc trưng ẩn
            kmeans = KMeans(n_clusters=k, n_init='auto', random_state=42)
            labels = kmeans.fit_predict(H_np)

            # bước 4: tính Sw
            Sw = np.zeros((H_np.shape[1], H_np.shape[1]))
            for i in range(k):
                cluster_points = H_np[labels == i]
                mu_i = np.mean(cluster_points, axis=0, keepdims=True)
                for h in cluster_points:
                    diff = (h - mu_i).reshape(-1, 1)
                    Sw += diff @ diff.T

            # bước 5: tính eigenvectors (Equation 5 trong paper)
            eigvals, eigvecs = eigh(Sw)
            V = eigvecs[:, :model.encoder[-1].out_features]  # chọn các vector nhỏ nhất

            # bước 6: tối ưu hóa encoder bằng loss mới
            # L_new = L_AE + lambda * ||H - H*V*V^T||^2
            V = torch.tensor(V, dtype=torch.float32)
            optimizer = optim.Adam(model.parameters(), lr=1e-3)
            lambda_reg = 0.1

            for epoch in range(30):
                model.train()
                optimizer.zero_grad()
                out, H = model(X)
                reconstruction_loss = nn.MSELoss()(out, X)
                projection = H @ V @ V.T
                constraint_loss = torch.norm(H - projection)
                total_loss = reconstruction_loss + lambda_reg * constraint_loss
                total_loss.backward()
                optimizer.step()

            print(f"Iteration {it+1}, Total Loss: {total_loss.item():.4f}")

        return labels, H_np
    
# Huyen luyen
input_dim = X_scaled.shape[1]
model = AutoEncoder(input_dim=input_dim)
cluster_labels, H_np = AutoEncoder.DEKM(model, X_tensor, k=4, Iter=10)

# Gán vào dataframe gốc
df['Cluster'] = cluster_labels

# Chi so danh gia
from sklearn.metrics import silhouette_score,calinski_harabasz_score,davies_bouldin_score

score = silhouette_score(H_np, cluster_labels)
ch_score = calinski_harabasz_score(H_np, cluster_labels)
db_score = davies_bouldin_score(H_np, cluster_labels)
print(f"Silhouette Score: {score:.4f}")
print(f"Calinski-Harabasz Index: {ch_score:.4f}")
print(f"Davies-Bouldin Index: {db_score:.4f}")
