import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns
import os
from sklearn.neighbors import NearestNeighbors

# Đọc dữ liệu
df = pd.read_csv('/home/thien/Coding/NCKH2425/fuzzy-2425/datasets/Country-data.csv')

# Chọn các features để phân cụm
features = ['child_mort', 'exports', 'health', 'imports', 'income', 'inflation', 'life_expec', 'total_fer', 'gdpp']
X = df[features]

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Tạo thư mục để lưu kết quả
if not os.path.exists('results'):
    os.makedirs('results')

# Tìm khoảng cách đến k điểm gần nhất
k = 5  # Có thể điều chỉnh
neigh = NearestNeighbors(n_neighbors=k)
nbrs = neigh.fit(X_scaled)
distances, indices = nbrs.kneighbors(X_scaled)

# Sắp xếp khoảng cách và vẽ đồ thị
distances = np.sort(distances, axis=0)
distances = distances[:,1]
plt.figure(figsize=(10, 6))
plt.plot(distances)
plt.title('K-distance Graph')
plt.xlabel('Data Points sorted by distance')
plt.ylabel('Epsilon')
plt.savefig('results/k_distance_graph.png')
plt.close()

# Thử nghiệm với nhiều giá trị eps
eps_values = [0.5, 1, 1.5, 2, 2.5, 3]
results = []

for eps in eps_values:
    dbscan = DBSCAN(eps=eps, min_samples=5)
    clusters = dbscan.fit_predict(X_scaled)
    n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
    n_noise = list(clusters).count(-1)
    results.append({'eps': eps, 'n_clusters': n_clusters, 'n_noise': n_noise})
    
    print(f'Epsilon: {eps}')
    print(f'Number of clusters: {n_clusters}')
    print(f'Number of noise points: {n_noise}\n')

# Chọn eps tốt nhất (ví dụ: eps với số cụm lớn nhất và số điểm nhiễu ít nhất)
best_result = max(results, key=lambda x: x['n_clusters'] - x['n_noise']/len(X))
best_eps = best_result['eps']

print(f"Best eps: {best_eps}")

# Áp dụng DBSCAN với eps tốt nhất
dbscan = DBSCAN(eps=best_eps, min_samples=5)
clusters = dbscan.fit_predict(X_scaled)

# Thêm nhãn cụm vào dataframe gốc
df['Cluster'] = clusters

# Lưu kết quả phân cụm vào file CSV
df.to_csv('results/clustered_data.csv', index=False)

# Áp dụng PCA để giảm chiều dữ liệu xuống 2D để có thể trực quan hóa
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Trực quan hóa kết quả
plt.figure(figsize=(12, 8))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis')
plt.colorbar(scatter)
plt.title(f'DBSCAN Clustering Results (eps={best_eps})')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.savefig('results/dbscan_clusters.png')
plt.close()

# Phân tích và trực quan hóa các cụm
for feature in features:
    plt.figure(figsize=(12, 8))
    sns.boxplot(x='Cluster', y=feature, data=df)
    plt.title(f'Distribution of {feature} across clusters')
    plt.savefig(f'results/{feature}_distribution.png')
    plt.close()

# Tạo báo cáo tổng quan
with open('results/cluster_summary.txt', 'w') as f:
    f.write("DBSCAN Clustering Summary\n")
    f.write("=========================\n\n")
    f.write(f"Best epsilon: {best_eps}\n")
    f.write(f"Total number of clusters: {len(set(clusters)) - (1 if -1 in clusters else 0)}\n")
    f.write(f"Number of noise points: {list(clusters).count(-1)}\n\n")
    
    for cluster in sorted(set(clusters)):
        if cluster == -1:
            f.write("Outliers:\n")
        else:
            f.write(f"Cluster {cluster}:\n")
        cluster_data = df[df['Cluster'] == cluster]
        f.write(f"Number of countries: {len(cluster_data)}\n")
        f.write("Top 5 countries: " + ", ".join(cluster_data['country'].head().tolist()) + "\n")
        f.write("Cluster statistics:\n")
        f.write(cluster_data[features].describe().to_string() + "\n\n")

print("Clustering completed. Results saved in 'results' folder.")