import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

file_path = '/home/thien/Coding/NCKH2425/fuzzy-2425/Kmean/datasets/Country-data.csv'
data = pd.read_csv(file_path)

print(data.head())

X = data.select_dtypes(include=[float, int])

kmeans = KMeans(n_clusters=3, random_state=0)
kmeans.fit(X)

data['Cluster'] = kmeans.labels_

output_path = '/home/thien/Coding/NCKH2425/fuzzy-2425/Kmean/outputs/Country-data-clustered.csv'
data.to_csv(output_path, index=False)
print(f"Dữ liệu đã phân cụm được lưu vào {output_path}")

plt.figure(figsize=(10, 7))
plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=kmeans.labels_, cmap='viridis')
plt.title('K-means Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

image_path = '/home/thien/Coding/NCKH2425/fuzzy-2425/Kmean/outputs/kmeans_clusters.png'
plt.savefig(image_path)
print(f"Hình ảnh trực quan hóa cụm đã được lưu vào {image_path}")
