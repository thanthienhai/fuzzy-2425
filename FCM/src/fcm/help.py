import numpy as np
import pandas as pd
import skfuzzy as fuzz
import matplotlib.pyplot as plt

# Bước 1: Đọc dữ liệu
df = pd.read_csv('Country-data.csv')

# Lấy các đặc điểm để phân cụm
features = ['exports', 'imports']  # Điều chỉnh tên cột nếu cần
X = df[features].values.T  # Chuyển vị dữ liệu để phù hợp với skfuzzy

# Bước 2: Thực hiện Fuzzy C-Means
n_clusters = 3  # Số lượng cụm
cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(X, c=n_clusters, m=2, error=0.005, maxiter=1000)

# Bước 3: Tính chỉ số cụm dựa trên độ thành viên cao nhất
cluster_membership = np.argmax(u, axis=0)

# Bước 4: Tạo màu dựa trên độ thành viên
# Sử dụng giá trị thành viên lớn nhất để đại diện cho màu sắc của điểm dữ liệu
max_u = np.max(u, axis=0)

# Bước 5: Vẽ biểu đồ với hiệu ứng gradient
plt.figure(figsize=(10, 7))

# Mỗi điểm sẽ có màu dựa trên giá trị thành viên lớn nhất cho bất kỳ cụm nào
scatter = plt.scatter(X[0], X[1], c=max_u, cmap='plasma', s=100, alpha=0.6)

# Vẽ các tâm cụm (centroids)
plt.scatter(cntr[:, 0], cntr[:, 1], color='red', marker='x', s=150, label="Centers")

# Bước 6: Thêm colorbar chung
cbar = plt.colorbar(scatter)
cbar.set_label('Degree of Membership')

# Hoàn thiện biểu đồ
plt.title("Fuzzy C-Means Clustering with Dataset HELP")
plt.xlabel("Ex")
plt.ylabel("Im")
plt.legend()
plt.show()
