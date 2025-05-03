import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import skfuzzy as fuzz
import matplotlib.pyplot as plt

'''
    Du lieu
'''
# Đọc dữ liệu
orders = pd.read_csv('List of Orders.csv')
order_details = pd.read_csv('Order Details.csv')

orders.dropna(inplace=True)
order_details.dropna(inplace=True)

# Merge 2 bảng theo Order ID
df = pd.merge(order_details, orders, on='Order ID')

print(df.head(5))

# Chọn các đặc trưng số
features = ['Amount', 'Profit', 'Quantity']
data = df[features].copy()

# Chuẩn hóa dữ liệu
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(data)

'''
    Huan luyen
'''
cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(X_scaled.T, c=3, m=2, error=0.005, maxiter=1000, init=None)

# Gán nhãn cho mỗi điểm dữ liệu
cluster_membership = np.argmax(u, axis=0)

# Vẽ kết quả phân cụm
plt.figure()
for j in range(3):
    plt.scatter(X_scaled[cluster_membership == j, 0],
                X_scaled[cluster_membership == j, 1], label=f'Cluster {j}')
plt.legend()
plt.title('Kết quả phân cụm bằng FCM')
plt.show()          