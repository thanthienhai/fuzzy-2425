import numpy as np

# Step 1: Make a membership of values of feature in matrix
def initialize_membership_matrix(n_samples, n_clusters):
    U = np.random.rand(n_samples, n_clusters)
    U = U / np.sum(U, axis=1, keepdims=True)  # Đảm bảo tổng các giá trị trên mỗi hàng là 1
    return U

# Step 2: Calculate each centroids
def calculate_centroids(U, X, m):
    um = U ** m
    centroids = np.dot(um.T, X) / np.sum(um.T, axis=1, keepdims=True)
    return centroids

# Step 3: Update the membership's values
def update_membership_matrix(U, X, centroids, m):
    n_samples = X.shape[0]
    n_clusters = centroids.shape[0]
    
    for i in range(n_samples):
        for j in range(n_clusters):
            sum_terms = 0
            for k in range(n_clusters):
                dist_ratio = np.linalg.norm(X[i] - centroids[j]) / np.linalg.norm(X[i] - centroids[k])
                sum_terms += (dist_ratio ** (2 / (m - 1)))
            U[i, j] = 1 / sum_terms
    
    return U

# Step 4: Connect all of them to build FCM
def fuzzy_c_means(X, n_clusters, m=2, max_iter=100, error=1e-5):
    n_samples = X.shape[0]
    U = initialize_membership_matrix(n_samples, n_clusters)  # Khởi tạo ngẫu nhiên
    
    for iteration in range(max_iter):
        centroids = calculate_centroids(U, X, m)  # Tính toán các tâm cụm
        U_new = update_membership_matrix(U, X, centroids, m)  # Cập nhật ma trận membership
        
        # Calculate the norm, U_new minus U, when it small than error then break of the process
        if np.linalg.norm(U_new - U) < error:
            break
        
        U = U_new
    
    return U, centroids

# Simple dataset
X = np.array([[1, 2],
              [1, 4],
              [3, 3],
              [5, 4]])

# Cluster I want to try
n_clusters = 2

# Rup it up
U, centroids = fuzzy_c_means(X, n_clusters)

print("The final of memberships's values matrix:")
print(U)
print("\nThe final of centroids:")
print(centroids)
