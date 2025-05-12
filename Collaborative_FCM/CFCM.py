import numpy as np

def compute_distances(data, centers):
    """Tính khoảng cách Euclidean bình phương giữa các điểm dữ liệu và tâm cụm."""
    n_samples = data.shape[0]
    n_clusters = centers.shape[0]
    distances = np.zeros((n_samples, n_clusters))
    for k in range(n_clusters):
        distances[:, k] = np.sum((data - centers[k]) ** 2, axis=1)
    return distances

def fuzzy_clustering(data, n_clusters, beta=1.0, max_iter=100, tol=1e-4, random_state=42):
    """
    Thuật toán phân cụm mờ dựa trên công thức từ bài báo.
    
    Parameters:
    - data: numpy array, shape (n_samples, n_features)
    - n_clusters: số lượng cụm
    - beta: tham số điều chỉnh (mặc định 1.0)
    - max_iter: số lần lặp tối đa
    - tol: ngưỡng hội tụ
    - random_state: seed cho khởi tạo ngẫu nhiên
    
    Returns:
    - u: ma trận giá trị thành viên, shape (n_samples, n_clusters)
    - centers: tâm cụm, shape (n_clusters, n_features)
    """
    np.random.seed(random_state)
    n_samples, n_features = data.shape
    
    # Khởi tạo ngẫu nhiên giá trị thành viên
    u = np.random.rand(n_samples, n_clusters)
    u /= np.sum(u, axis=1, keepdims=True)  # Chuẩn hóa để tổng = 1
    
    # Khởi tạo ngẫu nhiên tâm cụm
    centers = data[np.random.choice(n_samples, n_clusters, replace=False)]
    
    # Giả định trọng số đặc trưng
    feature_weights = np.ones(n_features)  # |j_k^2| = 1
    
    for iteration in range(max_iter):
        u_old = u.copy()
        
        # Bước 1: Tính khoảng cách
        distances = compute_distances(data, centers)
        distances = np.maximum(distances, 1e-10)  # Tránh chia cho 0
        
        # Bước 2: Cập nhật giá trị thành viên u_rk
        u = np.zeros((n_samples, n_clusters))
        P = n_features
        denom = 1 + beta * (P - 1)
        
        for i in range(n_samples):
            for k in range(n_clusters):
                # Tính phần \sum_{j=1}^c (d_ik^2 / d_jk^2)
                ratio_sum = np.sum(distances[i, :] / distances[i, k])
                # Tính phần điều chuẩn
                reg_term = beta * np.sum(feature_weights * u[i, :] ** 2) / denom
                u[i, k] = (1 / ratio_sum) * (1 - reg_term)
                # Thêm phần điều chỉnh (giả định |j_k^2| = 1)
                u[i, k] += (beta * np.sum(feature_weights) * u[i, k] ** 2) / denom
        
        # Chuẩn hóa u để tổng = 1
        u_sum = np.sum(u, axis=1, keepdims=True)
        u = np.divide(u, u_sum, where=u_sum != 0)
        
        # Bước 3: Cập nhật tâm cụm v_rt
        centers = np.zeros((n_clusters, n_features))
        for k in range(n_clusters):
            u_k = u[:, k].reshape(-1, 1)
            numerator = np.sum(u_k * data, axis=0)
            denominator = np.sum(u_k) + beta * np.sum(feature_weights)
            centers[k] = numerator / max(denominator, 1e-10)
        
        # Kiểm tra hội tụ
        if np.max(np.abs(u - u_old)) < tol:
            break
    
    return u, centers