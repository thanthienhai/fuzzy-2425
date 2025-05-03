import matplotlib.pyplot as plt
import h5py 
import numpy as np
from functools import reduce


'''
    Du lieu
'''
def hdf5(path, data_key = "data", target_key = "target", flatten = True):
    """
        loads data from hdf5: 
        - hdf5 should have 'train' and 'test' groups 
        - each group should have 'data' and 'target' dataset or spcify the key
        - flatten means to flatten images N * (C * H * W) as N * D array
    """
    with h5py.File(path, 'r') as hf:
        train = hf.get('train')
        X_tr = train.get(data_key)[:]
        y_tr = train.get(target_key)[:]
        test = hf.get('test')
        X_te = test.get(data_key)[:]
        y_te = test.get(target_key)[:]
        if flatten:
            X_tr = X_tr.reshape(X_tr.shape[0], reduce(lambda a, b: a * b, X_tr.shape[1:]))
            X_te = X_te.reshape(X_te.shape[0], reduce(lambda a, b: a * b, X_te.shape[1:]))
    return X_tr, y_tr, X_te, y_te


X_tr, y_tr, X_te, y_te = hdf5("usps.h5")
X_tr.shape, X_te.shape

num_samples = 10
num_classes = len(set(y_tr))

classes = set(y_tr)
num_classes = len(classes)
fig, ax = plt.subplots(num_samples, num_classes, sharex = True, sharey = True, figsize=(num_classes, num_samples))

for label in range(num_classes):
    class_idxs = np.where(y_tr == label)
    for i, idx in enumerate(np.random.randint(0, class_idxs[0].shape[0], num_samples)):
        ax[i, label].imshow(X_tr[class_idxs[0][idx]].reshape([16, 16]), 'gray')
        ax[i, label].set_axis_off()

'''
    Giam chieu du lieu
'''
from sklearn.decomposition import PCA
X_tr = X_tr / 255.0
X_pca = PCA(n_components=50).fit_transform(X_tr)

'''
    Mo hinh
'''
def initialize_membership_matrix(n_samples, n_clusters):
    U = np.random.rand(n_samples, n_clusters)
    return U / np.sum(U, axis=1, keepdims=True)

def calculate_centroids(X, U, T, m, eta):
    num = (U**m + eta * T**m).T @ X
    denom = np.sum(U**m + eta * T**m, axis=0).reshape(-1, 1)
    return num / denom

def update_U_T(X, C, m, eta):
    dist = np.linalg.norm(X[:, np.newaxis] - C, axis=2) + 1e-10
    U_new = 1.0 / (dist ** (2/(m-1)))
    U_new = U_new / np.sum(U_new, axis=1, keepdims=True)
    
    T_new = np.exp(-dist**2 / np.mean(dist**2))
    T_new = T_new / np.sum(T_new, axis=1, keepdims=True)
    return U_new, T_new

def pfcm(X, n_clusters, m=2.0, eta=2.0, max_iter=100):
    n_samples = X.shape[0]
    U = initialize_membership_matrix(n_samples, n_clusters)
    T = U.copy()
    
    for _ in range(max_iter):
        C = calculate_centroids(X, U, T, m, eta)
        U, T = update_U_T(X, C, m, eta)
        
    labels = np.argmax(U, axis=1)
    return labels, U, T, C


'''
    Thuc hien mo hinh
'''
labels, U, T, centers = pfcm(X_pca, n_clusters=3)

'''
    Truc quan
'''
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', alpha=0.6)
plt.scatter(centers[:, 0], centers[:, 1], color='red', marker='x', s=100)
plt.title('Phân cụm với PFCM')
# plt.xlabel(selected_cols[0])
# plt.ylabel(selected_cols[1])
plt.grid(True)
plt.show()


'''
    Chi so danh gia
'''
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
sil_score = silhouette_score(X_pca, labels)
calinski_score = calinski_harabasz_score(X_pca, labels)
davies_score = davies_bouldin_score(X_pca, labels)

print(f"Silhouette Score: {sil_score:.4f}")
print(f"Calinski-Harabasz Index: {calinski_score:.4f}")
print(f"Davies-Bouldin Index: {davies_score:.4f}")

    




