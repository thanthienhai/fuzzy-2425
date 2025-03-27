import numpy as np


def calculate_xbi(U, X, V, m=2):
    """
    Tính toán Xie-Beni Index (XBI) từ ma trận thành viên U, dữ liệu X và tâm cụm V.

    Parameters:
    U : numpy array, shape (n, c)
        Ma trận thành viên, U[i,k] là giá trị thành viên của điểm i với cụm k.
    X : numpy array, shape (n, d)
        Tập hợp điểm dữ liệu, n là số điểm, d là số chiều.
    V : numpy array, shape (c, d)
        Tâm của các cụm, c là số cụm.
    m : float
        Hệ số mờ (mặc định là 2).

    Returns:
    float: Giá trị XBI
    """
    n, c = U.shape  # Số điểm dữ liệu và số cụm

    # Tính tử số: Tổng bình phương khoảng cách có trọng số
    numerator = 0
    for i in range(n):
        for k in range(c):
            diff = X[i] - V[k]  # Vector chênh lệch (x_i - v_k)
            dist_squared = np.sum(diff ** 2)  # Bình phương khoảng cách Euclid
            numerator += (U[i, k] ** m) * dist_squared

    # Tính mẫu số: Khoảng cách nhỏ nhất giữa các tâm cụm
    min_dist_squared = float('inf')
    for k in range(c):
        for l in range(k + 1, c):
            diff = V[k] - V[l]
            dist_squared = np.sum(diff ** 2)
            min_dist_squared = min(min_dist_squared, dist_squared)

    # Tính XBI
    xbi = numerator / (n * min_dist_squared)
    return xbi


def main():
    # Dữ liệu mẫu
    U = np.array([[0.9, 0.1],  # Ma trận thành viên
                  [0.3, 0.7],
                  [0.6, 0.4]])
    X = np.array([[1, 2],  # Dữ liệu 2D (3 điểm, 2 chiều)
                  [2, 3],
                  [3, 1]])
    V = np.array([[1.5, 2],  # Tâm của 2 cụm
                  [2.5, 2]])

    # Tính XBI
    print(">>> calculate_xbi(U, X, V, m=2)")
    xbi_value = calculate_xbi(U, X, V, m=2)
    print(xbi_value)


if __name__ == "__main__":
    main()