import numpy as np


def calculate_fhv(U, X, V, m=2):
    """
    Tính toán Fuzzy Hypervolume (FHV) từ ma trận thành viên U, dữ liệu X và tâm cụm V.

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
    float: Giá trị FHV
    """
    n, c = U.shape  # Số điểm dữ liệu và số cụm
    d = X.shape[1]  # Số chiều của dữ liệu
    fhv = 0

    for k in range(c):
        # Tính tử số: Ma trận hiệp phương sai mờ F_k
        numerator = np.zeros((d, d))
        denominator = 0

        for i in range(n):
            u_ik_m = U[i, k] ** m
            diff = X[i] - V[k]  # Vector chênh lệch (x_i - v_k)
            numerator += u_ik_m * np.outer(diff, diff)  # (x_i - v_k)(x_i - v_k)^T
            denominator += u_ik_m

        F_k = numerator / denominator  # Ma trận hiệp phương sai mờ

        # Tính căn bậc hai của định thức
        fhv += np.sqrt(np.linalg.det(F_k))

    return fhv


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

    # Tính FHV
    print(">>> calculate_fhv(U, X, V, m=2)")
    fhv_value = calculate_fhv(U, X, V, m=2)
    print(fhv_value)


if __name__ == "__main__":
    main()