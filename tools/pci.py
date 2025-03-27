def calculate_pci(U):
    """
    Tính toán Partition Coefficient Index (PCI) từ ma trận thành viên U.

    Parameters:
    U : list hoặc numpy array, shape (n, c)
        Ma trận thành viên, trong đó U[i][k] là giá trị thành viên của điểm i với cụm k.

    Returns:
    float: Giá trị PCI
    """
    n = len(U)  # Số lượng điểm dữ liệu
    pci_sum = 0

    # Tính tổng bình phương các giá trị thành viên
    for i in range(n):
        for k in range(len(U[i])):
            pci_sum += U[i][k] ** 2

    # Chia cho số lượng điểm dữ liệu
    pci = pci_sum / n
    return pci


def main():
    # Dữ liệu mẫu: Ma trận thành viên U với 3 điểm dữ liệu và 2 cụm
    U = [
        [0.9, 0.1],
        [0.3, 0.7],
        [0.6, 0.4]
    ]

    print(">>> calculate_pci(U)")
    pci_value = calculate_pci(U)
    print(pci_value)


if __name__ == "__main__":
    main()