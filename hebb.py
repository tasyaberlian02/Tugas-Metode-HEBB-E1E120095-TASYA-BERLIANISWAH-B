import numpy as np

# Fungsi aktivasi hard limit
def hard_limit(x):
    return np.where(x >= 0, 1, -1)

# Data training
data_training1 = np.array([[1, -1, -1], [1, -1, -1], [1, 1, 1]])
target1 = np.array([[1]])

data_training2 = np.array([[-1, -1, -1], [1, -1, 1], [1, -1, 1]])
target2 = np.array([[-1]])

# Menghitung bobot menggunakan metode HEBB
def calculate_weights(data_training, target):
    weights = np.zeros((3, 3))

    for i in range(3):
        for j in range(3):
            weights[i, j] = data_training[i, j] * target

    return weights

# Menggabungkan bobot dari kedua data training
weights1 = calculate_weights(data_training1, target1)
weights2 = calculate_weights(data_training2, target2)
final_weights = weights1 + weights2

# Menampilkan hasil bobot
print("Bobot yang telah ditraining:")
print(final_weights)

# Inputan user
user_input = np.array([int(input(f"Masukkan nilai baris ke-{i//3 + 1}, kolom ke-{i%3 + 1} (1 atau -1): ")) for i in range(9)]).reshape(3, 3)

# Menghitung output dengan fungsi aktivasi hard limit
output = hard_limit(np.dot(user_input.flatten(), final_weights.flatten()))

# Menampilkan hasil output
print("Hasil Output:")
print(output)