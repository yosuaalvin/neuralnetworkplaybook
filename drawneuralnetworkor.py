import numpy as np
import matplotlib.pyplot as plt

# Inisialisasi data
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
targets = np.array([0, 1, 1, 1])

# Inisialisasi parameter
w = np.array([-0.5, 0.5])
b = 0

# Inisialisasi learning rate
learning_rate = 0.1

# Inisialisasi list untuk menyimpan perubahan weight dan error
w_changes = []
error_values = []

# Definisikan fungsi sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Definisikan fungsi error
def calculate_error(y, y_hat):
    return 0.5 * np.mean((y - y_hat)**2)

# Loop untuk gradient descent
for epoch in range(50):
    # Forward pass
    linear_output = np.dot(inputs, w) + b
    output = sigmoid(linear_output)

    # Hitung error
    error = calculate_error(targets, output)
    error_values.append(error)

    # Backward pass
    delta_output = (output - targets) * output * (1 - output)
    grad_w = np.dot(inputs.T, delta_output)
    grad_b = np.sum(delta_output)

    # Update parameter
    w -= learning_rate * grad_w
    b -= learning_rate * grad_b

    # Simpan perubahan weight
    w_changes.append(w.copy())

# Menggambar struktur jaringan saraf tiruan
fig, ax = plt.subplots()
ax.set_aspect('equal')

# Menggambar neuron pada lapisan input
ax.scatter([0, 0], [0.8, 0.2], color='b', s=100)
ax.text(0, 0.85, 'Input Layer', ha='center', va='bottom')

# Menggambar neuron pada lapisan tersembunyi
ax.scatter([0.5], [0.5], color='g', s=100)
ax.text(0.5, 0.55, 'Hidden Layer', ha='center', va='bottom')

# Menggambar neuron pada lapisan output
ax.scatter([1], [0.5], color='r', s=100)
ax.text(1, 0.55, 'Output Layer', ha='center', va='bottom')

# Menggambar koneksi antar neuron
for i in range(4):
    ax.plot([0, 0.5], [0.8, 0.5], 'k-')
    ax.plot([0, 0.5], [0.2, 0.5], 'k-')
    ax.plot([0.5, 1], [0.5, 0.5], 'k-')

# Menampilkan gambar
plt.show()
