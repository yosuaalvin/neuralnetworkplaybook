import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Inisialisasi parameter error kecil
w1 = np.array([0.5, 0.3])
w2 = np.array([0.4, 0.7])
w3 = np.array([0.6, 0.9])
b1 = 0.2
b2 = 0.4

# Inisialisasi data
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
targets = np.array([0, 1, 1, 0])

# Inisialisasi learning rate
learning_rate = 0.1

# Inisialisasi list untuk menyimpan perubahan weight dan error
w1_changes = []
w2_changes = []
w3_changes = []
b1_changes = []
b2_changes = []
error_changes = []

# Definisikan fungsi sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Definisikan turunan sigmoid
def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

# Definisikan fungsi error
def calculate_error(y, y_hat):
    return 0.5 * np.mean((y - y_hat)**2)

# Loop untuk gradient descent
for epoch in range(50):
    # Forward pass
    h1 = sigmoid(np.dot(inputs, w1) + b1)
    h2 = sigmoid(np.dot(inputs, w2) + b2)
    output = sigmoid(np.dot(np.column_stack((h1, h2)), w3))

    # Hitung error
    error = calculate_error(targets, output)
    print(error)
    error_changes.append(error)

    # Backward pass
    delta_output = (output - targets) * sigmoid_derivative(output)
    delta_h2 = np.dot(delta_output, w3[1]) * sigmoid_derivative(h2)
    delta_h1 = np.dot(delta_output, w3[0]) * sigmoid_derivative(h1)

    # Hitung gradien
    grad_w3 = np.dot(np.column_stack((h1, h2)).T, delta_output)
    grad_b2 = np.sum(delta_output, axis=0)
    grad_w2 = np.dot(inputs.T, delta_h2)
    grad_w1 = np.dot(inputs.T, delta_h1)
    grad_b1 = np.sum(delta_h1, axis=0)

    # Update parameter
    w3 -= learning_rate * grad_w3
    b2 -= learning_rate * grad_b2
    w2 -= learning_rate * grad_w2
    w1 -= learning_rate * grad_w1
    b1 -= learning_rate * grad_b1
    print(w1)
    # Simpan perubahan weight
    w1_changes.append(w1.copy())
    w2_changes.append(w2.copy())
    w3_changes.append(w3.copy())
    b1_changes.append(b1)
    b2_changes.append(b2)

fig, ax = plt.subplots()
ax.set_xlim(0, len(w1_changes))
ax.set_ylim(0, 2)
ax.set_xlabel('Weight Update')
ax.set_ylabel('Error')

# Inisialisasi scatter plot
sc_w1 = ax.scatter([], [], c='blue', label='w1')
sc_w2 = ax.scatter([], [], c='green', label='w2')
sc_w3 = ax.scatter([], [], c='red', label='w3')
sc_b1 = ax.scatter([], [], c='purple', label='b1')
sc_b2 = ax.scatter([], [], c='orange', label='b2')
ax.legend()

# Inisialisasi plot grafik fungsi error
line, = ax.plot([], [], c='black')

# Fungsi update scatter plot
def update_scatter_plot(i):
    # Mengambil data perubahan weight pada indeks i
    w1_val = w1_changes[i][0]
    w2_val = w2_changes[i][0]
    w3_val = w3_changes[i][0]
    b1_val = b1_changes[i]
    b2_val = b2_changes[i]
    error_val = error_changes[i]

    # Mengatur posisi pada masing-masing scatter plot
    sc_w1.set_offsets(np.array([[i, w1_val], [i, error_val]]))
    sc_w2.set_offsets(np.array([[i, w2_val], [i, error_val]]))
    sc_w3.set_offsets(np.array([[i, w3_val], [i, error_val]]))
    sc_b1.set_offsets(np.array([[i, b1_val], [i, error_val]]))
    sc_b2.set_offsets(np.array([[i, b2_val], [i, error_val]]))

    # Memperbarui data pada plot grafik fungsi error
    line.set_data(range(i+1), error_changes[:i+1])

    return sc_w1, sc_w2, sc_w3, sc_b1, sc_b2, line

# Animasi scatter plot
ani = animation.FuncAnimation(fig, update_scatter_plot, frames=len(w1_changes), interval=200, blit=True)

# Tampilkan animasi
plt.show()


