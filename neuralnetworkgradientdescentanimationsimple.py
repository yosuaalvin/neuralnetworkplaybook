import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

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

# Inisialisasi scatter plot
fig, axs = plt.subplots(2, 1, figsize=(6, 8))

# Plot data asli
axs[0].scatter(inputs[:, 0], inputs[:, 1], c=targets, cmap='viridis')
axs[0].set_xlim(-0.5, 1.5)
axs[0].set_ylim(-0.5, 1.5)
axs[0].set_xlabel('Input 1')
axs[0].set_ylabel('Input 2')
axs[0].set_title('Gradient Descent Visualization')

# Plot fungsi error
axs[1].set_xlim(0, len(error_values))
axs[1].set_ylim(0, 0.5)
axs[1].set_xlabel('Epoch')
axs[1].set_ylabel('Error')
axs[1].set_title('Error Progression')

# Inisialisasi plot weight
line, = axs[0].plot([], [], 'r-', label='Decision Boundary')

# Inisialisasi plot fungsi error
error_line, = axs[1].plot([], [], 'g-', label='Error')
axs[1].legend()

# Fungsi update plot
def update_plot(i):
    # Mengambil data weight pada indeks i
    current_w = w_changes[i]

    # Menghitung garis decision boundary dari weight
    x = np.linspace(-0.5, 1.5, 100)
    y = -(current_w[0] * x + b) / current_w[1]

    # Memperbarui data pada plot decision boundary
    line.set_data(x, y)

    # Memperbarui data pada plot fungsi error
    error_line.set_data(range(i+1), error_values[:i+1])
    print("Cost (Epoch {}): y = {:.2f}x + {:.2f}".format(i, current_w[0], b))

    return line, error_line

# Animasi plot
ani = animation.FuncAnimation(fig, update_plot, frames=len(w_changes), interval=500, blit=True)

# Tampilkan animasi
plt.show()
