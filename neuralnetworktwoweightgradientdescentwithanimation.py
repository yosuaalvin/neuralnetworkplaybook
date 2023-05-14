import numpy as np
import matplotlib
matplotlib.use("Qt5Agg")  # Specify the backend (e.g., TkAgg, Qt5Agg, etc.)
from IPython.display import HTML

import matplotlib.pyplot as plt

import matplotlib.animation as animation

# Inisialisasi parameter error kecil
w1 = np.array([0.5, 0.3])
w2 = np.array([0.4, 0.7])
w3 = np.array([0.6, 0.9])
b1 = 0.1
b2 = 0.1

# Inisialisasi data
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
targets = np.array([0, 1, 1, 1])

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
for epoch in range(100):
    # Forward pass
    h1 = sigmoid(np.dot(inputs, w1) + b1)
    h2 = sigmoid(np.dot(inputs, w2) + b2)
    output = sigmoid(np.dot(np.column_stack((h1, h2)), w3))

    # Hitung error
    error = calculate_error(targets, output)
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
    #print(w1)
    # Simpan perubahan weight
    w1_changes.append(w1.copy())
    w2_changes.append(w2.copy())
    w3_changes.append(w3.copy())
    b1_changes.append(b1)
    b2_changes.append(b2)

# Inisialisasi scatter plot dan plot cost function
fig, axs = plt.subplots(3, 1, figsize=(6, 10))

# Plot data asli
axs[0].scatter(inputs[:, 0], inputs[:, 1], c=targets, cmap='viridis')
axs[0].set_xlim(-0.5, 1.5)
axs[0].set_ylim(-0.5, 1.5)
axs[0].set_xlabel('Input 1')
axs[0].set_ylabel('Input 2')
axs[0].set_title('Gradient Descent Visualization')

# Plot cost function
axs[1].plot(range(len(error_changes)), error_changes)
axs[1].set_xlabel('Epoch')
axs[1].set_ylabel('Error')
axs[1].set_title('Cost Function')

# Plot weights
x = np.arange(len(w1_changes))
axs[2].scatter(x, [w[0] for w in w1_changes], c='blue', label='w1')
axs[2].scatter(x, [w[0] for w in w2_changes], c='green', label='w2')
axs[2].scatter(x, [w[0] for w in w3_changes], c='red', label='w3')
axs[2].scatter(x, b1_changes, c='purple', label='b1')
axs[2].scatter(x, b2_changes, c='orange', label='b2')
axs[2].set_xlabel('Epoch')
axs[2].set_ylabel('Weight Value')
axs[2].set_title('Weight Progression')
axs[2].legend()

# Initialize the decision boundary line
line, = axs[0].plot([], [], 'r-', label='Decision Boundary')

# Initialize the error line
error_line, = axs[1].plot([], [], 'g-', label='Error')
#plt.ion()
def update_plot(i):
# Mengambil data weight pada indeks i
    current_w1 = w1_changes[i]
    current_w2 = w2_changes[i]
    current_w3 = w3_changes[i]
    current_b1 = b1_changes[i]
    current_b2 = b2_changes[i]
    # Menghitung garis decision boundary dari weight
    #y = -(current_w1[0] * x + current_b1) / current_w2[0]

    # Memperbarui data pada plot decision boundary
    #line = axs[0].plot(x, y, 'r-', label='Decision Boundary')

    # Memperbarui data pada plot cost function
    #error_line, = axs[1].plot([], [], 'g-', label='Error')
    #y = -(current_w1[0] * x + current_w1[1] * current_b1) / (current_w2[0] * current_w2[1])
    y = -(current_w1[0] * x + current_w1[1] * current_b1 + current_w3[0] * current_b2) / (current_w2[0] * current_w2[1])
    m = -current_w1[0] / current_w2[0]
    c = -(current_w1[1] * current_b1 + current_w3[0] * current_b2) / (current_w2[0] * current_w2[1])
    print("Persamaan garis: y = {}x + {}".format(m, c))    
    print(y)
    # Memperbarui data pada plot decision boundary
    line.set_data(x, y)

    # Memperbarui data pada plot cost function
    error_line.set_data(range(i + 1), error_changes[:i + 1])

    # Mencetak nilai weight
    print("Epoch {}: w1 = {:.2f}, w2 = {:.2f}, w3 = {:.2f}, b1 = {:.2f}, b2 = {:.2f}".format(i, current_w1[0], current_w2[0], current_w3[0], current_b1, current_b2))


    return line, error_line
ani = animation.FuncAnimation(fig, update_plot, frames=len(w1_changes), interval=100, blit=True)
#ani.save('animation2.mp4', writer='ffmpeg')
#HTML(ani.to_html5_video())

plt.show()