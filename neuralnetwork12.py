import numpy as np
import matplotlib.pyplot as plt

# Define the neural network function y = x * w
def nn(x, w1, w2, w3, b1, b2):
    h1 = 1 / (1 + np.exp(-(x[0] * w1 + x[1] * w1 + b1)))
    h2 = 1 / (1 + np.exp(-(x[0] * w2 + x[1] * w2 + b2)))
    y = 1 / (1 + np.exp(-(h1 * w3[0] + h2 * w3[1])))
    return y

# Define the cost function
def cost(w1, w2, w3, b1, b2):
    total_cost = 0
    for x, t in zip(inputs, targets):
        y = nn(x, w1, w2, w3, b1, b2)
        total_cost += (t - y) ** 2
    return total_cost

# Define the inputs and targets
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
targets = np.array([0, 1, 1, 0])

# Define the range for the weights or biases you want to vary
w1_range = np.linspace(-5, 5, num=100)
w2_range = np.linspace(-5, 5, num=100)

# Fix the other weights and biases to specific values
w3 = [0.5, 0.3]
b1 = 0.2
b2 = 0.4

# Compute the cost values
cost_values = np.zeros((len(w1_range), len(w2_range)))
for i, w1 in enumerate(w1_range):
    for j, w2 in enumerate(w2_range):
        cost_values[i, j] = cost(w1, w2, w3, b1, b2)

# Plot the cost values in a 2D plot
plt.imshow(cost_values, extent=[w2_range.min(), w2_range.max(), w1_range.min(), w1_range.max()], origin='lower', cmap='viridis')
plt.colorbar(label='Cost')
plt.xlabel('w2')
plt.ylabel('w1')
plt.title('Cost Function')
plt.show()
