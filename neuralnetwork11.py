import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the neural network function for OR gate
def nn(x, w1, w2, w3, b1, b2):
    h1 = 1 / (1 + np.exp(-(x[0] * w1 + x[1] * w2 + b1)))
    h2 = 1 / (1 + np.exp(-(x[0] * w1 + x[1] * w2 + b2)))
    y = 1 / (1 + np.exp(-(h1 * w3 + h2 * w3)))
    return y

# Define the cost function
def cost(w1, w2, w3, b1, b2):
    total_cost = 0
    for x, t in zip(inputs, targets):
        y = nn(x, w1, w2, w3, b1, b2)
        total_cost += (t - y) ** 2
    return total_cost

# Inisialisasi data
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
targets = np.array([0, 1, 1, 0])

# Define the weights for which we want to plot the cost
num_samples = 1000
w1_range = np.random.uniform(0, 4, size=num_samples)
w2_range = np.random.uniform(0, 4, size=num_samples)
w3_range = np.random.uniform(0, 4, size=num_samples)
b1_range = np.random.uniform(0, 4, size=num_samples)
b2_range = np.random.uniform(0, 4, size=num_samples)

cost_values = np.zeros(num_samples)

# Calculate the cost for each combination of weights
for i in range(num_samples):
    cost_values[i] = cost(w1_range[i], w2_range[i], w3_range[i], b1_range[i], b2_range[i])

# Find the minimum cost and its corresponding weights
min_cost = np.min(cost_values)
min_cost_index = np.argmin(cost_values)
best_w1 = w1_range[min_cost_index]
best_w2 = w2_range[min_cost_index]
best_w3 = w3_range[min_cost_index]
best_b1 = b1_range[min_cost_index]
best_b2 = b2_range[min_cost_index]

# Plot the cost function
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(best_w1, best_w2, best_w3, c='red', label='Minimum Cost')
ax.scatter(w1_range, w2_range, cost_values, c=cost_values, cmap='viridis')
ax.set_xlabel('w1')
ax.set_ylabel('w2')
ax.set_zlabel('Cost')
ax.set_title('Cost Function')
plt.legend()
plt.show()
