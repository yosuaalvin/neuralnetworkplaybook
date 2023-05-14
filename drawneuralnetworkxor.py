import matplotlib.pyplot as plt
import numpy as np

# Fungsi aktivasi sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Menggambar jaringan saraf tiruan dengan fungsi aktivasi
def draw_neural_network():
    # Membuat objek figure dan axes
    fig, ax = plt.subplots()

    # Menentukan posisi neuron pada setiap lapisan
    input_layer_pos = [(0, 0.8), (0, 0.2)]
    hidden_layer_pos = [(0.5, 0.6), (0.5, 0.4)]
    output_layer_pos = [(1, 0.5)]

    # Menggambar neuron pada setiap lapisan
    for pos in input_layer_pos:
        circle = plt.Circle(pos, 0.1, color='b', ec='k')
        ax.add_patch(circle)
    for pos in hidden_layer_pos:
        circle = plt.Circle(pos, 0.1, color='g', ec='k')
        ax.add_patch(circle)
    for pos in output_layer_pos:
        circle = plt.Circle(pos, 0.1, color='r', ec='k')
        ax.add_patch(circle)

    # Menggambar koneksi antar neuron
    for input_pos in input_layer_pos:
        for hidden_pos in hidden_layer_pos:
            ax.plot([input_pos[0], hidden_pos[0]], [input_pos[1], hidden_pos[1]], 'k-')
    for hidden_pos in hidden_layer_pos:
        for output_pos in output_layer_pos:
            ax.plot([hidden_pos[0], output_pos[0]], [hidden_pos[1], output_pos[1]], 'k-')

    # Menggambar fungsi aktivasi sigmoid
    ax.text(hidden_layer_pos[0][0], hidden_layer_pos[0][1] + 0.1, 'sigmoid', ha='center', va='center')

    # Menyembunyikan axis dan menentukan batas gambar
    ax.axis('off')
    ax.set_xlim([-0.2, 1.2])
    ax.set_ylim([-0.2, 1.2])

    # Menampilkan gambar
    plt.show()

# Memanggil fungsi untuk menggambar jaringan saraf tiruan
draw_neural_network()
