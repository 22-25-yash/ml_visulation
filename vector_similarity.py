import numpy as np
import matplotlib.pyplot as plt

def dot_product(v1, v2):
    return np.dot(v1, v2)

def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def euclidean_distance(v1, v2):
    return np.linalg.norm(v1 - v2)

def plot_vectors(v1, v2):
    fig, ax = plt.subplots()

    ax.quiver(0, 0, v1[0], v1[1], angles='xy', scale_units='xy', scale=1, color='r')
    ax.quiver(0, 0, v2[0], v2[1], angles='xy', scale_units='xy', scale=1, color='b')

    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.grid()

    return fig