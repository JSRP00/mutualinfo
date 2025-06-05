# examples/synthetic_data.py

import numpy as np
import matplotlib.pyplot as plt
from mutualinfo.utils import generate_dependent_data, generate_independent_data


def plot_data(x, y, title, ax):
    ax.scatter(x, y, s=10, alpha=0.6)
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")


def main():
    seed = 42
    n = 500

    x1, y1 = generate_independent_data(n=n, seed=seed)
    x2, y2 = generate_dependent_data(n=n, noise=0.1, seed=seed)
    x3, y3 = generate_dependent_data(n=n, noise=0.5, seed=seed)
    x4 = np.random.normal(0, 1, n)
    y4 = x4 + np.random.normal(0, 0.2, n)

    fig, axs = plt.subplots(2, 2, figsize=(10, 8))

    plot_data(x1, y1, "Independientes", axs[0, 0])
    plot_data(x2, y2, "Dependencia No Lineal (ruido bajo)", axs[0, 1])
    plot_data(x3, y3, "Dependencia No Lineal (ruido alto)", axs[1, 0])
    plot_data(x4, y4, "Dependencia Lineal", axs[1, 1])

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
