import numpy as np
from matplotlib import pyplot as plt


def generate_plot():
    fig, ax = plt.subplots(figsize=(1, 1), dpi=50, tight_layout=True)
    fig.patch.set_facecolor('#f4f0e7')

    for _ in range(100):
        r = np.random.uniform(0, 1, 1)
        theta = np.random.uniform(0, 2*np.pi, 1)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        ax.scatter(x, y, color='k', lw=0, s=np.random.uniform(1, 6))
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)

    ax.axis('off')
    plt.savefig("random_eigen.svg", format='svg', bbox_inches='tight', pad_inches=0, transparent=True)



if __name__ == "__main__":
    generate_plot()
