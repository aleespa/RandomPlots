import numpy as np
from matplotlib import pyplot as plt


def generate_plot():
    t = np.linspace(0, 2 * np.pi, 2000)
    fig, ax = plt.subplots(figsize=(1, 1), dpi=50, tight_layout=True)
    fig.patch.set_facecolor('#f4f0e7')

    for x in [0.3,
              0.6499999999999999,
              0.825]:
        k, l = (x, 0.3)
        plot_spiro(t, k, l, ax)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)

    ax.axis('off')
    plt.savefig("cont_spirograph.svg", format='svg', bbox_inches='tight', pad_inches=0, transparent=True)


def spiro(t: np.array,
          k: float = 0.5,
          l: float = 0.5):
    return ((1 - k) * np.exp(1j * t)
            + k * l * np.exp(- 1j * t * (1 - k) / k))


def plot_spiro(t, k, l, ax):
    s = spiro(100 * t, k, l)
    ax.plot(s.real, s.imag, lw=0.5, alpha=1,
            color="k")


if __name__ == "__main__":
    generate_plot()
