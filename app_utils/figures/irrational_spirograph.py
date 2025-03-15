import io

import matplotlib.colors as mcolors
import numpy as np
from matplotlib import pyplot as plt

colors_dark = [
    "#8488d7",
    "#a86f86",
    "#8ba886",
    "#4b6673",
    "#e1e1e1",
]
cmap_dark = mcolors.LinearSegmentedColormap.from_list("custom_cmap", colors_dark, N=250)

colors_light = [
    "#e67e22",  # Muted orange
    "#c0392b",  # Muted red
    "#2980b9",  # Muted blue
    "#8e44ad",  # Muted purple
    "#38823E",  # Muted green
]
cmap_light = mcolors.LinearSegmentedColormap.from_list("custom_cmap", colors_light, N=250)


def generate_plot(seed, bg_color, dark_mode) -> io.BytesIO:
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 2 * np.pi, 10000)
    fig, ax = plt.subplots(figsize=(12, 12), dpi=200, tight_layout=True)
    fig.patch.set_facecolor(bg_color)

    for _ in range(4):
        k, l = rng.uniform(0.01, 0.99, 2)
        if dark_mode:
            color = cmap_dark(rng.uniform())
        else:
            color = cmap_light(rng.uniform())
        plot_spiro(t, k, l, ax, color)

    ax.axis('off')
    buffer = io.BytesIO()
    plt.savefig(buffer, format='jpg', bbox_inches='tight', pad_inches=0)
    buffer.seek(0)

    return buffer


def spiro(t: np.array,
          k: float = 0.5,
          l: float = 0.5):
    return ((1 - k) * np.exp(1j * t)
            + k * l * np.exp(- 1j * t * (1 - k) / k))


def plot_spiro(t, k, l, ax, color):
    s = spiro(100 * t, k, l)
    ax.plot(s.real, s.imag, lw=1, alpha=0.9,
            color=color)


def create_image(seed=0, dark_mode=True, bg_color=(0, 0, 0)):
    buffer = generate_plot(seed, bg_color, dark_mode)
    plt.close()
    return buffer.getvalue()
