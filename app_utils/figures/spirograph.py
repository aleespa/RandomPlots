import numpy as np
from matplotlib import pyplot as plt
import io
import base64
import matplotlib.colors as mcolors

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


def generate_plot(seed=0, dark=False, color=(0, 0, 0)):
    rng = np.random.default_rng(seed)
    # Generate data
    t = np.linspace(0, 2 * np.pi, 5000)
    fig, ax = plt.subplots(figsize=(12, 12), dpi=200, tight_layout=True)

    # Set background color
    fig.patch.set_facecolor(color)
    n_spirographs = rng.integers(4, 5)
    for _ in range(n_spirographs):
        a, b, c, d = generate_k_l(rng)
        if dark:
            line_color = cmap_dark(rng.uniform())
        else:
            line_color = cmap_light(rng.uniform())
        plot_spiro(t, a, b, c, d, ax, line_color)

    # Ensure the center is always at (0, 0)
    ax.set_aspect('equal')  # Ensure equal scaling for x and y axes
    ax.spines['left'].set_position('zero')  # Move y-axis to x=0
    ax.spines['bottom'].set_position('zero')  # Move x-axis to y=0
    ax.spines['top'].set_visible(False)  # Hide top spine
    ax.spines['right'].set_visible(False)  # Hide right spine

    # Automatically calculate limits and make them symmetrical
    x_limits = ax.get_xlim()
    y_limits = ax.get_ylim()
    max_limit = max(abs(x_limits[0]), x_limits[1], abs(y_limits[0]), y_limits[1])
    ax.set_xlim(-max_limit, max_limit)
    ax.set_ylim(-max_limit, max_limit)

    # Remove axis ticks and labels
    ax.axis('off')

    # Save the figure to a buffer without extra white space
    buffer = io.BytesIO()
    plt.savefig(buffer, format='jpg', bbox_inches='tight', pad_inches=0)
    buffer.seek(0)

    return buffer


def spiro(t: np.array,
          k: float = 0.5,
          l: float = 0.5):
    return ((1 - k) * np.exp(1j * t)
            + k * l * np.exp(- 1j * t * (1 - k) / k))


def plot_spiro(t, a, b, c, d, ax, color):
    scaled_t = t * a
    s = spiro(scaled_t, a / b, c / d)
    ax.plot(s.real, s.imag, lw=3.5, alpha=0.9,
            color=color)


def generate_k_l(rng):
    while True:
        a, b, c, d = sorted(rng.integers(3, 35, 4))
        k, l = a / b, c / d
        if k != 1 and l != 1 and k != l:  # Ensure k and l are not 1
            return a, b, c, d


def create_image(seed=0, dark_mode=True, color=(0, 0, 0)):
    buffer = generate_plot(seed, dark_mode, color)
    image_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close()
    return image_data

