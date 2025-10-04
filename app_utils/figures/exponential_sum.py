import io

import numpy as np
from matplotlib import pyplot as plt


def create_image(seed=0, bg_color=(0, 0, 0), cmap=None):
    buffer = generate_plot(seed, bg_color, cmap)
    plt.close()
    return buffer.getvalue()


def generate_plot(seed, bg_color=(0, 0, 0), cmap=None):
    rng = np.random.default_rng(seed)

    fig, ax = plt.subplots(figsize=(12, 12), dpi=200, tight_layout=True)
    fig.patch.set_facecolor(bg_color)

    n = 4500
    num_segments = 50
    step = 90
    n_range = np.arange(0, n, 1)
    param_1 = rng.uniform(12, 15)
    param_2 = rng.uniform(90, 120)
    s = np.cumsum(
        np.exp((n_range / param_2 + np.cos(n_range * param_1)) * (2 * np.pi * 1j))
    )
    s_real = s.real
    s_imag = s.imag

    for z in range(num_segments):
        start = step * z
        end = step * (z + 1) + 1
        plt.plot(
            s_real[start:end],
            s_imag[start:end],
            lw=1,
            color=cmap(z / num_segments + 0.2),
            alpha=0.8,
        )
    ax.axis('off')

    buffer = io.BytesIO()
    plt.savefig(buffer, format='jpg', bbox_inches='tight', pad_inches=0)
    buffer.seek(0)
    plt.close()
    return buffer
