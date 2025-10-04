import io

import numpy as np
from matplotlib import pyplot as plt


def create_image(seed=0, bg_color=(0, 0, 0), cmap=None):
    buffer = generate_plot(seed, bg_color, cmap)
    plt.close()
    return buffer.getvalue()


def generate_plot(seed, bg_color=(0, 0, 0), cmap=None):
    rng = np.random.default_rng(seed)
    cc = rng.uniform(-0.02, 0.02)
    cc2 = rng.uniform(-0.02, 0.02)
    c = cc + cc2 * 1j
    s = -1

    fig, ax = plt.subplots(figsize=(12, 12), dpi=200, tight_layout=True)
    fig.patch.set_facecolor(bg_color)

    n_loops = 400
    n_points = 256
    r = rng.normal(0, 1, (2, n_loops))
    x = np.zeros((n_points, n_loops), dtype=np.complex64)
    y = np.zeros((n_points, n_loops), dtype=np.complex64)
    x[0, :] = 1 + r[1, :] - c * np.abs(r[0, :])
    y[0, :] = s * r[0, :]

    for j in range(1, n_points):
        x[j, :] = 1 + y[j - 1, :] - c * np.abs(x[j - 1, :])
        y[j, :] = s * x[j - 1, :]

    x_flat = x.flatten()
    y_flat = y.flatten()
    colors = np.repeat(cmap(np.linspace(0, 1, n_loops)), n_points, axis=0)

    # Single scatter call
    ax.scatter(
        x_flat.real,
        y_flat.real,
        s=1.1,
        color=colors,
        alpha=0.95,
        lw=0,
    )
    ax.axis('off')
    buffer = io.BytesIO()
    plt.savefig(buffer, format='jpg', bbox_inches='tight', pad_inches=0)
    buffer.seek(0)

    return buffer
