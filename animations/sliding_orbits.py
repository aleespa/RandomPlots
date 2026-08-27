import numpy as np
from matplotlib import pyplot as plt

from colors.palettes import RedWht
from common.image_processing import ImageProcessingSettings

FIGURE_SIZE = (12, 12)
DPI = 100


def generate(settings: ImageProcessingSettings = None):
    settings = settings or ImageProcessingSettings(1)
    rng = settings.rng
    t = rng.uniform(0, 2 * np.pi, 580)
    color_list = rng.choice(RedWht, 580)

    for i, s in enumerate(np.linspace(0, 2, 20)):
        frame(s, t, color_list)
        settings.save_numbered_frame(i, 'black')

    settings.save_video(30)


def frame(s, t: np.ndarray, color_list: list):
    fig, _ = plt.subplots(figsize=FIGURE_SIZE, dpi=DPI)
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0), facecolor='k')
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)

    r_line = np.linspace(0, 0.5, 50)
    r_sin = np.linspace(0, s, 50)

    for m in range(1, 10):
        for k in range(m * 6 + 1):
            idx = k * m
            if idx >= len(t):
                continue  # Avoid index overflow

            angle_line = t[idx] + r_line
            angle_sin = t[idx] + r_sin

            x = m * np.cos(angle_line)
            y = m * np.sin(angle_sin)

            ax.plot(x, y, color=color_list[idx], alpha=0.8, lw=2)
            ax.scatter(
                [m * np.cos(t[idx])],
                [m * np.sin(t[idx])],
                s=22,
                zorder=10,
                color='#00b0b0',
                alpha=0.8,
            )


if __name__ == '__main__':
    generate()
