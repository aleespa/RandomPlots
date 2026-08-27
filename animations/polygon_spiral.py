import numpy as np
from matplotlib import pyplot as plt

from colors.palettes import ECOSPL
from common.image_processing import ImageProcessingSettings

FIGURE_SIZE = (12, 12)
DPI = 100


def generate(settings: ImageProcessingSettings = None):
    settings = settings or ImageProcessingSettings(1)
    rng = settings.rng

    fig, _ = plt.subplots(figsize=FIGURE_SIZE, dpi=DPI)
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0), facecolor='#f4f0e7')

    y1, y2 = -260, 260
    x1, x2 = -260, 260
    ax.set_xlim(x1, x2)
    ax.set_ylim(y1, y2)
    n = 350

    def cubo(a, b, t):
        c_x = np.array([[a, -a], [b, -a], [b, -b], [a, -b], [a, -a]])
        c_y = np.array([[a, a], [a, b], [b, b], [b, a], [a, a]])
        trig_matrix = np.vstack((np.cos(t), np.sin(t)))
        x = (c_x @ trig_matrix).reshape(-1, )
        y = (c_y @ trig_matrix).reshape(-1, )

        ax.plot(x, y, color=rng.choice(ECOSPL), alpha=0.9, lw=2)

    for i, t in enumerate(np.linspace(0, 20 * np.pi, n)[::-1]):
        cubo(1 * t, 3 * t, t)
        plt.savefig(settings.frames_path / f'frame{i:04d}.png', facecolor='black')

    settings.save_video(30)


if __name__ == '__main__':
    generate()
