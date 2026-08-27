import matplotlib.pylab as plt
import numpy as np

from common.image_processing import ImageProcessingSettings

colors = (
        ['#ff6f4b'] * 20
        + ['#fd4c55'] * 20
        + ['#e13661'] * 20
        + ['#c1246b'] * 20
        + ['#a11477'] * 20
        + ['#c1246b'] * 20
        + ['#e13661'] * 20
        + ['#fd4c55'] * 20
        + ['#ff6f4b'] * 40
)


def norm(x, t):
    return np.exp(-((x - t) ** 2) / 550) * (x < t)


def generate(settings: ImageProcessingSettings = None):
    settings = settings or ImageProcessingSettings()
    u1 = settings.rng.uniform(0, 1, 200)
    u2 = settings.rng.uniform(0, 1, 200)
    for t in range(200):
        fig, _ = plt.subplots(figsize=(12, 12), dpi=150)
        ax = fig.add_axes((0.0, 0.0, 1.0, 1.0), facecolor='k')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.scatter(
            u1, u2, s=[2600 * norm(x, t) for x in range(200)], alpha=0.7, color=colors
        )
        settings.save_numbered_frame(t, 'black')
    settings.save_video(30)


if __name__ == '__main__':
    generate()
