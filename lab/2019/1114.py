import gc
from datetime import datetime

import matplotlib.pylab as plt
import numpy as np

from common.image_processing import ImageProcessingSettings
from common.technology import images_to_video

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


def generate(settings=ImageProcessingSettings()):
    u1 = np.random.uniform(0, 1, 200)
    u2 = np.random.uniform(0, 1, 200)
    for t in range(200):
        fig, _ = plt.subplots(figsize=(12, 12), dpi=150)
        ax = fig.add_axes((0.0, 0.0, 1.0, 1.0), facecolor='k')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.scatter(
            u1, u2, s=[2600 * norm(x, t) for x in range(200)], alpha=0.7, color=colors
        )

        time_string = datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')
        plt.savefig(
            f'outputs/{settings.filename}/{time_string}.png',
            facecolor='black',
        )
        plt.close()
        gc.collect()
    images_to_video(f'outputs/{settings.filename}',
                    f'{settings.filename}.mp4',
                    30)

