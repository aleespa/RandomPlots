from datetime import datetime
from math import pi

import matplotlib.pylab as plt
import numpy as np
from loguru import logger

from common.image_processing import ImageProcessingSettings
from common.technology import images_to_video

colors = ['#e26000', '#228B46', '#5092B8', '#ff9b9b', '#c9d06c', '#22ba5a', '#58c0e7']


def generate(settings=ImageProcessingSettings()):
    fig, _ = plt.subplots(figsize=(12, 12), dpi=150)
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0), facecolor='#f4f0e7')

    y1, y2 = -260, 260
    x1, x2 = -260, 260
    ax.set_xlim(x1, x2)
    ax.set_ylim(y1, y2)

    def cubo(a, b, t):
        coeff_x = np.array([[a, -a], [b, -a], [b, -b], [a, -b], [a, -a]])
        coeff_y = np.array([[a, a], [a, b], [b, b], [b, a], [a, a]])
        trig_matrix = np.vstack((np.cos(t), np.sin(t)))
        x = (coeff_x @ trig_matrix).reshape(
            -1,
        )
        y = (coeff_y @ trig_matrix).reshape(
            -1,
        )

        ax.plot(x, y, color=np.random.choice(colors), alpha=0.9, lw=2)

    for t in np.linspace(0, 20 * pi, 350)[::-1]:
        cubo(
            1 * t,
            3 * t,
            t,
        )
        time_string = datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')
        logger.info(f"{time_string}.png Saved")
        fig.savefig(f'outputs/{settings.filename}/{time_string}.png', facecolor='k')

    images_to_video(f'outputs/{settings.filename}', f'{settings.filename}.mp4', 60)
