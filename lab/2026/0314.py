import gc
from datetime import datetime

import numpy as np
from matplotlib import pyplot as plt

from common.image_processing import ImageProcessingSettings


def generate(settings=ImageProcessingSettings(1)):
    filename = settings.filename
    bg_color = "k"
    fig, ax = plt.subplots(figsize=(12, 12), dpi=200, tight_layout=True)
    fig.patch.set_facecolor(bg_color)
    ax.axis('off')

    t = np.linspace(0, 10 * np.pi, 1000)
    for i in np.linspace(0, 2 * np.pi, 1):
        x = np.cos(t * np.cos(t) + np.pi / 2)
        y = np.sin(t * np.sin(t) - np.cos(t))
        plt.plot(x,y,lw=2, color='white')

    time_string = datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')
    fig.savefig(f'outputs/{filename}/{time_string}.png', facecolor=bg_color)
    plt.close()
    gc.collect()
