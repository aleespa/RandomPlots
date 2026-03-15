import gc

import numpy as np
from matplotlib import pyplot as plt

from common.image_processing import ImageProcessingSettings


def generate(settings=ImageProcessingSettings(1)):
    bg_color = "k"
    fig, ax = plt.subplots(figsize=(12, 12), dpi=200, tight_layout=True)
    fig.patch.set_facecolor(bg_color)
    ax.axis('off')

    t = np.linspace(0, 10 * np.pi, 1000)
    for i in np.linspace(0, 2 * np.pi, 1):
        x = np.cos(t * np.cos(t) + np.pi / 2)
        y = np.sin(t * np.sin(t) - np.cos(t))
        plt.plot(x,y,lw=2, color='white')

    settings.save_to_png(fig, bg_color)
    plt.close()
    gc.collect()

if __name__ == '__main__':
    generate()