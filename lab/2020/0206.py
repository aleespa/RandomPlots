from math import cos, sin, pi

import matplotlib.pylab as plt
import numpy as np

from common.image_processing import ImageProcessingSettings


def spiral(t):
    p = plt.figure(figsize=(15, 15), facecolor='black', dpi=100)
    p = plt.axis('off')
    for k, i in zip(np.linspace(0, 1.5, 60), range(60)):
        p = plt.plot(
            [k * cos(x) * sin(x) for x in np.linspace(0, 2 * pi, 700)],
            [k * sin(x + t) ** 2 for x in np.linspace(0, 2 * pi, 700)],
            alpha=0.85,
            lw=random_lw[i],
            color=random_colors[i],
        )

def generate(settings: ImageProcessingSettings = None):
    settings = settings or ImageProcessingSettings(1)


    colors = ['#3b567d', '#67a23f', '#f6ae33', '#f94333', '#249596']
    random_colors = settings.rng.choice(colors, 60)
    random_lw = settings.rng.uniform(0.5, 4.5, 60)




    for t, i in zip(np.linspace(0, pi / 2, 360), range(360)):
        spiral(t)
        settings.save_to_png(fig, 'black')
        plt.close(fig)
        import gc
        gc.collect()

if __name__ == '__main__':
    generate()
