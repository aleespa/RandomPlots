from math import cos, pi

import matplotlib.pylab as plt
import numpy as np

from common.image_processing import ImageProcessingSettings


def rayos(u):
    p = plt.figure(figsize=(14, 14), facecolor='black')
    p = plt.axis('off')
    plt.ylim(-2, 2)
    for i in range(20):
        plt.plot(
            np.linspace(0, 2 * pi, 1000),
            [
                r3[i] * cos((t) * r2[i] + r1[i] * u)
                for t in np.linspace(0, 2 * pi, 1000)
            ],
            alpha=r4[i],
            lw=r5[i],
        )

def generate(settings: ImageProcessingSettings = None):
    settings = settings or ImageProcessingSettings(1)


    r1, r2, r3 = (
        settings.rng.uniform(2, 8, 20),
        settings.rng.uniform(0.8, 1.3, 20),
        settings.rng.uniform(0.5, 1, 20),
    )
    r4, r5 = settings.rng.uniform(0.5, 1, 20), settings.rng.uniform(8, 20, 20)




    for u, i in zip(np.linspace(0, 6 * pi, 500), range(500)):
        rayos(u)
        settings.save_to_png(fig, 'black')
        plt.close(fig)
        import gc
        gc.collect()

if __name__ == '__main__':
    generate()
