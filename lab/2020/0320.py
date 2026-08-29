from math import pi

import matplotlib.pylab as plt
import numpy as np

from common.image_processing import ImageProcessingSettings


def generate(settings: ImageProcessingSettings = None):
    settings = settings or ImageProcessingSettings(1)


    p = plt.figure(figsize=(14, 14), facecolor='black', dpi=400)
    p = plt.axis('off')
    for z in np.linspace(1, 20, 100):
        n = 60
        c, s = np.cos(np.linspace(0, 2 * pi, n)), np.sin(np.linspace(0, 2 * pi, n))
        r0 = settings.rng.uniform(1 * z, 1.2 * z)
        r = [r0] + list(settings.rng.uniform(1 * z, 1.2 * z, n - 2)) + [r0]

        plt.plot(
            [c[i] * r[i] for i in range(n)],
            [s[i] * r[i] for i in range(n)],
            lw=1.1,
            color=plt.cm.cool(z / 20),
        )
    settings.save_frame('black')

if __name__ == '__main__':
    generate()
