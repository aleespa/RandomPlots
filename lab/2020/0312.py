import matplotlib.pylab as plt
import numpy as np

from common.image_processing import ImageProcessingSettings


def generate(settings: ImageProcessingSettings = None):
    settings = settings or ImageProcessingSettings(1)

    p = plt.figure(figsize=(14, 14), facecolor='black', dpi=400)
    p = plt.axis('off')

    for u in np.linspace(0.02, 0.1, 55):
        K = settings.rng.uniform(1, 15)
        Sn = np.append([1], abs(settings.rng.normal(1 + u, 0.01, 60)))
        plt.plot(
            K * Sn.cumprod() / (K + (Sn.cumprod() - 1)),
            color=plt.cm.rainbow(settings.rng.uniform(0, 1)),
            lw=settings.rng.uniform(2, 4.5),
            alpha=0.85,
        )

    settings.save_frame('black')

if __name__ == '__main__':
    generate()
