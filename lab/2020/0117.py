import matplotlib.pylab as plt
import numpy as np

from common.image_processing import ImageProcessingSettings


def generate(settings: ImageProcessingSettings = None):
    settings = settings or ImageProcessingSettings(1)

    p = plt.figure(figsize=(14, 14), facecolor='black', dpi=400)
    p = plt.axis('off')
    for i in range(50):
        plt.scatter(
            settings.rng.normal(0, 1, 100),
            np.random.exponential(1, 100),
            s=settings.rng.normal(45, 70, 100),
            alpha=0.7,
            color=plt.cm.YlOrRd(i / (50)),
        )
    settings.save_frame('black')

if __name__ == '__main__':
    generate()
