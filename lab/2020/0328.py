import matplotlib.pylab as plt
import numpy as np
from scipy.stats import gamma

from common.image_processing import ImageProcessingSettings


def generate(settings: ImageProcessingSettings = None):
    settings = settings or ImageProcessingSettings(1)

    p = plt.figure(figsize=(14, 14), facecolor='black', dpi=400)
    p = plt.axis('off')
    colors = ['#64418b', '#a1378b', '#d42173', '#fb294b', '#ff5800']
    plt.ylim(-0.05, 0.45)
    for a in np.linspace(1.1, 12):
        plt.plot(
            [gamma.pdf(a=a, x=x) for x in np.linspace(1, 15, 150)],
            lw=settings.rng.uniform(1, 4),
            alpha=0.85,
            color=settings.rng.choice(colors),
        )
    p = settings.save_frame('black')

if __name__ == '__main__':
    generate()
