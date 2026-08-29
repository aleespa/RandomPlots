import matplotlib.pylab as plt
import numpy as np
from scipy.stats import norm

from common.image_processing import ImageProcessingSettings


def generate(settings: ImageProcessingSettings = None):
    settings = settings or ImageProcessingSettings(1)

    p = plt.figure(figsize=(15, 15), facecolor='black', dpi=400)
    p = plt.axis('off')
    p = plt.ylim(0, 1)
    for i, z in enumerate(np.linspace(-4, 4, 150)):
        p = plt.plot(
            np.linspace(-4, 4, 2),
            [norm.cdf(z) + norm.pdf(z) * (x - z) for x in np.linspace(-4, 4, 2)],
            color=plt.cm.RdPu((z + 4) / (8)),
            lw=2,
        )

    p = settings.save_frame('black')

if __name__ == '__main__':
    generate()
