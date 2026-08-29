import matplotlib.pylab as plt
import numpy as np

from common.image_processing import ImageProcessingSettings


def generate(settings: ImageProcessingSettings = None):
    settings = settings or ImageProcessingSettings(1)

    n = 400
    random1 = settings.rng.uniform(0, 1, n)
    random2 = settings.rng.uniform(0, 1, n)
    random3 = np.random.exponential(6, n)
    p = plt.figure(figsize=(14, 14), facecolor='black', dpi=400)
    p = plt.axis('off')
    p = plt.scatter(random1, random2, s=random3, color='white', zorder=n**2 + 10)
    for i in range(n):
        for j in range(n):
            if settings.rng.choice([False, True], p=[0.99, 0.01]):
                plt.plot(
                    [random1[i], random1[j]],
                    [random2[i], random2[j]],
                    alpha=0.3,
                    lw=1,
                    color=plt.cm.inferno(settings.rng.uniform(0, 1)),
                )
    settings.save_frame('black')

if __name__ == '__main__':
    generate()
