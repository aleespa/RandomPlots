import matplotlib.pylab as plt
import numpy as np

from common.image_processing import ImageProcessingSettings


def generate(settings: ImageProcessingSettings = None):
    settings = settings or ImageProcessingSettings(1)

    p = plt.figure(figsize=(14, 14), facecolor='black', dpi=400)
    p = plt.axis('off')
    p = plt.xlim(-2, 2)
    p = plt.ylim(-2, 2)
    color1 = ['#003087', '#005eb8', '#0072ce', '#41b6e6', '#00a9ce']
    color2 = ['#ffa500', '#c83200', '#ffb76d', '#ffa264', '#ff875f']
    for z in range(1000):
        plt.plot(
            np.repeat(settings.rng.normal(0, 0.8), 20),
            np.linspace(-1, 1, 20),
            alpha=0.15,
            lw=settings.rng.uniform(0.5, 5),
            color=settings.rng.choice(color1),
        )
        plt.plot(
            np.linspace(-1, 1, 20),
            np.repeat(settings.rng.normal(0, 0.8), 20),
            alpha=0.15,
            lw=settings.rng.uniform(0.5, 5),
            color=settings.rng.choice(color2),
        )
    settings.save_frame('black')

if __name__ == '__main__':
    generate()
