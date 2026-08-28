import matplotlib.pylab as plt
import numpy as np

from common.image_processing import ImageProcessingSettings


def generate(settings: ImageProcessingSettings = None):
    settings = settings or ImageProcessingSettings(1)
    colors = ['#e74c3c', '#f1c40f', '#2ecc71', '#e67e22', '#27ae60']
    fig = plt.figure(figsize=(16, 9), facecolor='black', dpi=400)
    plt.axis('off')
    plt.xlim(1.3, 1.8)
    plt.ylim(0, 0.5)
    for i in np.linspace(0.1, 5, 150):
        plt.plot(
            np.linspace(1, 2, 100),
            [x**i for x in np.linspace(0, 1, 100)],
            alpha=settings.rng.beta(1, 1),
            lw=settings.rng.choice([1, 5, 3, 2, 5]),
            color=settings.rng.choice(colors),
        )

    settings.save_to_png(fig, 'black')


if __name__ == '__main__':
    generate()
