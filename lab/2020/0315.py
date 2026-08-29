import matplotlib.pylab as plt
import numpy as np
from scipy.stats import binom

from common.image_processing import ImageProcessingSettings


def generate(settings: ImageProcessingSettings = None):
    settings = settings or ImageProcessingSettings(1)

    colors = ['#6fa6ce', '#9799ba', '#feadb9', '#f9e1e0']
    for p, i in zip(np.linspace(0.01, 0.99, 120), range(120)):
        plt.figure(figsize=(13, 13), facecolor='black', dpi=100)
        plt.axis('off')
        plt.ylim(-0.2, 0.4)
        plt.xlim(0, 40)
        for n in [10, 20, 30, 40]:
            X = range(n)
            plt.bar(
                x=X,
                height=binom.pmf(k=X, n=n, p=p),
                alpha=0.85,
                color=colors[int(n / 10 - 1)],
            )
        settings.save_frame('black')

    for p, i in zip(np.linspace(0.99, 0.01, 120), range(120, 240)):
        plt.figure(figsize=(13, 13), facecolor='black', dpi=100)
        plt.axis('off')
        plt.ylim(-0.2, 0.4)
        plt.xlim(0, 40)
        for n in [10, 20, 30, 40]:
            X = range(n)
            plt.bar(
                x=X,
                height=binom.pmf(k=X, n=n, p=p),
                alpha=0.85,
                color=colors[int(n / 10 - 1)],
            )
        settings.save_frame('black')

if __name__ == '__main__':
    generate()
