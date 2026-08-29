import matplotlib.pylab as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from common.image_processing import ImageProcessingSettings


def generate(settings: ImageProcessingSettings = None):
    settings = settings or ImageProcessingSettings(1)

    path = '/mnt/c/Users/Alejandro Lopez/Pictures/RandomPlots/'

    ending_color = ['#6ad29e', '#ffeead', '#ff6f69', '#ffcc5c', '#4fb09f']
    t = np.linspace(0, 2 * np.pi, 1000)
    cmap = LinearSegmentedColormap.from_list("", ending_color, 1000)

    list = [2 * np.pi * (34 / 50), np.pi, 2 * np.pi]
    print(list)
    for i, curv in enumerate(list):

        plt.figure(figsize=(12, 12), facecolor='black', dpi=400)

        plt.axis('off')
        for alpha in np.linspace(0, curv, 20):

            plt.plot(-np.cos(t + alpha), np.sin(t), color=cmap(alpha / curv), lw=2)
            plt.plot(np.cos(t + alpha) + 2, np.sin(t), color=cmap(alpha / curv), lw=2)
            plt.plot(np.cos(t + alpha), np.sin(t) + 2, color=cmap(alpha / curv), lw=2)
            plt.plot(-np.cos(t + alpha) + 2, np.sin(t) + 2, color=cmap(alpha / curv), lw=2)

        settings.save_frame('black')
        plt.close()

if __name__ == '__main__':
    generate()
