import matplotlib.pylab as plt

from common.image_processing import ImageProcessingSettings


def generate(settings: ImageProcessingSettings = None):
    settings = settings or ImageProcessingSettings(1)

    for i in range(510):
        p = plt.figure(figsize=(13, 13), facecolor='black', dpi=100)
        p = plt.axis('off')
        x, y = settings.rng.normal(0, 1, i**2 + 20), settings.rng.normal(0, 1, i**2 + 20)
        p = plt.hist2d(x, y, bins=100, cmap='afmhot')
        p = settings.save_frame('black')

if __name__ == '__main__':
    generate()
