import matplotlib.pylab as plt

from common.image_processing import ImageProcessingSettings


def generate(settings: ImageProcessingSettings = None):
    settings = settings or ImageProcessingSettings(1)

    p = plt.figure(figsize=(14, 14), facecolor='black', dpi=400)
    p = plt.axis('off')
    for z in range(-25, 25):
        plt.plot(
            [0, 1],
            [0, z],
            lw=0.35 * (abs(z) + 3) ** 0.5,
            alpha=1,
            zorder=200 - z,
            color=plt.cm.rainbow(((z + 25) / 50)),
        )
        plt.plot(
            [2, 1],
            [0, z],
            lw=0.35 * (abs(z) + 3) ** 0.5,
            alpha=1,
            zorder=200 - z,
            color=plt.cm.rainbow(1 - ((z + 25) / 50)),
        )
    settings.save_frame('black')

if __name__ == '__main__':
    generate()
