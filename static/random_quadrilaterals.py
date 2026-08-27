import gc

import matplotlib.pylab as plt

from common.image_processing import ImageProcessingSettings

FIGURE_SIZE = (12, 12)
DPI = 150

colors = ['#3a3663', '#414977', '#476589', '#4c7c9a', '#58c0e7']


def generate(settings: ImageProcessingSettings = None):
    settings = settings or ImageProcessingSettings(1)
    rng = settings.rng

    for fig_num in [1, 2]:
        fig, _ = plt.subplots(figsize=FIGURE_SIZE, dpi=DPI)
        ax = fig.add_axes((0.0, 0.0, 1.0, 1.0), facecolor='k')
        for _ in range(100):
            R = rng.uniform(-5, 5, 4)
            if fig_num == 1:
                x = [0, -R[0], -R[1], 0]
                y = [0, -R[2], R[3], 0]
            else:
                x = [0, -R[0], -R[0], 0]
                y = [0, -R[1], R[1], 0]
            ax.fill_between(x, y, alpha=0.25, color=rng.choice(colors))
            ax.plot(x, y, alpha=0.2, color='white', lw=1)
        settings.save_to_png(fig, 'black')
        plt.close()
        gc.collect()


if __name__ == '__main__':
    generate()
