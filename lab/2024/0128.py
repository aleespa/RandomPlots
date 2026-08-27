import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial import ConvexHull

from common.image_processing import ImageProcessingSettings


def random_walk(n, rng: np.random.Generator):
    return np.cumsum(rng.normal(0, 0.5, (n, 2)), axis=0)


def generate_plot(Z: np.array, settings: ImageProcessingSettings):
    fig, _ = plt.subplots(figsize=(12, 12), dpi=200)
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0), facecolor='#f4f0e7')
    # ax.plot(Z[:,0], Z[:,1], color='k', lw=1.5)
    for k in range(10, 10000, 5):
        hull = ConvexHull(Z[:k, :])
        for simplex in hull.simplices:
            ax.plot(Z[:k, :][simplex, 0],
                     Z[:k, :][simplex, 1],
                     lw=1, color='k')
    hull = ConvexHull(Z[:, :])
    for simplex in hull.simplices:
        ax.plot(Z[:, :][simplex, 0],
                Z[:, :][simplex, 1],
                lw=6, color='#162807')
        # 800000
        # 162807
        # 003366
    settings.save_to_png(fig, 'k')
    plt.close()


def generate(settings: ImageProcessingSettings = None):
    settings = settings or ImageProcessingSettings(1)
    n = 10000
    Z = random_walk(n, settings.rng)
    generate_plot(Z, settings)


if __name__ == '__main__':
    generate()
