import numpy as np
from matplotlib import pyplot as plt

from common.image_processing import ImageProcessingSettings


def plot(settings: ImageProcessingSettings):
    num_points = 800
    number_centers = 20
    # Sphere parameters
    radius = 0.5  # Radius of the sphere
    center = np.stack(
        [
            np.cos(np.linspace(0, 2 * np.pi, number_centers)),
            np.sin(np.linspace(0, 2 * np.pi, number_centers)),
            np.zeros(number_centers),
        ],
        axis=0,
    )

    theta = settings.rng.uniform(0, np.pi, num_points)
    phi = settings.rng.uniform(0, 2 * np.pi, num_points)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d', facecolor='#f4f0e7')

    for j in range(number_centers):
        x = center[0, j] + (j / number_centers) * np.sin(theta) * np.cos(phi)
        y = center[1, j] + (j / number_centers) * np.sin(theta) * np.sin(phi)
        z = center[2, j] + (j / number_centers) * np.cos(theta)
        ax.scatter(x, y, z, alpha=1, lw=0, color='k', s=1)

    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    ax.axis('off')
    ax.view_init(elev=10, azim=0)

    settings.save_to_png(fig, '#f4f0e7')


def generate(settings: ImageProcessingSettings = None):
    settings = settings or ImageProcessingSettings(1)
    plot(settings)


if __name__ == '__main__':
    generate()
