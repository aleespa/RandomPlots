import matplotlib.pylab as plt
import numpy as np
from matplotlib.tri import Triangulation

from common.image_processing import ImageProcessingSettings


def generate(settings: ImageProcessingSettings = None):
    settings = settings or ImageProcessingSettings(1)

    theta = np.linspace(0, 2 * np.pi, 200)
    w = np.linspace(-0.25, 0.25, 20)
    w, theta = np.meshgrid(w, theta)
    phi = 0.5 * theta
    r = 1 + w * np.cos(phi)

    x = np.ravel(r * np.cos(theta))
    y = np.ravel(r * np.sin(theta))
    z = np.ravel(w * np.sin(phi))

    tri = Triangulation(np.ravel(w), np.ravel(theta))
    colors = [plt.cm.winter(settings.rng.uniform(0, 1)) for i in range(4000)]

    i = 0
    for t in np.linspace(0, 360, 300):
        fig = plt.figure(figsize=(15, 15))
        ax = plt.gca(projection='3d', facecolor='black')
        ax.scatter(x, y, z, s=32, color=colors)
        p = plt.axis('off')
        ax.set_xlim(-0.8, 0.8)
        ax.set_ylim(-0.8, 0.8)
        ax.set_zlim(-0.4, 0.4)
        ax.view_init(20, t)
        settings.save_frame('black')
        i += 1

if __name__ == '__main__':
    generate()
