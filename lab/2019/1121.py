import matplotlib.pylab as plt
import numpy as np

from common.image_processing import ImageProcessingSettings


def Mandelbrot(a):
    z = 0
    n = 0
    for i in range(200):
        n += 1
        if abs(z) > 2:
            break
        else:
            z = z**2 + a
    return n


def generate(settings: ImageProcessingSettings = None):
    settings = settings or ImageProcessingSettings(1)
    x = np.linspace(-2, 0.5, 2000)
    y = np.linspace(-1, 1, 2000)
    xx, yy = np.meshgrid(x, y, sparse=True)
    z = np.array([[Mandelbrot(complex(s, t)) for s in xx[0]] for t in yy[:, 0]])

    fig = plt.figure(figsize=(12, 12), facecolor='black', dpi=400)
    plt.axis('off')
    plt.contourf(x, y, z, cmap='magma')
    settings.save_to_png(fig, 'black')


if __name__ == '__main__':
    generate()
