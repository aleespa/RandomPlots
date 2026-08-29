import matplotlib.pylab as plt
import numpy as np

from common.image_processing import ImageProcessingSettings


def iterator(r, x0, n):
    y = [x0]
    for z in range(n):
        y.append(r * (y[-1]) * (1 - y[-1]))
    return y

def generate(settings: ImageProcessingSettings = None):
    settings = settings or ImageProcessingSettings(1)

    p = plt.figure(figsize=(14, 14), facecolor='black', dpi=600)
    p = plt.axis('off')




    n = 3000
    for z, i in zip(np.linspace(1, 4, n), range(n)):
        lista = iterator(z, 0.8, 100)
        plt.scatter(
            np.linspace(0.5, 4, 10 * n)[i * 10 : (i + 1) * 10],
            lista[-10:],
            s=0.7,
            color='w',
        )
    settings.save_frame('black')

if __name__ == '__main__':
    generate()
