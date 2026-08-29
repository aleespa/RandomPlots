from math import sqrt

import matplotlib.pylab as plt
import numpy as np

from common.image_processing import ImageProcessingSettings


def brownian_path(N):
    Δt_sqrt = sqrt(1 / N)
    Z = np.random.randn(N)
    Z[0] = 0
    B = np.cumsum(Δt_sqrt * Z)
    return B

def generate(settings: ImageProcessingSettings = None):
    settings = settings or ImageProcessingSettings(1)


    p = plt.figure(figsize=(14, 14), facecolor='black', dpi=400)
    p = plt.axis('off')




    X = brownian_path(80)
    for c in np.linspace(0, 0.5):
        plt.plot(c * X + c * abs(min(X)), color='white', lw=0.5, zorder=50)
    for c in np.linspace(0, 0.5):
        plt.plot(-(c * X + c * abs(min(X))), color='red', lw=0.5)
    plt.show()
    # settings.save_frame('black')

if __name__ == '__main__':
    generate()
