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





    n = 50
    X, Y = [brownian_path(600) for i in range(n)], [brownian_path(600) for i in range(n)]
    fig = plt.figure(figsize=(13, 13), facecolor='black', dpi=100)
    p = plt.axis('off')
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    for k in range(600):
        p = plt.scatter(
            [x[k] for x in X], [y[k] for y in Y], alpha=0.8, color=plt.cm.rainbow(k / 600)
        )
        settings.save_to_png(fig, 'black')
        plt.close(fig)
        import gc
        gc.collect()

if __name__ == '__main__':
    generate()
