import numpy as np
from matplotlib import pyplot as plt

from tools.image_processing import ImageProcessingSettings


def generate(settings=ImageProcessingSettings(5)):

    B = -1
    C = 0.6

    fig, ax = plt.subplots(1, 1, figsize=(14, 14), facecolor='#000000', dpi=200)
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0), facecolor='#000000')
    nloops = 400
    n_points = 4000
    for eloop in range(0, nloops):

        xlast = settings.rng.normal(0, 1, 1)
        ylast = settings.rng.normal(0, 1, 1)

        xnew = np.zeros(shape=(n_points,))
        ynew = np.zeros(shape=(n_points,))
        for loop in range(0, n_points):
            xnew[loop] = 1 + ylast - C * abs(xlast)
            ynew[loop] = B * xlast
            xlast = xnew[loop]
            ylast = ynew[loop]

        plt.scatter(
            np.real(xnew),
            np.real(ynew),
            s=0.3,
            color=plt.cm.plasma(eloop / nloops),
            alpha=0.9,
            lw=0,
        )
        plt.xlim(-1.35, 2.1)
        plt.ylim(-2.1, 1.35)
    settings.save_to_png()
