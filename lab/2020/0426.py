import matplotlib.pylab as plt
import numpy as np
from scipy.stats import norm

for k, loc in enumerate(np.linspace(-16, 16, 120)):

    p = plt.figure(figsize=(13, 13), facecolor='black', dpi=100)
    p = plt.axis('off')
    p = plt.ylim(0, 4.3)
    for i, z in enumerate(np.linspace(1, 4, 30)):
        plt.plot(
            np.linspace(-10, 10, 500),
            [
                z
                - z * norm.pdf(x=x, loc=loc, scale=z / 2)
                - z * norm.pdf(x=x, loc=-loc, scale=z / 2)
                for x in np.linspace(-10, 10, 500)
            ],
            lw=3,
            color=plt.cm.Spectral((z - 1) / 3),
        )
    p = plt.savefig(
        f'C:/Users/Alejandro/Pictures/RandomPlots/26042020/plot{k+120}.PNG',
        facecolor='black',
    )
