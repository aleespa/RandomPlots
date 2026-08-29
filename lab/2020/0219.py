from math import cos, pi

import matplotlib.pylab as plt
import numpy as np

from common.image_processing import ImageProcessingSettings


def generate(settings: ImageProcessingSettings = None):
    settings = settings or ImageProcessingSettings(1)


    p = plt.figure(figsize=(14,14),facecolor='black',dpi=400)
    p = plt.axis('off')
    n = 80
    for z in  list(np.linspace(0,pi,n)):
        plt.scatter([cos(x*z) for x in np.linspace(0,1,n)],[x for x in np.linspace(0,10,n)],
                    alpha=0.8,color=[plt.cm.rainbow(settings.rng.uniform(0,z/(pi))) for _ in range(n)],s=17)
        plt.scatter([-cos(x*z) for x in np.linspace(0,1,n)],[x+10 for x in -np.linspace(0,10,n)],
                    alpha=0.8,color=[plt.cm.rainbow(settings.rng.uniform(0,z/(pi))) for _ in range(n)],s=17)
    settings.save_frame('black')

if __name__ == '__main__':
    generate()
