from math import cos, sin, pi

import matplotlib.pylab as plt
import numpy as np

from common.image_processing import ImageProcessingSettings


def generate(settings: ImageProcessingSettings = None):
    settings = settings or ImageProcessingSettings(1)


    p = plt.figure(figsize=(12,12),facecolor='black',dpi=800)
    p = plt.axis('off')
    n=0
    for p in np.linspace(0,5,60):
        plt.plot(np.linspace(0,1,800),[cos(x+p)+sin(x) for x in np.linspace(-3*pi,3*pi,800)],
                 lw=6,alpha=0.8,color =plt.cm.rainbow(settings.rng.uniform(0,1)-0.1))
    settings.save_frame('black')

if __name__ == '__main__':
    generate()
