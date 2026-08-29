from math import cos, sin, pi

import matplotlib.pylab as plt
import numpy as np

from common.image_processing import ImageProcessingSettings


def generate(settings: ImageProcessingSettings = None):
    settings = settings or ImageProcessingSettings(1)


    p = plt.figure(figsize=(14,14),facecolor='black',dpi=400)
    p = plt.axis('off')
    n =5
    for z in np.linspace(0,2*pi,80):
        plt.plot([cos(x)*cos(z*2) for x in np.linspace(0,2*pi,n)],
                 [sin(x)*sin(z*2) for x in np.linspace(0,2*pi,n)],lw=2,alpha=.8)

    settings.save_frame('black')

if __name__ == '__main__':
    generate()
