from math import cos, sin, pi

import matplotlib.pylab as plt
import numpy as np

from common.image_processing import ImageProcessingSettings


def generate(settings: ImageProcessingSettings = None):
    settings = settings or ImageProcessingSettings(1)


    p = plt.figure(figsize=(12,12),facecolor='black',dpi=400)
    p = plt.axis('off')
    i = 0
    for x in np.linspace(-10,10,10):
        i+=1
        for y in np.linspace(-10,10,10):
            plt.plot([2*cos(t)+y for t in np.linspace(0,2*pi,100)],
                     [2*sin(t)+x for t in np.linspace(0,2*pi,100)],
                     alpha=0.8,
                     lw=i**0.8+1,
                     color=((i/10),0,1))
    settings.save_frame('black')

if __name__ == '__main__':
    generate()
