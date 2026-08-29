from math import cos, sin, pi

import matplotlib.pylab as plt
import numpy as np

from common.image_processing import ImageProcessingSettings


def generate(settings: ImageProcessingSettings = None):
    settings = settings or ImageProcessingSettings(1)



    p = plt.figure(figsize=(12,12),facecolor='black',dpi=800)
    p = plt.axis('off')
    for z in np.linspace(0,2*pi,15):
        for i in np.linspace(1,20,20):
            plt.plot([i*cos(x+z) for x in np.linspace(0,2*pi,4)],
                     [i*sin(x+z) for x in np.linspace(0,2*pi,4)],
                     color=plt.cm.viridis(i/(20)-0.05),
                     lw=4,alpha=0.85)
    settings.save_frame('black')

if __name__ == '__main__':
    generate()
