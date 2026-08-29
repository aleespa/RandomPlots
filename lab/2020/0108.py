from math import cos, sin, pi

import matplotlib.pylab as plt
import numpy as np

from common.image_processing import ImageProcessingSettings


def generate(settings: ImageProcessingSettings = None):
    settings = settings or ImageProcessingSettings(1)


    p = plt.figure(figsize=(12,12),facecolor='black',dpi=400)
    p = plt.axis('off')

    for i in range(1,21):
        plt.plot([i*cos(x) for x in np.linspace(0,2*pi,500)],
                 [i*sin(x) for x in np.linspace(0,2*pi,500)],
                 color=plt.cm.plasma(i/(20)-0.25),
                 lw=i/3+2)
    for i in np.linspace(-1,1,55):
        plt.plot(np.linspace(-20,20,2) ,[i*x for x in np.linspace(-20,20,2)],color='black'
                 ,alpha=0.7,lw=2)
        plt.plot([i*x for x in np.linspace(-20,20,2)],np.linspace(-20,20,2),color='black'
                 ,alpha=0.7,lw=4)
    settings.save_frame('black')

if __name__ == '__main__':
    generate()
