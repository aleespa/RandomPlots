from math import cos, sin, pi

import matplotlib.pylab as plt
import numpy as np

from common.image_processing import ImageProcessingSettings


def generate(settings: ImageProcessingSettings = None):
    settings = settings or ImageProcessingSettings(1)


    p = plt.figure(figsize=(14,14),facecolor='black',dpi=400)
    p = plt.axis('off')
    for k in np.linspace(0.5,22,100):
        plt.plot([k*cos(z)+cos(k) for z in np.linspace(0,2*pi,1500)],
                 [k*sin(z)+sin(k) for z in np.linspace(0,2*pi,1500)],
                 alpha=0.8,
                 lw=settings.rng.uniform(0.8,3),
                 color=plt.cm.rainbow(((k+25)/50)))

    settings.save_frame('black')

if __name__ == '__main__':
    generate()
