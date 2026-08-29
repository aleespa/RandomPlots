from math import cos, sin, pi

import matplotlib.pylab as plt
import numpy as np

from common.image_processing import ImageProcessingSettings


def generate(settings: ImageProcessingSettings = None):
    settings = settings or ImageProcessingSettings(1)


    p = plt.figure(figsize=(14,14),facecolor='black',dpi=400)
    p = plt.axis('off')
    for z in np.linspace(-2*pi,2*pi,35):
        for w in np.linspace(-2*pi,2*pi,35):
            plt.plot([0.5*sin(z+w)*cos(x) +z for x in np.linspace(0,2*pi,300)],
                     [0.5*sin(z+w)*sin(x) + w for x in np.linspace(0,2*pi,300)],
                     alpha=0.9,color=settings.rng.choice(['#b8ffb9','#93eac4','#75dbd8','#6dc5df','#7eaee3']),
                     lw=2.2)
    settings.save_frame('black')

if __name__ == '__main__':
    generate()
