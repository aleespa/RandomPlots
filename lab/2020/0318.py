from math import cos, sin, pi

import matplotlib.pylab as plt
import numpy as np

from common.image_processing import ImageProcessingSettings


def generate(settings: ImageProcessingSettings = None):
    settings = settings or ImageProcessingSettings(1)


    p = plt.figure(figsize=(14,14),facecolor='black',dpi=400)

    p = plt.axis('off')
    colors = ['#ffe7f8','#ffcfee','#c8bbff','#bff0ff','#dff8ff']
    for u in np.linspace(0,2*pi,100):
        for z in np.linspace(0,2*pi,9):
            plt.plot([cos(u),cos(u)+cos(z)],[sin(u),sin(u)+sin(z)],alpha=1,lw=2,
                     color=settings.rng.choice(colors),zorder=settings.rng.randomint(50))

    settings.save_frame('black')

if __name__ == '__main__':
    generate()
