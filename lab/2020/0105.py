from math import cos, sin, pi

import matplotlib.pylab as plt
import numpy as np

from common.image_processing import ImageProcessingSettings


def generate(settings: ImageProcessingSettings = None):
    settings = settings or ImageProcessingSettings(1)


    p = plt.figure(figsize=(12,12),facecolor='black',dpi=400)
    p = plt.axis('off')
    colors = ['#e1d18e','#ce4343','#dd653c','#6d45d6','#f4c0a9','#77bd98','#00c2c7']
    n = 0
    for p in np.linspace(0,4,60):
        plt.plot([cos(x) for x in np.linspace(0,4*pi,1000)],[cos(x*p)+sin(x*p) + sin(x) for x in np.linspace(0,4*pi,1000)],
                 lw=2,color =settings.rng.choice(colors),alpha=0.8)
        n+=1
    settings.save_frame('black')

if __name__ == '__main__':
    generate()
