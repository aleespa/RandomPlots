from math import cos, sin, pi

import matplotlib.pylab as plt
import numpy as np

from common.image_processing import ImageProcessingSettings


def generate(settings: ImageProcessingSettings = None):
    settings = settings or ImageProcessingSettings(1)


    p = plt.figure(figsize=(12,12),facecolor='black',dpi=400)
    p = plt.axis('off')
    for x,y,z in [(5*cos(t),5*sin(t),t) for t in np.linspace(0,2*pi)]:
        plt.scatter(settings.rng.normal(x,1.5,400),settings.rng.normal(y,1.5,400),s=settings.rng.uniform(5,100,400),alpha=0.6,
                    color=plt.cm.winter(z/(2*pi)))
    settings.save_frame('black')

if __name__ == '__main__':
    generate()
