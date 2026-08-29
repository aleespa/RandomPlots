from math import cos, sin, pi

import matplotlib.pylab as plt
import numpy as np

from common.image_processing import ImageProcessingSettings


def generate(settings: ImageProcessingSettings = None):
    settings = settings or ImageProcessingSettings(1)


    p = plt.figure(figsize=(12,12),facecolor='black',dpi=400)
    p = plt.axis('off')
    for p in np.linspace(0,5,60):
        plt.scatter(np.linspace(0,1,900),[p*cos(x+p**2)+sin(x) for x in np.linspace(-2*pi,2*pi,900)],
                    s=1,alpha=0.8,color =plt.cm.Spectral(settings.rng.uniform(0,1)-0.1))
    settings.save_frame('black')

if __name__ == '__main__':
    generate()
