from math import cos, pi

import matplotlib.pylab as plt
import numpy as np

from common.image_processing import ImageProcessingSettings


def generate(settings: ImageProcessingSettings = None):
    settings = settings or ImageProcessingSettings(1)


    p = plt.figure(figsize=(14,14),facecolor='black',dpi=400)
    p = plt.axis('off')
    for z in np.linspace(-3*pi,3*pi,200):
        plt.plot([0,0,z*cos(z),z*cos(z),0],[0,z*cos(z),z*cos(z),0,0],lw=1.2)
        plt.plot([0,0,-z*cos(z),-z*cos(z),0],[0,z*cos(z),z*cos(z),0,0],lw=1.2)
    settings.save_frame('black')

if __name__ == '__main__':
    generate()
