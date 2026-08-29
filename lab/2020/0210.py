from math import cos, sin, pi

import matplotlib.pylab as plt
import numpy as np

from common.image_processing import ImageProcessingSettings


def generate(settings: ImageProcessingSettings = None):
    settings = settings or ImageProcessingSettings(1)



    p = plt.figure(figsize=(14,14),facecolor='black',dpi=400)
    p = plt.axis('off')
    for z in np.linspace(0,1,100):
        plt.plot([z*cos(x)*settings.rng.uniform(0.95,1.05) for x in np.linspace(0,2*pi,1000)],
                 [z*sin(x)*settings.rng.uniform(0.95,1.05) for x in np.linspace(0,2*pi,1000)],alpha=0.8,
                 color = plt.cm.Blues(settings.rng.uniform(0,1)))
    settings.save_frame('black')

if __name__ == '__main__':
    generate()
