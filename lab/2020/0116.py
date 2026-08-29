from math import cos, sin, pi

import matplotlib.pylab as plt
import numpy as np

from common.image_processing import ImageProcessingSettings


def gal(u):
    p = plt.figure(figsize=(14,14),facecolor='black')
    p = plt.axis('off')
    for i in np.linspace(0,5,120):
        plt.scatter([i*sin(x+u*i) for x in np.linspace(0,6*pi,100)],
                    [i*cos(x+i) for x in np.linspace(0,6*pi,100)],s=10,alpha=0.8,color=colors)

def generate(settings: ImageProcessingSettings = None):
    settings = settings or ImageProcessingSettings(1)


    colors = ['#f28db3','#69edde',	'#ffddff'	,'#ffffff'	,'#ED5F94']*20


    for u, i in zip([x**2 for x in np.linspace(0,5,600)],range(600)):
        gal(u)
        settings.save_to_png(fig, 'black')
        plt.close(fig)
        import gc
        gc.collect()

if __name__ == '__main__':
    generate()
