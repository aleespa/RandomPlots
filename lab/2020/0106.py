from math import cos, sin, pi

import matplotlib.pylab as plt
import numpy as np

from common.image_processing import ImageProcessingSettings


def generate(settings: ImageProcessingSettings = None):
    settings = settings or ImageProcessingSettings(1)


    p = plt.figure(figsize=(12,12),facecolor='black',dpi=400)
    p = plt.axis('off')
    for r in np.linspace(0,2*pi,200):
        X = [0]
        Y = [0]
        for _ in range(1000):
            X.append(X[-1] + settings.rng.normal(cos(r),2))
            Y.append(Y[-1] + settings.rng.normal(sin(r),2))
        plt.plot(X,Y,alpha=0.65,lw=0.8,color=plt.cm.hsv(r/(2*pi)))
    settings.save_frame('black')

if __name__ == '__main__':
    generate()
