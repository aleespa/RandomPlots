from math import cos, sin, pi

import matplotlib.pylab as plt
import numpy as np

from common.image_processing import ImageProcessingSettings


def drawCircle(x,y,r):
    plt.plot([r*cos(t)+x for t in np.linspace(0,2*pi,70)],
             [r*sin(t)+y for t in np.linspace(0,2*pi,70)],lw=1,color="white")
    if r>2:
        drawCircle(x + r/2, y, r/2)
        drawCircle(x , y + r/2, r/2)
        drawCircle(x , y - r/2, r/2)
        drawCircle(x - r/2, y, r/2)

def generate(settings: ImageProcessingSettings = None):
    settings = settings or ImageProcessingSettings(1)


    p = plt.figure(figsize=(14,14),facecolor='black',dpi=500)
    p = plt.axis('off')
    drawCircle(0,0,50)

    p = settings.save_frame('black')

if __name__ == '__main__':
    generate()
