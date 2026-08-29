from math import cos, sin, pi

import matplotlib.pylab as plt
import numpy as np

from common.image_processing import ImageProcessingSettings


def cubo(a,b,t,color):
    p = plt.plot([a*cos(t)-a*sin(t),b*cos(t)-a*sin(t),b*cos(t)-b*sin(t),a*cos(t)-b*sin(t),a*cos(t)-a*sin(t)],
                 [a*sin(t)+a*cos(t),b*sin(t)+a*cos(t),b*sin(t)+b*cos(t),a*sin(t)+b*cos(t),a*sin(t)+a*cos(t)],
                 color=plt.cm.rainbow(a/(400*pi**2)),alpha=1,lw=1.2)

def generate(settings: ImageProcessingSettings = None):
    settings = settings or ImageProcessingSettings(1)


    p = plt.figure(figsize=(14,14),facecolor='black',dpi=400)
    p = plt.axis('off')
    p = plt.xlim(-400,400)
    p = plt.ylim(-400,400)
    for t in np.linspace(0,20*pi,180):
        cubo(t**2,t*2,t,'red')
    settings.save_frame('black')

if __name__ == '__main__':
    generate()
