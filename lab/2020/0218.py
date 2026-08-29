from math import cos, sin, pi

import matplotlib.pylab as plt
import numpy as np

from common.image_processing import ImageProcessingSettings


def generate(settings: ImageProcessingSettings = None):
    settings = settings or ImageProcessingSettings(1)


    i=0
    for z in np.linspace(0,2,900):
        p = plt.figure(figsize=(14,14),facecolor='black')
        p = plt.axis('off')
        p = plt.xlim(-280,280)
        p = plt.ylim(-280,280)
        def cubo(a,b,t,color):
            p = plt.plot([a*cos(t)-a*sin(t),b*cos(t)-a*sin(t),b*cos(t)-b*sin(t),a*cos(t)-b*sin(t),a*cos(t)-a*sin(t)],
                         [a*sin(t)+a*cos(t),b*sin(t)+a*cos(t),b*sin(t)+b*cos(t),a*sin(t)+b*cos(t),a*sin(t)+a*cos(t)],
                         color=plt.cm.cool(a/(20*pi)-0.3),alpha=0.8,lw=3)
        for t in np.linspace(0,20*pi,80):
            cubo(t,3*t+1,t*z,'red')
        settings.save_frame('black')
        i+=1

if __name__ == '__main__':
    generate()
