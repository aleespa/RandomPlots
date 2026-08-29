from math import pi

import matplotlib.pylab as plt
import numpy as np

from common.image_processing import ImageProcessingSettings


def helix(R,r,n):
    fig = plt.figure(figsize=(13,13),facecolor='black')
    ax = fig.gca(projection='3d',facecolor='black')
    ax.set_zlim(-5,5)
    ax.set_xlim(-5,5)
    ax.set_ylim(-5,5)
    p = plt.axis('off')
    t = np.linspace(0, 2*pi, 2000)
    x,y,z=(R+r*np.cos(n*t))*np.cos(t), (R+r*np.cos(n*t))*np.sin(t), r*np.sin(n*t)
    ax.plot(x, y, z,lw=6,alpha=1,color=plt.cm.hsv(n/70))

def generate(settings: ImageProcessingSettings = None):
    settings = settings or ImageProcessingSettings(1)



    for l,i in zip(np.linspace(0,70,390),range(390)):
        helix(5,2,l)
        settings.save_to_png(fig, 'black')
        plt.close(fig)
        import gc
        gc.collect()

if __name__ == '__main__':
    generate()
