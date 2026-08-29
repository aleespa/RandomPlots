from math import cos, sin

import matplotlib.pylab as plt

from common.image_processing import ImageProcessingSettings


def generate(settings: ImageProcessingSettings = None):
    settings = settings or ImageProcessingSettings(1)


    p = plt.figure(figsize=(14,14),facecolor='black',dpi=400)
    p = plt.axis('off')
    for z in range(200):
        rango =range(z*10,(z+1)*10)
        plt.plot([(z)*cos(z)*sin(z*2) for z in rango],
                 [z*sin(z)*sin(z) for z in rango],
                 lw=4,
                 alpha=0.7,
                 color = plt.cm.RdPu(settings.rng.uniform(0,1)))
    settings.save_frame('black')

if __name__ == '__main__':
    generate()
