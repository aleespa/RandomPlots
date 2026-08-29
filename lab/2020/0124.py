from math import cos, sin

import matplotlib.pylab as plt

from common.image_processing import ImageProcessingSettings


def generate(settings: ImageProcessingSettings = None):
    settings = settings or ImageProcessingSettings(1)


    p = plt.figure(figsize=(14,14),facecolor='black',dpi=400)
    p = plt.axis('off')
    plt.scatter([(z)*cos(z)*sin(z) for z in range(7000)],
                [z*sin(z)*sin(z) for z in range(7000)],
                s=4,
                color=[plt.cm.Accent(settings.rng.uniform(0,1)) for z in range(7000)])
    settings.save_frame('black')

    p = plt.figure(figsize=(14,14),facecolor='black',dpi=400)
    p = plt.axis('off')
    plt.scatter([(z)*cos(z)*sin(z*10) for z in range(7000)],
                [z*sin(z)*sin(z) for z in range(7000)],
                s=4,
                color=[plt.cm.Accent(settings.rng.uniform(0,1)) for z in range(7000)])
    settings.save_frame('black')

    p = plt.figure(figsize=(14,14),facecolor='black',dpi=400)
    p = plt.axis('off')
    plt.scatter([(z)*cos(z)*sin(z*10) for z in range(7000)],
                [z*sin(z)*sin(z*.4) for z in range(7000)],
                s=4,
                color=[plt.cm.Accent(settings.rng.uniform(0,1)) for z in range(7000)])
    settings.save_frame('black')

if __name__ == '__main__':
    generate()
