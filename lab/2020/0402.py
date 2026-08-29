import matplotlib.pylab as plt
import numpy as np

from common.image_processing import ImageProcessingSettings


def Cantor(n):
    if n == 0:
        return [[0, 1]]
    else:
        X = []
        for c in Cantor(n - 1):
            X.append([c[0], c[0] + 1 / (3**n)])
            X.append([c[1] - 1 / (3**n), c[1]])
        return X

def generate(settings: ImageProcessingSettings = None):
    settings = settings or ImageProcessingSettings(1)




    p = plt.figure(figsize=(15, 15), facecolor='black', dpi=400)
    p = plt.axis('off')
    m = 11
    for n, a in zip(range(m), np.linspace(1, 0, m)):
        for c in Cantor(n):
            plt.plot([c[0], c[1]], [a, a], lw=4, color=plt.cm.hsv(a))
            plt.fill_between([c[0], c[1]], [a, a], color='white', alpha=1 - a / 1.1)
    p = settings.save_frame('black')

if __name__ == '__main__':
    generate()
