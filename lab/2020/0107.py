import matplotlib.pylab as plt
import numpy as np
import seaborn as sns

from common.image_processing import ImageProcessingSettings


def generate(settings: ImageProcessingSettings = None):
    settings = settings or ImageProcessingSettings(1)

    p = plt.figure(figsize=(12, 12), facecolor='black', dpi=400)
    p = plt.axis('off')

    for m in np.linspace(-10, 10, 20):
        x = settings.rng.normal(m, 1, size=15)
        p = sns.kdeplot(x, shade=True)
    p = plt.ylim(-0.60, 1)
    settings.save_frame('black')

if __name__ == '__main__':
    generate()
