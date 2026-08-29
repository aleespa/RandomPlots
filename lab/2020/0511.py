import matplotlib.pylab as plt

from common.image_processing import ImageProcessingSettings


def barra(x, y, c):
    plt.fill_between([x, x + 1], [y, y], color=c)

def generate(settings: ImageProcessingSettings = None):
    settings = settings or ImageProcessingSettings(1)




    n = 35
    X = settings.rng.choice(range(1, n + 1), n, replace=False)
    l = 0
    for i in range(n - 1):
        for j in range(0, n - i - 1):
            if X[j] > X[j + 1]:
                X[j], X[j + 1] = X[j + 1], X[j]
            fig, ax = plt.subplots(figsize=(13, 13), facecolor='black', dpi=100)
            p = plt.axis('off')
            for k, x in enumerate(X):
                barra(k, x, c=plt.cm.rainbow(1 - x / n))
            p = settings.save_frame('black')
            l += 1

if __name__ == '__main__':
    generate()
