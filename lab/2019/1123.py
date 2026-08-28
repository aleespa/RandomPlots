import matplotlib.pylab as plt
import numpy as np

from common.image_processing import ImageProcessingSettings


def Lolka(VI, rng):
    alpha, beta, gamma, delta = -0.4, 0.01, 0.2, -0.01
    h = 0.01
    f = lambda x: np.array(
        [
            alpha * x[0] + beta * x[0] * x[1] + rng.uniform(-3, 3),
            gamma * x[1] + delta * x[0] * x[1],
        ]
    )
    U = [np.array(VI)]
    T = np.arange(0, 400, h)
    for j in range(len(T) - 1):
        k1 = f(U[j])
        k2 = f(U[j] + (h / 2) * k1)
        k3 = f(U[j] + (h / 2) * k2)
        k4 = f(U[j] + h * k3)
        U.append(U[j] + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4))

    plt.plot(
        [U[i][0] for i in range(len(U))],
        [U[i][1] for i in range(len(U))],
        alpha=0.8,
        lw=1,
    )


def generate(settings: ImageProcessingSettings = None):
    settings = settings or ImageProcessingSettings(1)
    fig = plt.figure(figsize=(12, 12), facecolor='black', dpi=400)
    plt.axis('off')
    for i in range(1, 11):
        Lolka([5 * i, 80], settings.rng)
    settings.save_to_png(fig, 'black')


if __name__ == '__main__':
    generate()
