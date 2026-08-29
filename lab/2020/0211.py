import matplotlib.pylab as plt
import numpy as np
from scipy.integrate import odeint

from common.image_processing import ImageProcessingSettings


def f(state, t):
    x, y, z = state  # Desempaqueta el vector de estado
    return sigma * (y - x), x * (rho - z) - y, x * y - beta * z  # Derivadas

def generate(settings: ImageProcessingSettings = None):
    settings = settings or ImageProcessingSettings(1)

    rho = 28.0
    sigma = 10.0
    beta = 8.0 / 3.0




    state0 = [1.0, 1.0, 1.0]
    t = np.arange(0.0, 100.0, 0.001)

    states = odeint(f, state0, t)

    for j in range(333):
        fig = plt.figure(figsize=(14, 14), facecolor='black', dpi=300)
        ax = fig.gca(projection='3d', facecolor='black')
        p = plt.axis('off')
        ax.set_zlim(10, 45)
        ax.set_xlim(-10, 20)
        ax.set_ylim(-45, 20)
        ax.plot(
            states[:, 0],
            states[:, 1],
            states[:, 2],
            color=plt.cm.bwr(1 - j / (333)),
            alpha=0.3,
        )
        ax.plot(
            states[:, 0][j * 300 : (j + 1) * 300],
            states[:, 1][j * 300 : (j + 1) * 300],
            states[:, 2][j * 300 : (j + 1) * 300],
            color=plt.cm.bwr(j / (333)),
            lw=4,
        )
        ax.scatter(
            states[:, 0][(j + 1) * 300],
            states[:, 1][(j + 1) * 300],
            states[:, 2][(j + 1) * 300],
            color='white',
        )
        p = settings.save_frame('black')

if __name__ == '__main__':
    generate()
