import matplotlib.pyplot as plt
import numpy as np

recorrido = np.array(
    [
        [63, 14, 37, 24, 51, 26, 35, 10],
        [22, 39, 62, 13, 36, 11, 50, 27],
        [15, 64, 23, 38, 25, 52, 9, 34],
        [40, 21, 16, 61, 12, 33, 28, 49],
        [17, 60, 1, 44, 29, 48, 53, 8],
        [2, 41, 20, 57, 6, 55, 32, 47],
        [59, 18, 43, 4, 45, 30, 7, 54],
        [42, 3, 58, 19, 56, 5, 46, 31],
    ]
)


def tablero(w, h):
    re = np.r_[w * [0, 1]]
    ro = np.r_[w * [1, 0]]
    return np.row_stack(h * (re, ro))


tablero_ajedrez = tablero(4, 4)


def camino_caballo(slider):
    tablero_ajedrez[recorrido == slider] = -1
    fig, ax = plt.subplots(facecolor='white', figsize=(12, 12), dpi=200)
    ax.imshow(tablero_ajedrez, cmap=plt.cm.gist_gray, interpolation='nearest')
    plt.axis('off')


for i in range(1, 65):
    camino_caballo(i)
    plt.savefig(
        f'C:/Users/Alejandro/Pictures/RandomPlots/29112019/plor{i}.png',
        facecolor='black',
    )
