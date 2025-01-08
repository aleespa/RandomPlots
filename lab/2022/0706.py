import itertools
import os
import sys

import matplotlib.pylab as plt
import numpy as np

filename = os.path.basename(sys.argv[0])[:-3]


def sim_sde(
    alpha=1,
    beta=1,
    a=1,
    b=20,
    sigma=1,
    n=100000,
    m=32**2,
    dt=0.0001,
    initial_value=None,
    random_state=None,
):
    a1 = np.array([[alpha, -beta], [beta, alpha]])
    a2 = np.array([[a, -b], [b, a]])
    if type(random_state) is int:
        np.random.seed(random_state)
    W = np.random.normal(loc=0, scale=np.sqrt(dt), size=(2 * n, 2))
    if type(initial_value) == np.ndarray:
        ic = initial_value
    else:
        ic = np.random.normal(loc=0, scale=1, size=(m, 2))
    z = np.zeros((n, 2, m))
    z[0, ...] = ic.T
    for i in range(1, n):
        z1 = z[i - 1, ...]
        z[i, ...] = (
            z1
            + (a1 @ z1 - (np.linalg.norm(z1, axis=0) ** 2) * (a2 @ z1)) * dt
            + sigma * W[i, :, np.newaxis]
        )
    return z


def cartesian_product_itertools(arrays):
    return np.array(list(itertools.product(*arrays)))


r = 1
m = 600
layers = 50
n = 10000
Ic = np.stack(
    [
        r * np.cos(np.linspace(0, 2 * np.pi, m)),
        r * np.sin(np.linspace(0, 2 * np.pi, m)),
    ],
    axis=-1,
)

Z = [
    sim_sde(
        alpha=1,
        beta=1,
        a=1,
        b=25,
        initial_value=Ic * x,
        m=m,
        random_state=1,
        n=15000,
        sigma=1,
    )
    for x in np.linspace(0, 2, layers)
]

time = 20
frame_rate = 30
frames = time * frame_rate
times = np.ceil(np.linspace(0, n, frames)).astype(int)
for k, j in enumerate(times):
    if k <= 355:
        plt.figure(num=1, clear=True, figsize=(12, 12), dpi=100, facecolor='black')
        plt.axis('off')
        for i, z in enumerate(Z):
            plt.scatter(
                z[j, 0, :], z[j, 1, :], color=plt.cm.turbo(i / layers), alpha=0.8, s=5
            )
        plt.xlim(Z[0][j, 0, 0] - 3, Z[0][j, 0, 0] + 3)
        plt.ylim(Z[0][j, 1, 0] - 3, Z[0][j, 1, 0] + 3)
        plt.tight_layout()
        plt.savefig(
            f"./../outputs/{filename}/plot{frames-k}.png",
            facecolor='black',
        )
        plt.close()
