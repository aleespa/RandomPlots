import os
import sys

import matplotlib.pylab as plt
import numpy as np
from numpy import pi, cos, sin

filename = os.path.basename(sys.argv[0])[:-3]


def rotation_matrix(angle):
    return np.array([[cos(angle), -sin(angle)], [sin(angle), cos(angle)]])


colors = ['b', 'r', 'g']
for i in range(3):
    fig, _ = plt.subplots(figsize=(12, 12), dpi=400)
    ax = fig.add_axes([0, 0, 1, 1], facecolor='black')
    for _ in range(35):
        initialX, initialY = np.random.uniform(-2e2, 2e2), np.random.uniform(-2e2, 2e2)
        intersections = np.random.randint(2, 15)
        for _ in range(intersections):
            n_simulations = np.random.randint(10, 500)
            rotation = rotation_matrix(np.random.uniform(0, 2 * pi))
            random_directions = np.stack(
                [
                    np.random.normal(3, 1.1, n_simulations).cumsum(axis=0),
                    np.random.normal(-1, 1.1, n_simulations).cumsum(axis=0),
                ]
            )
            road = rotation @ random_directions
            ax.plot(
                np.concatenate([[initialX], initialX + road[0, :]]),
                np.concatenate([[initialY], initialY + road[1, :]]),
                lw=np.random.exponential(n_simulations / 1200),
                color='w',
            )

    for _ in range(35):
        initialX, initialY = np.random.uniform(-1e2, 1e2), np.random.uniform(-1e2, 1e2)
        intersections = np.random.randint(2, 15)
        for _ in range(intersections):
            n_simulations = np.random.randint(10, 50)
            rotation = rotation_matrix(np.random.uniform(0, 2 * pi))
            random_directions = np.stack(
                [
                    np.random.normal(3, 1 / 3, n_simulations).cumsum(axis=0),
                    np.random.normal(-1, 1, n_simulations).cumsum(axis=0),
                ]
            )
            road = rotation @ random_directions
            ax.plot(
                np.concatenate([[initialX], initialX + road[0, :]]),
                np.concatenate([[initialY], initialY + road[1, :]]),
                lw=np.random.exponential(n_simulations / 100),
                color=colors[i],
            )

    ax.set_xlim(-250, 250)
    ax.set_ylim(-250, 250)
    fig.savefig(f'../outputs/{filename}_{i}.png', facecolor='k')
