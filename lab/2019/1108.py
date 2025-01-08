import random
import string
from math import cos, sin, pi

import matplotlib.pylab as plt
import numpy as np
import toml

from tools.technology import create_directory


def pol(n):
    config = toml.load('config.toml')
    filename = config['file_to_run']
    create_directory(f"outputs/{filename}")
    u = np.linspace(0, 2 * pi, n)
    fig, ax = plt.subplots(1, 1, figsize=(14, 14), dpi=200)
    ax = fig.add_axes([0, 0, 1, 1], facecolor='k')
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    for j in range(n):
        for i in range(n):
            ax.plot([cos(u[j]), cos(u[i])], [sin(u[j]), sin(u[i])], lw=0.7)
    name = ''.join(random.choices(string.ascii_letters + string.digits, k=8))
    fig.savefig(f'outputs/{filename}/{name}.png', facecolor='k')


def generate():
    pol(40)
