import gc
import sys
from datetime import datetime
from typing import List

import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from numpy import pi, cos, sin

matplotlib.use('Agg')


def points(n: int, seed: int) -> tuple[np.array, np.array]:
    t = np.linspace(0, +2 * pi, n)
    np.random.seed(seed)
    k_elements = np.random.randint(1, 30)
    speed = np.random.randint(1, 80, k_elements)
    width = np.random.normal(0, 5, k_elements)
    sine_contributions = width * sin(np.outer(t, speed))
    r = np.random.uniform(40, 500) + np.sum(sine_contributions, axis=1)
    x = cos(t) * r
    y = sin(t) * r

    return x, y


def generate_plot(X: List[np.array], Y: List[np.array], directory: str):
    current_time = datetime.now()
    time_string = current_time.strftime('%Y-%m-%d_%H-%M-%S-%f')
    fig, _ = plt.subplots(figsize=(12, 12), dpi=200)
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0), facecolor='#f4f0e7')
    for x, y in zip(X, Y):
        plt.plot(x, y, color='k', lw=1.5)
    fig.savefig(f'outputs/{directory}/{time_string}.png', facecolor='k')
    plt.close()
    del x, y
    gc.collect()


def generate():
    filename = sys.argv[1]
    n = 1000
    n_lines = 40
    X, Y = [], []
    for seed in range(n_lines):
        x, y = points(n, seed)
        X.append(x)
        Y.append(y)
    generate_plot(X, Y, filename)
