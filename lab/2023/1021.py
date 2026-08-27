from typing import List

import numpy as np
from matplotlib import pyplot as plt
from numpy import pi, cos, sin

from common.image_processing import ImageProcessingSettings


def points(n: int, rng: np.random.Generator) -> tuple[np.array, np.array]:
    t = np.linspace(0, +2 * pi, n)
    k_elements = rng.integers(1, 30)
    speed = rng.integers(1, 80, k_elements)
    width = rng.normal(0, 5, k_elements)
    sine_contributions = width * sin(np.outer(t, speed))
    r = rng.uniform(40, 500) + np.sum(sine_contributions, axis=1)
    x = cos(t) * r
    y = sin(t) * r

    return x, y


def generate_plot(X: List[np.array], Y: List[np.array], settings: ImageProcessingSettings):
    fig, _ = plt.subplots(figsize=(12, 12), dpi=200)
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0), facecolor='#f4f0e7')
    for x, y in zip(X, Y):
        plt.plot(x, y, color='k', lw=1.5)
    settings.save_to_png(fig, 'k')
    plt.close()
    del x, y


def generate(settings: ImageProcessingSettings = None):
    settings = settings or ImageProcessingSettings(1)
    n = 1000
    n_lines = 40
    X, Y = [], []
    for _ in range(n_lines):
        x, y = points(n, settings.rng)
        X.append(x)
        Y.append(y)
    generate_plot(X, Y, settings)


if __name__ == '__main__':
    generate()
