import subprocess

import numpy as np


def julia_java(
        z: float,
        size: int,
        x_centre: float,
        y_center: float,
        radius: float,
        filename: str) -> None:
    r = subprocess.run(["java", "Julia",
                        f"{z.real}",
                        f'{z.real}',
                        f'{size}',
                        f'{x_centre}',
                        f'{y_center}',
                        f'{radius}',
                        f'{filename}'],
                       capture_output=True, text=True)


def julia_set_v1(
        z: float,
        size: int,
        x_centre: float,
        y_center: float,
        radius: float,
        filename: str) -> None:
    m = n = size

    x = np.linspace(x_centre - radius, x_centre + radius, num=m).reshape((1, m))
    y = np.linspace(y_center - radius, y_center + radius, num=n).reshape((n, 1))
    grid = np.tile(x, (n, 1)) + 1j * np.tile(y, (1, m))

    index = np.full((n, m), True, dtype=bool)
    number_iterations = np.zeros((n, m))

    for i in range(256):
        grid[index] = grid[index] ** 2 + z
        index[np.abs(z) > 2] = False
        number_iterations[index & (grid > 1)] = i - np.log(np.log(np.abs(grid[index & (z > 1)]))) / np.log(2)
        number_iterations[index & (grid <= 1)] = i

    np.savetxt(filename, number_iterations, fmt='%.2f')


def julia_set_v2(
        z: float,
        size: int,
        x_centre: float,
        y_center: float,
        radius: float,
        filename: str) -> None:
    m = n = size

    x = np.linspace(x_centre - radius, x_centre + radius, num=m).reshape((1, m))
    y = np.linspace(y_center - radius, y_center + radius, num=n).reshape((n, 1))
    grid = np.tile(x, (n, 1)) + 1j * np.tile(y, (1, m))

    index = np.full((n, m), True, dtype=bool)
    number_iterations = np.zeros((n, m))

    for i in range(256):
        grid[index] = grid[index] ** 2 + z
        index[np.abs(grid) > 2] = False
        number_iterations[index & (grid > 1)] = i - np.log(np.log(np.abs(grid[index & (grid > 1)]))) / np.log(2)
        number_iterations[index & (grid <= 1)] = i

    np.savetxt(filename, number_iterations, fmt='%.2f')
