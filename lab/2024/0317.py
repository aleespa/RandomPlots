import gc
import sys
import time
from datetime import datetime

import numpy as np
from loguru import logger
from matplotlib import pyplot as plt
from scipy.integrate import odeint

from tools.technology import create_directory, clear_folder, images_to_video


def generate():
    filename = sys.argv[1]
    n_frames = 60
    create_directory(f"outputs/{filename}")
    clear_folder(f"outputs/{filename}")
    fig, _ = plt.subplots(figsize=(9, 16), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1], facecolor='#f4f0e7')
    t = np.linspace(0, 2, 20)
    x_initial = np.linspace(-2, 2, n_frames)
    y_initial = np.linspace(-2, 2, n_frames)
    X, Y = np.meshgrid(x_initial, y_initial)
    initial_conditions = np.vstack([X.flatten(), Y.flatten()]).T
    mus = np.linspace(-1, 1, n_frames)
    mu = 0.4
    solutions = []
    for initial_state in initial_conditions:
        solution = odeint(system, initial_state, t, args=(mu,))
        solutions.append(np.array(solution))
    for i, mu in enumerate(mus):
        ax.clear()
        t1 = time.time()
        for solution in solutions:
            ax.scatter(solution[i, 0], solution[i, 1], s=1, color='k')
        y1, y2 = -10, 10
        x1, x2 = -10, 10
        w = x2 - x1
        h = y2 - y1
        z = (16 / 18) * w - (1 / 2) * h
        ax.set_xlim(x1, x2)
        ax.set_ylim(y1 - z, y2 + z)

        time_string = datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')
        fig.savefig(f'outputs/{filename}/{time_string}.png', facecolor='k')
        t2 = time.time()
        logger.info(
            f"theta = {mu:.8f} "
            f"frame {str(i + 1).zfill(3)}/{n_frames} "
            f"time = {t2 - t1:.2f} seconds"
        )
        gc.collect()

    images_to_video(f'outputs/{filename}', f'{filename}.mp4', 6)


def system(state, t, mu):
    x, y = state
    dxdt = mu * x + y - x**2
    dydt = -x + mu * y + 2 * x**2
    return [dxdt, dydt]
