import gc
from datetime import datetime
from math import sin, cos

import matplotlib.pylab as plt
import numpy as np
import toml
from loguru import logger

from tools.technology import create_directory, images_to_video, clear_folder

t = np.random.uniform(0, 2 * np.pi, 580)
colors = [
    '#ff817a',
    '#ff8d87',
    '#ff9a94',
    '#ffa6a1',
    '#ffb3af',
    '#ffc0bc',
    '#ffccc9',
    '#ffd9d7',
    '#ffe5e4',
    '#fff2f1',
    '#ffffff',
]
colorslist = np.random.choice(colors, 580)


def plots(s):
    config = toml.load('config.toml')
    filename = config['file_to_run']
    fig, _ = plt.subplots(figsize=(12, 12), dpi=100)
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0), facecolor='k')
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    for m in range(1, 10):
        for k in range(m * 6 + 1):
            ax.plot(
                [m * cos(t[k * m] + r) for r in np.linspace(0, 0.5, 50)],
                [m * sin(t[k * m] + r) for r in np.linspace(0, s, 50)],
                color=colorslist[k * m],
                alpha=0.8,
                lw=2,
            )
            ax.scatter(
                [m * cos(t[k * m])],
                [m * sin(t[k * m])],
                s=22,
                zorder=10,
                color='#00b0b0',
                alpha=0.8,
            )

    time_string = datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')
    plt.savefig(f'outputs/{filename}/plot{time_string}.PNG', facecolor='black')
    logger.info(f"{time_string}.png Saved")
    plt.close()
    gc.collect()


def generate():
    config = toml.load('config.toml')
    filename = config['file_to_run']
    create_directory(f"outputs/{filename}")
    clear_folder(f"outputs/{filename}")

    for s in np.linspace(0, 2, 180):
        plots(s)
    images_to_video(f'outputs/{filename}', f'{filename}.mp4', 30)
    logger.info(f"Finished")
