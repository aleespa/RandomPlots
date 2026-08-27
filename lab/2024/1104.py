import matplotlib.colors as mcolors
import matplotlib.pylab as plt
import numpy as np
from loguru import logger

from common.image_processing import ImageProcessingSettings

colors = ["#f4f0e7", "#000007", "#EB251F", '#ffffff']  # Red, Green, Blue, Yellow, Cyan

cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", colors, N=1000)


def generate(settings: ImageProcessingSettings = None):
    settings = settings or ImageProcessingSettings(1)

    logger.info(f"Starting calculation")
    n = 600
    theta = np.linspace(0, 2 * np.pi, n)
    # z = 1.1 * np.exp(1j * theta) * ((2 - np.exp(1j * theta)) / 4)
    # for i in range(n):
    #
    #     subprocess.run(["java", "julia", f"{z[i].real}", f'{z[i].imag}',
    #                     f'outputs/{settings.filename}/temp/{i}.txt'],
    #                    capture_output=True, text=True)
    #
    #     logger.info(f"Calculation finished for theta = {theta[i]}")

    temp_dir = settings.output_path / settings.filename / 'temp'
    for i in range(n):
        number_iterations_1 = np.loadtxt(
            temp_dir / f"{i}.txt", delimiter=",", dtype=float
        )
        number_iterations_2 = np.loadtxt(
            temp_dir / f"{n-i-1}.txt", delimiter=",", dtype=float
        )
        fig, ax = plt.subplots(figsize=(12, 12), dpi=200)
        ax = fig.add_axes((0, 0, 1, 1), facecolor="#f4f0e7")
        ax.set_xticks([])
        ax.set_yticks([])

        number_iterations = number_iterations_1 + number_iterations_2
        ax.imshow(number_iterations[500:3500, 500:3500], cmap=cmap, vmax=200, vmin=3)

        settings.save_numbered_frame(i, "#f4f0e7")
    settings.save_video(30)
    logger.info(f"Finished")


if __name__ == '__main__':
    generate()
