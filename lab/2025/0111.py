import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.tri import Triangulation

from tools.image_processing import ImageProcessingSettings


def generate(settings=ImageProcessingSettings(1)):

    x = settings.rng.exponential(0.2, 200)
    y = settings.rng.exponential(0.2, 200)

    for i in range(200):
        fig, ax = plt.subplots(figsize=(12, 12), dpi=200)
        ax = fig.add_axes((0, 0, 1, 1), facecolor="#f4f0e7")

        triang = Triangulation(x, y)
        edges = []
        widths = []
        for triangle in triang.triangles:
            for i in range(3):
                p1 = triangle[i]
                p2 = triangle[(i + 1) % 3]
                dist = np.linalg.norm([x[p1] - x[p2], y[p1] - y[p2]])
                edges.append([(x[p1], y[p1]), (x[p2], y[p2])])
                widths.append(1 / (1 + dist/2))  # Scale the width for better visibility
        line_collection = LineCollection(edges, linewidths=widths, colors='#000000')
        ax.add_collection(line_collection)

        ax.set(xlim=(0, 2), ylim=(0, 2))
        settings.save_frame()
        x *= (1 - 0.01 * np.log(y))
        y *= (1 - 0.01 * np.log(x))

    settings.images_to_video()
