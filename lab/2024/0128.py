import sys

import numpy as np
from scipy.spatial import ConvexHull
import matplotlib

from matplotlib import pyplot as plt

matplotlib.use('Agg')

def random_walk(n):
    # Seed = 1,2 and 3
    np.random.seed(3)
    return np.cumsum(np.random.normal(0,0.5, (n, 2)),
                     axis=0)

def convex_hull(points):
   return ConvexHull(points)

def generate_plot(Z: np.array,
                  filename: str):
    fig, _ = plt.subplots(figsize=(12, 12), dpi=200)
    ax = fig.add_axes([0, 0, 1, 1], facecolor='#f4f0e7')
    # ax.plot(Z[:,0], Z[:,1], color='k', lw=1.5)
    for k in range(10,10000,5):
        hull = ConvexHull(Z[:k,:])
        for simplex in hull.simplices:
            ax.plot(Z[:k,:][simplex, 0],
                     Z[:k,:][simplex, 1],
                     lw=1, color='k')
    hull = ConvexHull(Z[:, :])
    for simplex in hull.simplices:
        ax.plot(Z[:, :][simplex, 0],
                Z[:, :][simplex, 1],
                lw=6, color='#162807')
        # 800000
        # 162807
        # 003366
    fig.savefig(f'outputs/{filename}.png', facecolor='k')
    plt.close()

def generate():
    filename = sys.argv[1]
    n = 10000
    Z = random_walk(n)
    generate_plot(Z, filename)
