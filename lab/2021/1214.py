import matplotlib.pylab as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

path = '/mnt/c/Users/Alejandro Lopez/Pictures/RandomPlots/'

for i, d in enumerate(np.linspace(1, 3, 180)):
    plt.figure(figsize=(12, 12), facecolor='black', dpi=200)
    plt.axis('off')
    ending_color = ['#6ad29e', '#ffeead', '#ff6f69', '#ffcc5c', '#4fb09f']
    for r in [4, 6, 8, 10, 12]:
        cmap = LinearSegmentedColormap.from_list("", [ending_color[int(r / 2) - 2], ending_color[int(r / 2) - 3],
                                                      ending_color[int(r / 2) - 2]], 100)
        for theta in np.linspace(0, 2 * np.pi, r * 10):
            x = np.cos(np.linspace(0, 2 * np.pi, r)) + (r - 2) * np.cos(theta * d)
            y = np.sin(np.linspace(0, 2 * np.pi, r)) + (r - 2) * np.sin(theta)
            plt.plot(y, x, color=cmap(theta / (2 * np.pi)), alpha=0.8, lw=4)
    plt.xlim(-12, 12)
    plt.ylim(-12, 12)
    plt.savefig(path + f'14122021/plot{i}.png', facecolor='black')
    plt.close()
