from math import cos, sin, pi, sqrt

import matplotlib.pylab as plt
import numpy as np

ran1 = np.random.beta(1.5, 1, 1000) * 40 * pi
p = plt.figure(figsize=(13, 13), facecolor='black')
p = plt.axis('off')
lin1x, lin1y = [cos(t) * t for t in np.linspace(0, 40 * pi, 1100)], [
    sin(t) * t - 10 * sqrt((cos(t) * t) ** 2 + (sin(t) * t) ** 2)
    for t in np.linspace(0, 40 * pi, 1100)
]
lin2x, lin2y = [cos(t + 1) * t for t in np.linspace(0, 40 * pi, 1100)], [
    sin(t + 1) * t - 10 * sqrt((cos(t + 1) * t) ** 2 + (sin(t + 1) * t) ** 2)
    for t in np.linspace(0, 40 * pi, 1100)
]

for i in range(100):
    plt.plot(
        lin1x[i * 11 : (i + 1) * 11 + 1],
        lin1y[i * 11 : (i + 1) * 11 + 1],
        lw=6,
        alpha=1,
        color='#2f7a37',
    )
    plt.plot(
        lin2x[i * 11 : (i + 1) * 11 + 1],
        lin2y[i * 11 : (i + 1) * 11 + 1],
        lw=6,
        alpha=1,
        color='#3ba347',
    )
    plt.savefig(
        f'C:/Users/Alejandro/Pictures/RandomPlots//24122019/plot{i}.png',
        facecolor='black',
    )
m = 100
for z in np.linspace(0, 1, 30):
    p = plt.figure(figsize=(13, 13), facecolor='black')
    p = plt.axis('off')
    plt.plot(lin1x, lin1y, lw=6, alpha=1, color='#2f7a37')
    plt.plot(lin2x, lin2y, lw=6, alpha=1, color='#3ba347')
    plt.scatter([0], [0], s=9200 * z, marker="*", color='#f0ff00', zorder=1000)
    plt.scatter([0], [0], s=4000 * z, marker="*", color='#f6ff61', zorder=1001)
    plt.scatter([0], [0], s=2000 * z, marker="*", color='#fbffb3', zorder=1002)
    plt.savefig(
        f'C:/Users/Alejandro/Pictures/RandomPlots//24122019/plot{m}.png',
        facecolor='black',
    )
    m += 1
for i in range(100, 130):
    p = plt.figure(figsize=(13, 13), facecolor='black')
    p = plt.axis('off')
    plt.plot(lin1x, lin1y, lw=6, alpha=1, color='#2f7a37')
    plt.plot(lin2x, lin2y, lw=6, alpha=1, color='#3ba347')
    plt.scatter([0], [0], s=9200, marker="*", color='#f0ff00', zorder=1000)
    plt.scatter([0], [0], s=4000, marker="*", color='#f6ff61', zorder=1001)
    plt.scatter([0], [0], s=2000, marker="*", color='#fbffb3', zorder=1002)
    ran1 = np.random.beta(1.5, 1, 1000) * 40 * pi
    plt.scatter(
        [cos(t) * t for t in ran1],
        [sin(t) * t - 10 * sqrt((cos(t) * t) ** 2 + (sin(t) * t) ** 2) for t in ran1],
        alpha=0.7,
        s=[abs(np.random.normal(70, 10)) for _ in ran1],
        color=[plt.cm.hsv(np.random.uniform(0, 1)) for _ in ran1],
        zorder=20,
    )
    plt.scatter(
        [cos(t) * t + 0.3 for t in ran1],
        [sin(t) * t - 10 * sqrt((cos(t) * t) ** 2 + (sin(t) * t) ** 2) for t in ran1],
        alpha=0.7,
        s=[abs(np.random.normal(10, 2)) for _ in ran1],
        color='white',
        zorder=20,
        marker="*",
    )
    for _ in range(6):
        plt.savefig(
            f'C:/Users/Alejandro/Pictures/RandomPlots//24122019/plot{m}.png',
            facecolor='black',
        )
        m += 1
