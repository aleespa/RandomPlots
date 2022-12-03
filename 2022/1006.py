import numpy as np
import matplotlib.pylab as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from numpy import sqrt, pi, cos, sin
from scipy import interpolate
from scipy.stats import norm
from scipy.interpolate import interp1d
import os
import sys
import itertools

filename = os.path.basename(sys.argv[0])[:-3]

def ikeda_map(x0, y0, u):
    tn = 0.4 - 6 / (1 + x0 ** 2 + y0 ** 2)
    x1 = 1 + u * (x0 * cos(tn) - y0 * sin(tn))
    y1 = u * (x0 * sin(tn) + y0 * cos(tn))
    return x1, y1


def plot_1():
    u = 0.918
    m = 20
    n = 380
    n_points = 10000
    n_frames = 300
    initial_anchor = 5100
    ppf = int((n_points + initial_anchor) / n_frames)
    Z_initial = np.concatenate((8 * np.cos(np.linspace(0, 2 * pi, m + 1)).reshape(-1, 1)[:-1] + 2,
                                8 * np.sin(np.linspace(0, 2 * pi, m + 1)).reshape(-1, 1)[:-1] + 2), axis=1).T
    Z = np.zeros((2, m, n))
    Z_smooth = np.zeros((2, m, n_points + initial_anchor))
    Z[:, :, 0] = Z_initial
    Z_smooth[:, :, :initial_anchor] = np.repeat(Z_initial[:, :, np.newaxis], initial_anchor, axis=2)
    for i in range(n - 1):
        Z[0, :, i + 1], Z[1, :, i + 1] = ikeda_map(Z[0, :, i], Z[1, :, i], u)

    for j in range(m):
        tck, un = interpolate.splprep([Z[0, j, :], Z[1, j, :]], s=0)
        t = np.linspace(0, 1, n_points)
        Z_smooth[:, j, initial_anchor:] = interpolate.splev(t, tck)

    for frame in range(233, n_frames):
        fig = plt.figure(figsize=(12, 12), dpi=200)
        ax = fig.add_axes([0, 0, 1, 1], facecolor='black')

        for j in range(m):
            plt.scatter(Z_smooth[0, j, (min(n_frames - 1, frame + 100)) * ppf + 1],
                        Z_smooth[1, j, (min(n_frames - 1, frame + 100)) * ppf + 1], color='r', zorder=2, s=50)

            ax.plot(Z_smooth[0, j, frame * ppf: (min(n_frames - 1, frame + 100)) * ppf],
                    Z_smooth[1, j, frame * ppf: (min(n_frames - 1, frame + 100)) * ppf],
                    color='#EBF5FE', lw=3, alpha=1, zorder=1)
        ax.set_xlim(-8 + 2 - 0.1, 8 + 2.1)
        ax.set_ylim(-8 + 2 - 0.1, 8 + 2.1)
        fig.savefig(f'../outputs/{filename}/plot{frame}.png')
        plt.close()
        plt.cla()
        plt.clf()


def plot_2():
    u = 0.5
    m = 20
    n = 12
    n_points = 20000
    n_frames = 300
    ppf = int(n_points / n_frames)
    fig = plt.figure(figsize=(12, 12), dpi=400)
    ax = fig.add_axes([0, 0, 1, 1], facecolor='black')
    newcmp = LinearSegmentedColormap.from_list("",
                                               ["#ffd47f", "#ffe9f8", "#60c0f0", "#ff9a9a", "#ff0303"])
    for k, r in enumerate(np.linspace(25, 30, 35)):

        # Z_initial = np.concatenate((r * np.cos(np.linspace(0, 2 * pi, m + 1) + r/8).reshape(-1, 1)[:-1] + 2,
        #                             r * np.sin(np.linspace(0, 2 * pi, m + 1) + r/8).reshape(-1, 1)[:-1] + 2), axis=1).T

        Z_initial = np.random.normal(0, r ** 2, (2, m))
        Z = np.zeros((2, m, n))
        Z_smooth = np.zeros((2, m, n_points))
        Z[:, :, 0] = Z_initial

        for i in range(n - 1):
            Z[0, :, i + 1], Z[1, :, i + 1] = ikeda_map(Z[0, :, i], Z[1, :, i], u)

        for j in range(m):
            tck, un = interpolate.splprep([Z[0, j, :], Z[1, j, :]], s=0)
            t = np.linspace(0, 1, n_points)
            Z_smooth[:, j, :] = interpolate.splev(t, tck)
        for j in range(m):
            ax.plot(Z_smooth[0, j, :], Z_smooth[1, j, :],
                    color=newcmp(k / 35), lw=1.2, alpha=0.9, zorder=int(r * 10))
        ax.set_xlim(-6 + 1, 6 + 1)
        ax.set_ylim(-6, 6)
    fig.savefig(f'../outputs/{filename}_v2.png')
    plt.close()
    plt.cla()
    plt.clf()


def plot_3():
    u = 0.7
    m = 25 * 4
    n = 13
    n_points = 1000
    n_frames = 300
    ppf = int(n_points / n_frames)
    fig = plt.figure(figsize=(9, 16), dpi=400)
    ax = fig.add_axes([0.1, 3.5/16 - 0.1, 1 - 0.2, 9/16 - 0.2], facecolor='black')
    newcmp = LinearSegmentedColormap.from_list("",
                                               ["#ffd47f", "#ffe9f8", "#60c0f0", "#ff9a9a", "#ff0303"])
    for k, r in enumerate(np.linspace(25, 30, 35)):
        m_4 = int(m / 4)
        Z_initial = np.concatenate((np.linspace(-6, 6, m_4).reshape(-1, 1),
                                    np.zeros(m_4).reshape(-1, 1) - 6), axis=1).T
        Z_initial = np.concatenate((Z_initial,
                                    np.concatenate((np.linspace(-6, 6, m_4).reshape(-1, 1),
                                                    np.zeros(m_4).reshape(-1, 1) + 6), axis=1).T
                                    ), axis=1)
        Z_initial = np.concatenate((Z_initial,
                                    np.concatenate((np.zeros(m_4).reshape(-1, 1) - 6,
                                                    np.linspace(-6, 6, m_4).reshape(-1, 1)), axis=1).T
                                    ), axis=1)
        Z_initial = np.concatenate((Z_initial,
                                    np.concatenate((np.zeros(m_4).reshape(-1, 1) + 6,
                                                    np.linspace(-6, 6, m_4).reshape(-1, 1)), axis=1).T
                                    ), axis=1)
        Z = np.zeros((2, m, n))
        Z_smooth = np.zeros((2, m, n_points))
        Z[:, :, 0] = Z_initial

        for i in range(n - 1):
            Z[0, :, i + 1], Z[1, :, i + 1] = ikeda_map(Z[0, :, i], Z[1, :, i], u)

        for j in range(m):
            tck, un = interpolate.splprep([Z[0, j, :], Z[1, j, :]], s=0)
            t = np.linspace(0, 1, n_points)
            Z_smooth[:, j, :] = interpolate.splev(t, tck)
        for j in range(m):
            if j < m_4:
                ax.plot(Z_smooth[0, j, :], Z_smooth[1, j, :],
                        color='#83dedc', lw=2, alpha=0.8, zorder=j % 4)
            elif j < 2 * m_4:
                ax.plot(Z_smooth[0, j, :], Z_smooth[1, j, :],
                        color='#fcc859', lw=2, alpha=0.8, zorder=j % 4)
            elif j < 3 * m_4:
                ax.plot(Z_smooth[0, j, :], Z_smooth[1, j, :],
                        color='#ef6585', lw=2, alpha=0.8, zorder=j % 4)
            else:
                ax.plot(Z_smooth[0, j, :], Z_smooth[1, j, :],
                        color='#9d6adf', lw=2, alpha=0.8, zorder=j % 4)
        ax.plot([-6, -6, 6, 6, -6], [-6, 6, 6, -6, -6], color='w', lw=20, zorder=5)
        ax.set_xlim(-6, 6)
        ax.set_ylim(-6, 6)
    fig.savefig(f'../outputs/{filename}_v3.png', facecolor='k')
    plt.close()
    plt.cla()
    plt.clf()

class Frame:
    def __int__(self):
        pass

    def get_social_life(self):
        pass




if __name__ == '__main__':
    # plot_1()
    plot_3()