from math import cos, sin, pi

import matplotlib.pylab as plt
import numpy as np

from common.image_processing import ImageProcessingSettings


def generate(settings: ImageProcessingSettings = None):
    settings = settings or ImageProcessingSettings(1)
    for _ in range(4, 8):
        n = 6
        plt.figure(figsize=(12, 12), facecolor='black', dpi=500)
        plt.axis('off')
        plt.xlim(-2, 2)
        plt.ylim(-2, 2)
        plt.plot(
            [cos(x) for x in np.linspace(0, 2 * pi, n)],
            [sin(x) for x in np.linspace(0, 2 * pi, n)],
            lw=3,
            alpha=1,
        )

        for y in np.linspace(0, 2 * pi, n):
            plt.plot(
                [(cos(x)) * 0.5 + cos(y) for x in np.linspace(0, 2 * pi, n)],
                [(sin(x)) * 0.5 + sin(y) for x in np.linspace(0, 2 * pi, n)],
                lw=2,
                alpha=1,
            )
            for z in np.linspace(0, 2 * pi, n):
                plt.plot(
                    [
                        (cos(x)) * 0.25 + cos(y) * 0.5 + cos(z)
                        for x in np.linspace(0, 2 * pi, n)
                    ],
                    [
                        (sin(x)) * 0.25 + sin(y) * 0.5 + sin(z)
                        for x in np.linspace(0, 2 * pi, n)
                    ],
                    lw=1.8,
                    alpha=1,
                )
                for w in np.linspace(0, 2 * pi, n):
                    plt.plot(
                        [
                            (cos(x)) * 0.125 + cos(y) * 0.5 + cos(z) * 0.25 + cos(w)
                            for x in np.linspace(0, 2 * pi, n)
                        ],
                        [
                            (sin(x)) * 0.125 + sin(y) * 0.5 + sin(z) * 0.25 + sin(w)
                            for x in np.linspace(0, 2 * pi, n)
                        ],
                        lw=1,
                        alpha=1,
                    )
                    for u in np.linspace(0, 2 * pi, n):
                        plt.plot(
                            [
                                (cos(x)) * 0.0625
                                + cos(y) * 0.5
                                + cos(z) * 0.25
                                + cos(w) * 0.125
                                + cos(u)
                                for x in np.linspace(0, 2 * pi, n)
                            ],
                            [
                                (sin(x)) * 0.0625
                                + sin(y) * 0.5
                                + sin(z) * 0.25
                                + sin(w) * 0.125
                                + sin(u)
                                for x in np.linspace(0, 2 * pi, n)
                            ],
                            lw=1,
                            alpha=1,
                        )
                        if n < 6:
                            for r in np.linspace(0, 2 * pi, n):
                                plt.plot(
                                    [
                                        (cos(x)) * 0.0625 / 2
                                        + cos(y) * 0.5
                                        + cos(z) * 0.25
                                        + cos(w) * 0.125
                                        + cos(u) * 0.0625
                                        + cos(r)
                                        for x in np.linspace(0, 2 * pi, n)
                                    ],
                                    [
                                        (sin(x)) * 0.0625 / 2
                                        + sin(y) * 0.5
                                        + sin(z) * 0.25
                                        + sin(w) * 0.125
                                        + sin(u) * 0.0625
                                        + sin(r)
                                        for x in np.linspace(0, 2 * pi, n)
                                    ],
                                    lw=1,
                                    alpha=1,
                                )
        settings.save_frame('black')


if __name__ == '__main__':
    generate()
