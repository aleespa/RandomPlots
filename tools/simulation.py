from typing import Union

import numpy as np

def gaussian_random_walk(
        n_steps: int,
        dimensions: int = 1,
        drift: Union[float, np.array] = 0,
        var: Union[float, np.array] = 1) -> np.array:
    return drift + np.random.normal(size=(n_steps, dimensions),
                                    loc=0,
                                    scale=var).cumsum(axis=0)

def brownian_bridge(n_steps: int,
                    random_seed: Union[int, None] = None) -> np.array:
    t = np.linspace(0, 1, n_steps)
    ii = np.arange(1, n_steps + 1)
    np.random.seed(random_seed)
    xi = np.random.randn(n_steps) / (ii * np.pi)
    ui = np.sin(np.outer(ii, np.pi * t))
    B = np.sum(xi * ui, axis=1)
    return np.sqrt(2) * B