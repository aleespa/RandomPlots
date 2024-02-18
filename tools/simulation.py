from typing import Union

import numpy as np

def gaussian_random_walk(
        n_steps: int,
        dimensions: int = 1,
        drift: Union[float, np.array] = 0,
        var: Union[float, np.array] =1):
    return drift + np.random.normal(size=(n_steps, dimensions),
                            loc=0,
                            scale=var).cumsum(axis=0)
