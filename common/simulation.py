from typing import Union

import numpy as np
from loguru import logger
from numpy.random import Generator


def get_rng(seed=None):
    seed_seq = (
        np.random.SeedSequence() if seed is None
        else np.random.SeedSequence(seed) if not isinstance(seed, np.random.SeedSequence)
        else seed
    )
    logger.info(f"Seed: {seed_seq.entropy}")
    return np.random.default_rng(seed_seq)

def gaussian_random_walk(
    n_steps: int,
    dimensions: int = 1,
    drift: Union[float, np.array] = 0,
    var: Union[float, np.array] = 1,
) -> np.array:
    normal = np.random.normal(size=(n_steps, dimensions), loc=0, scale=var)
    return drift + normal.cumsum(axis=0)


def brownian_bridge(n_steps: int, rng: Generator) -> np.array:
    t = np.linspace(0, 1, n_steps - 2)
    ii = np.arange(1, n_steps - 1)
    xi = rng.normal(loc=0, scale=1, size=n_steps - 2) / (ii * np.pi)
    ui = np.sin(np.outer(ii, np.pi * t))
    b = np.concatenate([np.zeros(1), np.sum(xi * ui, axis=1), np.zeros(1)])
    return np.sqrt(2) * b
