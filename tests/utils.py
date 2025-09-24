import gurobipy as gp
import numpy as np
import pandas as pd

from ocean.abc import Mapper
from ocean.feature import Feature, parse_features


def generate_data(
    seed: int,
    n_samples: int,
    n_classes: int,
) -> tuple[
    pd.DataFrame,
    np.ndarray[tuple[int], np.dtype[np.int64 | np.float64]],
    Mapper[Feature],
]:
    generator = np.random.default_rng(seed)

    discrete_values = {
        "discrete_0": generator.integers(0, 4, n_samples),
        "discrete_1": generator.choice([0.5, 3.5, 8.75, 9.25], n_samples),
    }

    binary_values = {
        "binary_0": generator.integers(0, 2, n_samples),
        "binary_1": generator.integers(0, 2, n_samples),
    }

    continuous_values = {
        "continuous_0": generator.uniform(0, 1, n_samples),
        "continuous_1": generator.uniform(-1, 1, n_samples),
        "continuous_2": generator.uniform(-3, 1, n_samples),
        "continuous_3": generator.uniform(1, 4, n_samples),
        "continuous_4": generator.uniform(-2, -1, n_samples),
    }

    encoded_values = {
        "encoded_0": generator.choice(["a", "b", "c", "d"], n_samples),
        "encoded_1": generator.choice(["0", "a", "1", "b"], n_samples),
    }

    data = pd.DataFrame({
        **discrete_values,
        **binary_values,
        **continuous_values,
        **encoded_values,
    })
    data, mapper = parse_features(data, discretes=tuple(discrete_values.keys()))

    if n_classes == -1:
        return data, generator.uniform(0, 1, n_samples).flatten(), mapper
    return data, generator.integers(0, n_classes, n_samples).flatten(), mapper


ENV = gp.Env(empty=True)
ENV.setParam("OutputFlag", 0)
ENV.start()
