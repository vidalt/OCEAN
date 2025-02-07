import numpy as np
import pytest

from ocean.feature import Feature
from ocean.mip import FeatureVar
from ocean.mip.base import BaseModel

from ....feature.utils import BOUNDS, CHOICES, N_CODES, N_LEVELS, SEEDS
from ....utils import ENV


@pytest.mark.parametrize("seed", SEEDS)
def test_binary(seed: int) -> None:
    generator = np.random.default_rng(seed)
    model = BaseModel(env=ENV)
    feature = Feature(Feature.Type.BINARY)
    var = FeatureVar(feature=feature, name="x")
    var.build(model)

    val = generator.uniform(0.0, 0.4)
    objective = (var.x - val) ** 2
    model.setObjective(objective)
    model.optimize()
    assert var.X == 0.0

    val = generator.uniform(0.6, 1.0)
    objective = (var.x - val) ** 2
    model.setObjective(objective)
    model.optimize()
    assert var.X == 1.0


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("n_levels", N_LEVELS)
@pytest.mark.parametrize(("lower", "upper"), BOUNDS)
def test_discrete(
    seed: int,
    n_levels: int,
    lower: int,
    upper: int,
) -> None:
    generator = np.random.default_rng(seed)
    levels = generator.uniform(lower, upper, n_levels)
    model = BaseModel(env=ENV)
    feature = Feature(Feature.Type.DISCRETE, levels=levels)
    var = FeatureVar(feature=feature, name="x")
    var.build(model)

    val = generator.choice(levels)
    objective = (var.x - val) ** 2
    model.setObjective(objective)
    model.optimize()
    assert np.isclose(var.X, val)

    val = generator.uniform(float(levels.min()), float(levels.max()))
    objective = (var.x - val) ** 2
    model.setObjective(objective)
    model.optimize()
    assert np.isclose(var.X, levels[np.abs(levels - val).argmin()])


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("n_levels", N_LEVELS)
@pytest.mark.parametrize(("lower", "upper"), BOUNDS)
def test_continuous(seed: int, n_levels: int, lower: int, upper: int) -> None:
    generator = np.random.default_rng(seed)
    levels = generator.uniform(lower, upper, n_levels)
    model = BaseModel(env=ENV)
    feature = Feature(Feature.Type.CONTINUOUS, levels=levels)
    var = FeatureVar(feature=feature, name="x")
    var.build(model)

    val = generator.uniform(float(levels.min()), float(levels.max()))
    objective = (var.x - val) ** 2
    model.setObjective(objective)
    model.optimize()
    assert np.isclose(var.X, val)

    val = generator.uniform(
        float(levels.min()) - 1.0, float(levels.min()) - 0.5
    )
    objective = (var.x - val) ** 2
    model.setObjective(objective)
    model.optimize()
    assert np.isclose(var.X, levels.min())

    val = generator.uniform(
        float(levels.max()) + 0.5, float(levels.max()) + 1.0
    )
    objective = (var.x - val) ** 2
    model.setObjective(objective)
    model.optimize()
    assert np.isclose(var.X, levels.max())


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("n_codes", N_CODES)
def test_one_hot_encoded(seed: int, n_codes: int) -> None:
    generator = np.random.default_rng(seed)
    codes = generator.choice(CHOICES, n_codes)
    model = BaseModel(env=ENV)
    feature = Feature(ftype=Feature.Type.ONE_HOT_ENCODED, codes=codes)
    var = FeatureVar(feature=feature, name="x")
    var.build(model)

    code = generator.choice(codes)
    val = generator.uniform(0.0, 0.4)
    objective = (var[code] - val) ** 2
    model.setObjective(objective)
    model.optimize()
    assert var[code].X == 0.0

    assert sum(1 for code in var.codes if var[code].X == 1.0) == 1

    val = generator.uniform(0.6, 1.0)
    objective = (var[code] - val) ** 2
    model.setObjective(objective)
    model.optimize()
    assert var[code].X == 1.0

    assert sum(1 for code in var.codes if var[code].X == 1.0) == 1
