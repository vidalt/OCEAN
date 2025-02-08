import numpy as np
import pytest

from ocean.feature import Feature
from ocean.mip import BaseModel, FeatureVar

from ....feature.utils import BOUNDS, CHOICES, N_CODES, N_LEVELS, SEEDS
from ....utils import ENV


@pytest.mark.parametrize("seed", SEEDS)
def test_binary(seed: int) -> None:
    generator = np.random.default_rng(seed)
    model = BaseModel(env=ENV)
    feature = Feature(Feature.Type.BINARY)
    var = FeatureVar(feature=feature, name="x")
    var.build(model)

    v = var.xget()
    val = generator.uniform(0.0, 0.4)
    objective = (v - val) ** 2
    model.setObjective(objective)
    model.optimize()
    assert v.X == 0.0

    val = generator.uniform(0.6, 1.0)
    objective = (v - val) ** 2
    model.setObjective(objective)
    model.optimize()
    assert v.X == 1.0


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

    v = var.xget()
    val = generator.choice(levels)
    objective = (v - val) ** 2
    model.setObjective(objective)
    model.optimize()
    assert np.isclose(v.X, val)

    val = generator.uniform(float(levels.min()), float(levels.max()))
    objective = (v - val) ** 2
    model.setObjective(objective)
    model.optimize()
    assert np.isclose(v.X, levels[np.abs(levels - val).argmin()])


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

    v = var.xget()
    lb, ub = np.min(levels), np.max(levels)

    val = generator.uniform(lb, ub)
    objective = (v - val) ** 2
    model.setObjective(objective)
    model.optimize()
    assert np.isclose(v.X, val)

    val = generator.uniform(lb - 1.0, lb - 0.5)
    objective = (v - val) ** 2
    model.setObjective(objective)
    model.optimize()
    assert np.isclose(v.X, lb)

    val = generator.uniform(ub + 0.5, ub + 1.0)
    objective = (v - val) ** 2
    model.setObjective(objective)
    model.optimize()
    assert np.isclose(v.X, ub)


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
    v = var.xget(code)
    val = generator.uniform(0.0, 0.4)
    objective = (v - val) ** 2
    model.setObjective(objective)
    model.optimize()
    assert v.X == 0.0

    assert sum(1 for code in var.codes if var.xget(code).X == 1.0) == 1.0

    val = generator.uniform(0.6, 1.0)
    objective = (v - val) ** 2
    model.setObjective(objective)
    model.optimize()
    assert v.X == 1.0

    assert sum(1 for code in var.codes if var.xget(code).X == 1.0) == 1.0
