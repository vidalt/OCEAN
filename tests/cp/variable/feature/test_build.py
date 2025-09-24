import numpy as np
import pytest
from ortools.sat.python import cp_model as cp

from ocean.cp import BaseModel, FeatureVar
from ocean.feature import Feature

from ....feature.utils import BOUNDS, CHOICES, N_CODES, N_LEVELS, SEEDS


def test_binary() -> None:
    model = BaseModel()
    feature = Feature(Feature.Type.BINARY)
    var = FeatureVar(feature=feature, name="x")
    var.build(model)
    v = var.xget()
    assert isinstance(v, cp.IntVar)
    msg = r"The 'mget' method is only supported for numeric features"
    with pytest.raises(ValueError, match=msg):
        _ = var.mget(0)

    msg = r"Get by code is only supported for one-hot encoded features"
    with pytest.raises(ValueError, match=msg):
        _ = var.xget("a")


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("n_levels", N_LEVELS)
@pytest.mark.parametrize(("lower", "upper"), BOUNDS)
def test_discrete(seed: int, n_levels: int, lower: int, upper: int) -> None:
    generator = np.random.default_rng(seed)
    levels = generator.uniform(lower, upper, n_levels)
    model = BaseModel()
    feature = Feature(Feature.Type.DISCRETE, levels=levels)
    var = FeatureVar(feature=feature, name="x")
    var.build(model)
    v = var.xget()
    assert isinstance(v, cp.IntVar)
    n = len(var.levels)
    for i in range(n - 1):
        mu = var.mget(i)
        assert isinstance(mu, cp.IntVar)

    msg = r"Get by code is only supported for one-hot encoded features"
    with pytest.raises(ValueError, match=msg):
        _ = var.xget("a")


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("n_levels", N_LEVELS)
@pytest.mark.parametrize(("lower", "upper"), BOUNDS)
def test_continuous(seed: int, n_levels: int, lower: int, upper: int) -> None:
    generator = np.random.default_rng(seed)
    levels = generator.uniform(lower, upper, n_levels)
    model = BaseModel()
    feature = Feature(Feature.Type.CONTINUOUS, levels=levels)
    var = FeatureVar(feature=feature, name="x")
    var.build(model)
    v = var.xget()
    assert isinstance(v, cp.IntVar)
    n = len(var.levels)
    for i in range(n - 1):
        mu = var.mget(i)
        assert isinstance(mu, cp.IntVar)

    msg = r"Get by code is only supported for one-hot encoded features"
    with pytest.raises(ValueError, match=msg):
        _ = var.xget("a")


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("n_codes", N_CODES)
def test_one_hot_encoded(seed: int, n_codes: int) -> None:
    generator = np.random.default_rng(seed)
    codes = generator.choice(CHOICES, n_codes)
    model = BaseModel()
    feature = Feature(ftype=Feature.Type.ONE_HOT_ENCODED, codes=codes)
    var = FeatureVar(feature=feature, name="x")
    var.build(model)

    msg = r"The 'mget' method is only supported for numeric features"
    with pytest.raises(ValueError, match=msg):
        _ = var.mget(0)

    msg = r"Code is required for one-hot encoded features get"
    with pytest.raises(ValueError, match=msg):
        _ = var.xget()

    for code in var.codes:
        v = var.xget(code)
        assert isinstance(v, cp.IntVar)

    code = "none"
    msg = r"Code 'none' not found in the feature codes"
    with pytest.raises(ValueError, match=msg):
        _ = var.xget(code)
