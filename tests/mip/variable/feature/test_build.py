import gurobipy as gp
import numpy as np
import pytest

from ocean.feature import Feature
from ocean.mip import FeatureVar
from ocean.mip.base import BaseModel

from ....feature.utils import BOUNDS, CHOICES, N_CODES, N_LEVELS, SEEDS
from ....utils import ENV


def test_binary() -> None:
    model = BaseModel(env=ENV)
    feature = Feature(Feature.Type.BINARY)
    var = FeatureVar(feature=feature, name="x")
    var.build(model)
    model.update()
    assert isinstance(var.x, gp.Var)
    assert var.x.VType == gp.GRB.BINARY
    msg = r"This feature does not support indexing"
    with pytest.raises(ValueError, match=msg):
        var.mget(0)

    msg = r"Indexing is only supported for one-hot encoded features"
    with pytest.raises(ValueError, match=msg):
        _ = var[0]


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("n_levels", N_LEVELS)
@pytest.mark.parametrize(("lower", "upper"), BOUNDS)
def test_discrete(seed: int, n_levels: int, lower: int, upper: int) -> None:
    generator = np.random.default_rng(seed)
    levels = generator.uniform(lower, upper, n_levels)
    model = BaseModel(env=ENV)
    feature = Feature(Feature.Type.DISCRETE, levels=levels)
    var = FeatureVar(feature=feature, name="x")
    var.build(model)
    model.update()
    assert isinstance(var.x, gp.Var)
    assert var.x.VType == gp.GRB.CONTINUOUS
    n = len(var.levels)
    for i in range(n - 1):
        assert isinstance(var.mget(i), gp.Var)
        assert var.mget(i).VType == gp.GRB.BINARY

    msg = r"Indexing is only supported for one-hot encoded features"
    with pytest.raises(ValueError, match=msg):
        _ = var[0]


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
    model.update()
    assert isinstance(var.x, gp.Var)
    assert var.x.VType == gp.GRB.CONTINUOUS
    n = len(var.levels)
    for i in range(n - 1):
        assert isinstance(var.mget(i), gp.Var)
        assert var.mget(i).VType == gp.GRB.CONTINUOUS
        assert var.mget(i).LB == 0.0
        assert var.mget(i).UB == 1.0

    msg = r"Indexing is only supported for one-hot encoded features"
    with pytest.raises(ValueError, match=msg):
        _ = var[0]


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("n_codes", N_CODES)
def test_one_hot_encoded(seed: int, n_codes: int) -> None:
    generator = np.random.default_rng(seed)
    codes = generator.choice(CHOICES, n_codes)
    model = BaseModel(env=ENV)
    feature = Feature(ftype=Feature.Type.ONE_HOT_ENCODED, codes=codes)
    var = FeatureVar(feature=feature, name="x")
    var.build(model)
    model.update()

    msg = "This feature does not support indexing"
    with pytest.raises(ValueError, match=msg):
        var.mget(0)

    msg = "x property is not available for one-hot encoded features"
    with pytest.raises(ValueError, match=msg):
        _ = var.x

    for code in var.codes:
        assert isinstance(var[code], gp.Var)
        assert var[code].VType == gp.GRB.BINARY
