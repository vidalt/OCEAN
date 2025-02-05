import numpy as np
import pytest

from ocean.feature import Feature
from ocean.mip import FeatureVar
from ocean.mip.base import BaseModel

from ..utils import ENV


def test_binary() -> None:
    model = BaseModel(env=ENV)
    feature = Feature(ftype=Feature.Type.BINARY)
    var = FeatureVar(feature=feature, name="x")
    var.build(model)

    objective = (var.x - 1.0) ** 2
    model.setObjective(objective)
    model.optimize()
    assert var.X == 1.0

    objective = (var.x - 0.0) ** 2
    model.setObjective(objective)
    model.optimize()
    assert var.X == 0.0

    objective = (var.x - 0.3) ** 2
    model.setObjective(objective)
    model.optimize()
    assert var.X == 0.0

    objective = (var.x - 0.7) ** 2
    model.setObjective(objective)
    model.optimize()
    assert var.X == 1.0

    with pytest.raises(
        ValueError, match="This feature does not support indexing"
    ):
        var.mget(0)

    with pytest.raises(
        ValueError,
        match="Indexing is only supported for one-hot encoded features",
    ):
        var["a"]


def test_discrete() -> None:
    model = BaseModel(env=ENV)
    feature = Feature(ftype=Feature.Type.DISCRETE, levels=(2, 1, 3))
    var = FeatureVar(feature=feature, name="test")
    var.build(model)

    objective = (var.x - 1.0) ** 2
    model.setObjective(objective)
    model.optimize()
    assert np.isclose(var.X, 1.0)

    objective = (var.x - 2.0) ** 2
    model.setObjective(objective)
    model.optimize()
    assert np.isclose(var.X, 2.0)

    objective = (var.x - 3.0) ** 2
    model.setObjective(objective)
    model.optimize()
    assert np.isclose(var.X, 3.0)

    objective = (var.x - 0.9) ** 2
    model.setObjective(objective)
    model.optimize()
    assert np.isclose(var.X, 1.0)

    objective = (var.x - 2.7) ** 2
    model.setObjective(objective)
    model.optimize()
    assert np.isclose(var.X, 3.0)

    objective = (var.x - 1.8) ** 2
    model.setObjective(objective)
    model.optimize()
    assert np.isclose(var.X, 2.0)
    assert var.mget(0).X == 1.0
    assert var.mget(1).X == 0.0


def test_continuous() -> None:
    model = BaseModel(env=ENV)
    feature = Feature(
        ftype=Feature.Type.CONTINUOUS,
        levels=(-0.65, 0.35, 0.64, 1.28, 1.92, 2.92),
    )
    var = FeatureVar(feature=feature, name="x")
    var.build(model)

    objective = (var.x + 1.0) ** 2
    model.setObjective(objective)
    model.optimize()
    assert np.isclose(var.X, -0.65)

    objective = (var.x - 0.0) ** 2
    model.setObjective(objective)
    model.optimize()
    assert np.isclose(var.X, 0.0)

    objective = (var.x - 0.5) ** 2
    model.setObjective(objective)
    model.optimize()
    assert np.isclose(var.X, 0.5)

    objective = (var.x - 1.0) ** 2
    model.setObjective(objective)
    model.optimize()
    assert np.isclose(var.X, 1.0)

    objective = (var.x - 1.5) ** 2
    model.setObjective(objective)
    model.optimize()
    assert np.isclose(var.X, 1.5)
    assert var.mget(0).X >= var.mget(1).X >= var.mget(2).X

    objective = (var.x - 2.30) ** 2
    model.setObjective(objective)
    model.optimize()
    assert np.isclose(var.X, 2.30)

    objective = (var.x - 3.5) ** 2
    model.setObjective(objective)
    model.optimize()
    assert np.isclose(var.X, 2.92)


def test_one_hot_encoded() -> None:
    model = BaseModel(env=ENV)
    feature = Feature(ftype=Feature.Type.ONE_HOT_ENCODED, codes=("a", "b", "c"))
    var = FeatureVar(feature=feature, name="x")
    var.build(model)

    objective = (var["a"] - 0.7) ** 2
    model.setObjective(objective)
    model.optimize()
    assert var["a"].X == 1.0
    assert var["b"].X == 0.0
    assert var["c"].X == 0.0

    objective = (var["b"] - 0.7) ** 2
    model.setObjective(objective)
    model.optimize()
    assert var["a"].X == 0.0
    assert var["b"].X == 1.0
    assert var["c"].X == 0.0

    objective = (var["c"] - 0.7) ** 2
    model.setObjective(objective)
    model.optimize()
    assert var["a"].X == 0.0
    assert var["b"].X == 0.0
    assert var["c"].X == 1.0

    with pytest.raises(
        ValueError,
        match="x property is not available for one-hot encoded features",
    ):
        var.x  # noqa: B018
