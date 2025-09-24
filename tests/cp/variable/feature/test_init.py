import numpy as np
import pytest

from ocean.cp import FeatureVar
from ocean.feature import Feature

from ....feature.utils import BOUNDS, CHOICES, N_CODES, N_LEVELS, SEEDS


def test_binary() -> None:
    feature = Feature(Feature.Type.BINARY)
    var = FeatureVar(feature=feature, name="x")
    assert var.is_binary
    assert not var.is_continuous
    assert not var.is_discrete
    assert not var.is_numeric
    assert not var.is_one_hot_encoded
    msg = r"Levels can only be accessed for numeric features."
    with pytest.raises(AttributeError, match=msg):
        _ = var.levels

    msg = r"Codes can only be accessed for one-hot encoded features."
    with pytest.raises(AttributeError, match=msg):
        _ = var.codes


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("n_levels", N_LEVELS)
@pytest.mark.parametrize(("lower", "upper"), BOUNDS)
def test_continuous(seed: int, n_levels: int, lower: int, upper: int) -> None:
    generator = np.random.default_rng(seed)
    levels = generator.uniform(lower, upper, n_levels)
    expected_levels = np.sort(np.unique(levels))
    feature = Feature(Feature.Type.CONTINUOUS, levels=levels)
    var = FeatureVar(feature=feature, name="x")
    assert var.is_continuous
    assert not var.is_binary
    assert not var.is_discrete
    assert var.is_numeric
    assert not var.is_one_hot_encoded
    assert (var.levels == expected_levels).all()
    msg = r"Codes can only be accessed for one-hot encoded features."
    with pytest.raises(AttributeError, match=msg):
        _ = var.codes

    feature = Feature(ftype=Feature.Type.CONTINUOUS)
    var = FeatureVar(feature=feature, name="x")
    assert var.is_continuous
    msg = r"Levels have not been defined for this feature."
    with pytest.raises(AttributeError, match=msg):
        _ = var.levels

    feature = Feature(ftype=Feature.Type.CONTINUOUS)
    var = FeatureVar(feature=feature, name="x")
    assert var.is_continuous
    additional_levels = generator.uniform(lower, upper, 2)
    feature.add(*additional_levels)
    assert (var.levels == np.sort(np.unique(additional_levels))).all()


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("n_levels", N_LEVELS)
@pytest.mark.parametrize(("lower", "upper"), BOUNDS)
def test_discrete(seed: int, n_levels: int, lower: int, upper: int) -> None:
    generator = np.random.default_rng(seed)
    levels = generator.uniform(lower, upper, n_levels)
    expected_levels = np.sort(np.unique(levels))
    feature = Feature(Feature.Type.DISCRETE, levels=levels)
    var = FeatureVar(feature=feature, name="x")
    assert var.is_discrete
    assert not var.is_binary
    assert not var.is_continuous
    assert var.is_numeric
    assert not var.is_one_hot_encoded
    assert (var.levels == expected_levels).all()
    msg = r"Codes can only be accessed for one-hot encoded features."
    with pytest.raises(AttributeError, match=msg):
        _ = var.codes


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("n_codes", N_CODES)
def test_one_hot_encoded(seed: int, n_codes: int) -> None:
    generator = np.random.default_rng(seed)
    codes = generator.choice(CHOICES, n_codes, replace=False)
    feature = Feature(Feature.Type.ONE_HOT_ENCODED, codes=codes)
    var = FeatureVar(feature=feature, name="x")
    assert var.is_one_hot_encoded
    assert not var.is_binary
    assert not var.is_continuous
    assert not var.is_discrete
    assert not var.is_numeric
    assert set(var.codes) == set(codes)
    assert len(var.codes) == len(set(codes))

    feature = Feature(ftype=Feature.Type.ONE_HOT_ENCODED)
    var = FeatureVar(feature=feature, name="x")
    assert var.is_one_hot_encoded

    msg = r"Codes have not been defined for this feature."
    with pytest.raises(AttributeError, match=msg):
        _ = var.codes

    msg = r"Levels can only be accessed for numeric features."
    with pytest.raises(AttributeError, match=msg):
        _ = var.levels
