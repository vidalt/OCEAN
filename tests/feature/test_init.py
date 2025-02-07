import numpy as np
import pytest

from ocean.feature import Feature

from .utils import BOUNDS, CHOICES, N_CODES, N_LEVELS, SEEDS


def test_binary() -> None:
    feature = Feature(Feature.Type.BINARY)
    assert feature.is_binary
    assert not feature.is_continuous
    assert not feature.is_discrete
    assert not feature.is_numeric
    assert not feature.is_one_hot_encoded
    msg = r"Levels can only be accessed for numeric features."
    with pytest.raises(AttributeError, match=msg):
        _ = feature.levels

    msg = r"Codes can only be accessed for one-hot encoded features."
    with pytest.raises(AttributeError, match=msg):
        _ = feature.codes


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("n_levels", N_LEVELS)
@pytest.mark.parametrize(("lower", "upper"), BOUNDS)
def test_continuous(seed: int, n_levels: int, lower: int, upper: int) -> None:
    generator = np.random.default_rng(seed)
    levels = generator.uniform(lower, upper, n_levels)
    expected_levels = np.sort(np.unique(levels))
    feature = Feature(Feature.Type.CONTINUOUS, levels=levels)
    assert feature.is_continuous
    assert not feature.is_binary
    assert not feature.is_discrete
    assert feature.is_numeric
    assert not feature.is_one_hot_encoded
    assert (feature.levels == expected_levels).all()
    msg = r"Codes can only be accessed for one-hot encoded features."
    with pytest.raises(AttributeError, match=msg):
        _ = feature.codes

    feature = Feature(ftype=Feature.Type.CONTINUOUS)
    assert feature.is_continuous
    msg = r"Levels have not been defined for this feature."
    with pytest.raises(AttributeError, match=msg):
        _ = feature.levels

    feature = Feature(ftype=Feature.Type.CONTINUOUS)
    assert feature.is_continuous
    additional_levels = generator.uniform(lower, upper, 2)
    feature.add(*additional_levels)
    assert (feature.levels == np.sort(np.unique(additional_levels))).all()

    msg = r"Levels cannot contain NaN values."
    with pytest.raises(ValueError, match=msg):
        feature.add(np.nan)


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("n_levels", N_LEVELS)
@pytest.mark.parametrize(("lower", "upper"), BOUNDS)
def test_discrete(seed: int, n_levels: int, lower: int, upper: int) -> None:
    generator = np.random.default_rng(seed)
    levels = generator.uniform(lower, upper, n_levels)
    expected_levels = np.sort(np.unique(levels))
    feature = Feature(Feature.Type.DISCRETE, levels=levels)
    assert feature.is_discrete
    assert not feature.is_binary
    assert not feature.is_continuous
    assert feature.is_numeric
    assert not feature.is_one_hot_encoded
    assert (feature.levels == expected_levels).all()
    msg = r"Codes can only be accessed for one-hot encoded features."
    with pytest.raises(AttributeError, match=msg):
        _ = feature.codes

    msg = r"Levels can only be added to continuous features."
    with pytest.raises(AttributeError, match=msg):
        feature.add(0)


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("n_codes", N_CODES)
def test_one_hot_encoded(seed: int, n_codes: int) -> None:
    generator = np.random.default_rng(seed)
    codes = generator.choice(CHOICES, n_codes)
    expected_codes = set(codes)
    feature = Feature(Feature.Type.ONE_HOT_ENCODED, codes=codes)
    assert feature.is_one_hot_encoded
    assert not feature.is_binary
    assert not feature.is_continuous
    assert not feature.is_discrete
    assert not feature.is_numeric
    assert set(feature.codes) == expected_codes
    assert len(feature.codes) == len(expected_codes)

    feature = Feature(ftype=Feature.Type.ONE_HOT_ENCODED)
    assert feature.is_one_hot_encoded

    msg = r"Codes have not been defined for this feature."
    with pytest.raises(AttributeError, match=msg):
        _ = feature.codes

    msg = r"Levels can only be accessed for numeric features."
    with pytest.raises(AttributeError, match=msg):
        _ = feature.levels
