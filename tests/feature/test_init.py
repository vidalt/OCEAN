import numpy as np
import pytest

from ocean.feature import Feature


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


def test_continuous() -> None:
    levels = (0, 1)
    feature = Feature(Feature.Type.CONTINUOUS, levels=levels)
    assert feature.is_continuous
    assert not feature.is_binary
    assert not feature.is_discrete
    assert feature.is_numeric
    assert not feature.is_one_hot_encoded
    assert (feature.levels == np.array([0, 1])).all()
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
    feature.add(0.0, 2.0)
    assert (feature.levels == np.array([0.0, 2.0])).all()

    msg = r"Levels cannot contain NaN values."
    with pytest.raises(ValueError, match=msg):
        feature.add(np.nan)


def test_discrete() -> None:
    levels = (2, 1, 3)
    feature = Feature(Feature.Type.DISCRETE, levels=levels)
    assert feature.is_discrete
    assert not feature.is_binary
    assert not feature.is_continuous
    assert feature.is_numeric
    assert not feature.is_one_hot_encoded
    assert (feature.levels == np.sort(levels)).all()
    msg = r"Codes can only be accessed for one-hot encoded features."
    with pytest.raises(AttributeError, match=msg):
        _ = feature.codes

    msg = r"Levels can only be added to continuous features."
    with pytest.raises(AttributeError, match=msg):
        feature.add(0)


def test_one_hot_encoded() -> None:
    codes = ("a", "b", "c")
    feature = Feature(Feature.Type.ONE_HOT_ENCODED, codes=codes)
    assert feature.is_one_hot_encoded
    assert not feature.is_binary
    assert not feature.is_continuous
    assert not feature.is_discrete
    assert not feature.is_numeric
    assert set(feature.codes) == set(codes)
    assert len(feature.codes) == len(codes)

    feature = Feature(ftype=Feature.Type.ONE_HOT_ENCODED)
    assert feature.is_one_hot_encoded

    msg = r"Codes have not been defined for this feature."
    with pytest.raises(AttributeError, match=msg):
        _ = feature.codes

    msg = r"Levels can only be accessed for numeric features."
    with pytest.raises(AttributeError, match=msg):
        _ = feature.levels
