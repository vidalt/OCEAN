import numpy as np
import pytest

from ocean.feature import Feature


def test_binary() -> None:
    feature = Feature(ftype=Feature.Type.BINARY)
    assert feature.is_binary
    assert not feature.is_continuous
    assert not feature.is_discrete
    assert not feature.is_numeric
    assert not feature.is_one_hot_encoded
    with pytest.raises(
        AttributeError,
        match=r"Levels can only be accessed for numeric features.",
    ):
        feature.levels  # noqa: B018
    with pytest.raises(
        AttributeError,
        match=r"Codes can only be accessed for one-hot encoded features.",
    ):
        feature.codes  # noqa: B018


def test_continuous() -> None:
    feature = Feature(ftype=Feature.Type.CONTINUOUS, levels=(0, 1))
    assert feature.is_continuous
    assert not feature.is_binary
    assert not feature.is_discrete
    assert feature.is_numeric
    assert not feature.is_one_hot_encoded
    assert (feature.levels == np.array([0, 1])).all()
    with pytest.raises(
        AttributeError,
        match=r"Codes can only be accessed for one-hot encoded features.",
    ):
        feature.codes  # noqa: B018

    feature = Feature(ftype=Feature.Type.CONTINUOUS)
    assert feature.is_continuous
    with pytest.raises(
        AttributeError,
        match=r"Levels have not been defined for this feature.",
    ):
        feature.levels  # noqa: B018

    feature = Feature(ftype=Feature.Type.CONTINUOUS)
    assert feature.is_continuous
    feature.add(0.0, 2.0)
    assert (feature.levels == np.array([0.0, 2.0])).all()

    with pytest.raises(
        ValueError,
        match=r"Levels cannot contain NaN values.",
    ):
        feature.add(np.nan)


def test_discrete() -> None:
    feature = Feature(ftype=Feature.Type.DISCRETE, levels=(1, 2, 3))
    assert feature.is_discrete
    assert not feature.is_binary
    assert not feature.is_continuous
    assert feature.is_numeric
    assert not feature.is_one_hot_encoded
    with pytest.raises(
        AttributeError,
        match=r"Codes can only be accessed for one-hot encoded features.",
    ):
        feature.codes  # noqa: B018

    with pytest.raises(
        AttributeError,
        match=r"Levels can only be added to continuous features.",
    ):
        feature.add(0)


def test_one_hot_encoded() -> None:
    feature = Feature(
        ftype=Feature.Type.ONE_HOT_ENCODED,
        codes=("a", "b"),
    )
    assert feature.is_one_hot_encoded
    assert not feature.is_binary
    assert not feature.is_continuous
    assert not feature.is_discrete
    assert not feature.is_numeric
    assert set(feature.codes) == {"a", "b"}
    assert len(feature.codes) == len({"a", "b"})

    feature = Feature(ftype=Feature.Type.ONE_HOT_ENCODED)
    assert feature.is_one_hot_encoded

    with pytest.raises(
        AttributeError,
        match=r"Codes have not been defined for this feature.",
    ):
        feature.codes  # noqa: B018
