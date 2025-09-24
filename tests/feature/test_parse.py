import pandas as pd
import pytest

from ocean.abc import Mapper
from ocean.feature import Feature, parse_features

DATA = pd.DataFrame({
    "a": [1.0, 2.0, 3.0, 2.0, 1.0],
    "b": ["a", "b", "c", "b", "a"],
    "c": [True, False, True, False, True],
    "d": [0.25, 0.34, -0.12, 0.654, 0.98],
    "e": [0, 1, 0, 1, 0],
    "f": [1, 1, 1, 1, 1],
    "g": [pd.NA, 3, "a", pd.NA, 1],
})
discretes = ("a",)
invalid_discretes = ("h",)
expected_shape = (5, 7)
expected_columns = {
    ("a", ""),
    ("b", "a"),
    ("b", "b"),
    ("b", "c"),
    ("c", ""),
    ("d", ""),
    ("e", ""),
}
expected_features = {"a", "b", "c", "d", "e"}
n_features = 5


def test_parse() -> None:
    data, mapper = parse_features(data=DATA)
    assert isinstance(mapper, Mapper)
    assert isinstance(data, pd.DataFrame)
    assert data.shape == expected_shape
    assert set(mapper.columns) == expected_columns
    names = set(mapper.keys())
    assert names == expected_features
    assert mapper["a"].is_continuous
    assert mapper["b"].is_one_hot_encoded
    assert mapper["c"].is_binary
    assert mapper["d"].is_continuous
    assert mapper["e"].is_binary
    assert len(mapper) == n_features
    assert "a" in mapper
    assert "f" not in mapper
    assert set(mapper.names) == expected_features
    assert set(mapper.codes) == {"a", "b", "c", ""}


def test_parse_valid() -> None:
    data, mapper = parse_features(data=DATA, discretes=discretes)
    assert isinstance(mapper, Mapper)
    assert isinstance(data, pd.DataFrame)
    assert data.shape == (5, 7)
    assert set(mapper.columns) == expected_columns
    names = set(mapper.keys())
    assert names == expected_features
    assert mapper["a"].ftype == Feature.Type.DISCRETE
    assert mapper["b"].ftype == Feature.Type.ONE_HOT_ENCODED
    assert mapper["c"].ftype == Feature.Type.BINARY
    assert mapper["d"].ftype == Feature.Type.CONTINUOUS
    assert mapper["e"].ftype == Feature.Type.BINARY
    assert len(mapper) == n_features
    assert "a" in mapper
    assert "f" not in mapper


def test_parse_invalid() -> None:
    msg = r"Columns not found in the data: \['h'\]"
    with pytest.raises(ValueError, match=msg):
        parse_features(DATA, discretes=invalid_discretes)
