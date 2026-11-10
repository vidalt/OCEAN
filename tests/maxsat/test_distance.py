import itertools
import sys
from typing import TYPE_CHECKING, cast

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from ocean import MaxSATExplainer
from ocean.feature import parse_features

from ..distance_utils import manual_postprocessed_distance
from ..utils import generate_data

if TYPE_CHECKING:
    from ocean.typing import Array1D

pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="tests for non-windows platforms"
)


def test_get_distance_requires_explanation() -> None:
    data, y, mapper = generate_data(seed=42, n_samples=50, n_classes=2)
    clf = RandomForestClassifier(
        random_state=42,
        n_estimators=3,
        max_depth=2,
    )
    clf.fit(data, y)
    model = MaxSATExplainer(clf, mapper=mapper)

    with pytest.raises(RuntimeError, match="No explanation has been computed"):
        model.get_distance()


@pytest.mark.parametrize("seed", [42, 43, 44])
def test_get_distance_matches_manual_postprocessed_distance(seed: int) -> None:
    data, y, mapper = generate_data(seed=seed, n_samples=100, n_classes=2)
    clf = RandomForestClassifier(
        random_state=seed,
        n_estimators=5,
        max_depth=3,
    )
    clf.fit(data, y)
    model = MaxSATExplainer(clf, mapper=mapper)

    x = data.iloc[0, :].to_numpy(dtype=float).flatten()
    prediction = np.asarray(cast("Array1D", clf.predict([x])), dtype=np.int64)
    target = int(1 - prediction[0])
    explanation = model.explain(
        x,
        y=target,
        norm=1,
        random_seed=seed,
    )

    assert explanation is not None

    expected = manual_postprocessed_distance(explanation, norm=1)
    assert model.get_distance() == pytest.approx(expected)
    assert model.get_distance() == pytest.approx(expected)


@pytest.mark.parametrize("n_categories", [2, 3, 4])
def test_one_hot_change_costs_one_unit(n_categories: int) -> None:
    """A category change costs one unit, including after query cleanup."""
    raw = pd.DataFrame({
        "category": [f"category_{i}" for i in range(n_categories)]
    })
    data, mapper = parse_features(raw, encoded=tuple(raw.columns))
    clf = RandomForestClassifier(
        n_estimators=1, max_depth=3, bootstrap=False, random_state=42
    )
    clf.fit(data, [0, *([1] * (n_categories - 1))])
    model = MaxSATExplainer(clf, mapper=mapper)
    query = data.iloc[0].to_numpy(dtype=float)
    for target in (1, 0):
        explanation = model.explain(query, y=target, norm=1)
        assert explanation is not None
        assert model.get_objective_value() == pytest.approx(1.0)
        assert model.get_distance() == pytest.approx(1.0)
        query = explanation.to_numpy().copy()
        assert (
            int(
                np.asarray(
                    cast(
                        "Array1D",
                        clf.predict(
                            pd.DataFrame([query], columns=data.columns)
                        ),
                    ),
                    dtype=np.int64,
                ).item()
            )
            == target
        )


def test_category_change_beats_larger_numeric_change() -> None:
    """The objective must optimize the right metric, not only report it."""
    raw = pd.DataFrame({
        "amount": [0.0, 3.0, 6.0, 0.0, 3.0, 6.0],
        "category": ["a", "a", "a", "b", "b", "b"],
    })
    data, mapper = parse_features(raw, encoded=("category",), scale=False)
    clf = RandomForestClassifier(
        n_estimators=1, max_depth=2, bootstrap=False, random_state=42
    )
    clf.fit(data, [0, 1, 1, 1, 1, 1])
    model = MaxSATExplainer(clf, mapper=mapper)
    query = data.iloc[0].to_numpy(dtype=float)
    explanation = model.explain(query, y=1, norm=1)
    assert explanation is not None
    result = explanation.to_numpy()
    assert result[mapper.idx.get("amount")] == 0.0
    assert result[mapper.idx.get("category", "b")] == 1.0
    assert model.get_objective_value() == pytest.approx(1.0)
    assert model.get_distance() == pytest.approx(1.0)


def test_three_category_changes_cost_three_units() -> None:

    raw = pd.DataFrame(
        itertools.product(["a", "b"], repeat=3),
        columns=["first", "second", "third"],
    )
    data, mapper = parse_features(raw, encoded=tuple(raw.columns))
    clf = RandomForestClassifier(
        n_estimators=1, max_depth=3, bootstrap=False, random_state=42
    )
    clf.fit(data, (raw.to_numpy() == "b").all(axis=1).astype(int))
    model = MaxSATExplainer(clf, mapper=mapper)
    explanation = model.explain(data.iloc[0].to_numpy(dtype=float), y=1, norm=1)
    assert explanation is not None
    np.testing.assert_array_equal(
        explanation.to_numpy(), data.iloc[-1].to_numpy()
    )
    assert model.get_objective_value() == pytest.approx(3.0)
    assert model.get_distance() == pytest.approx(3.0)
