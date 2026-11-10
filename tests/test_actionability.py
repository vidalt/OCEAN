"""Behavioral parity of query-specific actionability across CP and MIP."""

from collections.abc import Iterator
from typing import Literal, cast

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from ocean import ConstraintProgrammingExplainer, MixedIntegerProgramExplainer
from ocean.abc import Mapper
from ocean.cp import Explainer as CPExplainer
from ocean.feature import Feature
from ocean.mip import Explainer as MIPExplainer
from ocean.typing import Actionability, ActionabilityRule, Array1D

from .utils import ENV

type Explainer = CPExplainer | MIPExplainer


@pytest.fixture(params=["cp", "mip"])
def backend(request: pytest.FixtureRequest) -> str:
    return str(request.param)


def make_explainer(
    backend: str, data: pd.DataFrame, mapper: Mapper[Feature]
) -> Explainer:
    clf = RandomForestClassifier(
        n_estimators=1, max_depth=1, random_state=0, bootstrap=False
    )
    clf.fit(data, (data.iloc[:, 0] > data.iloc[:, 0].median()).astype(int))
    if backend == "cp":
        return ConstraintProgrammingExplainer(clf, mapper=mapper)
    return MixedIntegerProgramExplainer(clf, mapper=mapper, env=ENV)


@pytest.fixture(params=["continuous", "discrete", "binary"])
def explainer(
    backend: str, request: pytest.FixtureRequest
) -> Iterator[Explainer]:
    kind = request.param
    if kind == "binary":
        feature = Feature(Feature.Type.BINARY)
        values = [0.0, 1.0]
    else:
        feature = Feature(
            Feature.Type.CONTINUOUS
            if kind == "continuous"
            else Feature.Type.DISCRETE,
            levels=[0.0, 6.0] if kind == "continuous" else [0.0, 2.0, 4.0, 6.0],
        )
        values = [0.0, 2.0, 4.0, 6.0]
    data = pd.DataFrame({"feature": values})
    model = make_explainer(
        backend, data, Mapper({"feature": feature}, columns=data.columns)
    )
    yield model
    if isinstance(model, MIPExplainer):
        model.dispose()


def query_for(explainer: Explainer, *, high: bool = False) -> Array1D:
    binary = explainer.mapper["feature"].is_binary
    return np.array(
        [1.0 if binary else 5.0] if high else [0.0 if binary else 1.0]
    )


@pytest.mark.parametrize("norm", [0, 1, 2])
@pytest.mark.parametrize("rule", ["fixed", "increase", "decrease"])
@pytest.mark.parametrize("high", [False, True])
def test_unchanged_is_allowed(
    explainer: Explainer, norm: int, rule: ActionabilityRule, *, high: bool
) -> None:
    query = query_for(explainer, high=high)
    result = explainer.explain(
        query, y=int(high), norm=norm, actionability={"feature": rule}
    )
    assert result is not None
    np.testing.assert_array_equal(result.x, query)


@pytest.mark.parametrize("norm", [0, 1, 2])
@pytest.mark.parametrize("direction", ["increase", "decrease"])
def test_direction_and_cleanup(
    explainer: Explainer, norm: int, direction: Literal["increase", "decrease"]
) -> None:
    high = direction == "decrease"
    query = query_for(explainer, high=high)
    target = 0 if high else 1
    result = explainer.explain(
        query, y=target, norm=norm, actionability={"feature": direction}
    )
    assert result is not None
    assert result.x[0] < query[0] if high else result.x[0] > query[0]
    forbidden: ActionabilityRule = "increase" if high else "decrease"
    with pytest.warns(UserWarning, match="[Nn]o feasible"):
        result = explainer.explain(
            query, y=target, norm=norm, actionability={"feature": forbidden}
        )
    assert result is None
    # Both feasible and infeasible queries must release their constraints.
    result = explainer.explain(query, y=target, norm=norm)
    assert result is not None


def test_fixed_prevents_class_change_and_manual_cleanup(
    explainer: Explainer,
) -> None:
    query = query_for(explainer)
    with pytest.warns(UserWarning, match="[Nn]o feasible"):
        result = explainer.explain(
            query,
            y=1,
            norm=1,
            actionability={"feature": "fixed"},
            clean_up=False,
        )
    assert result is None
    with pytest.warns(UserWarning, match="[Nn]o feasible"):
        assert explainer.explain(query, y=1, norm=1, clean_up=False) is None
    explainer.cleanup()
    assert explainer.explain(query, y=1, norm=1) is not None


@pytest.mark.parametrize(
    "rules", [{"missing": "fixed"}, {"feature": "invalid"}]
)
def test_invalid_rules_do_not_pollute_model(
    explainer: Explainer, rules: dict[str, str]
) -> None:
    query = query_for(explainer)
    with pytest.raises(ValueError, match="actionability"):
        explainer.explain(
            query, y=1, norm=1, actionability=cast("Actionability", rules)
        )
    assert explainer.explain(query, y=1, norm=1) is not None


def test_continuous_split_boundary(backend: str) -> None:
    data = pd.DataFrame({"feature": [0.0, 2.0, 4.0, 6.0]})
    mapper = Mapper(
        {"feature": Feature(Feature.Type.CONTINUOUS, levels=[0, 6])},
        columns=data.columns,
    )
    model = make_explainer(backend, data, mapper)
    try:
        # The fitted stump splits at 3; equality belongs to the left region.
        query = np.array([3.0])
        result = model.explain(
            query, y=0, norm=1, actionability={"feature": "fixed"}
        )
        assert result is not None
        np.testing.assert_array_equal(result.x, query)
        with pytest.warns(UserWarning, match="[Nn]o feasible"):
            assert (
                model.explain(
                    query, y=1, norm=1, actionability={"feature": "decrease"}
                )
                is None
            )
        result = model.explain(
            query, y=1, norm=1, actionability={"feature": "increase"}
        )
        assert result is not None
        assert result.x[0] > query[0]
    finally:
        if isinstance(model, MIPExplainer):
            model.dispose()


def test_one_hot_group(backend: str) -> None:
    columns = pd.MultiIndex.from_tuples([("category", "a"), ("category", "b")])
    data = pd.DataFrame([[0.0, 1.0], [1.0, 0.0]], columns=columns)
    mapper = Mapper(
        {"category": Feature(Feature.Type.ONE_HOT_ENCODED, codes=["a", "b"])},
        columns=columns,
    )
    model = make_explainer(backend, data, mapper)
    query = np.array([0.0, 1.0])
    try:
        with pytest.raises(ValueError, match="only 'fixed'"):
            model.explain(
                query, y=1, norm=1, actionability={"category": "increase"}
            )
        result = model.explain(
            query, y=0, norm=1, actionability={"category": "fixed"}
        )
        assert result is not None
        np.testing.assert_array_equal(result.x, query)
        with pytest.warns(UserWarning, match="[Nn]o feasible"):
            assert (
                model.explain(
                    query, y=1, norm=1, actionability={"category": "fixed"}
                )
                is None
            )
        assert model.explain(query, y=1, norm=1) is not None
    finally:
        if isinstance(model, MIPExplainer):
            model.dispose()


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1.0, 7.0, 0.5])
def test_invalid_constrained_values(explainer: Explainer, value: float) -> None:
    if value == 0.5 and explainer.mapper["feature"].is_continuous:
        return
    with pytest.raises(ValueError, match=r"requires|Expected|outside"):
        explainer.explain(
            np.array([value]), y=1, norm=1, actionability={"feature": "fixed"}
        )
    assert explainer.explain(query_for(explainer), y=1, norm=1) is not None


def test_empty_rules_match_default(explainer: Explainer) -> None:
    query = query_for(explainer)
    result = explainer.explain(query, y=1, norm=1)
    assert result is not None
    expected = result.x.copy()
    result = explainer.explain(query, y=1, norm=1, actionability={})
    assert result is not None
    np.testing.assert_array_equal(result.x, expected)


def test_fixed_feature_allows_alternative_feature(backend: str) -> None:
    # Reversed mapping order and integer keys exercise name-to-column lookup.
    columns = pd.Index([10, 20])
    data = pd.DataFrame(
        [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], columns=columns
    )
    mapper = Mapper(
        {20: Feature(Feature.Type.BINARY), 10: Feature(Feature.Type.BINARY)},
        columns=columns,
    )
    clf = RandomForestClassifier(
        n_estimators=1, max_depth=2, bootstrap=False, random_state=0
    )
    clf.fit(data, [0, 1, 1, 1])
    model = (
        ConstraintProgrammingExplainer(clf, mapper=mapper)
        if backend == "cp"
        else MixedIntegerProgramExplainer(clf, mapper=mapper, env=ENV)
    )
    try:
        result = model.explain(
            np.array([0.0, 0.0]), y=1, norm=1, actionability={10: "fixed"}
        )
        assert result is not None
        np.testing.assert_array_equal(result.x, [0.0, 1.0])
        assert clf.predict(pd.DataFrame([result.x], columns=columns))[0] == 1
    finally:
        if isinstance(model, MIPExplainer):
            model.dispose()
