import sys
from typing import TYPE_CHECKING

import gurobipy as gp
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from ocean import (
    ConstraintProgrammingExplainer,
    MaxSATExplainer,
    MixedIntegerProgramExplainer,
)
from ocean.abc import Mapper
from ocean.feature import Feature, parse_features

from .distance_utils import (
    manual_postprocessed_distance,
    manual_weighted_postprocessed_distance,
)
from .utils import ENV, generate_data

if TYPE_CHECKING:
    from ocean.typing import BaseExplanation

type ExplainerT = (
    ConstraintProgrammingExplainer
    | MixedIntegerProgramExplainer
    | MaxSATExplainer
)


def build_explainer(
    explainer_class: type[
        ConstraintProgrammingExplainer | MixedIntegerProgramExplainer
    ],
    clf: RandomForestClassifier | XGBClassifier,
    mapper: Mapper[Feature],
) -> ConstraintProgrammingExplainer | MixedIntegerProgramExplainer:
    if explainer_class is MixedIntegerProgramExplainer:
        return explainer_class(clf, mapper=mapper, env=ENV)
    return explainer_class(clf, mapper=mapper)


@pytest.mark.parametrize(
    "explainer_class",
    [ConstraintProgrammingExplainer, MixedIntegerProgramExplainer],
)
def test_get_distance_requires_explanation(
    explainer_class: type[
        ConstraintProgrammingExplainer | MixedIntegerProgramExplainer
    ],
) -> None:
    data, y, mapper = generate_data(seed=42, n_samples=50, n_classes=2)
    clf = RandomForestClassifier(
        random_state=42,
        n_estimators=3,
        max_depth=2,
    )
    clf.fit(data, y)
    model = build_explainer(explainer_class, clf, mapper)

    with pytest.raises(RuntimeError, match="No explanation has been computed"):
        model.get_distance()


@pytest.mark.parametrize(
    "explainer_class",
    [ConstraintProgrammingExplainer, MixedIntegerProgramExplainer],
)
@pytest.mark.parametrize(
    "classifier_class",
    [RandomForestClassifier, XGBClassifier],
)
@pytest.mark.parametrize("seed", [42, 43])
@pytest.mark.parametrize("norm", [0, 1, 2])
def test_get_distance_matches_manual_postprocessed_distance(
    explainer_class: type[
        ConstraintProgrammingExplainer | MixedIntegerProgramExplainer
    ],
    classifier_class: type[RandomForestClassifier | XGBClassifier],
    seed: int,
    norm: int,
) -> None:
    data, y, mapper = generate_data(seed=seed, n_samples=100, n_classes=2)
    clf = classifier_class(
        random_state=seed,
        n_estimators=5,
        max_depth=3,
    )
    clf.fit(data, y)
    model = build_explainer(explainer_class, clf, mapper)

    x = data.iloc[0, :].to_numpy(dtype=float).flatten()
    prediction = np.asarray(clf.predict([x]), dtype=np.int64)
    target = int(1 - prediction[0])

    try:
        explanation = model.explain(
            x,
            y=target,
            norm=norm,
            random_seed=seed,
            clean_up=False,
        )
    except gp.GurobiError as exc:
        pytest.skip(f"Skipping test due to {exc}")

    assert explanation is not None

    expected = manual_postprocessed_distance(explanation, norm=norm)
    assert model.get_distance() == pytest.approx(expected)

    model.cleanup()
    assert model.get_distance() == pytest.approx(expected)


@pytest.mark.parametrize(
    "explainer_class",
    [ConstraintProgrammingExplainer, MixedIntegerProgramExplainer],
)
@pytest.mark.parametrize(
    "weighted_norms",
    [[2.0], [1.0, 0.5], [0.25, 1.5, 2.0]],
)
def test_get_distance_matches_manual_weighted_norms(
    explainer_class: type[
        ConstraintProgrammingExplainer | MixedIntegerProgramExplainer
    ],
    weighted_norms: list[float],
) -> None:
    seed = 7
    data, y, mapper = generate_data(seed=seed, n_samples=100, n_classes=2)
    clf = RandomForestClassifier(
        random_state=seed,
        n_estimators=5,
        max_depth=3,
    )
    clf.fit(data, y)
    model = build_explainer(explainer_class, clf, mapper)

    x = data.iloc[0, :].to_numpy(dtype=float).flatten()
    prediction = np.asarray(clf.predict([x]), dtype=np.int64)
    target = int(1 - prediction[0])

    try:
        explanation = model.explain(
            x,
            y=target,
            weighted_norms=weighted_norms,
            random_seed=seed,
            clean_up=False,
        )
    except gp.GurobiError as exc:
        pytest.skip(f"Skipping test due to {exc}")

    assert explanation is not None

    expected = manual_weighted_postprocessed_distance(
        explanation,
        weighted_norms=weighted_norms,
    )
    assert model.get_distance() == pytest.approx(expected)

    model.cleanup()
    assert model.get_distance() == pytest.approx(expected)


@pytest.mark.parametrize(
    "explainer_class",
    [ConstraintProgrammingExplainer, MixedIntegerProgramExplainer],
)
@pytest.mark.parametrize(
    "weighted_norms",
    [[], [1.0, 1.0, 1.0, 1.0], [-1.0], [float("inf")], [float("nan")]],
)
def test_explain_rejects_invalid_weighted_norms(
    explainer_class: type[
        ConstraintProgrammingExplainer | MixedIntegerProgramExplainer
    ],
    weighted_norms: list[float],
) -> None:
    seed = 7
    data, y, mapper = generate_data(seed=seed, n_samples=50, n_classes=2)
    clf = RandomForestClassifier(
        random_state=seed,
        n_estimators=3,
        max_depth=2,
    )
    clf.fit(data, y)
    model = build_explainer(explainer_class, clf, mapper)

    x = data.iloc[0, :].to_numpy(dtype=float).flatten()
    prediction = np.asarray(clf.predict([x]), dtype=np.int64)
    target = int(1 - prediction[0])

    with pytest.raises(ValueError, match="weighted_norms"):
        model.explain(x, y=target, weighted_norms=weighted_norms)


@pytest.mark.skipif(
    sys.platform == "win32", reason="tests for non-windows platforms"
)
def test_get_distance_matches_across_backends_on_discrete_data() -> None:
    raw = pd.DataFrame({
        "age_bucket": [0, 0, 1, 1, 2, 2, 3, 3, 1, 2, 0, 3],
        "owns_home": [0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1],
        "job_type": [
            "office",
            "office",
            "manual",
            "manual",
            "service",
            "service",
            "office",
            "manual",
            "service",
            "office",
            "manual",
            "service",
        ],
    })
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1], dtype=np.int64)
    data, mapper = parse_features(
        raw,
        discretes=("age_bucket",),
        encoded=("job_type",),
        scale=False,
    )
    clf = RandomForestClassifier(
        random_state=7,
        n_estimators=5,
        max_depth=3,
    )
    clf.fit(data, y)

    x = data.iloc[0, :].to_numpy(dtype=float).flatten()
    prediction = np.asarray(clf.predict([x]), dtype=np.int64)
    target = int(1 - prediction[0])

    explainers: dict[str, ExplainerT] = {
        "mip": MixedIntegerProgramExplainer(clf, mapper=mapper, env=ENV),
        "cp": ConstraintProgrammingExplainer(clf, mapper=mapper),
        "maxsat": MaxSATExplainer(clf, mapper=mapper),
    }

    try:
        explanations: dict[str, BaseExplanation | None] = {
            name: explainer.explain(x, y=target, norm=1, random_seed=7)
            for name, explainer in explainers.items()
        }
    except gp.GurobiError as exc:
        pytest.skip(f"Skipping test due to {exc}")

    assert all(explanation is not None for explanation in explanations.values())

    distances = {
        name: explainer.get_distance()
        for name, explainer in explainers.items()
    }

    for name, explanation in explanations.items():
        assert explanation is not None
        assert distances[name] == pytest.approx(
            manual_postprocessed_distance(explanation, norm=1)
        )

    assert distances["mip"] == pytest.approx(distances["cp"])
    assert distances["mip"] == pytest.approx(distances["maxsat"])


@pytest.mark.parametrize("norm", [0, 2])
def test_get_distance_matches_across_mip_cp_on_discrete_data(norm: int) -> None:
    raw = pd.DataFrame({
        "age_bucket": [0, 0, 1, 1, 2, 2, 3, 3, 1, 2, 0, 3],
        "owns_home": [0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1],
        "job_type": [
            "office",
            "office",
            "manual",
            "manual",
            "service",
            "service",
            "office",
            "manual",
            "service",
            "office",
            "manual",
            "service",
        ],
    })
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1], dtype=np.int64)
    data, mapper = parse_features(
        raw,
        discretes=("age_bucket",),
        encoded=("job_type",),
        scale=False,
    )
    clf = RandomForestClassifier(
        random_state=7,
        n_estimators=5,
        max_depth=3,
    )
    clf.fit(data, y)

    x = data.iloc[0, :].to_numpy(dtype=float).flatten()
    prediction = np.asarray(clf.predict([x]), dtype=np.int64)
    target = int(1 - prediction[0])

    explainers: dict[
        str,
        ConstraintProgrammingExplainer | MixedIntegerProgramExplainer,
    ] = {
        "mip": MixedIntegerProgramExplainer(clf, mapper=mapper, env=ENV),
        "cp": ConstraintProgrammingExplainer(clf, mapper=mapper),
    }

    try:
        explanations = {
            name: explainer.explain(x, y=target, norm=norm, random_seed=7)
            for name, explainer in explainers.items()
        }
    except gp.GurobiError as exc:
        pytest.skip(f"Skipping test due to {exc}")

    assert all(explanation is not None for explanation in explanations.values())

    distances = {
        name: explainer.get_distance()
        for name, explainer in explainers.items()
    }

    for name, explanation in explanations.items():
        assert explanation is not None
        assert distances[name] == pytest.approx(
            manual_postprocessed_distance(explanation, norm=norm)
        )

    assert distances["mip"] == pytest.approx(distances["cp"])
