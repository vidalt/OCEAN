from typing import TYPE_CHECKING, cast

import gurobipy as gp
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from ocean import ConstraintProgrammingExplainer, MixedIntegerProgramExplainer
from ocean.abc import Mapper
from ocean.feature import Feature, parse_features
from ocean.typing import BaseExplanation

from .distance_utils import manual_postprocessed_distance
from .utils import ENV

if TYPE_CHECKING:
    from ocean.cp import Explainer as CPExplainer
    from ocean.mip import Explainer as MIPExplainer

SEED = 42
N_FEATURES = 10
N_SAMPLES = 500
N_QUERY_EXAMPLES = 10
N_ESTIMATORS = 10
MAX_DEPTH = 5
MAX_TIME = 10
DISTANCE_TOL = 1e-4
BOUND_TOL = 1e-8
BOUNDS = (
    (-10.0, -2.0),
    (-6.0, 6.0),
    (-3.0, 0.5),
    (0.0, 1.0),
    (1.5, 8.0),
    (5.0, 25.0),
    (-20.0, -5.0),
    (10.0, 30.0),
    (-1.5, 2.5),
    (50.0, 120.0),
)
REGRESSION_SEED = 0
REGRESSION_QUERY_INDICES = (20, 34)


def build_continuous_dataset(
    *,
    seed: int,
    n_samples: int,
) -> tuple[pd.DataFrame, np.ndarray, Mapper[Feature]]:
    rng = np.random.default_rng(seed)
    raw = pd.DataFrame({
        f"feature_{i}": rng.uniform(low, high, n_samples)
        for i, (low, high) in enumerate(BOUNDS)
    })
    normalized = (raw - raw.mean()) / raw.std()
    score = sum(
        (N_FEATURES - i) * normalized[f"feature_{i}"] for i in range(N_FEATURES)
    )
    y = (np.asarray(score) > np.median(np.asarray(score))).astype(np.int64)
    data, mapper = parse_features(raw, scale=False)
    return data, y, mapper


def select_query_indices(predictions: np.ndarray) -> np.ndarray:
    per_class = N_QUERY_EXAMPLES // 2
    negative = np.flatnonzero(predictions == 0)
    positive = np.flatnonzero(predictions == 1)

    assert negative.size >= per_class
    assert positive.size >= per_class

    return np.sort(np.concatenate([negative[:per_class], positive[:per_class]]))


def predict_processed(
    clf: RandomForestClassifier,
    x: np.ndarray,
    columns: pd.Index,
) -> int:
    frame = pd.DataFrame([x], columns=columns)
    prediction = np.asarray(clf.predict(frame), dtype=np.int64)
    return int(prediction[0])


def next_float32_up(value: float) -> float:
    return float(
        np.nextafter(
            np.float32(value),
            np.float32(np.inf),
            dtype=np.float32,
        )
    )


def next_float32_down_or_equal(value: float) -> float:
    candidate = float(np.float32(value))
    if candidate <= value:
        return candidate
    return float(
        np.nextafter(
            np.float32(value),
            np.float32(-np.inf),
            dtype=np.float32,
        )
    )


def continuous_interval_optimality_gap_bound(
    levels: np.ndarray,
    *,
    num_epsilon: float,
    feasibility_tol: float,
) -> float:
    levels_arr = np.asarray(levels, dtype=np.float64)
    gaps = np.diff(levels_arr)
    n_intervals = gaps.size
    feature_bound = 0.0

    for idx, gap in enumerate(gaps):
        effective_epsilon = max(num_epsilon, 2.0 * feasibility_tol / gap)
        raw_slack = max(
            0.0,
            gap * effective_epsilon - gap * feasibility_tol - feasibility_tol,
        )
        left_ulp = next_float32_up(float(levels_arr[idx])) - float(
            levels_arr[idx]
        )
        right_ulp = float(levels_arr[idx + 1]) - next_float32_down_or_equal(
            float(levels_arr[idx + 1])
        )

        left_penalty = 0.0
        if idx > 0:
            left_penalty = max(0.0, raw_slack - left_ulp)

        right_penalty = 0.0
        if idx < n_intervals - 1:
            right_penalty = max(0.0, raw_slack - right_ulp)

        feature_bound = max(feature_bound, left_penalty, right_penalty)

    return feature_bound


def theoretical_mip_optimality_gap_bound(
    mapper: Mapper[Feature],
    *,
    num_epsilon: float,
    feasibility_tol: float,
    class_epsilon: float,
) -> float:
    # The class-margin epsilon is shared by CP and MIP, so it changes the
    # feasible set but not the MIP-vs-CP excess distance on that set.
    _ = class_epsilon
    return float(
        sum(
            continuous_interval_optimality_gap_bound(
                feature.levels,
                num_epsilon=num_epsilon,
                feasibility_tol=feasibility_tol,
            )
            for feature in mapper.values()
            if feature.is_continuous
        )
    )


def assert_continuous_explanation_is_valid(
    explanation: BaseExplanation,
    *,
    query: np.ndarray,
    target: int,
    clf: RandomForestClassifier,
    mapper: Mapper[Feature],
    columns: pd.Index,
) -> None:
    counterfactual = explanation.x

    np.testing.assert_allclose(explanation.query, query, atol=0.0, rtol=0.0)
    assert counterfactual.shape == query.shape
    assert np.isfinite(counterfactual).all()
    assert predict_processed(clf, counterfactual, columns) == target

    for name, feature in mapper.items():
        assert feature.is_continuous
        idx = mapper.idx.get(name)
        value = float(counterfactual[idx])
        assert float(feature.levels[0]) <= value <= float(feature.levels[-1])


def test_continuous_cp_and_mip_explainers_match_distance() -> None:  # noqa: PLR0914
    data, y, mapper = build_continuous_dataset(seed=SEED, n_samples=N_SAMPLES)
    assert mapper.n_columns == N_FEATURES
    assert all(feature.is_continuous for feature in mapper.values())

    clf = RandomForestClassifier(
        random_state=SEED,
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
    )
    clf.fit(data, y)

    importances = np.asarray(clf.feature_importances_, dtype=np.float64)
    assert importances.shape == (N_FEATURES,)
    assert np.count_nonzero(importances > 0.0) == N_FEATURES

    predictions = np.asarray(clf.predict(data), dtype=np.int64)
    query_indices = select_query_indices(predictions)
    assert query_indices.size == N_QUERY_EXAMPLES

    try:
        mip_explainer: MIPExplainer = MixedIntegerProgramExplainer(
            clf,
            mapper=mapper,
            env=ENV,
        )
        cp_explainer: CPExplainer = ConstraintProgrammingExplainer(
            clf,
            mapper=mapper,
        )
    except gp.GurobiError as exc:
        pytest.skip(f"Skipping test due to {exc}")

    explainers = (
        ("mip", mip_explainer),
        ("cp", cp_explainer),
    )
    distances: dict[str, list[float]] = {name: [] for name, _ in explainers}
    worst_case_optimality_gap = 0.0
    theoretical_gap_bound = theoretical_mip_optimality_gap_bound(
        mapper,
        num_epsilon=float(mip_explainer.num_epsilon),
        feasibility_tol=cast(
            "float",
            mip_explainer.getParamInfo("FeasibilityTol")[2],
        ),
        class_epsilon=float(mip_explainer.epsilon),
    )

    try:
        for idx in query_indices:
            query = data.iloc[int(idx)].to_numpy(dtype=float)
            target = int(1 - predictions[int(idx)])

            try:
                for name, explainer in explainers:
                    explanation = explainer.explain(
                        query,
                        y=target,
                        norm=1,
                        max_time=MAX_TIME,
                        num_workers=1,
                        random_seed=SEED,
                        clean_up=False,
                    )
                    assert explanation is not None

                    assert_continuous_explanation_is_valid(
                        explanation,
                        query=query,
                        target=target,
                        clf=clf,
                        mapper=mapper,
                        columns=data.columns,
                    )

                    if name == "mip":
                        raw_counterfactual = np.array(
                            [
                                mip_explainer.vget(i).X
                                for i in range(mip_explainer.n_columns)
                            ],
                            dtype=np.float64,
                        )
                        assert (
                            predict_processed(
                                clf,
                                raw_counterfactual,
                                data.columns,
                            )
                            == target
                        )
                        assert (
                            float(np.abs(explanation.x - query).sum())
                            <= float(np.abs(raw_counterfactual - query).sum())
                            + DISTANCE_TOL
                        )

                    direct_l1 = float(np.abs(explanation.x - query).sum())
                    manual_distance = manual_postprocessed_distance(
                        explanation, norm=1
                    )
                    solver_distance = explainer.get_distance()

                    assert manual_distance == pytest.approx(
                        direct_l1, abs=DISTANCE_TOL
                    )
                    assert solver_distance == pytest.approx(
                        direct_l1, abs=DISTANCE_TOL
                    )
                    distances[name].append(solver_distance)

                optimal_distance = distances["cp"][-1]
                mip_distance = distances["mip"][-1]
                optimality_gap = max(0.0, mip_distance - optimal_distance)
                worst_case_optimality_gap = max(
                    worst_case_optimality_gap,
                    optimality_gap,
                )
                assert optimality_gap <= theoretical_gap_bound + BOUND_TOL, (
                    f"MIP distance exceeded optimal CP distance by "
                    f"{optimality_gap:.8f} on query index {idx}, "
                    f"above the theoretical bound {theoretical_gap_bound:.8f}"
                )
            finally:
                for _, explainer in explainers:
                    explainer.cleanup()
    except gp.GurobiError as exc:
        pytest.skip(f"Skipping test due to {exc}")

    assert len(distances["mip"]) == N_QUERY_EXAMPLES
    assert len(distances["cp"]) == N_QUERY_EXAMPLES
    assert worst_case_optimality_gap <= theoretical_gap_bound + BOUND_TOL


def test_continuous_mip_regression_seed_returns_valid_cf() -> None:
    data, y, mapper = build_continuous_dataset(
        seed=REGRESSION_SEED,
        n_samples=N_SAMPLES,
    )
    clf = RandomForestClassifier(
        random_state=REGRESSION_SEED,
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
    )
    clf.fit(data, y)

    predictions = np.asarray(clf.predict(data), dtype=np.int64)

    try:
        mip_explainer: MIPExplainer = MixedIntegerProgramExplainer(
            clf,
            mapper=mapper,
            env=ENV,
        )
        cp_explainer: CPExplainer = ConstraintProgrammingExplainer(
            clf,
            mapper=mapper,
        )
    except gp.GurobiError as exc:
        pytest.skip(f"Skipping test due to {exc}")

    explainers = (
        ("mip", mip_explainer),
        ("cp", cp_explainer),
    )

    try:
        for idx in REGRESSION_QUERY_INDICES:
            query = data.iloc[idx].to_numpy(dtype=float)
            target = int(1 - predictions[idx])
            distances: dict[str, float] = {}

            try:
                for name, explainer in explainers:
                    explanation = explainer.explain(
                        query,
                        y=target,
                        norm=1,
                        max_time=MAX_TIME,
                        num_workers=1,
                        random_seed=REGRESSION_SEED,
                        clean_up=False,
                    )
                    assert explanation is not None

                    assert_continuous_explanation_is_valid(
                        explanation,
                        query=query,
                        target=target,
                        clf=clf,
                        mapper=mapper,
                        columns=data.columns,
                    )

                    direct_l1 = float(np.abs(explanation.x - query).sum())
                    manual_distance = manual_postprocessed_distance(
                        explanation, norm=1
                    )
                    solver_distance = explainer.get_distance()

                    assert manual_distance == pytest.approx(
                        direct_l1, abs=DISTANCE_TOL
                    )
                    assert solver_distance == pytest.approx(
                        direct_l1, abs=DISTANCE_TOL
                    )
                    distances[name] = solver_distance
            finally:
                for _, explainer in explainers:
                    explainer.cleanup()

            assert distances["mip"] == pytest.approx(
                distances["cp"], abs=DISTANCE_TOL
            )
    except gp.GurobiError as exc:
        pytest.skip(f"Skipping test due to {exc}")
