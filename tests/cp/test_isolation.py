import warnings
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest
from ortools.sat.python import cp_model as cp
from sklearn.ensemble import IsolationForest, RandomForestClassifier

from ocean.abc import Mapper
from ocean.cp import ENV as CP_ENV
from ocean.cp import Explainer as ConstraintProgrammingExplainer
from ocean.cp import Model
from ocean.feature import Feature
from ocean.mip import Explainer as MixedIntegerProgramExplainer
from ocean.tree import parse_ensembles

from .utils import (
    MAX_DEPTH,
    MAX_SAMPLES,
    N_CLASSES,
    N_ESTIMATORS,
    N_ISOLATORS,
    N_SAMPLES,
    SEEDS,
    train_rf_isolation,
    validate_paths,
    validate_sklearn_paths,
    validate_sklearn_pred,
    validate_solution,
)

if TYPE_CHECKING:
    from ocean.typing import Array1D


def selected_isolation_length(model: Model) -> float:
    solver = CP_ENV.solver
    length = 0.0
    for tree in model.isolators:
        for leaf in tree.leaves:
            if solver.Value(tree[leaf.node_id]) == 1:
                length += leaf.length
                break
    return length


def _match_cp_and_mip_distance(
    clf: RandomForestClassifier,
    isolation: IsolationForest,
    mapper: Mapper[Feature],
    data: pd.DataFrame,
    *,
    seed: int,
    isolation_threshold: float | None = None,
) -> (
    tuple[
        ConstraintProgrammingExplainer,
        MixedIntegerProgramExplainer,
    ]
    | None
):
    for row in range(len(data)):
        x: Array1D = np.asarray(data.iloc[row, :], dtype=np.float64).ravel()
        row_frame = pd.DataFrame([x], columns=data.columns)
        target = int(1 - clf.predict(row_frame)[0])  # pyright: ignore[reportUnknownArgumentType]
        cp_model = ConstraintProgrammingExplainer(
            clf,
            mapper=mapper,
            isolation=isolation,
            isolation_threshold=isolation_threshold,
        )
        mip_model = MixedIntegerProgramExplainer(
            clf,
            mapper=mapper,
            isolation=isolation,
            isolation_threshold=isolation_threshold,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            cp_explanation = cp_model.explain(
                x,
                y=target,
                norm=1,
                max_time=10,
                random_seed=seed,
            )
            mip_explanation = mip_model.explain(
                x,
                y=target,
                norm=1,
                max_time=10,
                random_seed=seed,
            )

        if cp_explanation is None or mip_explanation is None:
            continue

        assert clf.predict(np.asarray([cp_explanation.to_numpy()]))[0] == target
        assert (
            clf.predict(np.asarray([mip_explanation.to_numpy()]))[0] == target
        )
        assert cp_model.get_distance() == pytest.approx(
            mip_model.get_distance(),
            abs=1e-6,
        )
        return cp_model, mip_model
    return None


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("n_estimators", N_ESTIMATORS)
@pytest.mark.parametrize("max_depth", MAX_DEPTH)
@pytest.mark.parametrize("n_samples", N_SAMPLES)
@pytest.mark.parametrize("n_classes", N_CLASSES)
@pytest.mark.parametrize("n_isolators", N_ISOLATORS)
@pytest.mark.parametrize("max_samples", MAX_SAMPLES)
class TestIsolation:
    @staticmethod
    def test_build(
        seed: int,
        n_estimators: int,
        max_depth: int,
        n_samples: int,
        n_classes: int,
        n_isolators: int,
        max_samples: int,
    ) -> None:
        clf, isolation, mapper = train_rf_isolation(
            seed,
            n_estimators,
            max_depth,
            n_isolators,
            max_samples,
            n_samples,
            n_classes,
        )
        trees = parse_ensembles(clf, isolation, mapper=mapper)
        model = Model(
            trees=trees,
            mapper=mapper,
            n_isolators=n_isolators,
            max_samples=max_samples,
        )
        model.build()

        status = CP_ENV.solver.Solve(model)
        assert status == cp.OPTIMAL, CP_ENV.solver.StatusName()

        explanation = model.explanation

        validate_solution(explanation)
        validate_paths(*model.trees, explanation=explanation)
        validate_sklearn_paths(clf, explanation, model.estimators)

        assert model.n_estimators == n_estimators
        assert model.n_isolators == n_isolators
        assert len(model.isolators) == n_isolators
        assert selected_isolation_length(model) + 1e-6 >= model.min_length

    @staticmethod
    def test_set_majority_class(
        seed: int,
        n_estimators: int,
        max_depth: int,
        n_samples: int,
        n_classes: int,
        n_isolators: int,
        max_samples: int,
    ) -> None:
        clf, isolation, mapper, data = train_rf_isolation(
            seed,
            n_estimators,
            max_depth,
            n_isolators,
            max_samples,
            n_samples,
            n_classes,
            return_data=True,
        )
        trees = parse_ensembles(clf, isolation, mapper=mapper)
        model = Model(
            trees=trees,
            mapper=mapper,
            n_isolators=n_isolators,
            max_samples=max_samples,
        )
        model.build()

        predictions = np.array(clf.predict(data), dtype=np.int64)
        classes = set(map(int, predictions.flatten()))
        n_optimal = 0

        for class_ in classes:
            model.set_majority_class(y=class_)

            status = CP_ENV.solver.Solve(model)
            assert status in {cp.OPTIMAL, cp.INFEASIBLE}, (
                f"{CP_ENV.solver.ResponseStats()} for class {class_}"
            )

            if status == cp.INFEASIBLE:  # pyright: ignore[reportUnnecessaryComparison]  # type: ignore[unreachable]
                model.cleanup()
                continue

            n_optimal += 1

            explanation = model.explanation

            validate_solution(explanation)
            validate_paths(*model.trees, explanation=explanation)
            validate_sklearn_paths(clf, explanation, model.estimators)
            validate_sklearn_pred(clf, explanation, m_class=class_, model=model)
            assert selected_isolation_length(model) + 1e-6 >= model.min_length

            model.cleanup()

        assert n_optimal > 0


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("n_estimators", [4])
@pytest.mark.parametrize("max_depth", MAX_DEPTH)
@pytest.mark.parametrize("n_samples", [100, 200])
@pytest.mark.parametrize("n_isolators", N_ISOLATORS)
@pytest.mark.parametrize("max_samples", MAX_SAMPLES)
def test_cp_and_mip_isolation_explanations_match_distance(
    seed: int,
    n_estimators: int,
    max_depth: int,
    n_samples: int,
    n_isolators: int,
    max_samples: int,
) -> None:
    clf, isolation, mapper, data = train_rf_isolation(
        seed,
        n_estimators,
        max_depth,
        n_isolators,
        max_samples,
        n_samples,
        2,
        return_data=True,
    )

    if (
        _match_cp_and_mip_distance(
            clf,
            isolation,
            mapper,
            data,
            seed=seed,
        )
        is not None
    ):
        return

    msg = (
        "Could not find a query whose CP and MIP isolation-forest "
        f"counterfactuals were both feasible for seed={seed}, "
        f"n_estimators={n_estimators}, max_depth={max_depth}, "
        f"n_samples={n_samples}, n_isolators={n_isolators}, "
        f"max_samples={max_samples}."
    )
    raise AssertionError(msg)


def test_cp_and_mip_custom_isolation_threshold_match_distance() -> None:
    seed = 43
    n_isolators = 2
    max_samples = 8
    isolation_threshold = 0.51
    clf, isolation, mapper, data = train_rf_isolation(
        seed,
        4,
        3,
        n_isolators,
        max_samples,
        500,
        2,
        return_data=True,
    )
    baseline_model = MixedIntegerProgramExplainer(
        clf,
        mapper=mapper,
        isolation=isolation,
    )
    expected_min_average_length = -baseline_model.min_average_length * np.log2(
        isolation_threshold
    )
    matched_models = _match_cp_and_mip_distance(
        clf,
        isolation,
        mapper,
        data,
        seed=seed,
        isolation_threshold=isolation_threshold,
    )
    if matched_models is not None:
        cp_model, mip_model = matched_models
        assert cp_model.min_average_length == pytest.approx(
            expected_min_average_length,
            abs=1e-12,
        )
        assert mip_model.min_average_length == pytest.approx(
            expected_min_average_length,
            abs=1e-12,
        )
        return

    msg = (
        "Could not find a query whose CP and MIP custom-threshold "
        "isolation-forest counterfactuals were both feasible."
    )
    raise AssertionError(msg)
