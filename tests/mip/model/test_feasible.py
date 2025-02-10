import gurobipy as gp
import numpy as np
import pytest

from ocean.mip import Model
from ocean.tree import parse_ensembles, parse_trees

from ...utils import ENV
from ..utils import (
    MAX_DEPTH,
    MAX_SAMPLES,
    N_CLASSES,
    N_ESTIMATORS,
    N_ISOLATORS,
    N_SAMPLES,
    SEEDS,
    train_rf,
    train_rf_isolation,
    validate_paths,
    validate_sklearn_paths,
    validate_sklearn_pred,
    validate_solution,
)


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("n_estimators", N_ESTIMATORS)
@pytest.mark.parametrize("max_depth", MAX_DEPTH)
@pytest.mark.parametrize("n_samples", N_SAMPLES)
@pytest.mark.parametrize("n_classes", N_CLASSES)
class TestNoIsolation:
    @staticmethod
    def test_build(
        seed: int,
        n_estimators: int,
        max_depth: int,
        n_samples: int,
        n_classes: int,
    ) -> None:
        clf, mapper = train_rf(
            seed,
            n_estimators,
            max_depth,
            n_samples,
            n_classes,
        )
        trees = tuple(parse_trees(clf, mapper=mapper))
        model = Model(trees=trees, mapper=mapper, env=ENV)
        model.build()

        try:
            model.optimize()
        except gp.GurobiError as e:
            pytest.skip(f"Skipping test due to {e}")

        assert model.Status == gp.GRB.OPTIMAL

        explanation = model.explanation

        validate_solution(explanation)
        validate_paths(*model.trees, explanation=explanation)
        validate_sklearn_paths(clf, explanation, model.estimators)

    @staticmethod
    def test_set_majority_class(
        seed: int,
        n_estimators: int,
        max_depth: int,
        n_samples: int,
        n_classes: int,
    ) -> None:
        clf, mapper, data = train_rf(
            seed,
            n_estimators,
            max_depth,
            n_samples,
            n_classes,
            return_data=True,
        )
        trees = tuple(parse_trees(clf, mapper=mapper))
        model = Model(trees=trees, mapper=mapper, env=ENV)
        model.build()

        predictions = np.array(clf.predict(data), dtype=np.int64)
        classes = set(map(int, predictions.flatten()))

        n_skipped = 0
        for class_ in classes:
            model.set_majority_class(y=class_)

            try:
                model.optimize()
            except gp.GurobiError:
                n_skipped += 1
                continue

            assert model.Status == gp.GRB.OPTIMAL

            explanation = model.explanation

            validate_solution(explanation)
            validate_paths(*model.trees, explanation=explanation)
            validate_sklearn_paths(clf, explanation, model.estimators)
            validate_sklearn_pred(clf, explanation, m_class=class_, model=model)

            model.clear_majority_class()

        if n_skipped > 0:
            msg = f"Skipped {n_skipped} tests due to GurobiErrors"
            # This test passes but some tests were skipped
            pytest.skip(msg)


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
        clf, ilf, mapper = train_rf_isolation(
            seed,
            n_estimators,
            max_depth,
            n_isolators,
            max_samples,
            n_samples,
            n_classes,
        )
        trees = parse_ensembles(clf, ilf, mapper=mapper)
        model = Model(
            trees=trees,
            mapper=mapper,
            n_isolators=n_isolators,
            env=ENV,
        )
        model.build()

        try:
            model.optimize()
        except gp.GurobiError as e:
            pytest.skip(f"Skipping test due to {e}")

        assert model.Status == gp.GRB.OPTIMAL

        explanation = model.explanation

        validate_solution(explanation)
        validate_paths(*model.trees, explanation=explanation)
        validate_sklearn_paths(clf, explanation, model.estimators)

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
        clf, ilf, mapper, data = train_rf_isolation(
            seed,
            n_estimators,
            max_depth,
            n_isolators,
            max_samples,
            n_samples,
            n_classes,
            return_data=True,
        )
        trees = parse_ensembles(clf, ilf, mapper=mapper)
        model = Model(
            trees=trees,
            mapper=mapper,
            n_isolators=n_isolators,
            env=ENV,
        )
        model.build()

        predictions = np.array(clf.predict(data), dtype=np.int64)
        classes = set(map(int, predictions.flatten()))

        n_skipped = 0
        for class_ in classes:
            model.set_majority_class(y=class_)

            try:
                model.optimize()
            except gp.GurobiError:
                n_skipped += 1
                continue

            assert model.Status == gp.GRB.OPTIMAL

            explanation = model.explanation

            validate_solution(explanation)
            validate_paths(*model.trees, explanation=explanation)
            validate_sklearn_paths(clf, explanation, model.estimators)
            validate_sklearn_pred(clf, explanation, m_class=class_, model=model)

            model.clear_majority_class()

        if n_skipped > 0:
            msg = f"Skipped {n_skipped} tests due to GurobiErrors"
            # This test passes but some tests were skipped
            pytest.skip(msg)
