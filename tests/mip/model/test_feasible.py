import gurobipy as gp
import numpy as np
import pytest

from ocean.mip import Model
from ocean.tree import parse_trees

from ...utils import ENV
from ..utils import (
    MAX_DEPTH,
    N_CLASSES,
    N_ESTIMATORS,
    N_SAMPLES,
    SEEDS,
    train_rf,
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

        solution = model.solution

        validate_solution(solution)
        validate_paths(*model.trees, solution=solution)
        validate_sklearn_paths(clf, solution, model.trees)

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

        predictions = np.array(clf.predict(data.to_numpy()), dtype=np.int64)
        classes = set(map(int, predictions.flatten()))

        n_skipped = 0
        for class_ in classes:
            model.set_majority_class(m_class=class_)

            try:
                model.optimize()
            except gp.GurobiError:
                n_skipped += 1
                continue

            assert model.Status == gp.GRB.OPTIMAL

            solution = model.solution

            validate_solution(solution)
            validate_paths(*model.trees, solution=solution)
            validate_sklearn_paths(clf, solution, model.trees)
            validate_sklearn_pred(clf, solution, m_class=class_, model=model)

            model.clear_majority_class()

        if n_skipped > 0:
            msg = f"Skipped {n_skipped} tests due to GurobiErrors"
            # This test passes but some tests were skipped
            pytest.skip(msg)
