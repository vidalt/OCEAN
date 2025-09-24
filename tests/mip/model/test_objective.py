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
    check_solution,
    train_rf,
    validate_paths,
    validate_sklearn_paths,
    validate_sklearn_pred,
    validate_solution,
)

P_QUERIES = 0.2


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("n_estimators", N_ESTIMATORS)
@pytest.mark.parametrize("max_depth", MAX_DEPTH)
@pytest.mark.parametrize("n_samples", N_SAMPLES)
@pytest.mark.parametrize("n_classes", N_CLASSES)
@pytest.mark.parametrize("norm", [1, 2])
class TestNoIsolation:
    @staticmethod
    def test_build(
        seed: int,
        n_estimators: int,
        max_depth: int,
        n_samples: int,
        n_classes: int,
        norm: int,
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

        n_queries = int(data.shape[1] * P_QUERIES)
        generator = np.random.default_rng(seed)
        queries = generator.choice(
            range(len(data)), size=n_queries, replace=False
        )

        n_skipped = 0
        for query in queries:
            x = np.array(data.to_numpy()[query], dtype=np.float64).flatten()

            model.add_objective(x=x, norm=norm)

            try:
                model.optimize()
            except gp.GurobiError:
                n_skipped += 1
                continue

            assert model.Status == gp.GRB.OPTIMAL

            explanation = model.explanation

            validate_solution(explanation)
            validate_paths(*model.trees, explanation=explanation)
            validate_sklearn_paths(clf, explanation, model.trees)
            check_solution(x, explanation)

            model.cleanup()

        if n_skipped > 0:
            msg = f"Skipped {n_skipped} tests due to GurobiErrors"
            # This test passes but some tests were skipped
            pytest.skip(msg)

    @staticmethod
    def test_set_majority_class(
        seed: int,
        n_estimators: int,
        max_depth: int,
        n_samples: int,
        n_classes: int,
        norm: int,
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

        generator = np.random.default_rng(seed)
        query = generator.integers(len(data))

        x = np.array(data.to_numpy()[query], dtype=np.float64).flatten()
        y = int(predictions[query])

        n_skipped = 0
        for class_ in classes:
            model.set_majority_class(y=class_)

            model.add_objective(x=x, norm=norm)

            try:
                model.optimize()
            except gp.GurobiError:
                n_skipped += 1
                continue

            assert model.Status == gp.GRB.OPTIMAL

            explanation = model.explanation
            validate_solution(explanation)
            validate_paths(*model.trees, explanation=explanation)
            validate_sklearn_paths(clf, explanation, model.trees)
            validate_sklearn_pred(clf, explanation, m_class=class_, model=model)
            if class_ == y:
                check_solution(x, explanation)

            model.clear_majority_class()
            model.cleanup()

        if n_skipped > 0:
            msg = f"Skipped {n_skipped} tests due to GurobiErrors"
            # This test passes but some tests were skipped
            pytest.skip(msg)
