import numpy as np
import pytest

from ocean.maxsat import ENV, Model
from ocean.tree import parse_trees

from ..utils import (
    MAX_DEPTH,
    N_CLASSES,
    N_ESTIMATORS,
    N_SAMPLES,
    SEEDS,
    train_rf,
    validate_paths,
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
        clf, mapper, data = train_rf(
            seed,
            n_estimators,
            max_depth,
            n_samples,
            n_classes,
            return_data=True,
        )
        trees = tuple(parse_trees(clf, mapper=mapper))

        # Get predictions and pick a class to target
        predictions = np.array(clf.predict(data), dtype=np.int64)
        target_class = int(predictions[0])

        model = Model(trees=trees, mapper=mapper)
        model.build()
        model.set_majority_class(y=target_class)

        solver = ENV.solver
        solver_model = solver.solve(model)

        explanation = model.explanation

        validate_solution(explanation)
        validate_paths(
            *model.trees, explanation=explanation, solver_model=solver_model
        )
        validate_sklearn_pred(clf, explanation, m_class=target_class)

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

        predictions = np.array(clf.predict(data), dtype=np.int64)
        classes = set(map(int, predictions.flatten()))

        for class_ in classes:
            model = Model(trees=trees, mapper=mapper)
            model.build()
            model.set_majority_class(y=class_)

            solver = ENV.solver
            solver_model = solver.solve(model)

            explanation = model.explanation

            validate_solution(explanation)
            validate_paths(
                *model.trees, explanation=explanation, solver_model=solver_model
            )
            validate_sklearn_pred(clf, explanation, m_class=class_)
            model.cleanup()
