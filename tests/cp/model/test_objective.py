import numpy as np
import pytest
from ortools.sat.python import cp_model as cp

from ocean.cp import ENV, Model
from ocean.tree import parse_trees

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
        clf, mapper, data = train_rf(
            seed,
            n_estimators,
            max_depth,
            n_samples,
            n_classes,
            return_data=True,
        )
        trees = tuple(parse_trees(clf, mapper=mapper))
        model = Model(trees=trees, mapper=mapper)
        model.build()

        x = np.array(data.iloc[0].to_numpy(), dtype=np.float64).flatten()
        model.add_objective(x=x)

        solver = ENV.solver
        status = solver.Solve(model)
        assert status == cp.OPTIMAL
        model.explanation.query = x
        explanation = model.explanation

        validate_solution(explanation)
        validate_paths(*model.trees, explanation=explanation)
        validate_sklearn_paths(clf, explanation, model.estimators)

        model.cleanup()

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
        x = np.array(data.iloc[0].to_numpy(), dtype=np.float64).flatten()
        for class_ in classes:
            if class_ == int(predictions[0]):
                continue
            model = Model(trees=trees, mapper=mapper)
            model.build()
            model.set_majority_class(y=class_)
            model.add_objective(x=x)

            solver = ENV.solver
            status = solver.Solve(model)
            assert status == cp.OPTIMAL, (
                f"Status: {solver.status_name()}"
                f" for class {class_}, x = {x}, y={predictions[0]}"
            )

            model.explanation.query = x
            explanation = model.explanation

            validate_solution(explanation)
            validate_paths(*model.trees, explanation=explanation)
            validate_sklearn_paths(clf, explanation, model.estimators)
            validate_sklearn_pred(clf, explanation, m_class=class_, model=model)
            model.cleanup()
