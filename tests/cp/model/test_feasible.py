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
        clf, mapper = train_rf(
            seed,
            n_estimators,
            max_depth,
            n_samples,
            n_classes,
        )
        trees = tuple(parse_trees(clf, mapper=mapper))
        model = Model(trees=trees, mapper=mapper)
        model.build()
        n_nodes = sum(tree.n_nodes for tree in model.trees)
        n_leaves = sum(len(tree.leaves) for tree in model.trees)
        feature_vars = 0
        feature_constraints = 0
        for feature in model.mapper.values():
            if feature.is_binary:
                feature_vars += 1
            elif feature.is_continuous:
                feature_vars += len(feature.levels)
                feature_constraints += 2 * (len(feature.levels) - 1)
            elif feature.is_discrete:
                feature_vars += len(feature.levels) + 2
                feature_constraints += 2 * len(feature.levels)
            else:
                feature_vars += len(feature.codes)
                feature_constraints += 1
        lb = 2 * (n_nodes - n_leaves)
        ub = (n_nodes - n_leaves) * (n_nodes - n_leaves + 1)
        lb += feature_constraints + n_estimators
        ub += feature_constraints + n_estimators
        assert len(model.Proto().variables) == n_leaves + feature_vars
        assert len(model.Proto().constraints) >= lb
        assert len(model.Proto().constraints) <= ub

        solver = ENV.solver
        status = solver.Solve(model)
        assert status == cp.OPTIMAL

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
        model = Model(trees=trees, mapper=mapper)

        predictions = np.array(clf.predict(data), dtype=np.int64)
        classes = set(map(int, predictions.flatten()))
        model.build()
        for class_ in classes:
            model.set_majority_class(y=class_)

            solver = ENV.solver
            status = solver.Solve(model)
            assert status == cp.OPTIMAL, (
                f"{solver.ResponseStats()} for class {class_} with constraint "
            )

            explanation = model.explanation

            validate_solution(explanation)
            validate_paths(*model.trees, explanation=explanation)
            validate_sklearn_paths(clf, explanation, model.estimators)
            validate_sklearn_pred(clf, explanation, m_class=class_, model=model)

            model.cleanup()
