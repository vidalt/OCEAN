import sys

import numpy as np
import pytest

from ocean.abc import Mapper
from ocean.maxsat import Model
from ocean.tree import parse_trees

from ..utils import (
    MAX_DEPTH,
    N_CLASSES,
    N_ESTIMATORS,
    N_SAMPLES,
    SEEDS,
    train_rf,
)

pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="tests for non-windows platforms"
)


def test_no_trees() -> None:
    msg = r"At least one tree is required."
    with pytest.raises(ValueError, match=msg):
        Model(trees=[], mapper=Mapper())


def test_no_features() -> None:
    msg = r"At least one feature is required."
    rf, mapper = train_rf(42, 2, 2, 100, 2)
    trees = tuple(parse_trees(rf, mapper=mapper))
    with pytest.raises(ValueError, match=msg):
        Model(trees=trees, mapper=Mapper())


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("n_estimators", N_ESTIMATORS)
@pytest.mark.parametrize("max_depth", MAX_DEPTH)
@pytest.mark.parametrize("n_samples", N_SAMPLES)
@pytest.mark.parametrize("n_classes", N_CLASSES)
class TestNoIsolation:
    @staticmethod
    def test_no_weights(
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
        trees = parse_trees(clf, mapper=mapper)
        model = Model(trees=trees, mapper=mapper)
        expected_weights = np.ones(n_estimators, dtype=float)
        assert model is not None
        assert model.n_estimators == n_estimators
        assert model.n_classes == n_classes
        assert model.weights.shape == expected_weights.shape
        assert np.isclose(model.weights, expected_weights).all()

    @staticmethod
    def test_weights(
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
        trees = parse_trees(clf, mapper=mapper)
        generator = np.random.default_rng(seed)
        weights = generator.random(n_estimators).flatten()
        model = Model(trees=trees, mapper=mapper, weights=weights)
        assert model is not None
        assert model.n_estimators == n_estimators
        assert model.n_classes == n_classes
        assert model.weights.shape == weights.shape
        assert np.isclose(model.weights, weights).all()

    @staticmethod
    def test_invalid_weights(
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
        generator = np.random.default_rng(seed)
        shapes = [generator.integers(n_estimators + 1, 2 * n_estimators + 1)]
        if n_estimators > 2:
            shapes += [generator.integers(1, n_estimators - 1)]
        for shape in shapes:
            weights = generator.random(shape).flatten()
            msg = r"The number of weights must match the number of trees."
            with pytest.raises(ValueError, match=msg):
                Model(trees=trees, mapper=mapper, weights=weights)


def test_hard_voting_function_uses_tree_class_variables() -> None:
    clf, mapper = train_rf(42, 3, 2, 100, 2)
    for estimator in clf.estimators_:
        tree = estimator.tree_
        children_left = np.asarray(tree.children_left, dtype=np.int64)
        children_right = np.asarray(tree.children_right, dtype=np.int64)
        leaf_ids = np.flatnonzero(children_left == children_right)
        for node_id in leaf_ids:
            leaf_value = np.asarray(tree.value[node_id], dtype=np.float64)
            winners = np.argmax(leaf_value, axis=1)
            tree.value[node_id] = 0.0
            for output_idx, winner in enumerate(winners):
                tree.value[node_id, output_idx, winner] = 1.0

    trees = parse_trees(clf, mapper=mapper)
    model = Model(trees=trees, mapper=mapper, hard_voting=True)
    model.build()

    for class_id in range(model.n_classes):
        assert len(model.function[0, class_id]) == model.n_trees
