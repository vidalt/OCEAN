import numpy as np
import pytest
from ortools.sat.python import cp_model as cp

from ocean.cp import BaseModel, TreeManager
from ocean.tree import Node, Tree

SEEDS = range(5)
N_ESTIMATORS = [2, 3, 4]
N_CLASSES = [2, 3]


def create_simple_tree(seed: int, n_classes: int) -> Tree:
    generator = np.random.default_rng(seed)
    values = generator.uniform(0.0, 1.0, (4, n_classes))
    values /= values.sum(axis=1, keepdims=True)
    values = values.reshape(4, 1, n_classes)
    thresholds = generator.uniform(0.0, 1.0, 3)
    left = Node(2, value=values[0])
    right = Node(3, value=values[1])
    root1 = Node(
        1, threshold=thresholds[0], feature="x", left=left, right=right
    )

    left = Node(5, value=values[2])
    right = Node(6, value=values[3])
    root2 = Node(
        4, threshold=thresholds[1], feature="x", left=left, right=right
    )

    root = Node(
        0, threshold=thresholds[2], feature="x", left=root1, right=root2
    )
    return Tree(root=root)


def test_no_trees() -> None:
    msg = r"At least one tree is required."
    with pytest.raises(ValueError, match=msg):
        TreeManager(trees=[])


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("n_estimators", N_ESTIMATORS)
@pytest.mark.parametrize("n_classes", N_CLASSES)
class TestTreeManager:
    @staticmethod
    def test_init_default_weights(
        seed: int,
        n_estimators: int,
        n_classes: int,
    ) -> None:
        trees = [
            create_simple_tree(seed + i, n_classes) for i in range(n_estimators)
        ]
        manager = TreeManager(trees=trees)

        assert manager.n_trees == n_estimators
        assert manager.n_estimators == n_estimators
        assert manager.n_classes == n_classes
        assert manager.weights.shape == (n_estimators,)
        assert np.all(manager.weights == 1.0)

    @staticmethod
    def test_init_custom_weights(
        seed: int,
        n_estimators: int,
        n_classes: int,
    ) -> None:
        generator = np.random.default_rng(seed)
        trees = [
            create_simple_tree(seed + i, n_classes) for i in range(n_estimators)
        ]
        weights = generator.random(n_estimators).flatten()

        manager = TreeManager(trees=trees, weights=weights)
        assert manager.n_trees == n_estimators
        assert manager.weights.shape == (n_estimators,)
        assert np.allclose(manager.weights, weights)

    @staticmethod
    def test_invalid_weights(
        seed: int,
        n_estimators: int,
        n_classes: int,
    ) -> None:
        generator = np.random.default_rng(seed)
        trees = [
            create_simple_tree(seed + i, n_classes) for i in range(n_estimators)
        ]

        # Test with wrong number of weights
        invalid_weights = generator.random(n_estimators + 1).flatten()
        msg = r"The number of weights must match the number of trees."
        with pytest.raises(ValueError, match=msg):
            TreeManager(trees=trees, weights=invalid_weights)

    @staticmethod
    def test_build_trees(
        seed: int,
        n_estimators: int,
        n_classes: int,
    ) -> None:
        trees = [
            create_simple_tree(seed + i, n_classes) for i in range(n_estimators)
        ]
        manager = TreeManager(trees=trees)
        model = BaseModel()

        manager.build_trees(model)

        # Check that function dictionary is created
        assert isinstance(manager.function, dict)
        # Check function shape
        assert len(manager.function) == n_classes
        # Check that all expressions are LinearExpr
        for expr in manager.function.values():
            assert isinstance(expr, cp.LinearExpr)

    @staticmethod
    def test_weighted_function(
        seed: int,
        n_estimators: int,
        n_classes: int,
    ) -> None:
        generator = np.random.default_rng(seed)
        trees = [
            create_simple_tree(seed + i, n_classes) for i in range(n_estimators)
        ]
        manager = TreeManager(trees=trees)
        model = BaseModel()
        manager.build_trees(model)

        # Test with new weights
        new_weights = generator.random(n_estimators).flatten()
        weighted_func = manager.weighted_function(new_weights)

        assert isinstance(weighted_func, dict)
        assert len(weighted_func) == n_classes
        for expr in weighted_func.values():
            assert isinstance(expr, cp.LinearExpr)

    @staticmethod
    def test_custom_scale(
        seed: int,
        n_estimators: int,
        n_classes: int,
    ) -> None:
        trees = [
            create_simple_tree(seed + i, n_classes) for i in range(n_estimators)
        ]
        custom_scale = 1000
        manager = TreeManager(trees=trees, scale=custom_scale)

        model = BaseModel()
        manager.build_trees(model)

        assert manager.score_scale == custom_scale
