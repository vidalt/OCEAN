from collections import defaultdict
from typing import TYPE_CHECKING, cast

import gurobipy as gp
import numpy as np
import pytest
from sklearn.ensemble import IsolationForest, RandomForestClassifier

from ocean.abc import Mapper
from ocean.feature import Feature
from ocean.mip import FeatureVar, Model, Solution, TreeVar
from ocean.tree import Node, parse_trees

from ..utils import ENV, generate_data

if TYPE_CHECKING:
    import scipy.sparse as sp

    from ocean.typing import Key


def _check_solution(solution: Solution) -> None:
    x = solution.to_numpy()
    n = solution.n_columns
    codes: dict[Key, float] = defaultdict(float)
    for i in range(n):
        name = solution.names[i]
        feature = solution[name]
        value = x[i]
        if feature.is_one_hot_encoded:
            assert np.any(np.isclose(value, [0.0, 1.0]))
            codes[name] += value

        if feature.is_binary:
            assert np.any(np.isclose(value, [0.0, 1.0]))
        elif feature.is_numeric:
            assert feature.levels[0] <= value <= feature.levels[-1]
            if feature.is_discrete:
                assert np.any(np.isclose(value, feature.levels))

    for value in codes.values():
        assert np.isclose(value, 1.0)


def _check_node(tree: TreeVar, node: Node, solution: Solution) -> None:
    if node.is_leaf:
        return

    left = node.left
    right = node.right
    x = solution.to_numpy()
    next_node = left if tree[left.node_id].X == 1.0 else right
    assert tree[next_node.node_id].X == 1.0

    name = node.feature
    if solution[name].is_one_hot_encoded:
        code = node.code
        i = solution.idx[name, code]
        value = x[i]
    else:
        i = solution.idx[name]
        value = x[i]

    if solution[name].is_numeric:
        threshold = node.threshold
        if value <= threshold:
            assert tree[left.node_id].X == 1.0
        else:
            assert tree[right.node_id].X == 1.0
    elif np.isclose(value, 0.0):
        assert tree[left.node_id].X == 1.0
    else:
        assert tree[right.node_id].X == 1.0

    _check_node(tree, next_node, solution=solution)


def _check_paths(
    clf: RandomForestClassifier,
    X: np.ndarray[tuple[int], np.dtype[np.float64]],
    trees: tuple[TreeVar, ...],
) -> None:
    x = X.reshape((1, -1))
    ind, ptr = clf.decision_path(x)  # pyright: ignore[reportUnknownVariableType]
    ind = cast("sp.csr_matrix", ind)
    ptr = np.array(ptr, dtype=np.int64)

    for t, tree in enumerate(trees):
        # Get the leaf node from the tree
        node = tree.root
        while not node.is_leaf:
            node = node.left if tree[node.left.node_id].X == 1.0 else node.right
            is_path_valid: bool = bool(ind[0, ptr[t] + node.node_id])
            assert is_path_valid


def _check_prediction(
    clf: RandomForestClassifier,
    solution: Solution,
    m_class: int,
    model: Model,
) -> None:
    x = solution.to_numpy().reshape(1, -1)
    prediction = np.asarray(clf.predict(x), dtype=np.int64)
    function = np.asarray(model.function.getValue(), dtype=np.float64)
    proba = function / np.sum(function)
    expected_proba = np.asarray(clf.predict_proba(x), dtype=np.float64)
    assert (prediction == m_class).all()
    assert np.isclose(expected_proba.flatten(), proba).all()


def _get_objective(mapper: Mapper[FeatureVar]) -> gp.QuadExpr:
    obj = gp.QuadExpr()
    for feature in mapper.values():
        if feature.is_one_hot_encoded:
            for j, code in enumerate(feature.codes):
                if j == 0:
                    obj += (0.75 - feature[code]) ** 2
        elif feature.is_numeric:
            level = feature.levels.mean()
            obj += (level - feature.x) ** 2
        else:
            obj += (0.0 - feature.x) ** 2
    return obj


def _train_rf(
    seed: int,
    n_estimators: int,
    max_depth: int,
    n_samples: int,
    n_classes: int,
) -> tuple[RandomForestClassifier, Mapper[Feature]]:
    data, y, mapper = generate_data(seed, n_samples, n_classes)
    clf = RandomForestClassifier(
        random_state=seed,
        n_estimators=n_estimators,
        max_depth=max_depth,
    )
    clf.fit(data.to_numpy(), y)
    return clf, mapper


def test_model_init_with_no_trees() -> None:
    msg = r"At least one tree is required."
    with pytest.raises(ValueError, match=msg):
        Model(trees=[], mapper=Mapper(), env=ENV)


def test_model_init_with_no_features() -> None:
    msg = r"At least one feature is required."
    rf, mapper = _train_rf(42, 2, 2, 100, 2)
    trees = tuple(parse_trees(rf, mapper=mapper))
    with pytest.raises(ValueError, match=msg):
        Model(trees=trees, mapper=Mapper(), env=ENV)


@pytest.mark.parametrize("seed", [42, 43, 44])
@pytest.mark.parametrize("n_estimators", [1, 5, 10])
@pytest.mark.parametrize("max_depth", [2, 3])
@pytest.mark.parametrize("n_samples", [100, 200, 500])
@pytest.mark.parametrize("n_classes", [2, 3, 4])
class TestModelInitWeightsNoIsolation:
    @staticmethod
    def test_model_init_with_no_weights(
        seed: int,
        n_estimators: int,
        max_depth: int,
        n_samples: int,
        n_classes: int,
    ) -> None:
        clf, mapper = _train_rf(
            seed,
            n_estimators,
            max_depth,
            n_samples,
            n_classes,
        )
        trees = parse_trees(clf, mapper=mapper)
        model = Model(trees=trees, mapper=mapper, env=ENV)
        expected_weights = np.ones(n_estimators, dtype=float)
        assert model is not None
        assert model.n_estimators == n_estimators
        assert model.n_classes == n_classes
        assert model.weights.shape == expected_weights.shape
        assert np.isclose(model.weights, expected_weights).all()

    @staticmethod
    def test_model_init_with_weights(
        seed: int,
        n_estimators: int,
        max_depth: int,
        n_samples: int,
        n_classes: int,
    ) -> None:
        clf, mapper = _train_rf(
            seed,
            n_estimators,
            max_depth,
            n_samples,
            n_classes,
        )
        trees = parse_trees(clf, mapper=mapper)
        generator = np.random.default_rng(seed)
        weights = generator.random(n_estimators).flatten()
        model = Model(trees=trees, mapper=mapper, weights=weights, env=ENV)
        assert model is not None
        assert model.n_estimators == n_estimators
        assert model.n_classes == n_classes
        assert model.weights.shape == weights.shape
        assert np.isclose(model.weights, weights).all()

    @staticmethod
    def test_model_init_with_invalid_weights(
        seed: int,
        n_estimators: int,
        max_depth: int,
        n_samples: int,
        n_classes: int,
    ) -> None:
        clf, mapper = _train_rf(
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
                Model(trees=trees, mapper=mapper, weights=weights, env=ENV)


@pytest.mark.parametrize("seed", [42, 43, 44])
@pytest.mark.parametrize("n_estimators", [1])
@pytest.mark.parametrize("max_depth", [2, 3])
@pytest.mark.parametrize("n_classes", [2, 3, 4])
@pytest.mark.parametrize("n_samples", [100, 200, 500])
def test_model_no_isolation(
    seed: int,
    n_estimators: int,
    max_depth: int,
    n_classes: int,
    n_samples: int,
) -> None:
    data, y, mapper = generate_data(seed, n_samples, n_classes)
    clf = RandomForestClassifier(
        random_state=seed,
        n_estimators=n_estimators,
        max_depth=max_depth,
    )
    clf.fit(data.to_numpy(), y)
    trees = tuple(parse_trees(clf, mapper=mapper))
    weights = (np.ones(n_estimators, dtype=float)).flatten()
    model = Model(
        trees=trees,
        mapper=mapper,
        weights=weights,
        model_type=Model.Type.MIP,
        env=ENV,
    )
    assert model is not None
    assert model.n_estimators == n_estimators
    assert model.n_classes == n_classes

    model.build()

    objective = _get_objective(model.mapper)
    model.setObjective(objective, sense=gp.GRB.MINIMIZE)

    try:
        model.optimize()
    except gp.GurobiError as e:
        pytest.skip(f"Skipping test due to {e}")

    assert model.Status == gp.GRB.OPTIMAL

    solution = model.solution
    assert solution is not None

    x = solution.to_numpy()

    _check_solution(solution)

    for tree in model.trees:
        _check_node(tree, tree.root, solution=solution)

    _check_paths(clf, x, model.trees)
    available_classes: set[int] = set()
    for tree in model.trees:
        for leaf in tree.leaves:
            class_ = int(np.argmax(leaf.value[0]))
            available_classes.add(class_)

    for class_ in available_classes:
        model.set_majority_class(m_class=class_)
        try:
            model.optimize()
        except gp.GurobiError as e:
            pytest.skip(f"Skipping test due to {e}")

        assert model.Status == gp.GRB.OPTIMAL

        solution = model.solution
        assert solution is not None

        x = solution.to_numpy()
        _check_paths(clf, x, model.trees)
        _check_prediction(
            clf,
            solution,
            m_class=class_,
            model=model,
        )

        model.clear_majority_class()


@pytest.mark.parametrize("seed", [42, 43, 44])
@pytest.mark.parametrize("n_estimators", [10])
@pytest.mark.parametrize("max_depth", [2, 3])
@pytest.mark.parametrize("n_classes", [2, 3, 4])
@pytest.mark.parametrize("n_samples", [100, 200, 500])
@pytest.mark.parametrize("n_isolators", [5])
@pytest.mark.parametrize("max_samples", [4, 8])
def test_model_isolation(
    seed: int,
    n_estimators: int,
    max_depth: int,
    n_classes: int,
    n_samples: int,
    n_isolators: int,
    max_samples: int,
) -> None:
    data, y, mapper = generate_data(seed, n_samples, n_classes)
    clf = RandomForestClassifier(
        random_state=seed,
        n_estimators=n_estimators,
        max_depth=max_depth,
    )
    clf.fit(data.to_numpy(), y)
    ilf = IsolationForest(
        random_state=seed,
        n_estimators=n_isolators,
        max_samples=max_samples,  # pyright: ignore[reportArgumentType]
    )
    ilf.fit(data.to_numpy())

    trees = tuple(parse_trees(clf, mapper=mapper))
    trees += tuple(parse_trees(ilf, mapper=mapper))
    weights = (np.ones(n_estimators, dtype=float) * 1e5).flatten()
    model = Model(
        trees=trees,
        mapper=mapper,
        weights=weights,
        n_isolators=n_isolators,
        delta=0.1,
        model_type=Model.Type.MIP,
        env=ENV,
    )
    assert model is not None
    assert model.n_estimators == n_estimators
    assert model.n_classes == n_classes

    model.build()

    objective = _get_objective(model.mapper)
    model.setObjective(objective, sense=gp.GRB.MINIMIZE)

    try:
        model.optimize()
    except gp.GurobiError as e:
        pytest.skip(f"Skipping test due to {e}")

    assert model.Status == gp.GRB.OPTIMAL

    solution = model.solution
    assert solution is not None

    x = solution.to_numpy()

    _check_solution(solution)

    for tree in model.trees:
        _check_node(tree, tree.root, solution=solution)

    _check_paths(clf, x, model.estimators)

    for class_ in (0, 1):
        model.set_majority_class(m_class=class_)

        try:
            model.optimize()
        except gp.GurobiError as e:
            pytest.skip(f"Skipping test due to {e}")

        if model.Status != gp.GRB.OPTIMAL:
            continue

        solution = model.solution
        assert solution is not None

        x = solution.to_numpy()

        _check_paths(clf, x, model.estimators)
        _check_prediction(
            clf,
            solution,
            m_class=class_,
            model=model,
        )

        model.clear_majority_class()
