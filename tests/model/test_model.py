from collections.abc import Hashable, Mapping

import gurobipy as gp
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from ocean.feature import FeatureMapper
from ocean.model.model import Model
from ocean.tree import Node, TreeVar, parse_tree

from ..utils import ENV, generate_data


def _check_solution(
    solution: Mapping[Hashable, float],
    mapper: FeatureMapper,
) -> None:
    for name, feature in mapper.items():
        if feature.is_one_hot_encoded:
            assert sum(solution[name, code] for code in feature.codes) == 1.0
        elif feature.is_binary:
            assert np.any(np.isclose(solution[name], [0.0, 1.0]))
        elif feature.is_numeric:
            assert feature.levels[0] <= solution[name] <= feature.levels[-1]
            if feature.is_discrete:
                assert np.any(np.isclose(solution[name], feature.levels))


def _check_node(
    tree: TreeVar,
    node: Node,
    *,
    mapper: FeatureMapper,
    solution: Mapping[Hashable, float],
) -> None:
    if node.is_leaf:
        return

    left = node.left
    right = node.right
    next_node = left if tree[left.node_id].X == 1.0 else right
    assert tree[next_node.node_id].X == 1.0

    feature = node.feature
    if mapper[feature].is_binary:
        if solution[feature] == 0:
            assert tree[left.node_id].X == 1.0
        else:
            assert tree[right.node_id].X == 1.0
    elif mapper[feature].is_one_hot_encoded:
        code = node.code
        if solution[feature, code] == 1.0:
            assert tree[right.node_id].X == 1.0
        else:
            assert tree[left.node_id].X == 1.0
    elif mapper[feature].is_numeric:
        threshold = node.threshold
        if solution[feature] <= threshold:
            assert tree[left.node_id].X == 1.0, (
                f"{solution[feature]} <= {threshold}"
            )
        else:
            assert tree[right.node_id].X == 1.0, (
                f"{solution[feature]} > {threshold}"
            )
    _check_node(tree, next_node, mapper=mapper, solution=solution)


def _check_paths(
    clf: RandomForestClassifier,
    solution: Mapping[Hashable, float],
    trees: tuple[TreeVar, ...],
    mapper: FeatureMapper,
) -> None:
    solution_: dict[tuple[Hashable, Hashable | str], float] = {}
    for name, feature in mapper.items():
        if not feature.is_one_hot_encoded:
            solution_[name, ""] = solution[name]
            continue

        for code in feature.codes:
            key = (name, code)
            solution_[name, code] = solution[key]

    x = pd.DataFrame([solution_], columns=mapper.columns)
    paths = clf.decision_path(x.to_numpy())  # pyright: ignore[reportUnknownVariableType]

    for t, tree in enumerate(trees):
        # Get the leaf node from the tree
        node = tree.root
        while not node.is_leaf:
            to = "left" if tree[node.left.node_id].X == 1.0 else "right"
            node = node.left if tree[node.left.node_id].X == 1.0 else node.right
            idx = paths[0][0, paths[1][t] + node.node_id]  # pyright: ignore[reportIndexIssue, reportUnknownVariableType]
            assert idx == 1, (
                "The paths break for:"
                f"tree={t}, node={node.parent.node_id}, "
                f"feature={node.parent.feature}\n"
                f"Model levels={mapper[node.parent.feature].levels}\n"
                f"Model threshold={node.parent.threshold}\n"
                f"Solution={solution[node.parent.feature]}\n"
                f"Path went to {to}"
            )


def _check_prediction(
    clf: RandomForestClassifier,
    solution: Mapping[Hashable, float],
    mapper: FeatureMapper,
    m_class: int,
    model: Model | None = None,
) -> None:
    solution_: dict[tuple[Hashable, Hashable | str], float] = {}
    for name, feature in mapper.items():
        if not feature.is_one_hot_encoded:
            solution_[name, ""] = solution[name]
            continue

        for code in feature.codes:
            key = (name, code)
            solution_[name, code] = solution[key]

    x = pd.DataFrame([solution_], columns=mapper.columns)
    prediction = clf.predict(x.to_numpy())[0]
    assert prediction == m_class, (
        clf.predict_proba(x.to_numpy())[0],
        (
            model.function.getValue() / model.function.getValue().sum()
            if model is not None
            else None
        ),
    )


def test_model_init() -> None:
    model = Model(
        trees=[],
        features={},
        weights=None,
        env=ENV,
    )
    assert model is not None
    assert model.n_trees == 0

    seed = 42
    n_estimators = 10
    max_depth = 4
    n_classes = 2
    n_samples = 100
    data, y, mapper = generate_data(seed, n_samples, n_classes)
    clf = RandomForestClassifier(
        random_state=seed,
        n_estimators=n_estimators,
        max_depth=max_depth,
    )
    clf.fit(data.to_numpy(), y)
    trees_ = [est.tree_ for est in clf.estimators_]
    trees = [parse_tree(tree, mapper=mapper) for tree in trees_]
    model = Model(
        trees=trees,
        features=mapper,
        weights=None,
        model_type=Model.Type.MIP,
        env=ENV,
    )
    assert model is not None
    assert model.n_trees == n_estimators
    assert model.n_classes == n_classes

    with pytest.raises(
        ValueError,
        match=r"The number of weights must match the number of trees.",
    ):
        model = Model(
            trees=trees,
            features=mapper,
            weights=np.ones(n_estimators + 1, dtype=float),
            model_type=Model.Type.MIP,
            env=ENV,
        )


@pytest.mark.parametrize("seed", [42, 43, 44])
@pytest.mark.parametrize("n_estimators", [10])
@pytest.mark.parametrize("max_depth", [2, 3, 4])
@pytest.mark.parametrize("n_classes", [2, 3, 4])
@pytest.mark.parametrize("n_samples", [100, 200, 500])
def test_model(
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
    trees_ = [est.tree_ for est in clf.estimators_]
    trees = [parse_tree(tree, mapper=mapper) for tree in trees_]
    weights = (np.ones(n_estimators, dtype=float) * 1e5).flatten()
    model = Model(
        trees=trees,
        features=mapper,
        weights=weights,
        model_type=Model.Type.MIP,
        env=ENV,
    )
    assert model is not None
    assert model.n_trees == n_estimators
    assert model.n_classes == n_classes

    model.build()

    objective = gp.QuadExpr()
    for feature in model.features.values():
        if feature.is_one_hot_encoded:
            for j, code in enumerate(feature.codes):
                if j == 0:
                    objective += (0.75 - feature[code]) ** 2
        elif feature.is_numeric:
            level = feature.levels.mean()
            objective += (level - feature.x) ** 2
        else:
            objective += (0.0 - feature.x) ** 2

    model.optimize()
    assert model.Status == gp.GRB.OPTIMAL

    solution = model.solution
    assert solution is not None

    _check_solution(solution, mapper=mapper)

    for tree in model.trees:
        _check_node(tree, tree.root, mapper=mapper, solution=solution)

    _check_paths(clf, solution, model.trees, mapper)

    model.set_majority_class(m_class=0)

    model.optimize()

    assert model.Status == gp.GRB.OPTIMAL

    solution = model.solution
    assert solution is not None

    _check_paths(clf, solution, model.trees, mapper)
    _check_prediction(clf, solution, mapper=mapper, m_class=0)

    model.clear_majority_class()

    model.set_majority_class(m_class=1)

    model.optimize()

    assert model.Status == gp.GRB.OPTIMAL

    solution = model.solution
    assert solution is not None

    _check_paths(clf, solution, model.trees, mapper)
    _check_prediction(clf, solution, mapper=mapper, m_class=1, model=model)
