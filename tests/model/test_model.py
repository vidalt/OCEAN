from collections.abc import Hashable, Mapping

import gurobipy as gp
import numpy as np
import pytest
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from ocean.feature import FeatureMapper
from ocean.mip import FeatureVar, Model, Solution, TreeVar
from ocean.tree import Node, parse_trees

from ..utils import ENV, generate_data


def _check_solution(
    solution: Solution,
    *,
    mapper: FeatureMapper,
) -> None:
    series = solution.to_series()
    for name, feature in mapper.items():
        if feature.is_one_hot_encoded:
            assert series.xs(name, level=0).sum() == 1.0
            continue

        value = series.xs(name, level=0).to_numpy()[0]
        if feature.is_binary:
            assert np.any(np.isclose(value, [0.0, 1.0]))
        elif feature.is_numeric:
            assert feature.levels[0] <= value <= feature.levels[-1]
            if feature.is_discrete:
                assert np.any(np.isclose(value, feature.levels))


def _check_node(
    tree: TreeVar,
    node: Node,
    *,
    mapper: FeatureMapper,
    solution: Solution,
) -> None:
    if node.is_leaf:
        return

    left = node.left
    right = node.right
    series = solution.to_series()
    next_node = left if tree[left.node_id].X == 1.0 else right
    assert tree[next_node.node_id].X == 1.0

    feature = node.feature
    if mapper[feature].is_one_hot_encoded:
        code = node.code
        value = series.xs((feature, code), level=(0, 1)).to_numpy()[0]
        if value == 0.0:
            assert tree[left.node_id].X == 1.0
        else:
            assert tree[right.node_id].X == 1.0
    else:
        value = series.xs(feature, level=0).to_numpy()[0]
        if mapper[feature].is_binary:
            if value == 0.0:
                assert tree[left.node_id].X == 1.0
            else:
                assert tree[right.node_id].X == 1.0
        elif mapper[feature].is_numeric:
            threshold = node.threshold
            if value <= threshold:
                assert tree[left.node_id].X == 1.0, f"{value} <= {threshold}"
            else:
                assert tree[right.node_id].X == 1.0, f"{value} > {threshold}"
    _check_node(tree, next_node, mapper=mapper, solution=solution)


def _check_paths(
    clf: RandomForestClassifier,
    X: np.ndarray[tuple[int], np.dtype[np.float64]],
    trees: tuple[TreeVar, ...],
) -> None:
    paths = clf.decision_path(X.reshape(1, -1))  # pyright: ignore[reportUnknownVariableType]

    for t, tree in enumerate(trees):
        # Get the leaf node from the tree
        node = tree.root
        while not node.is_leaf:
            node = node.left if tree[node.left.node_id].X == 1.0 else node.right
            idx: np.float64 = paths[0][0, paths[1][t] + node.node_id]  # pyright: ignore[reportIndexIssue, reportUnknownVariableType]
            assert idx == 1


def _check_prediction(
    clf: RandomForestClassifier,
    solution: Solution,
    mapper: FeatureMapper,
    m_class: int,
    model: Model | None = None,
) -> None:
    x = solution.to_numpy(columns=mapper.columns)
    prediction = clf.predict(x.reshape(1, -1))[0]  # pyright: ignore[reportUnknownVariableType]
    assert prediction == m_class, (
        clf.predict_proba(x.reshape(1, -1))[0],
        (
            model.function.getValue() / model.function.getValue().sum()
            if model is not None
            else None
        ),
    )


def _get_objective(features: Mapping[Hashable, FeatureVar]) -> gp.QuadExpr:
    obj = gp.QuadExpr()
    for feature in features.values():
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


def test_model_init() -> None:
    with pytest.raises(
        ValueError,
        match=r"At least one tree is required.",
    ):
        model = Model(
            trees=[],
            features={},
            weights=None,
            env=ENV,
        )

    seed = 42
    n_estimators = 10
    max_depth = 4
    n_classes = 2
    n_samples = 100
    data, y, mapper = generate_data(seed, n_samples, n_classes)
    dt = DecisionTreeClassifier()
    dt.fit(data.to_numpy(), y)
    trees = tuple(parse_trees([dt], mapper=mapper))
    model = Model(
        trees=trees,
        features=mapper,
        weights=None,
        model_type=Model.Type.MIP,
        env=ENV,
    )
    assert model is not None
    assert model.n_estimators == 1

    clf = RandomForestClassifier(
        random_state=seed,
        n_estimators=n_estimators,
        max_depth=max_depth,
    )
    clf.fit(data.to_numpy(), y)
    trees = tuple(parse_trees(clf, mapper=mapper))
    model = Model(
        trees=trees,
        features=mapper,
        weights=None,
        model_type=Model.Type.MIP,
        env=ENV,
    )
    assert model is not None
    assert model.n_estimators == n_estimators
    assert model.n_classes == n_classes

    with pytest.raises(
        ValueError,
        match=r"The number of weights must match the number of trees.",
    ):
        model = Model(
            trees=trees,
            features=mapper,
            weights=np.ones(n_estimators + 1).astype(np.float64),
            model_type=Model.Type.MIP,
            env=ENV,
        )


@pytest.mark.parametrize("seed", [42, 43, 44])
@pytest.mark.parametrize("n_estimators", [10])
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
    weights = (np.ones(n_estimators, dtype=float) * 1e5).flatten()
    model = Model(
        trees=trees,
        features=mapper,
        weights=weights,
        model_type=Model.Type.MIP,
        env=ENV,
    )
    assert model is not None
    assert model.n_estimators == n_estimators
    assert model.n_classes == n_classes

    model.build()

    objective = _get_objective(model.features)
    model.setObjective(objective, sense=gp.GRB.MINIMIZE)

    try:
        model.optimize()
    except gp.GurobiError as e:
        pytest.skip(f"Skipping test due to {e}")

    assert model.Status == gp.GRB.OPTIMAL

    solution = model.solution
    assert solution is not None

    x = solution.to_numpy(columns=mapper.columns)

    _check_solution(solution, mapper=mapper)

    for tree in model.trees:
        _check_node(tree, tree.root, mapper=mapper, solution=solution)

    _check_paths(clf, x, model.trees)

    for class_ in (0, 1):
        model.set_majority_class(m_class=class_)
        try:
            model.optimize()
        except gp.GurobiError as e:
            pytest.skip(f"Skipping test due to {e}")

        assert model.Status == gp.GRB.OPTIMAL

        solution = model.solution
        assert solution is not None

        x = solution.to_numpy(columns=mapper.columns)
        _check_paths(clf, x, model.trees)
        _check_prediction(
            clf,
            solution,
            mapper=mapper,
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
        features=mapper,
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

    objective = _get_objective(model.features)
    model.setObjective(objective, sense=gp.GRB.MINIMIZE)

    try:
        model.optimize()
    except gp.GurobiError as e:
        pytest.skip(f"Skipping test due to {e}")

    assert model.Status == gp.GRB.OPTIMAL

    solution = model.solution
    assert solution is not None

    x = solution.to_numpy(columns=mapper.columns)

    _check_solution(solution, mapper=mapper)

    for tree in model.trees:
        _check_node(tree, tree.root, mapper=mapper, solution=solution)

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

        x = solution.to_numpy(columns=mapper.columns)

        _check_paths(clf, x, model.estimators)
        _check_prediction(
            clf,
            solution,
            mapper=mapper,
            m_class=class_,
            model=model,
        )

        model.clear_majority_class()
