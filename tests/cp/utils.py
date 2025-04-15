from collections import defaultdict
from typing import TYPE_CHECKING, Literal, overload

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from ocean.abc import Mapper
from ocean.cp import ENV, Explanation, Model, TreeVar
from ocean.feature import Feature
from ocean.typing import Array1D, NonNegativeInt

from ..utils import generate_data

if TYPE_CHECKING:
    from ocean.typing import Key


def check_solution(x: Array1D, explanation: Explanation) -> None:
    n = explanation.n_columns
    x_sol = explanation.x
    for i in range(n):
        name = explanation.names[i]
        # For now we only check the non continuous features
        # as the continuous features are epsilon away from
        # the explanation
        if not explanation[name].is_continuous:
            assert np.isclose(x[i], x_sol[i])


def validate_solution(explanation: Explanation) -> None:
    x = explanation.x
    n = explanation.n_columns
    codes: dict[Key, float] = defaultdict(float)
    for i in range(n):
        name = explanation.names[i]
        feature = explanation[name]
        value = x[i]
        if feature.is_one_hot_encoded:
            assert np.any(np.isclose(value, [0.0, 1.0]))
            codes[name] += value

        if feature.is_binary:
            assert np.any(np.isclose(value, [0.0, 1.0]))
        elif feature.is_continuous:
            assert feature.levels[0] <= value <= feature.levels[-1]
        elif feature.is_discrete:
            assert np.any(np.isclose(value, feature.levels)), (
                f"Value {value} not in levels {feature.levels}"
            )

    for value in codes.values():
        assert np.isclose(value, 1.0)


def check_leafs(tree: TreeVar, explanation: Explanation) -> None:
    n_active = 0
    id_leaf = tree.root.node_id
    solver = ENV.solver
    for node in tree.leaves:
        assert node.is_leaf
        v = solver.Value(tree[node.node_id])
        n_active += v
        id_leaf = node.node_id if v else id_leaf

    assert n_active == 1, (
        f"Expected one leaf to be active, but {n_active} were found."
    )
    x_id_leaf = find_leaf(tree, explanation)
    assert id_leaf == x_id_leaf, (
        f"Expected leaf {id_leaf}, but found {x_id_leaf}."
        f" explanation {explanation}"
    )


def find_leaf(tree: TreeVar, explanation: Explanation) -> NonNegativeInt:
    node = tree.root
    x = explanation.x
    while not node.is_leaf:
        name = node.feature
        if explanation[name].is_one_hot_encoded:
            code = node.code
            i = explanation.idx.get(name, code)
            value = x[i]
        else:
            i = explanation.idx.get(name)
            value = x[i]

        if explanation[name].is_numeric:
            threshold = node.threshold
            node = node.left if value <= threshold else node.right
        elif np.isclose(value, 0.0):
            node = node.left
        else:
            node = node.right
    return node.node_id


def validate_path(tree: TreeVar, explanation: Explanation) -> None:
    check_leafs(tree, explanation)


def validate_paths(*trees: TreeVar, explanation: Explanation) -> None:
    for tree in trees:
        validate_path(tree, explanation)


def validate_sklearn_paths(
    clf: RandomForestClassifier,
    explanation: Explanation,
    trees: tuple[TreeVar, ...],
) -> None:
    x = explanation.x.reshape(1, -1)
    leaf_ids = clf.apply(x)  # pyright: ignore[reportUnknownVariableType]
    solver = ENV.solver
    for t, tree in enumerate(trees):
        # Get the leaf node from the tree
        leaf_id = leaf_ids[0, t]
        v = solver.Value(tree[leaf_id])
        active_leaf = leaf_id
        for node in tree.leaves:
            v_1 = solver.Value(tree[node.node_id])
            if v_1 == 1.0:
                active_leaf = node.node_id
                break
        lf = find_leaf(tree, explanation)
        assert active_leaf == lf
        assert v == 1.0, (
            f"Expected leaf {leaf_id} to be active, but found {active_leaf}, "
            f"in tree {t}"
        )


def validate_sklearn_pred(
    clf: RandomForestClassifier,
    explanation: Explanation,
    m_class: NonNegativeInt,
    model: Model,
) -> None:
    x = explanation.x.reshape(1, -1)
    prediction = np.asarray(clf.predict(x), dtype=np.int64)
    solver = ENV.solver
    values = [solver.Value(model.function[key]) for key in model.function]
    function = np.asarray(values, dtype=np.float64)
    proba = function / np.sum(function)
    expected_proba = np.asarray(clf.predict_proba(x), dtype=np.float64)
    assert (prediction == m_class).all()
    assert np.isclose(expected_proba.flatten(), proba).all(), (
        f"Expected {expected_proba.flatten()}, got {proba}"
    )


@overload
def train_rf(
    seed: int,
    n_estimators: int,
    max_depth: int,
    n_samples: int,
    n_classes: int,
    *,
    return_data: Literal[False] = False,
) -> tuple[RandomForestClassifier, Mapper[Feature]]: ...


@overload
def train_rf(
    seed: int,
    n_estimators: int,
    max_depth: int,
    n_samples: int,
    n_classes: int,
    *,
    return_data: Literal[True],
) -> tuple[RandomForestClassifier, Mapper[Feature], pd.DataFrame]: ...


def train_rf(
    seed: int,
    n_estimators: int,
    max_depth: int,
    n_samples: int,
    n_classes: int,
    *,
    return_data: bool = False,
) -> (
    tuple[RandomForestClassifier, Mapper[Feature]]
    | tuple[RandomForestClassifier, Mapper[Feature], pd.DataFrame]
):
    data, y, mapper = generate_data(seed, n_samples, n_classes)
    clf = RandomForestClassifier(
        random_state=seed,
        n_estimators=n_estimators,
        max_depth=max_depth,
    )
    clf.fit(data, y)
    if return_data:
        return clf, mapper, data
    return clf, mapper


SEEDS = [43, 44, 45]
N_ESTIMATORS = [1, 4, 8]
MAX_DEPTH = [2, 3]
N_CLASSES = [2, 4]
N_SAMPLES = [100, 200, 500]
