from collections import defaultdict
from typing import TYPE_CHECKING, Literal, cast, overload

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier

from ocean.abc import Mapper
from ocean.feature import Feature
from ocean.mip import Explanation, Model, TreeVar
from ocean.tree import Node
from ocean.typing import Array1D, NonNegativeInt

from ..utils import generate_data

if TYPE_CHECKING:
    import scipy.sparse as sp

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
        elif feature.is_numeric:
            assert feature.levels[0] <= value <= feature.levels[-1]
            if feature.is_discrete:
                assert np.any(np.isclose(value, feature.levels))

    for value in codes.values():
        assert np.isclose(value, 1.0)


def check_node(tree: TreeVar, node: Node, explanation: Explanation) -> None:
    if node.is_leaf:
        return

    left = node.left
    right = node.right
    x = explanation.x
    next_node = left if tree[left.node_id].X == 1.0 else right
    assert tree[next_node.node_id].X == 1.0

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
        if value <= threshold:
            assert tree[left.node_id].X == 1.0
        else:
            assert tree[right.node_id].X == 1.0
    elif np.isclose(value, 0.0):
        assert tree[left.node_id].X == 1.0
    else:
        assert tree[right.node_id].X == 1.0

    check_node(tree, next_node, explanation=explanation)


def validate_path(tree: TreeVar, explanation: Explanation) -> None:
    check_node(tree, tree.root, explanation=explanation)


def validate_paths(*trees: TreeVar, explanation: Explanation) -> None:
    for tree in trees:
        validate_path(tree, explanation)


def validate_sklearn_paths(
    clf: RandomForestClassifier,
    explanation: Explanation,
    trees: tuple[TreeVar, ...],
) -> None:
    x = explanation.x.reshape(1, -1)
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


def validate_sklearn_pred(
    clf: RandomForestClassifier,
    explanation: Explanation,
    m_class: NonNegativeInt,
    model: Model,
) -> None:
    x = explanation.x.reshape(1, -1)
    prediction = np.asarray(clf.predict(x), dtype=np.int64)
    function = np.asarray(model.function.getValue(), dtype=np.float64)
    proba = function / np.sum(function)
    expected_proba = np.asarray(clf.predict_proba(x), dtype=np.float64)
    assert (prediction == m_class).all()
    assert np.isclose(expected_proba.flatten(), proba).all()


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


@overload
def train_rf_isolation(
    seed: int,
    n_estimators: int,
    max_depth: int,
    n_isolators: int,
    max_samples: int,
    n_samples: int,
    n_classes: int,
    *,
    return_data: Literal[False] = False,
) -> tuple[RandomForestClassifier, IsolationForest, Mapper[Feature]]: ...


@overload
def train_rf_isolation(
    seed: int,
    n_estimators: int,
    max_depth: int,
    n_isolators: int,
    max_samples: int,
    n_samples: int,
    n_classes: int,
    *,
    return_data: Literal[True],
) -> tuple[
    RandomForestClassifier,
    IsolationForest,
    Mapper[Feature],
    pd.DataFrame,
]: ...


def train_rf_isolation(
    seed: int,
    n_estimators: int,
    max_depth: int,
    n_isolators: int,
    max_samples: int,
    n_samples: int,
    n_classes: int,
    *,
    return_data: bool = False,
) -> (
    tuple[RandomForestClassifier, IsolationForest, Mapper[Feature]]
    | tuple[
        RandomForestClassifier,
        IsolationForest,
        Mapper[Feature],
        pd.DataFrame,
    ]
):
    data, y, mapper = generate_data(seed, n_samples, n_classes)
    clf = RandomForestClassifier(
        random_state=seed,
        n_estimators=n_estimators,
        max_depth=max_depth,
    )
    clf.fit(data, y)
    ilf = IsolationForest(
        random_state=seed,
        n_estimators=n_isolators,
        max_samples=max_samples,  # pyright: ignore[reportArgumentType]
    )
    ilf.fit(data)
    if return_data:
        return clf, ilf, mapper, data
    return clf, ilf, mapper


SEEDS = [43, 44, 45]
N_ESTIMATORS = [1, 4, 8]
MAX_DEPTH = [2, 3]
N_CLASSES = [2, 4]
N_SAMPLES = [100, 200, 500]
N_ISOLATORS = [1, 2, 4]
MAX_SAMPLES = [4, 8]
