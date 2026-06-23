from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numba.typed.typedlist import List as NumbaList

from .costs import fitness, hypercube_cost
from .numba import njit
from .tools import hash_leaves_fnv1a

if TYPE_CHECKING:
    from collections.abc import Mapping, MutableSequence, Sequence


TREE_LEAF_FEATURE = -2

type HillClimbResult = tuple[
    np.float32,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.float32,
    np.float32,
    int,
    np.ndarray,
    np.int64,
    bool,
    float,
    float,
]


@njit(cache=True)
def filtered_get_leaf_numba(
    point: np.ndarray,
    node_index: MutableSequence[int],
    features: np.ndarray,
    thresholds: np.ndarray,
    probas: np.ndarray,
    children_left: np.ndarray,
    children_right: np.ndarray,
    inf: np.ndarray,
    sup: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    a = inf.copy()
    b = sup.copy()

    leaf_id = -1

    nb = 0
    while nb < len(node_index):
        node_id = node_index[nb]
        d = features[node_id]

        if d == TREE_LEAF_FEATURE:
            leaf_id = node_id
            nb += 1
            continue

        theta = thresholds[node_id]

        if point[d] < theta:
            b[d] = min(b[d], theta)
            nb += 1

        elif point[d] == theta:
            node_index = node_index[: nb + 1]
            while children_left[node_index[-1]] != -1:
                k = node_index[-1]
                theta_ = thresholds[k]
                d_ = features[k]
                if point[d_] <= theta_:
                    node_index.append(children_left[k])
                else:
                    node_index.append(children_right[k])

            b[d] = min(b[d], theta)
            nb += 1

        else:
            a[d] = max(a[d], theta)
            nb += 1

    p = probas[leaf_id][0]
    p_norm = p / p.sum()

    return a, b, p_norm, leaf_id


@njit(cache=True)
def feasible_leaf(
    a: np.ndarray,
    b: np.ndarray,
    continuous_col: np.ndarray,
    binary_col: np.ndarray,
    discrete_col: np.ndarray,
    _one_hot_encoded_col: Sequence[Sequence[int]],
    inf: np.ndarray,
    sup: np.ndarray,
) -> bool:
    _ = binary_col
    validity = True

    for i__ in continuous_col:
        if np.isclose(np.nextafter(a[i__], sup[i__]), b[i__]):
            validity = False

    for j in discrete_col:
        if a[j] + 0.5 == b[j] and a[j] != inf[j] and np.floor(a[j]) == a[j]:
            validity = False

    return validity


@njit(fastmath=True, cache=True)
def leaf_numba(  # noqa: PLR0913, PLR0914, PLR0917
    point: np.ndarray,
    query: np.ndarray,
    query_class: int,
    max_distance: float | np.float32,
    continuous_col: np.ndarray,
    binary_col: np.ndarray,
    discrete_col: np.ndarray,
    one_hot_encoded_col: Sequence[Sequence[int]],
    lambda_: float | np.float32,
    n_estimators: int,
    tree_ptrs: np.ndarray,
    global_nodes: np.ndarray,
    features_: Sequence[np.ndarray],
    thresholds_: Sequence[np.ndarray],
    values_: Sequence[np.ndarray],
    children_left_: Sequence[np.ndarray],
    children_right_: Sequence[np.ndarray],
    thresh2idx: Sequence[Mapping[np.float32, int]],
    inf: np.ndarray,
    sup: np.ndarray,
    norm: int,
    rank_maps: Sequence[np.ndarray],
) -> tuple[
    np.float32,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.float32,
    np.float32,
    int,
    np.ndarray,
]:
    n_features = len(point)
    n_classes = values_[0][0].shape[1]

    A = np.empty((n_estimators, n_features), dtype=np.float32)
    B = np.empty((n_estimators, n_features), dtype=np.float32)
    probas = np.empty((n_estimators, n_classes), dtype=np.float32)
    leaves = np.empty(n_estimators, dtype=np.int32)

    for j in range(n_estimators):
        off, nxt = tree_ptrs[j], tree_ptrs[j + 1]
        mask = (off <= global_nodes) & (global_nodes < nxt)
        local_nodes = global_nodes[mask] - off

        local_nodes_list: list[int] = NumbaList()  # type: ignore[assignment,no-untyped-call]
        for i_ in local_nodes:
            local_nodes_list.append(i_)  # noqa: PERF402

        a_j, b_j, p, node_j = filtered_get_leaf_numba(
            point,
            local_nodes_list,
            features_[j],
            thresholds_[j],
            values_[j],
            children_left_[j],
            children_right_[j],
            inf,
            sup,
        )
        A[j, :] = a_j
        B[j, :] = b_j
        probas[j, :] = p
        leaves[j] = rank_maps[j][node_j]

    mean_probas = np.empty(n_classes, dtype=np.float32)
    for c in range(n_classes):
        mean_probas[c] = probas[:, c].mean()

    label = int(mean_probas.argmax())
    query_prob = np.float32(mean_probas[query_class])

    a = np.empty(n_features, dtype=np.float32)
    b = np.empty(n_features, dtype=np.float32)
    for i__ in range(n_features):
        a[i__] = A[:, i__].max()
        b[i__] = B[:, i__].min()

    distance, _ = hypercube_cost(
        a,
        b,
        query,
        continuous_col,
        binary_col,
        discrete_col,
        one_hot_encoded_col,
        max_distance,
        norm,
    )

    a_idx = np.empty(len(a), dtype=np.int32)
    b_idx = np.empty(len(b), dtype=np.int32)
    for d in range(len(a)):
        a_idx[d] = thresh2idx[d][a[d]]
        b_idx[d] = thresh2idx[d][b[d]]

    cost = fitness(
        distance, query_prob, label, query_class, lambda_, max_distance
    )

    return cost, a_idx, b_idx, a, b, distance, query_prob, label, leaves


@njit(fastmath=True, cache=True)
def filtered_hill_climbing_dls_numba(  # noqa: C901, PLR0913, PLR0914, PLR0915, PLR0917
    points: np.ndarray,
    best_cost: float | np.float32,
    continuous_col: np.ndarray,
    binary_col: np.ndarray,
    discrete_col: np.ndarray,
    one_hot_encoded_col: Sequence[Sequence[int]],
    best_a_idx: np.ndarray,
    best_b_idx: np.ndarray,
    best_a: np.ndarray,
    best_b: np.ndarray,
    max_distance: float | np.float32,
    best_distance: float | np.float32,
    best_query_prob: float | np.float32,
    best_label: int,
    best_hash_id: int | np.int64,
    best_validity: bool,  # noqa: FBT001
    query: np.ndarray,
    query_class: int,
    lambda_: float | np.float32,
    n_estimators: int,
    indptr: np.ndarray,
    indices: np.ndarray,
    tree_ptrs: np.ndarray,
    features_: Sequence[np.ndarray],
    thresholds_: Sequence[np.ndarray],
    values_: Sequence[np.ndarray],
    children_left_: Sequence[np.ndarray],
    children_right_: Sequence[np.ndarray],
    thresh2idx: Sequence[Mapping[np.float32, int]],
    inf: np.ndarray,
    sup: np.ndarray,
    norm: int,
    tabu_states: np.ndarray | None,
    rank_maps: Sequence[np.ndarray],
) -> HillClimbResult:
    validity_proportion = 0
    seen_regions = 0

    n_features = len(best_a_idx)
    n_classes = values_[0][0].shape[1]

    prev_id = best_hash_id
    best_leaves = np.empty(n_estimators, dtype=np.int32)

    for i, point in enumerate(points):
        global_nodes = indices[indptr[i] : indptr[i + 1]]
        A = np.empty((n_estimators, n_features), dtype=np.float32)
        B = np.empty((n_estimators, n_features), dtype=np.float32)
        probas = np.zeros((n_estimators, n_classes), dtype=np.float32)
        leaves = np.empty(n_estimators, dtype=np.int32)

        for j in range(n_estimators):
            off, nxt = tree_ptrs[j], tree_ptrs[j + 1]
            mask = (off <= global_nodes) & (global_nodes < nxt)
            local_nodes = global_nodes[mask] - off

            local_nodes_list: list[int] = NumbaList()  # type: ignore[assignment,no-untyped-call]
            for i_ in local_nodes:
                local_nodes_list.append(i_)  # noqa: PERF402
            a_j, b_j, p, node_j = filtered_get_leaf_numba(
                point,
                local_nodes_list,
                features_[j],
                thresholds_[j],
                values_[j],
                children_left_[j],
                children_right_[j],
                inf,
                sup,
            )
            A[j, :] = a_j
            B[j, :] = b_j
            probas[j, :] = p
            leaves[j] = rank_maps[j][node_j]

        mean_probas = np.empty(n_classes, dtype=np.float32)
        for c in range(n_classes):
            mean_probas[c] = probas[:, c].mean()
        label = int(mean_probas.argmax())
        query_prob = np.float32(mean_probas[query_class])

        a = np.empty(n_features, dtype=np.float32)
        b = np.empty(n_features, dtype=np.float32)

        for i__ in range(n_features):
            val_a = A[:, i__].max()
            val_b = B[:, i__].min()
            a[i__] = val_a
            b[i__] = val_b

        validity = feasible_leaf(
            a,
            b,
            continuous_col,
            binary_col,
            discrete_col,
            one_hot_encoded_col,
            inf,
            sup,
        )
        validity_proportion += 1 * validity

        distance, _ = hypercube_cost(
            a,
            b,
            query,
            continuous_col,
            binary_col,
            discrete_col,
            one_hot_encoded_col,
            max_distance,
            norm,
        )

        cost = fitness(
            distance, query_prob, label, query_class, lambda_, max_distance
        )

        a_idx = np.empty(len(a), dtype=np.int32)
        b_idx = np.empty(len(b), dtype=np.int32)
        for d in range(len(a)):
            a_idx[d] = thresh2idx[d][a[d]]
            b_idx[d] = thresh2idx[d][b[d]]

        new_hash_id = hash_leaves_fnv1a(leaves)

        valid_transition = True
        if tabu_states is not None:
            for tabu_idx in range(tabu_states.shape[0]):
                from_ = tabu_states[tabu_idx, 0]
                to_ = tabu_states[tabu_idx, 1]
                if from_ == -1:
                    continue
                if (from_ == new_hash_id and to_ == prev_id) or (
                    from_ == prev_id and to_ == new_hash_id
                ):
                    valid_transition = False

        if cost < best_cost and valid_transition:
            best_cost = cost
            best_a_idx = a_idx
            best_b_idx = b_idx
            best_a = a
            best_b = b
            best_distance = distance
            best_query_prob = query_prob
            best_label = label
            best_validity = validity
            best_leaves = leaves
            best_hash_id = new_hash_id

    return (
        np.float32(best_cost),
        best_a_idx,
        best_b_idx,
        best_a,
        best_b,
        np.float32(best_distance),
        np.float32(best_query_prob),
        int(best_label),
        best_leaves,
        np.int64(best_hash_id),
        best_validity,
        validity_proportion / len(points),
        seen_regions / len(points),
    )


@njit(fastmath=True, cache=True)
def filtered_hill_climbing_sls_numba(  # noqa: C901, PLR0913, PLR0914, PLR0915, PLR0917
    points: np.ndarray,
    best_cost: float | np.float32,
    continuous_col: np.ndarray,
    binary_col: np.ndarray,
    discrete_col: np.ndarray,
    one_hot_encoded_col: Sequence[Sequence[int]],
    best_a_idx: np.ndarray,
    best_b_idx: np.ndarray,
    best_a: np.ndarray,
    best_b: np.ndarray,
    max_distance: float | np.float32,
    best_distance: float | np.float32,
    best_query_prob: float | np.float32,
    best_label: int,
    best_hash_id: int | np.int64,
    best_validity: bool,  # noqa: FBT001
    query: np.ndarray,
    query_class: int,
    lambda_: float | np.float32,
    n_estimators: int,
    indptr: np.ndarray,
    indices: np.ndarray,
    tree_ptrs: np.ndarray,
    features_: Sequence[np.ndarray],
    thresholds_: Sequence[np.ndarray],
    values_: Sequence[np.ndarray],
    children_left_: Sequence[np.ndarray],
    children_right_: Sequence[np.ndarray],
    thresh2idx: Sequence[Mapping[np.float32, int]],
    inf: np.ndarray,
    sup: np.ndarray,
    norm: int,
    tabu_transitions: np.ndarray | None,
    rank_maps: Sequence[np.ndarray],
) -> HillClimbResult:
    validity_proportion = 0
    seen_regions = 0

    n_features = len(best_a_idx)
    n_classes = values_[0][0].shape[1]

    prev_id = best_hash_id
    best_leaves = np.empty(n_estimators, dtype=np.int32)

    for i, point in enumerate(points):
        global_nodes = indices[indptr[i] : indptr[i + 1]]
        A = np.empty((n_estimators, n_features), dtype=np.float32)
        B = np.empty((n_estimators, n_features), dtype=np.float32)
        probas = np.zeros((n_estimators, n_classes), dtype=np.float32)
        leaves = np.empty(n_estimators, dtype=np.int32)

        for j in range(n_estimators):
            off, nxt = tree_ptrs[j], tree_ptrs[j + 1]
            mask = (off <= global_nodes) & (global_nodes < nxt)
            local_nodes = global_nodes[mask] - off

            local_nodes_list: list[int] = NumbaList()  # type: ignore[assignment,no-untyped-call]
            for i_ in local_nodes:
                local_nodes_list.append(i_)  # noqa: PERF402
            a_j, b_j, p, node_j = filtered_get_leaf_numba(
                point,
                local_nodes_list,
                features_[j],
                thresholds_[j],
                values_[j],
                children_left_[j],
                children_right_[j],
                inf,
                sup,
            )
            A[j, :] = a_j
            B[j, :] = b_j
            probas[j, :] = p
            leaves[j] = rank_maps[j][node_j]

        mean_probas = np.empty(n_classes, dtype=np.float32)
        for c in range(n_classes):
            mean_probas[c] = probas[:, c].mean()
        label = int(mean_probas.argmax())
        query_prob = np.float32(mean_probas[query_class])

        a = np.empty(n_features, dtype=np.float32)
        b = np.empty(n_features, dtype=np.float32)

        for i__ in range(n_features):
            a[i__] = A[:, i__].max()
            b[i__] = B[:, i__].min()

        validity = feasible_leaf(
            a,
            b,
            continuous_col,
            binary_col,
            discrete_col,
            one_hot_encoded_col,
            inf,
            sup,
        )
        validity_proportion += 1 * validity

        distance, _ = hypercube_cost(
            a,
            b,
            query,
            continuous_col,
            binary_col,
            discrete_col,
            one_hot_encoded_col,
            max_distance,
            norm,
        )

        cost = fitness(
            distance, query_prob, label, query_class, lambda_, max_distance
        )

        a_idx = np.empty(len(a), dtype=np.int32)
        b_idx = np.empty(len(b), dtype=np.int32)
        for d in range(len(a)):
            a_idx[d] = thresh2idx[d][a[d]]
            b_idx[d] = thresh2idx[d][b[d]]

        new_hash_id = hash_leaves_fnv1a(leaves)

        valid_transition = True
        if tabu_transitions is not None:
            for t_idx in range(tabu_transitions.shape[0]):
                from_ = tabu_transitions[t_idx, 0]
                to_ = tabu_transitions[t_idx, 1]
                if from_ == -1:
                    continue
                if (from_ == new_hash_id and to_ == prev_id) or (
                    from_ == prev_id and to_ == new_hash_id
                ):
                    valid_transition = False
                    seen_regions += 1
                    break

        if cost < best_cost and valid_transition:
            best_cost = cost
            best_a_idx = a_idx
            best_b_idx = b_idx
            best_a = a
            best_b = b
            best_distance = distance
            best_query_prob = query_prob
            best_label = label
            best_validity = validity
            best_leaves = leaves
            best_hash_id = new_hash_id

    return (
        np.float32(best_cost),
        best_a_idx,
        best_b_idx,
        best_a,
        best_b,
        np.float32(best_distance),
        np.float32(best_query_prob),
        int(best_label),
        best_leaves,
        np.int64(best_hash_id),
        best_validity,
        validity_proportion / len(points),
        seen_regions / len(points),
    )


__all__ = [
    "HillClimbResult",
    "feasible_leaf",
    "filtered_get_leaf_numba",
    "filtered_hill_climbing_dls_numba",
    "filtered_hill_climbing_sls_numba",
    "leaf_numba",
]
