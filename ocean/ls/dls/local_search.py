import time

import numpy as np

from ...typing import LocalSearchExplainer
from ..utils.leaves import (
    HillClimbResult,
    feasible_leaf,
    filtered_hill_climbing_dls_numba,
    leaf_numba,
)
from ..utils.tools import set_seed
from .neighborhoods import optimal_cell_adjacents

type LeafBounds = tuple[np.ndarray, np.ndarray]
type Callback = list[dict[str, float]]
type InitialLeafResult = tuple[
    float | np.float32,
    float | np.float32,
    float | np.float32,
    LeafBounds,
    tuple[np.ndarray, np.ndarray],
    int,
]
type StoredLocalSearchResult = tuple[Callback, list[LeafBounds], list[int]]
type BestLeafResult = tuple[LeafBounds | None, int | None, float | np.float32]
type LocalSearchResult = (
    InitialLeafResult | StoredLocalSearchResult | BestLeafResult
)


def deterministic_local_search(  # noqa: PLR0914
    exp: LocalSearchExplainer,
    n_iter: int,
    point: np.ndarray,
    query: np.ndarray,
    query_class: int,
    norm: int,
    lambda_: float,
    tabu_size: int,
    global_best_distance: float | np.float32,
    seed: int,
    timeout: float | None,
    store: bool = False,  # noqa: FBT001, FBT002
) -> LocalSearchResult:
    set_seed(seed)
    np.random.seed(seed)  # noqa: NPY002

    callback: Callback = []
    sols: list[LeafBounds] = []
    labels: list[int] = []

    # Initialization
    node_indicator, tree_ptrs = exp.rf.decision_path([point])
    indptr, indices = node_indicator.indptr, node_indicator.indices
    global_nodes = indices[indptr[0] : indptr[1]]
    cost, a_idx, b_idx, a, b, distance, query_prob, label, leaves_rank = (
        leaf_numba(
            point,
            query,
            query_class,
            exp.max_distance,
            exp.continuous_col,
            exp.binary_col,
            exp.discrete_col,
            exp.one_hot_encoded_col,
            lambda_,
            exp.rf.n_estimators,
            tree_ptrs,
            global_nodes,
            exp.features_,
            exp.thresholds_,
            exp.values_,
            exp.children_left_,
            exp.children_right_,
            exp.thresh2idx,
            exp.inf,
            exp.sup,
            norm,
            exp.weights,
            exp.base_scores,
            exp.score_kind,
            exp.normalize_leaf_values,
            exp.rank_maps,
        )
    )
    hash_id = exp.encode(leaves_rank)

    if n_iter == 0:
        return cost, distance, query_prob, (a, b), (a_idx, b_idx), label

    validity = feasible_leaf(
        a,
        b,
        exp.continuous_col,
        exp.binary_col,
        exp.discrete_col,
        exp.one_hot_encoded_col,
        exp.inf,
        exp.sup,
    )

    if label != query_class and validity:
        # The start point is already a valid counterfactual.
        global_best_leaf = (a, b)
        global_best_label = label
        global_best_distance = distance
    else:
        # No valid solution yet.
        global_best_leaf = None
        global_best_label = None
        global_best_distance = exp.max_distance

    tabu_transitions = (
        np.full((tabu_size, 2), -1, dtype=np.int64) if tabu_size > 0 else None
    )
    tabu_idx = 0

    transitions: dict[frozenset[int | np.int64], int] = {}

    # Main loop
    start_time = time.time()
    for _ in range(n_iter):
        tau = time.time()

        if timeout is not None and (tau - start_time) > timeout:
            break

        prev_id = hash_id
        (
            cost,
            a_idx,
            b_idx,
            a,
            b,
            distance,
            query_prob,
            label,
            leaves_rank,
            hash_id,
            validity,
            _validity_proportion,
            _seen_proportion,
        ) = hill_climbing(
            a_idx,
            b_idx,
            a,
            b,
            distance,
            query_prob,
            label,
            hash_id,
            validity,
            query,
            query_class,
            exp,
            lambda_,
            norm,
            tabu_transitions,
        )

        if tabu_transitions is not None:
            tabu_transitions[tabu_idx, 0] = prev_id
            tabu_transitions[tabu_idx, 1] = hash_id
            tabu_idx = (tabu_idx + 1) % tabu_size

        transitions[frozenset({prev_id, hash_id})] = (
            transitions.get(frozenset({prev_id, hash_id}), 0) + 1
        )

        if (
            label != query_class
            and distance < global_best_distance
            and validity
        ):
            global_best_leaf = (a, b)
            global_best_label = label
            global_best_distance = distance

            if store and global_best_leaf:
                callback.append({
                    "objective_value": float(global_best_distance),
                    "time": tau - start_time,
                })
                sols.append(global_best_leaf)
                labels.append(global_best_label)

    return (
        (callback, sols, labels)
        if store
        else (global_best_leaf, global_best_label, global_best_distance)
    )


def hill_climbing(
    a_idx: np.ndarray,
    b_idx: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    distance: float | np.float32,
    query_prob: float | np.float32,
    label: int,
    hash_id: int | np.int64,
    validity: bool,  # noqa: FBT001
    query: np.ndarray,
    query_class: int,
    exp: LocalSearchExplainer,
    lambda_: float,
    norm: int,
    tabu_states: np.ndarray | None,
) -> HillClimbResult:
    # Initialization
    (
        best_a_idx,
        best_b_idx,
        best_a,
        best_b,
        best_distance,
        best_query_prob,
        best_label,
        best_validity,
    ) = a_idx, b_idx, a, b, distance, query_prob, label, validity
    best_cost = 100000000

    # Neighbor cells
    points = optimal_cell_adjacents(
        a,
        b,
        exp.continuous_col,
        exp.binary_col,
        exp.discrete_col,
        exp.one_hot_encoded_col,
        exp.offsets,
        exp.thresholds_concat,
        exp.lengths_list,
        query,
        exp.max_distance,
        norm,
    )

    if len(points) == 0:
        dummy_leaves = np.zeros(exp.rf.n_estimators, dtype=np.int32)
        return (
            np.float32(best_cost),
            best_a_idx,
            best_b_idx,
            best_a,
            best_b,
            np.float32(best_distance),
            np.float32(best_query_prob),
            int(best_label),
            dummy_leaves,
            np.int64(hash_id),
            best_validity,
            0.0,
            0.0,
        )

    # Decision paths
    node_indicator, tree_ptrs = exp.rf.decision_path(points)  # vectorized pass
    indptr, indices = node_indicator.indptr, node_indicator.indices

    return filtered_hill_climbing_dls_numba(
        points,
        best_cost,
        exp.continuous_col,
        exp.binary_col,
        exp.discrete_col,
        exp.one_hot_encoded_col,
        best_a_idx,
        best_b_idx,
        best_a,
        best_b,
        exp.max_distance,
        best_distance,
        best_query_prob,
        best_label,
        hash_id,
        best_validity,
        query,
        query_class,
        lambda_,
        exp.rf.n_estimators,
        indptr,
        indices,
        tree_ptrs,
        exp.features_,
        exp.thresholds_,
        exp.values_,
        exp.children_left_,
        exp.children_right_,
        exp.thresh2idx,
        exp.inf,
        exp.sup,
        norm,
        tabu_states,
        exp.weights,
        exp.base_scores,
        exp.score_kind,
        exp.normalize_leaf_values,
        exp.rank_maps,
    )
