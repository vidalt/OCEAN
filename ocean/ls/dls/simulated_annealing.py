import time
from collections.abc import Callable

import numpy as np

from ...typing import LocalSearchExplainer
from ..utils.leaves import feasible_leaf, leaf_numba
from ..utils.tools import set_seed
from .neighborhoods import optimal_cell_adjacents, random_single_neighbor

type LeafBounds = tuple[np.ndarray, np.ndarray]
type Callback = list[dict[str, float]]
type BestLeafResult = tuple[LeafBounds | None, int | None, float | np.float32]
type StoredSAResult = tuple[Callback, list[LeafBounds], list[int]]
type SAResult = BestLeafResult | StoredSAResult


def metropolis_criterion(
    delta_cost: float | np.float32,
    temperature: float | np.float32,
) -> bool:
    if delta_cost <= 0:
        return True

    if temperature <= 0:
        return False

    acceptance_probability = np.exp(-delta_cost / temperature)
    return bool(np.random.random() < acceptance_probability)  # noqa: NPY002


def get_temperature_schedule(
    schedule_type: str,
    T_init: float,
    T_min: float,
    n_iter: int,
    alpha: float = 0.95,
) -> Callable[[int], float]:
    if schedule_type == "exponential":

        def schedule(k: int) -> float:
            return float(max(T_min, T_init * (alpha**k)))

        return schedule

    if schedule_type == "linear":

        def schedule(k: int) -> float:
            return float(max(T_min, T_init - k * (T_init - T_min) / n_iter))

        return schedule

    if schedule_type == "logarithmic":

        def schedule(k: int) -> float:
            return float(max(T_min, T_init / (1 + np.log(1 + k))))

        return schedule

    msg = (
        f"Unknown schedule: {schedule_type}. "
        "Use 'exponential', 'linear', or 'logarithmic'."
    )
    raise ValueError(msg)


def simulated_annealing(  # noqa: C901, PLR0912, PLR0913, PLR0914, PLR0915, PLR0917
    exp: LocalSearchExplainer,
    n_iter: int,
    point: np.ndarray,
    query: np.ndarray,
    query_class: int,
    norm: int,
    lambda_: float,
    global_best_distance: float | np.float32,
    seed: int,
    timeout: float | None,
    T_init: float = 1.0,
    T_min: float = 0.001,
    alpha: float = 0.95,
    schedule: str = "exponential",
    M_k: int = 10,
    store: bool = False,  # noqa: FBT001, FBT002
) -> SAResult:
    set_seed(seed)
    np.random.seed(seed)  # noqa: NPY002

    # Anytime-mode history.
    callback: Callback = []
    sols: list[LeafBounds] = []
    labels: list[int] = []

    temp_schedule = get_temperature_schedule(
        schedule, T_init, T_min, n_iter, alpha
    )

    node_indicator, tree_ptrs = exp.rf.decision_path([point])
    indptr, indices = node_indicator.indptr, node_indicator.indices
    global_nodes = indices[indptr[0] : indptr[1]]

    cost, a_idx, b_idx, a, b, distance, _query_prob, label, leaves_rank = (
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

    exp.encode(leaves_rank)

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

    current_cost = cost
    _current_a_idx, _current_b_idx = a_idx, b_idx
    current_a, current_b = a, b
    current_distance = distance
    current_label = label
    current_validity = validity

    if label != query_class and validity:
        global_best_leaf = (a, b)
        global_best_label = label
        global_best_distance = distance
    else:
        global_best_leaf = None
        global_best_label = None
        global_best_distance = exp.max_distance

    accepted_moves = 0
    rejected_moves = 0
    accepted_worse = 0

    start_time = time.time()
    k = 0

    while k < n_iter:
        tau = time.time()
        if timeout is not None and (tau - start_time) > timeout:
            break

        temperature = temp_schedule(k)

        if temperature <= T_min:
            break

        for m in range(M_k):
            if timeout is not None and (time.time() - start_time) > timeout:
                break

            neighbor_point_raw, valid = random_single_neighbor(
                current_a,
                current_b,
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

            if not valid:
                continue

            neighbor_point = neighbor_point_raw.reshape(1, -1)

            node_indicator, tree_ptrs = exp.rf.decision_path(neighbor_point)
            indptr, indices = node_indicator.indptr, node_indicator.indices
            global_nodes = indices[indptr[0] : indptr[1]]

            (
                neighbor_cost,
                neighbor_a_idx,
                neighbor_b_idx,
                neighbor_a,
                neighbor_b,
                neighbor_distance,
                _neighbor_query_prob,
                neighbor_label,
                neighbor_leaves_rank,
            ) = leaf_numba(
                neighbor_point[0],
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

            exp.encode(neighbor_leaves_rank)
            neighbor_validity = feasible_leaf(
                neighbor_a,
                neighbor_b,
                exp.continuous_col,
                exp.binary_col,
                exp.discrete_col,
                exp.one_hot_encoded_col,
                exp.inf,
                exp.sup,
            )

            delta_cost = neighbor_cost - current_cost

            if metropolis_criterion(delta_cost, temperature):
                accepted_moves += 1
                if delta_cost > 0:
                    accepted_worse += 1

                current_cost = neighbor_cost
                _current_a_idx, _current_b_idx = neighbor_a_idx, neighbor_b_idx
                current_a, current_b = neighbor_a, neighbor_b
                current_distance = neighbor_distance
                current_label = neighbor_label
                current_validity = neighbor_validity
            else:
                rejected_moves += 1

            if (
                current_label != query_class
                and current_distance < global_best_distance
                and current_validity
            ):
                global_best_leaf = (current_a.copy(), current_b.copy())
                global_best_label = current_label
                global_best_distance = current_distance

                if store:
                    callback.append({
                        "objective_value": float(global_best_distance),
                        "time": time.time() - start_time,
                        "temperature": temperature,
                        "k": k,
                        "m": m,
                    })
                    sols.append(global_best_leaf)
                    labels.append(int(global_best_label))

        k += 1

    if store:
        return callback, sols, labels
    return global_best_leaf, global_best_label, global_best_distance


def simulated_annealing_exhaustive(  # noqa: C901, PLR0912, PLR0914, PLR0915
    exp: LocalSearchExplainer,
    n_iter: int,
    point: np.ndarray,
    query: np.ndarray,
    query_class: int,
    norm: int,
    lambda_: float,
    global_best_distance: float | np.float32,
    seed: int,
    timeout: float | None,
    T_init: float = 1.0,
    T_min: float = 0.001,
    alpha: float = 0.95,
    schedule: str = "exponential",
    store: bool = False,  # noqa: FBT001, FBT002
) -> SAResult:
    set_seed(seed)
    np.random.seed(seed)  # noqa: NPY002

    # Anytime-mode history.
    callback: Callback = []
    sols: list[LeafBounds] = []
    labels: list[int] = []

    temp_schedule = get_temperature_schedule(
        schedule, T_init, T_min, n_iter, alpha
    )

    node_indicator, tree_ptrs = exp.rf.decision_path([point])
    indptr, indices = node_indicator.indptr, node_indicator.indices
    global_nodes = indices[indptr[0] : indptr[1]]

    cost, a_idx, b_idx, a, b, distance, _query_prob, label, leaves_rank = (
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

    exp.encode(leaves_rank)

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

    current_cost = cost
    _current_a_idx, _current_b_idx = a_idx, b_idx
    current_a, current_b = a, b
    current_distance = distance
    current_label = label
    current_validity = validity

    if label != query_class and validity:
        global_best_leaf = (a, b)
        global_best_label = label
        global_best_distance = distance
    else:
        global_best_leaf = None
        global_best_label = None
        global_best_distance = exp.max_distance

    accepted_moves = 0
    rejected_moves = 0
    accepted_worse = 0
    total_neighbors_explored = 0

    start_time = time.time()
    k = 0

    while k < n_iter:
        tau = time.time()
        if timeout is not None and (tau - start_time) > timeout:
            break

        temperature = temp_schedule(k)

        if temperature <= T_min:
            break

        neighbors = optimal_cell_adjacents(
            current_a,
            current_b,
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

        n_neighbors = len(neighbors)

        if n_neighbors == 0:
            break

        shuffle_indices = np.random.permutation(n_neighbors)  # noqa: NPY002

        for m in range(n_neighbors):
            if timeout is not None and (time.time() - start_time) > timeout:
                break

            neighbor_idx = shuffle_indices[m]
            neighbor_point = neighbors[neighbor_idx : neighbor_idx + 1]

            total_neighbors_explored += 1

            node_indicator, tree_ptrs = exp.rf.decision_path(neighbor_point)
            indptr, indices = node_indicator.indptr, node_indicator.indices
            global_nodes = indices[indptr[0] : indptr[1]]

            (
                neighbor_cost,
                neighbor_a_idx,
                neighbor_b_idx,
                neighbor_a,
                neighbor_b,
                neighbor_distance,
                _neighbor_query_prob,
                neighbor_label,
                neighbor_leaves_rank,
            ) = leaf_numba(
                neighbor_point[0],
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

            exp.encode(neighbor_leaves_rank)
            neighbor_validity = feasible_leaf(
                neighbor_a,
                neighbor_b,
                exp.continuous_col,
                exp.binary_col,
                exp.discrete_col,
                exp.one_hot_encoded_col,
                exp.inf,
                exp.sup,
            )

            delta_cost = neighbor_cost - current_cost

            if metropolis_criterion(delta_cost, temperature):
                accepted_moves += 1
                if delta_cost > 0:
                    accepted_worse += 1

                current_cost = neighbor_cost
                _current_a_idx, _current_b_idx = neighbor_a_idx, neighbor_b_idx
                current_a, current_b = neighbor_a, neighbor_b
                current_distance = neighbor_distance
                current_label = neighbor_label
                current_validity = neighbor_validity
            else:
                rejected_moves += 1

            if (
                current_label != query_class
                and current_distance < global_best_distance
                and current_validity
            ):
                global_best_leaf = (current_a.copy(), current_b.copy())
                global_best_label = current_label
                global_best_distance = current_distance

                if store:
                    callback.append({
                        "objective_value": float(global_best_distance),
                        "time": time.time() - start_time,
                        "temperature": temperature,
                        "k": k,
                        "m": m,
                        "n_neighbors": n_neighbors,
                    })
                    sols.append(global_best_leaf)
                    labels.append(int(global_best_label))

        k += 1

    if store:
        return callback, sols, labels
    return global_best_leaf, global_best_label, global_best_distance
