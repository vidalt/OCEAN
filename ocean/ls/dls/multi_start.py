from __future__ import annotations

import time
from typing import TYPE_CHECKING, cast

import numpy as np

from ..utils.initialisation import (
    gini_grid_perturbation_initialisation,
    multi_start_initialisation,
    naive_perturbation_initialisation,
    start_point_diagnostics,
)
from ..utils.tools import point2cell
from .local_search import BestLeafResult, deterministic_local_search
from .simulated_annealing import (
    simulated_annealing,
    simulated_annealing_exhaustive,
)

if TYPE_CHECKING:
    from ...typing import LocalSearchExplainer

type LeafBounds = tuple[np.ndarray, np.ndarray]
type Callback = list[dict[str, float]]
type StartLog = list[dict[str, object]]
type MultiStartResult = (
    tuple[Callback, LeafBounds | None, int | None, float | np.float32]
    | tuple[LeafBounds | None, int | None, float | np.float32]
)


def multi_start_deterministic(  # noqa: C901, PLR0913, PLR0914, PLR0917
    exp: LocalSearchExplainer,
    n_iter: int,
    std: float,
    n_population: int,
    query: np.ndarray,
    query_class: int,
    lambda_: float,
    tabu_size: int,
    global_best_distance: float | np.float32,
    seed: int,
    timeout: float | None,
    store: bool,  # noqa: FBT001
    norm: int,
    init_type: str = "simple",
    k_features: int = 5,
    flip_prob: float = 0.5,
    perturb_ratio: float = 1.0,
    per_start_log: StartLog | None = None,
    total_timeout: float | None = None,
) -> MultiStartResult:
    np.random.seed(seed)  # noqa: NPY002
    S_0 = np.array(
        point2cell(query, exp.offsets, exp.lengths_list, exp.thresholds_concat),
        dtype=np.int32,
    )

    # Initialization based on init_type.
    if init_type == "simple":
        points = multi_start_initialisation(
            query,
            S_0,
            exp.continuous_col,
            exp.binary_col,
            exp.discrete_col,
            exp.one_hot_encoded_col,
            exp.lengths_list,
            exp.offsets,
            exp.thresholds_concat,
            n_population,
            std,
        )
    elif init_type == "gini":
        points = gini_grid_perturbation_initialisation(
            exp, query, n_population, k_features, std
        )
    elif init_type == "naive":
        points = naive_perturbation_initialisation(
            exp,
            query,
            n_population,
            std=std,
            flip_prob=flip_prob,
            perturb_ratio=perturb_ratio,
        )
    else:
        msg = (
            f"Unknown init_type: {init_type}. "
            "Choose 'simple', 'gini', or 'naive'."
        )
        raise ValueError(msg)

    global_best_leaf = None
    global_best_label = None
    global_best_distance = exp.max_distance

    # Per-start instrumentation (no RNG draws -> search trajectory is unchanged)
    init_labels = exp.rf.predict(points) if per_start_log is not None else None

    callbacks: Callback = []
    start_time = time.time()
    for k, start_point in enumerate(points):
        tau = time.time()

        # timeouts
        if total_timeout is not None and (tau - start_time) >= total_timeout:
            break
        ls_timeout = timeout
        if total_timeout is not None:
            remaining = total_timeout - (tau - start_time)
            ls_timeout = (
                remaining if timeout is None else min(timeout, remaining)
            )

        prev_best = global_best_distance

        # Run local searches.
        best_leaf, best_label, best_distance = cast(
            "BestLeafResult",
            deterministic_local_search(
                exp,
                n_iter,
                start_point,
                query,
                query_class,
                norm,
                lambda_,
                tabu_size,
                global_best_distance,
                seed,
                ls_timeout,
                store=False,
            ),
        )

        ls_valid = bool(best_leaf) and (best_label != query_class)
        if ls_valid and best_distance <= global_best_distance:
            global_best_leaf = best_leaf
            global_best_label = best_label
            global_best_distance = best_distance

        # Initial population logs.
        if per_start_log is not None:
            if init_labels is None:
                msg = "Initial labels must be available for per-start logging."
                raise RuntimeError(msg)
            diag = start_point_diagnostics(exp, query, start_point, norm)
            per_start_log.append({
                "start_idx": int(k),
                "init_valid": bool(init_labels[k] != query_class),
                "init_label": int(init_labels[k]),
                "dist_init": diag["dist_init"],
                "cont_l1": diag["cont_l1"],
                "disc_l1": diag["disc_l1"],
                "n_bin_flips": diag["n_bin_flips"],
                "n_cat_changes": diag["n_cat_changes"],
                "segment": diag["segment"],
                "ls_valid": ls_valid,
                "ls_dist": float(best_distance) if ls_valid else None,
                "improved": bool(ls_valid and best_distance < prev_best - 1e-9),
                "global_best_after": float(global_best_distance),
                "time": float(tau - start_time),
            })

        if not best_leaf:
            continue

        if store:
            callbacks.append({
                "objective_value": float(global_best_distance),
                "time": tau - start_time,
            })

    return (
        (callbacks, global_best_leaf, global_best_label, global_best_distance)
        if store
        else (global_best_leaf, global_best_label, global_best_distance)
    )


def multi_start_simulated_annealing(  # noqa: PLR0913, PLR0917
    exp: LocalSearchExplainer,
    n_iter: int,
    std: float,
    n_population: int,
    query: np.ndarray,
    query_class: int,
    lambda_: float,
    global_best_distance: float | np.float32,
    seed: int,
    timeout: float | None,
    store: bool,  # noqa: FBT001
    norm: int,
    T_init: float = 1.0,
    T_min: float = 0.001,
    alpha: float = 0.95,
    schedule: str = "exponential",
    M_k: int = 10,
) -> MultiStartResult:
    np.random.seed(seed)  # noqa: NPY002

    S_0 = np.array(
        point2cell(query, exp.offsets, exp.lengths_list, exp.thresholds_concat),
        dtype=np.int32,
    )

    points = multi_start_initialisation(
        query,
        S_0,
        exp.continuous_col,
        exp.binary_col,
        exp.discrete_col,
        exp.one_hot_encoded_col,
        exp.lengths_list,
        exp.offsets,
        exp.thresholds_concat,
        n_population,
        std,
    )

    global_best_leaf = None
    global_best_label = None
    global_best_distance = exp.max_distance

    callbacks: Callback = []
    start_time = time.time()

    for start_point in points:
        tau = time.time()

        # Run SA from this start point.
        best_leaf, best_label, best_distance = cast(
            "BestLeafResult",
            simulated_annealing(
                exp,
                n_iter,
                start_point,
                query,
                query_class,
                norm,
                lambda_,
                global_best_distance,
                seed,
                timeout,
                T_init,
                T_min,
                alpha,
                schedule,
                M_k,
                store=False,
            ),
        )

        if best_leaf is None:
            continue

        if best_label != query_class and best_distance <= global_best_distance:
            global_best_leaf = best_leaf
            global_best_label = best_label
            global_best_distance = best_distance

        if store:
            callbacks.append({
                "objective_value": float(global_best_distance),
                "time": tau - start_time,
            })

    if store:
        return (
            callbacks,
            global_best_leaf,
            global_best_label,
            global_best_distance,
        )
    return global_best_leaf, global_best_label, global_best_distance


def multi_start_simulated_annealing_exhaustive(  # noqa: PLR0913, PLR0917
    exp: LocalSearchExplainer,
    n_iter: int,
    std: float,
    n_population: int,
    query: np.ndarray,
    query_class: int,
    lambda_: float,
    global_best_distance: float | np.float32,
    seed: int,
    timeout: float | None,
    store: bool,  # noqa: FBT001
    norm: int,
    T_init: float = 1.0,
    T_min: float = 0.001,
    alpha: float = 0.95,
    schedule: str = "exponential",
) -> MultiStartResult:
    np.random.seed(seed)  # noqa: NPY002

    S_0 = np.array(
        point2cell(query, exp.offsets, exp.lengths_list, exp.thresholds_concat),
        dtype=np.int32,
    )

    points = multi_start_initialisation(
        query,
        S_0,
        exp.continuous_col,
        exp.binary_col,
        exp.discrete_col,
        exp.one_hot_encoded_col,
        exp.lengths_list,
        exp.offsets,
        exp.thresholds_concat,
        n_population,
        std,
    )

    global_best_leaf = None
    global_best_label = None
    global_best_distance = exp.max_distance

    callbacks: Callback = []
    start_time = time.time()

    for start_point in points:
        tau = time.time()

        # Run exhaustive SA from this start point.
        best_leaf, best_label, best_distance = cast(
            "BestLeafResult",
            simulated_annealing_exhaustive(
                exp,
                n_iter,
                start_point,
                query,
                query_class,
                norm,
                lambda_,
                global_best_distance,
                seed,
                timeout,
                T_init,
                T_min,
                alpha,
                schedule,
                store=False,
            ),
        )

        if best_leaf is None:
            continue

        if best_label != query_class and best_distance <= global_best_distance:
            global_best_leaf = best_leaf
            global_best_label = best_label
            global_best_distance = best_distance

        if store:
            callbacks.append({
                "objective_value": float(global_best_distance),
                "time": tau - start_time,
            })

    if store:
        return (
            callbacks,
            global_best_leaf,
            global_best_label,
            global_best_distance,
        )
    return global_best_leaf, global_best_label, global_best_distance
