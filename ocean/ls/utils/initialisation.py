from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict, cast

import numpy as np

from .costs import get_norm
from .tools import (
    cell_center_vectorized_numba,
    floor_discrete_idx,
    get_path_sklearn,
    point2cell,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ...typing import LocalSearchExplainer
    from .tools import DecisionPathForest

EPSILON = 1e-9


class StartDiagnostics(TypedDict):
    cont_l1: float
    disc_l1: float
    n_bin_flips: int
    n_cat_changes: int
    dist_init: float
    segment: str


# Diagnostics: composition of an initial start point relative to the query
def start_point_diagnostics(
    exp: LocalSearchExplainer,
    query: np.ndarray,
    point: np.ndarray,
    norm: int,
) -> StartDiagnostics:
    query = np.asarray(query, dtype=np.float32)
    point = np.asarray(point, dtype=np.float32)

    cont = exp.continuous_col
    disc = exp.discrete_col
    binc = exp.binary_col

    cont_l1 = (
        float(np.sum(np.abs(point[cont] - query[cont]))) if len(cont) else 0.0
    )
    disc_l1 = (
        float(np.sum(np.abs(point[disc] - query[disc]))) if len(disc) else 0.0
    )
    n_bin_flips = int(np.sum(point[binc] != query[binc])) if len(binc) else 0

    n_cat_changes = 0
    for group in exp.one_hot_encoded_col:
        g = np.asarray(group)
        if int(point[g].argmax()) != int(query[g].argmax()):
            n_cat_changes += 1

    dist_init = float(get_norm(norm, query, point))

    changed: list[str] = []
    if cont_l1 > EPSILON:
        changed.append("continuous")
    if disc_l1 > EPSILON:
        changed.append("discrete")
    if n_bin_flips > 0:
        changed.append("binary")
    if n_cat_changes > 0:
        changed.append("categorical")
    if len(changed) == 0:
        segment = "none"
    elif len(changed) == 1:
        segment = changed[0]
    else:
        segment = "mixed"

    return {
        "cont_l1": cont_l1,
        "disc_l1": disc_l1,
        "n_bin_flips": n_bin_flips,
        "n_cat_changes": n_cat_changes,
        "dist_init": dist_init,
        "segment": segment,
    }


# Simple initialization sampling
def multi_start_initialisation(  # noqa: C901, PLR0914, PLR0915
    query: np.ndarray,
    S_0: np.ndarray,
    continuous_col: np.ndarray,
    binary_col: np.ndarray,
    discrete_col: np.ndarray,
    one_hot_encoded_col: Sequence[Sequence[int]],
    lengths_list: np.ndarray,
    offsets: np.ndarray,
    thresholds_concat: np.ndarray,
    n_population: int,
    std: float,
) -> np.ndarray:
    rows = np.arange(n_population)
    sizes = {
        "continuous": len(continuous_col),
        "binary": len(binary_col),
        "discrete": len(discrete_col),
        "categorical": len(one_hot_encoded_col),
    }
    types = [t for t, size in sizes.items() if size > 0]
    weights = np.array([sizes[t] for t in types], dtype=float)
    weights /= weights.sum()
    counts = np.floor(weights * n_population).astype(int)
    rest = n_population - counts.sum()

    order = np.argsort(-weights)
    for k in range(rest):
        counts[order[k % len(order)]] += 1

    start = 0
    segments: dict[str, np.ndarray] = {}
    for t, c in zip(types, counts, strict=False):
        segments[t] = rows[start : start + c]
        start += c

    mask_cont = np.isin(rows, segments.get("continuous", []))
    mask_bin = np.isin(rows, segments.get("binary", []))
    mask_disc = np.isin(rows, segments.get("discrete", []))
    mask_cat = np.isin(rows, segments.get("categorical", []))

    initial_cells_continuous = np.empty(
        (0, len(continuous_col)), dtype=np.int32
    )
    initial_binary = np.empty((0, len(binary_col)), dtype=np.int32)
    initial_cells_discrete = np.empty((0, len(discrete_col)), dtype=np.int32)
    initial_categoric = np.empty((0, len(query)), dtype=np.float32)

    # Sampling
    if len(continuous_col) != 0:
        n_cont = mask_cont.sum()
        initial_cells_continuous = (
            (
                np.tile(S_0[continuous_col], (n_cont, 1))
                + np.random.normal(  # noqa: NPY002
                    0,
                    std * lengths_list[continuous_col],
                    size=(n_cont, len(continuous_col)),
                )
            )
            .round()
            .astype(np.int32)
        )
        upper_continuous = np.array([
            lengths_list[d] - 1 for d in continuous_col
        ])[None, :]
        initial_cells_continuous = np.clip(
            initial_cells_continuous, 1, upper_continuous
        )

    if len(binary_col) != 0:
        n_bin = mask_bin.sum()
        initial_binary = np.random.randint(  # noqa: NPY002
            0, 2, size=(n_bin, len(binary_col))
        ).astype(np.int32)

    if len(discrete_col) != 0:
        n_disc = mask_disc.sum()
        initial_cells_discrete = (
            (
                np.tile(S_0[discrete_col], (n_disc, 1))
                + np.random.normal(  # noqa: NPY002
                    0,
                    std * lengths_list[discrete_col],
                    size=(n_disc, len(discrete_col)),
                )
            )
            .round()
            .astype(np.int32)
        )
        upper_discrete = np.array([lengths_list[d] - 1 for d in discrete_col])[
            None, :
        ]
        initial_cells_discrete = np.clip(
            initial_cells_discrete, 1, upper_discrete
        )

    if len(one_hot_encoded_col) != 0:
        n_cat = mask_cat.sum()
        initial_categoric = np.zeros((n_cat, len(query)), dtype=np.float32)
        chosen_indices: np.ndarray = np.array(
            [
                np.random.randint(  # noqa: NPY002
                    one_hot_encoded_col[i][0],
                    one_hot_encoded_col[i][-1] + 1,
                    size=n_cat,
                )
                for i in range(len(one_hot_encoded_col))
            ],
            dtype=np.intp,
        ).T

        for row_idx, selected_col in enumerate(chosen_indices):
            initial_categoric[row_idx, selected_col] = 1

    # Centers decision paths
    points = np.tile(query, (n_population, 1))

    if len(continuous_col) != 0:
        cont_lines = np.where(mask_cont)[0]
        points[np.ix_(cont_lines, continuous_col)] = (
            cell_center_vectorized_numba(
                initial_cells_continuous,
                continuous_col,
                offsets,
                thresholds_concat,
            )
        )

    if len(binary_col) != 0:
        bin_lines = np.where(mask_bin)[0]
        points[np.ix_(bin_lines, binary_col)] = initial_binary.astype(
            np.float32
        )

    if len(discrete_col) != 0:
        disc_lines = np.where(mask_disc)[0]
        points[np.ix_(disc_lines, discrete_col)] = floor_discrete_idx(
            initial_cells_discrete, discrete_col, offsets, thresholds_concat
        ).astype(np.float32)

    cat_lines = np.where(mask_cat)[0]
    if len(one_hot_encoded_col) != 0:
        for category in one_hot_encoded_col:
            category_idx = np.asarray(category, dtype=np.intp)
            points[np.ix_(cat_lines, category_idx)] = initial_categoric[
                :, category_idx
            ].astype(np.float32)

    return points


# Joint perturbation of features used on the decision paths of the query


def gini_grid_perturbation_initialisation(  # noqa: C901, PLR0914
    exp: LocalSearchExplainer,
    query: np.ndarray,
    n_population: int,
    k: int = 3,
    std: float = 1.0,
) -> np.ndarray:
    query = np.asarray(query, dtype=np.float32)
    n_features = query.shape[0]

    # 1. Feature importance distribution
    _, _, probs = get_path_sklearn(
        query,
        cast("DecisionPathForest", exp.rf),
        exp.features_,
        exp.thresholds_,
        return_local_importance=True,
    )
    probs = np.asarray(probs, dtype=float)

    # 2. Selection mask
    perturb_mask = np.zeros((n_population, n_features), dtype=bool)

    for i in range(n_population):
        idx = np.random.choice(  # noqa: NPY002
            n_features, size=k, replace=False, p=probs
        )
        perturb_mask[i, idx] = True

    # 3. Grid preparation for the query
    S_0 = np.array(
        point2cell(query, exp.offsets, exp.lengths_list, exp.thresholds_concat),
        dtype=np.int32,
    )

    # 4. Initialization of the points (copy of the query everywhere)
    points = np.tile(query, (n_population, 1))

    # --- BLOCK CONTINUOUS ---
    if len(exp.continuous_col) > 0:
        mask_cont = perturb_mask[:, exp.continuous_col]

        if mask_cont.any():
            S0_cont = S_0[exp.continuous_col]
            lengths_cont = exp.lengths_list[exp.continuous_col]

            # Gaussian noise on grid indices, scaled by dimension length.
            noise = np.random.normal(  # noqa: NPY002
                0,
                std * lengths_cont,
                size=(n_population, len(exp.continuous_col)),
            )

            cells_cont = (
                (np.tile(S0_cont, (n_population, 1)) + noise)
                .round()
                .astype(np.int32)
            )
            upper_cont = lengths_cont - 1
            cells_cont = np.clip(cells_cont, 1, upper_cont)

            values_cont = cell_center_vectorized_numba(
                cells_cont,
                exp.continuous_col,
                exp.offsets,
                exp.thresholds_concat,
            )

            points_view = points[:, exp.continuous_col]
            points_view[mask_cont] = values_cont[mask_cont]
            points[:, exp.continuous_col] = points_view

    # --- BLOCK DISCRETE ---
    if len(exp.discrete_col) > 0:
        mask_disc = perturb_mask[:, exp.discrete_col]

        if mask_disc.any():
            S0_disc = S_0[exp.discrete_col]
            lengths_disc = exp.lengths_list[exp.discrete_col]

            noise = np.random.normal(  # noqa: NPY002
                0,
                std * lengths_disc,
                size=(n_population, len(exp.discrete_col)),
            )
            cells_disc = (
                (np.tile(S0_disc, (n_population, 1)) + noise)
                .round()
                .astype(np.int32)
            )

            upper_disc = lengths_disc - 1
            cells_disc = np.clip(cells_disc, 1, upper_disc)

            values_disc = floor_discrete_idx(
                cells_disc, exp.discrete_col, exp.offsets, exp.thresholds_concat
            ).astype(np.float32)

            points_view = points[:, exp.discrete_col]
            points_view[mask_disc] = values_disc[mask_disc]
            points[:, exp.discrete_col] = points_view

    # --- BLOCK BINARY ---
    if len(exp.binary_col) > 0:
        mask_bin = perturb_mask[:, exp.binary_col]
        if mask_bin.any():
            points_view = points[:, exp.binary_col]
            points_view[mask_bin] = 1.0 - points_view[mask_bin]
            points[:, exp.binary_col] = points_view

    # --- BLOCK CATEGORICAL (One-Hot) ---
    if len(exp.one_hot_encoded_col) > 0:
        for group in exp.one_hot_encoded_col:
            group_idx = np.asarray(group, dtype=np.int32)

            mask_group = perturb_mask[:, group_idx].any(axis=1)

            if mask_group.any():
                rows_to_mod = np.where(mask_group)[0]

                points[np.ix_(rows_to_mod, group_idx)] = 0.0

                new_idx_local = np.random.randint(  # noqa: NPY002
                    0, len(group_idx), size=rows_to_mod.size
                )
                new_cols = group_idx[new_idx_local]

                points[rows_to_mod, new_cols] = 1.0

    return points.astype(np.float32)


# Naive perturbation: perturbs all features simultaneously, ignoring subgroups
def naive_perturbation_initialisation(  # noqa: C901, PLR0914, PLR0915
    exp: LocalSearchExplainer,
    query: np.ndarray,
    n_population: int,
    std: float = 1.0,
    flip_prob: float = 0.5,
    perturb_ratio: float = 1.0,
) -> np.ndarray:
    query = np.asarray(query, dtype=np.float32)
    n_features = query.shape[0]

    S_0 = np.array(
        point2cell(query, exp.offsets, exp.lengths_list, exp.thresholds_concat),
        dtype=np.int32,
    )

    points = np.tile(query, (n_population, 1))

    # Masque global de perturbation (n_population x n_features)
    if perturb_ratio < 1.0:
        k = max(1, round(perturb_ratio * n_features))
        perturb_mask = np.zeros((n_population, n_features), dtype=bool)
        for i in range(n_population):
            idx = np.random.choice(  # noqa: NPY002
                n_features, size=k, replace=False
            )
            perturb_mask[i, idx] = True
    else:
        perturb_mask = np.ones((n_population, n_features), dtype=bool)

    # --- CONTINUOUS ---
    if len(exp.continuous_col) > 0:
        mask_cont = perturb_mask[:, exp.continuous_col]
        if mask_cont.any():
            S0_cont = S_0[exp.continuous_col]
            lengths_cont = exp.lengths_list[exp.continuous_col]
            noise = np.random.normal(  # noqa: NPY002
                0,
                std * lengths_cont,
                size=(n_population, len(exp.continuous_col)),
            )
            cells_cont = (
                (np.tile(S0_cont, (n_population, 1)) + noise)
                .round()
                .astype(np.int32)
            )
            cells_cont = np.clip(cells_cont, 1, lengths_cont - 1)
            values_cont = cell_center_vectorized_numba(
                cells_cont,
                exp.continuous_col,
                exp.offsets,
                exp.thresholds_concat,
            )
            points_view = points[:, exp.continuous_col]
            points_view[mask_cont] = values_cont[mask_cont]
            points[:, exp.continuous_col] = points_view

    # --- DISCRETE ---
    if len(exp.discrete_col) > 0:
        mask_disc = perturb_mask[:, exp.discrete_col]
        if mask_disc.any():
            S0_disc = S_0[exp.discrete_col]
            lengths_disc = exp.lengths_list[exp.discrete_col]
            noise = np.random.normal(  # noqa: NPY002
                0,
                std * lengths_disc,
                size=(n_population, len(exp.discrete_col)),
            )
            cells_disc = (
                (np.tile(S0_disc, (n_population, 1)) + noise)
                .round()
                .astype(np.int32)
            )
            cells_disc = np.clip(cells_disc, 1, lengths_disc - 1)
            values_disc = floor_discrete_idx(
                cells_disc, exp.discrete_col, exp.offsets, exp.thresholds_concat
            ).astype(np.float32)
            points_view = points[:, exp.discrete_col]
            points_view[mask_disc] = values_disc[mask_disc]
            points[:, exp.discrete_col] = points_view

    # --- BINARY ---
    if len(exp.binary_col) > 0:
        mask_bin = perturb_mask[:, exp.binary_col]
        flip_draws = (
            np.random.random(  # noqa: NPY002
                size=(n_population, len(exp.binary_col))
            )
            < flip_prob
        )
        flip_where = mask_bin & flip_draws
        if flip_where.any():
            points_view = points[:, exp.binary_col]
            points_view[flip_where] = 1.0 - points_view[flip_where]
            points[:, exp.binary_col] = points_view

    # --- CATEGORICAL (One-Hot) ---
    if len(exp.one_hot_encoded_col) > 0:
        for group in exp.one_hot_encoded_col:
            group_idx = np.asarray(group, dtype=np.int32)
            mask_group = perturb_mask[:, group_idx].any(axis=1)
            flip_draws = (
                np.random.random(size=n_population) < flip_prob  # noqa: NPY002
            )
            rows_to_mod = np.where(mask_group & flip_draws)[0]
            if rows_to_mod.size > 0:
                points[np.ix_(rows_to_mod, group_idx)] = 0.0
                new_idx_local = np.random.randint(  # noqa: NPY002
                    0, len(group_idx), size=rows_to_mod.size
                )
                new_cols = group_idx[new_idx_local]
                points[rows_to_mod, new_cols] = 1.0

    return points.astype(np.float32)
