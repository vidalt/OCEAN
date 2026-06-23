from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ..utils.costs import hypercube_cost
from ..utils.numba import njit
from ..utils.tools import point2cell

if TYPE_CHECKING:
    from collections.abc import Sequence


@njit(cache=True)
def random_single_neighbor(  # noqa: C901, PLR0912, PLR0914, PLR0915
    a: np.ndarray,
    b: np.ndarray,
    continuous_col: np.ndarray,
    binary_col: np.ndarray,
    discrete_col: np.ndarray,
    one_hot_encoded_col: Sequence[Sequence[int]],
    offsets: np.ndarray,
    thresholds_concat: np.ndarray,
    lengths_list: np.ndarray,
    query: np.ndarray,
    max_distance: np.float32,
    norm: int,
) -> tuple[np.ndarray, bool]:
    _dist, proj = hypercube_cost(
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
    proj_idx = point2cell(proj, offsets, lengths_list, thresholds_concat)

    n_continuous_moves = 2 * len(continuous_col)
    n_discrete_moves = 2 * len(discrete_col)
    n_binary_moves = len(binary_col)

    n_categorical_moves = 0
    for category in one_hot_encoded_col:
        n_categorical_moves += len(category) - 1

    total_moves = (
        n_continuous_moves
        + n_discrete_moves
        + n_binary_moves
        + n_categorical_moves
    )

    if total_moves == 0:
        return proj, False

    eps = np.float32(1e-6)
    max_attempts = 20

    for _attempt in range(max_attempts):  # noqa: PLR1702
        chosen = np.random.randint(total_moves)  # noqa: NPY002

        neighbor = proj.copy()
        valid_move = False

        if chosen < n_continuous_moves:
            move_idx = chosen
            d_idx = move_idx // 2
            direction = move_idx % 2
            d = continuous_col[d_idx]

            if direction == 0:
                if proj_idx[d] > 0:
                    thresh = thresholds_concat[offsets[d] + proj_idx[d] - 1]
                else:
                    thresh = a[d]
                new_val = np.nextafter(np.float32(thresh), np.float32(-np.inf))
            else:
                thresh = thresholds_concat[offsets[d] + proj_idx[d]]
                new_val = np.nextafter(np.float32(thresh), np.float32(np.inf))

            if not (a[d] < new_val <= b[d]):
                neighbor[d] = new_val
                valid_move = True

        elif chosen < n_continuous_moves + n_discrete_moves:
            move_idx = chosen - n_continuous_moves
            d_idx = move_idx // 2
            direction = move_idx % 2
            d = discrete_col[d_idx]
            d_int = int(d)
            proj_idx_d = int(proj_idx[d_int])

            if direction == 0:
                left_thresh = thresholds_concat[
                    offsets[d_int] + max(0, proj_idx_d - 1)
                ]
                new_val = np.float32(np.floor(left_thresh))
            else:
                right_thresh = thresholds_concat[offsets[d_int] + proj_idx_d]
                new_val = np.float32(np.ceil(right_thresh))

            if not (a[d] < new_val <= b[d]):
                neighbor[d] = new_val
                valid_move = True

        elif chosen < n_continuous_moves + n_discrete_moves + n_binary_moves:
            move_idx = chosen - n_continuous_moves - n_discrete_moves
            d = binary_col[move_idx]

            new_val = np.float32(1 - proj[d])

            if not (a[d] < new_val <= b[d]):
                neighbor[d] = new_val
                valid_move = True

        else:
            move_idx = (
                chosen - n_continuous_moves - n_discrete_moves - n_binary_moves
            )

            cumsum = 0
            target_cat_idx = -1
            target_k_local = -1

            for cat_idx in range(len(one_hot_encoded_col)):
                category = one_hot_encoded_col[cat_idx]
                n_moves_in_cat = len(category) - 1

                if cumsum + n_moves_in_cat > move_idx:
                    target_cat_idx = cat_idx
                    target_k_local = move_idx - cumsum
                    break
                cumsum += n_moves_in_cat

            if target_cat_idx >= 0:
                category = one_hot_encoded_col[target_cat_idx]

                current_active = -1
                for cat_col in category:
                    if proj[cat_col] == np.float32(1.0):
                        current_active = cat_col
                        break

                count = 0
                target_k = -1
                for k in category:
                    if k != current_active:
                        if count == target_k_local:
                            target_k = k
                            break
                        count += 1

                if target_k >= 0:
                    forced_current = False
                    if current_active != -1 and (
                        (a[current_active] >= np.float32(0.5) - eps)
                        and (b[current_active] >= np.float32(1.0) - eps)
                    ):
                        forced_current = True

                    forbidden_k = b[target_k] <= np.float32(0.5) + eps

                    if forced_current or forbidden_k:
                        for j in category:
                            neighbor[j] = np.float32(0.0)
                        neighbor[target_k] = np.float32(1.0)
                        valid_move = True

        if valid_move:
            return neighbor, True

    return proj, False


@njit(cache=True)
def optimal_cell_adjacents(  # noqa: C901, PLR0912, PLR0914, PLR0915
    a: np.ndarray,
    b: np.ndarray,
    continuous_col: np.ndarray,
    binary_col: np.ndarray,
    discrete_col: np.ndarray,
    one_hot_encoded_col: Sequence[Sequence[int]],
    offsets: np.ndarray,
    thresholds_concat: np.ndarray,
    lengths_list: np.ndarray,
    query: np.ndarray,
    max_distance: np.float32,
    norm: int,
) -> np.ndarray:
    _dist, proj = hypercube_cost(
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
    proj_idx = point2cell(proj, offsets, lengths_list, thresholds_concat)

    n_categorical_neighbors = 0
    for category in one_hot_encoded_col:
        n_categorical_neighbors += len(category)

    max_neighbors = (
        2 * len(continuous_col)
        + 2 * len(discrete_col)
        + len(binary_col)
        + n_categorical_neighbors
    )

    result = np.empty((max_neighbors, proj.shape[0]), dtype=np.float32)
    idx = 0

    for d in continuous_col:
        for s in [-1, 1]:
            neighbor = proj.copy()
            thresh = thresholds_concat[offsets[d] + proj_idx[d] + min(0, s)]
            new_val = np.nextafter(
                np.float32(thresh),
                np.float32(s * np.inf),
            )
            neighbor[d] = new_val
            if not (a[d] < new_val <= b[d]):
                result[idx] = neighbor
                idx += 1

    for d in discrete_col:
        d_int = int(d)
        proj_idx_d = int(proj_idx[d_int])
        neighbor = proj.copy()
        left_thresh = thresholds_concat[offsets[d_int] + max(0, proj_idx_d - 1)]
        new_val = np.float32(np.floor(left_thresh))
        neighbor[d_int] = new_val
        if not (a[d_int] < new_val <= b[d_int]):
            result[idx] = neighbor
            idx += 1

        neighbor = proj.copy()
        right_thresh = thresholds_concat[offsets[d_int] + proj_idx_d]
        new_val = np.float32(np.ceil(right_thresh))
        neighbor[d_int] = new_val
        if not (a[d_int] < new_val <= b[d_int]):
            result[idx] = neighbor
            idx += 1

    for d in binary_col:
        neighbor = proj.copy()
        new_val = np.float32(1 - proj[d])
        neighbor[d] = new_val
        if not (a[d] < new_val <= b[d]):
            result[idx] = neighbor
            idx += 1

    eps = np.float32(1e-6)

    for category in one_hot_encoded_col:
        current_active = -1
        for cat_idx in category:
            if proj[cat_idx] == np.float32(1.0):
                current_active = cat_idx
                break

        for k in category:
            if k == current_active:
                continue

            forced_current = False
            if current_active != -1 and (
                (a[current_active] >= np.float32(0.5) - eps)
                and (b[current_active] >= np.float32(1.0) - eps)
            ):
                forced_current = True

            forbidden_k = b[k] <= np.float32(0.5) + eps

            if forced_current or forbidden_k:
                neighbor = proj.copy()
                for j in category:
                    neighbor[j] = np.float32(0.0)
                neighbor[k] = np.float32(1.0)

                result[idx] = neighbor
                idx += 1

    return result[:idx]
