from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ..utils.numba import njit
from ..utils.tools import shuffled_copy

if TYPE_CHECKING:
    from collections.abc import Sequence


@njit(cache=True)
def random_face_cells_sampling(  # noqa: C901, PLR0912, PLR0914, PLR0915
    a_idx: np.ndarray,
    b_idx: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    continuous_col: np.ndarray,
    binary_col: np.ndarray,
    discrete_col: np.ndarray,
    one_hot_encoded_col: Sequence[Sequence[int]],
    offsets: np.ndarray,
    thresholds_concat: np.ndarray,
    lengths_list: np.ndarray,
    _query: np.ndarray,
    _global_best_distance: float | np.float32,
    k: int,
    n_samples_per_face: int,
) -> tuple[np.ndarray, np.ndarray]:
    n_features = a.shape[0]
    half = 0.5

    n_features_categories = 0
    for category in one_hot_encoded_col:
        n_features_categories += len(category)
    n_free_features_categorical = n_features_categories - len(
        one_hot_encoded_col
    )

    total_faces = np.empty(
        2 * len(continuous_col)
        + 2 * len(discrete_col)
        + len(binary_col)
        + n_free_features_categorical,
        dtype=np.int32,
    )
    n_faces = 0

    for dim in np.concatenate((continuous_col, binary_col, discrete_col)):
        if 1 <= a_idx[dim] <= lengths_list[dim] - 1:
            total_faces[n_faces] = 2 * dim
            n_faces += 1

        if 1 <= b_idx[dim] + 1 <= lengths_list[dim] - 1:
            total_faces[n_faces] = 2 * dim + 1
            n_faces += 1

    for category in one_hot_encoded_col:
        for dim in category:
            if a[dim] != half:
                total_faces[n_faces] = 2 * dim + 1
                n_faces += 1

    k = min(k, n_faces)

    valid_faces = total_faces[:n_faces]
    shuffled_faces = valid_faces[
        np.random.permutation(n_faces)  # noqa: NPY002
    ]
    sampled = shuffled_faces[:k]

    result = np.zeros((k * n_samples_per_face, n_features), dtype=np.int32)
    for i, face_id in enumerate(sampled):  # noqa: PLR1702
        dim = face_id // 2
        side = face_id % 2

        for n in range(n_samples_per_face):
            idex = i * n_samples_per_face + n

            for j in continuous_col:
                if j == dim:
                    result[idex, j] = a_idx[j] if side == 0 else b_idx[j] + 1
                else:
                    result[idex, j] = np.random.randint(  # noqa: NPY002
                        a_idx[j] + 1, b_idx[j] + 1
                    )

            for j in binary_col:
                if j == dim:
                    result[idex, j] = 0 if side == 0 else 1
                elif a[j] == half and b[j] == 1:
                    result[idex, j] = 1
                elif a[j] == 0 and b[j] == half:
                    result[idex, j] = 0
                elif a[j] == 0 and b[j] == 1:
                    result[idex, j] = np.random.randint(0, 2)  # noqa: NPY002

            for j in discrete_col:
                if j == dim:
                    if side == 0:
                        result[idex, j] = a_idx[j]

                    elif side == 1:
                        current_value = thresholds_concat[offsets[j] + b_idx[j]]
                        next_value = thresholds_concat[
                            offsets[j] + b_idx[j] + 1
                        ]
                        if np.floor(current_value) == np.floor(next_value):
                            result[idex, j] = b_idx[j] + 2
                        else:
                            result[idex, j] = b_idx[j] + 1
                else:
                    result[idex, j] = np.random.randint(  # noqa: NPY002
                        a_idx[j] + 1, b_idx[j] + 1
                    )

            for category in one_hot_encoded_col:
                if dim in category:
                    for j in category:
                        result[idex, j] = 0
                        if j == dim:
                            result[idex, j] = 1
                else:
                    shuffled_category = shuffled_copy(category)

                    complete = False
                    for j in shuffled_category:
                        if a[j] == half and b[j] == 1:
                            result[idex, j] = 1
                            complete = True
                            break

                    if not complete:
                        for j in shuffled_category:
                            if a[j] == 0 and b[j] == 1:
                                result[idex, j] = 1
                                break

    return result, sampled
