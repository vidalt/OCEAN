from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .numba import njit

if TYPE_CHECKING:
    from collections.abc import Sequence

HALF = np.float32(0.5)
ONE = np.float32(1.0)
ZERO = np.float32(0.0)
NORM_L0 = 0
NORM_L1 = 1
NORM_L2 = 2


@njit(fastmath=True, cache=True)
def L1(x: np.ndarray, y: np.ndarray) -> np.float32:
    return np.float32(np.sum(np.abs(x - y)))


@njit(fastmath=True, cache=True)
def L2(x: np.ndarray, y: np.ndarray) -> np.float32:
    return np.float32(np.sqrt(np.sum((x - y) ** 2)))


@njit(fastmath=True, cache=True)
def L0(x: np.ndarray, y: np.ndarray) -> np.float32:
    return np.float32(np.count_nonzero(x - y))


@njit(fastmath=True, cache=True)
def get_norm(norm: int, query: np.ndarray, proj: np.ndarray) -> np.float32:
    if norm == NORM_L0:
        return L0(query, proj)
    if norm == NORM_L1:
        return L1(query, proj)
    if norm == NORM_L2:
        return L2(query, proj)

    msg = "Norm must be 0, 1, or 2."
    raise ValueError(msg)


@njit(cache=True)
def get_final_explanation(
    a: np.ndarray,
    b: np.ndarray,
    query: np.ndarray,
    continuous_col: np.ndarray,
    binary_col: np.ndarray,
    discrete_col: np.ndarray,
    one_hot_encoded_col: Sequence[Sequence[int]],
    max_distance: float | np.float32,
    norm: int,
) -> tuple[np.float32, np.ndarray]:
    new_a = a.copy()
    new_b = b.copy()

    for i in continuous_col:
        new_a_i = np.nextafter(a[i], np.float32(100.0))
        if new_a_i == b[i]:
            new_b_i = b[i]
        else:
            new_b_i = np.nextafter(b[i], np.float32(-100.0))

        new_a[i] = new_a_i
        new_b[i] = new_b_i

    return hypercube_cost(
        new_a,
        new_b,
        query,
        continuous_col,
        binary_col,
        discrete_col,
        one_hot_encoded_col,
        max_distance,
        norm,
    )


@njit(cache=True)
def _format_discrete_cols(
    discrete_col: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    query: np.ndarray,
    proj: np.ndarray,
) -> None:
    clipped_discrete = np.clip(
        query[discrete_col],
        a[discrete_col] + HALF,
        b[discrete_col],
    )
    for i, clipp in enumerate(clipped_discrete):
        if clipp == query[discrete_col][i]:
            continue
        if clipp == a[discrete_col][i] + HALF:
            if a[discrete_col][i] == ZERO:
                clipped_discrete[i] = ZERO
            else:
                clipped_discrete[i] = np.ceil(clipp)
        elif clipp == b[discrete_col][i]:
            clipped_discrete[i] = np.floor(clipp)

    proj[discrete_col] = clipped_discrete.astype(np.float32)


@njit(cache=True)
def _format_ohe_cols(
    category: Sequence[int],
    a: np.ndarray,
    b: np.ndarray,
    query: np.ndarray,
    proj: np.ndarray,
) -> None:
    forced_j = -1
    for j in category:
        if a[j] == HALF and b[j] == ONE:
            proj[j] = ONE
            forced_j = j
            break

    if forced_j == -1:
        found = False
        for j_ in category:
            if b[j_] == ONE and query[j_] == ONE:
                proj[j_] = ONE
                found = True
                break

        if not found:
            for j_ in category:
                if b[j_] == ONE:
                    proj[j_] = ONE
                    break


@njit(cache=True)
def hypercube_cost(  # noqa: C901
    a: np.ndarray,
    b: np.ndarray,
    query: np.ndarray,
    continuous_col: np.ndarray,
    binary_col: np.ndarray,
    discrete_col: np.ndarray,
    one_hot_encoded_col: Sequence[Sequence[int]],
    max_distance: float | np.float32,
    norm: int,
) -> tuple[np.float32, np.ndarray]:
    proj = np.zeros(a.shape, dtype=np.float32)

    if len(continuous_col) != 0:
        proj[continuous_col] = np.clip(
            query[continuous_col],
            a[continuous_col],
            b[continuous_col],
        )

    if len(binary_col) != 0:
        clipped_binary = np.clip(
            query[binary_col],
            a[binary_col],
            b[binary_col],
        )
        for j, clipp_ in enumerate(clipped_binary):
            if clipp_ == HALF:
                if a[binary_col][j] == HALF:
                    clipped_binary[j] = ONE
                if b[binary_col][j] == HALF:
                    clipped_binary[j] = ZERO

        proj[binary_col] = clipped_binary.astype(np.float32)

    if len(discrete_col) != 0:
        _format_discrete_cols(discrete_col, a, b, query, proj)

    if len(one_hot_encoded_col) != 0:
        for category in one_hot_encoded_col:
            _format_ohe_cols(category, a, b, query, proj)

    dist = get_norm(norm, query, proj)
    if dist == ZERO:
        return np.float32(max_distance), query.astype(np.float32)
    return dist, proj


@njit(fastmath=True, cache=True)
def fitness(
    distance: np.float32,
    query_prob: np.float32,
    label: int,
    query_class: int,
    lambda_: float | np.float32,
    max_distance: float | np.float32,
) -> np.float32:
    return np.float32(
        distance
        + max_distance * (label == query_class) * (lambda_ * query_prob + 1)
    )


__all__ = [
    "L0",
    "L1",
    "L2",
    "fitness",
    "get_final_explanation",
    "get_norm",
    "hypercube_cost",
]
