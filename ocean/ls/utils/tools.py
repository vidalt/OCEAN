from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Protocol, cast, overload

import numpy as np
import pandas as pd
from numba import types  # type: ignore[attr-defined]
from numba.typed.typedlist import List as NumbaList

from .numba import njit

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, MutableSequence, Sequence

type ScalarFloat = float | np.float32 | np.float64
type TreePaths = dict[int, list[tuple[int, int]]]
type LocalPathResult = (
    tuple[TreePaths, dict[int, int]]
    | tuple[TreePaths, dict[int, int], np.ndarray]
)

CONTINUOUS_LOWER_BOUND = -0.5
CONTINUOUS_UPPER_BOUND = 0.5
OHE_ACTIVE_COUNT = 1


class _DecisionPathMatrix(Protocol):
    indptr: np.ndarray
    indices: np.ndarray


class _Tree(Protocol):
    feature: np.ndarray
    threshold: np.ndarray
    children_left: np.ndarray
    children_right: np.ndarray
    weighted_n_node_samples: np.ndarray
    impurity: np.ndarray


class _Estimator(Protocol):
    tree_: _Tree


class DecisionPathForest(Protocol):
    @property
    def estimators_(self) -> Sequence[_Estimator]: ...

    @property
    def n_features_in_(self) -> int: ...

    def decision_path(
        self,
        x: Sequence[np.ndarray] | np.ndarray,
    ) -> tuple[_DecisionPathMatrix, np.ndarray]: ...


@njit(cache=True)
def floor_strict(x: ScalarFloat) -> np.int32:
    fx = np.floor(x)
    return np.int32(fx - (x == fx))


@njit(cache=True)
def ceil_strict(x: ScalarFloat) -> np.int32:
    cx = np.ceil(x)
    return np.int32(cx + (x == cx))


@njit(cache=True)
def shuffle_typed_list(lst: MutableSequence[np.int64]) -> None:
    n = len(lst)
    for i in range(n - 1, 0, -1):
        j = np.random.randint(0, i + 1)  # noqa: NPY002
        tmp = lst[i]
        lst[i] = lst[j]
        lst[j] = tmp


@njit(cache=True, fastmath=True)
def dot_product_int64(x: np.ndarray, y: np.ndarray) -> np.int64:
    total = np.int64(0)
    for i in range(x.shape[0]):
        total += np.int64(x[i]) * np.int64(y[i])
    return total


@njit(cache=True)
def hash_leaves_fnv1a(leaves: np.ndarray) -> np.int64:
    h = np.uint64(14695981039346656037)
    prime = np.uint64(1099511628211)
    for i in range(leaves.shape[0]):
        h ^= np.uint64(leaves[i] + 1)
        h *= prime
    return np.int64(h)


@njit(cache=True)
def shuffled_copy(lst: Sequence[int | np.int64]) -> NumbaList[np.int64]:
    n = len(lst)
    idx: np.ndarray = np.empty(n, dtype=np.int64)
    for i in range(n):
        idx[i] = i

    for i in range(n - 1, 0, -1):
        j = np.random.randint(0, i + 1)  # noqa: NPY002
        tmp = idx[i]
        idx[i] = idx[j]
        idx[j] = tmp

    out: NumbaList[np.int64] = NumbaList.empty_list(types.int64)  # type: ignore[no-untyped-call]
    for i in range(n):
        out.append(np.int64(lst[int(idx[i])]))
    return out  # pyright: ignore[reportReturnType]


def build_cat_groups(
    colnames: Sequence[str],
    category_names: Sequence[str],
) -> NumbaList[NumbaList[np.int64]]:
    groups = cast(
        "NumbaList[NumbaList[np.int64]]",
        NumbaList.empty_list(  # type: ignore[no-untyped-call]
            types.ListType(types.int64)
        ),
    )
    for category in category_names:
        group = cast(
            "NumbaList[np.int64]",
            NumbaList.empty_list(types.int64),  # type: ignore[no-untyped-call]
        )
        prefix = f"{category}_"
        for i, colname in enumerate(colnames):
            if colname.startswith(prefix):
                group.append(np.int64(i))
        groups.append(group)
    return groups


def build_cat_groups_ocean(
    colnames_multi_index: Iterable[object],
    category_names: Iterable[object],
) -> NumbaList[NumbaList[np.int64]]:
    groups = cast(
        "NumbaList[NumbaList[np.int64]]",
        NumbaList.empty_list(  # type: ignore[no-untyped-call]
            types.ListType(types.int64)
        ),
    )
    for category in category_names:
        group = cast(
            "NumbaList[np.int64]",
            NumbaList.empty_list(types.int64),  # type: ignore[no-untyped-call]
        )
        for i, colname in enumerate(colnames_multi_index):
            if isinstance(colname, tuple) and colname[0] == category:
                group.append(np.int64(i))
        groups.append(group)
    return groups


@njit(cache=True)
def sum_numba_list(lst: Iterable[ScalarFloat]) -> float:
    total = 0.0
    for value in lst:
        total += float(value)
    return float(total)


@overload
def get_path_sklearn(
    x: np.ndarray,
    rf: DecisionPathForest,
    features_: Sequence[np.ndarray] | None = None,
    thresholds_: Sequence[np.ndarray] | None = None,
    *,
    return_local_importance: Literal[False] = False,
) -> tuple[TreePaths, dict[int, int]]: ...


@overload
def get_path_sklearn(
    x: np.ndarray,
    rf: DecisionPathForest,
    features_: Sequence[np.ndarray] | None = None,
    thresholds_: Sequence[np.ndarray] | None = None,
    *,
    return_local_importance: Literal[True],
) -> tuple[TreePaths, dict[int, int], np.ndarray]: ...


def get_path_sklearn(  # noqa: PLR0914
    x: np.ndarray,
    rf: DecisionPathForest,
    features_: Sequence[np.ndarray] | None = None,
    thresholds_: Sequence[np.ndarray] | None = None,
    return_local_importance: bool = False,  # noqa: FBT001, FBT002
) -> LocalPathResult:
    paths: TreePaths = {}
    leaves: dict[int, int] = {}

    node_indicator, tree_ptrs = rf.decision_path([x])
    global_nodes = node_indicator.indices[
        node_indicator.indptr[0] : node_indicator.indptr[1]
    ]
    global_nodes = np.sort(global_nodes)

    w_local: np.ndarray | None = (
        np.zeros(rf.n_features_in_, dtype=float)
        if return_local_importance
        else None
    )

    for tree_index, estimator in enumerate(rf.estimators_):
        offset = tree_ptrs[tree_index]
        next_offset = tree_ptrs[tree_index + 1]
        start = np.searchsorted(global_nodes, offset, side="left")
        stop = np.searchsorted(global_nodes, next_offset, side="left")
        local_nodes = global_nodes[start:stop] - offset

        leaves[tree_index] = int(local_nodes[-1])
        nodes = local_nodes[:-1].astype(int)
        if nodes.size == 0:
            paths[tree_index] = []
            continue

        if features_ is None and thresholds_ is None:
            feats = estimator.tree_.feature[nodes]
            thrs = estimator.tree_.threshold[nodes]
        elif features_ is not None and thresholds_ is not None:
            feats = np.asarray(features_[tree_index])[nodes]
            thrs = np.asarray(thresholds_[tree_index])[nodes]
        else:
            msg = "features_ and thresholds_ must be provided together."
            raise ValueError(msg)

        sides = (x[feats] > thrs).astype(int)
        paths[tree_index] = list(
            zip(nodes.tolist(), sides.tolist(), strict=True)
        )

        if return_local_importance and w_local is not None:
            tree = estimator.tree_
            left = tree.children_left[nodes]
            right = tree.children_right[nodes]
            weighted_total = tree.weighted_n_node_samples[nodes]
            weighted_left = tree.weighted_n_node_samples[left]
            weighted_right = tree.weighted_n_node_samples[right]
            impurity_total = tree.impurity[nodes]
            impurity_left = tree.impurity[left]
            impurity_right = tree.impurity[right]

            delta = (
                weighted_total * impurity_total
                - weighted_left * impurity_left
                - weighted_right * impurity_right
            )
            np.add.at(w_local, feats, np.maximum(delta, 0.0))

    if return_local_importance:
        local_importance = w_local
        if local_importance is None:
            msg = "Local importance array was not initialized."
            raise RuntimeError(msg)
        total = local_importance.sum()
        if total > 0:
            local_importance /= total
        return paths, leaves, local_importance
    return paths, leaves


def print_path(
    explanation_: np.ndarray,
    path: Sequence[tuple[int, int]],
    leaf: int,
    features: np.ndarray,
    thresholds: np.ndarray,
    children_left: np.ndarray,
    children_right: np.ndarray,
    index2type: Mapping[int, object],
) -> None:
    for node_id, right in path:
        feature = int(features[node_id])
        threshold = float(thresholds[node_id])
        value = float(explanation_[feature])
        threshold_sign = "<=" if value <= threshold else ">"
        arrow = "←" if right == 1 else "→"
        condition = (
            f"[{index2type[feature]}] Node {node_id} : "
            f"x[{feature}] = {value!r} {threshold_sign} {threshold!r}"
        )
        left_child = int(children_left[node_id])
        right_child = int(children_right[node_id])
        if right == 1:
            print(  # noqa: T201
                f"{condition:<60} {left_child} , {right_child} {arrow} "
            )
        else:
            print(  # noqa: T201
                f"{condition:<60} {arrow} {left_child} , {right_child}"
            )
    print("Leaf", leaf)  # noqa: T201


def process(
    data: Iterable[tuple[float, float | None]],
    n_features: int,
) -> tuple[list[float], list[float]]:
    times: list[float] = []
    values: list[float] = []
    fallback_value = float(np.sqrt(n_features))
    for time, distance in data:
        if distance is None:
            continue
        value = fallback_value if np.isinf(distance) else distance
        times.append(time)
        values.append(value)
    return times, values


def print_thresholds(
    thresholds: Mapping[int, Sequence[float]],
    index2type: Mapping[int, object],
) -> None:
    n_dims = len(thresholds)
    max_len = max(len(vals) for vals in thresholds.values())
    matrix: np.ndarray = np.full(
        (max_len, n_dims),
        np.nan,
        dtype=np.float32,
    )
    for dim, vals in thresholds.items():
        matrix[: len(vals), dim] = vals
    display_matrix = np.where(np.isnan(matrix), "", matrix.astype(str))
    df = pd.DataFrame(
        display_matrix,
        columns=[f"Dim {i}" for i in range(n_dims)],
    )
    types_row = [index2type.get(dim, "") for dim in range(n_dims)]
    df = pd.concat(
        [pd.DataFrame([types_row], columns=df.columns), df],
        ignore_index=True,
    )
    print(df.to_string(index=False))  # noqa: T201


def verify_explanation(
    explanation: np.ndarray,
    continuous_col: Sequence[int],
    discrete_col: Sequence[int],
    binary_col: Sequence[int],
    one_hot_encoded_col: Sequence[Sequence[int]],
    offsets: np.ndarray,
    thresholds_concat: np.ndarray,
    lengths_list: np.ndarray,
) -> None:
    continuous_values = explanation[continuous_col]
    continuous_feasible = bool(
        np.all(continuous_values >= CONTINUOUS_LOWER_BOUND)
        & np.all(continuous_values <= CONTINUOUS_UPPER_BOUND)
    )
    if not continuous_feasible:
        print("continuous infeasibility")  # noqa: T201

    upper_bounds: np.ndarray = np.zeros(
        len(discrete_col),
        dtype=np.float32,
    )
    lower_bounds: np.ndarray = np.empty(
        len(discrete_col),
        dtype=np.float32,
    )
    for k, dim in enumerate(discrete_col):
        dim_thresholds = thresholds_concat[
            offsets[dim] : offsets[dim] + lengths_list[dim]
        ]
        lower_bounds[k] = np.float32(dim_thresholds.min())
        upper_bounds[k] = np.float32(dim_thresholds.max())

    discrete_values = explanation[discrete_col]
    discrete_feasible = bool(
        np.all(lower_bounds < discrete_values)
        & np.all(discrete_values <= upper_bounds)
        & np.all(np.floor(discrete_values) == discrete_values)
    )
    if not discrete_feasible:
        print("discrete infeasibility")  # noqa: T201

    binary_values = explanation[binary_col]
    binary_feasible = bool(np.all((binary_values == 0) | (binary_values == 1)))
    if not binary_feasible:
        print("binary infeasibility")  # noqa: T201

    categorical_feasible = all(
        np.sum(explanation[category]) == OHE_ACTIVE_COUNT
        for category in one_hot_encoded_col
    )
    if not categorical_feasible:
        print("categorical infeasibility")  # noqa: T201


@njit(cache=True)
def set_seed(value: int) -> None:
    np.random.seed(value)  # noqa: NPY002


@njit(cache=True)
def idx2thresh(
    M: Sequence[int],
    thresholds: Sequence[np.ndarray],
) -> np.ndarray:
    values: np.ndarray = np.array([thresholds[d][M[d]] for d in range(len(M))])
    return values


@njit(cache=True)
def idx2thresh_vectorized(
    M: np.ndarray,
    offsets: np.ndarray,
    thresholds_concat: np.ndarray,
) -> np.ndarray:
    values: np.ndarray = np.array([
        thresholds_concat[offsets[d] + M[d]] for d in range(len(M))
    ])
    return values


@njit(fastmath=True, cache=True)
def cell_center(
    idx: Sequence[int],
    offsets: np.ndarray,
    thresholds_concat: np.ndarray,
    lengths_list: np.ndarray | None = None,
) -> np.ndarray:
    _ = lengths_list
    point: list[np.float32] = []
    for d in range(len(idx)):
        a_d = thresholds_concat[offsets[d] + idx[d] - 1]
        b_d = thresholds_concat[offsets[d] + idx[d]]
        point.append(np.float32(0.5 * (a_d + b_d)))
    center: np.ndarray = np.array(point, dtype=np.float32)
    return center


def cell_center_vectorized(
    array_idx: np.ndarray,
    continuous_col: Sequence[int],
    offsets: np.ndarray,
    thresholds_concat: np.ndarray,
) -> np.ndarray:
    lower_flat = offsets[None, continuous_col] + (array_idx - 1)
    upper_flat = offsets[None, continuous_col] + array_idx
    lower_vals = thresholds_concat[lower_flat]
    upper_vals = thresholds_concat[upper_flat]
    return cast("np.ndarray", 0.5 * (lower_vals + upper_vals))


def rint_discrete_idx(
    array_idx: np.ndarray,
    discrete_col: Sequence[int],
    offsets: np.ndarray,
    thresholds_concat: np.ndarray,
) -> np.ndarray:
    flat = offsets[None, discrete_col] + array_idx
    return cast(
        "np.ndarray",
        np.rint(thresholds_concat[flat]).astype(np.float32),
    )


@njit(cache=True, fastmath=True)
def floor_discrete_idx(
    array_idx: np.ndarray,
    discrete_col: Sequence[int] | np.ndarray,
    offsets: np.ndarray,
    thresholds_concat: np.ndarray,
) -> np.ndarray:
    flat = offsets[None, discrete_col] + array_idx
    vals = thresholds_concat[flat.ravel()]
    vals = np.floor(vals).astype(np.float32)
    reshaped: np.ndarray = vals.reshape(flat.shape)
    return reshaped


def ceil_discrete_idx(
    array_idx: np.ndarray,
    discrete_col: Sequence[int],
    offsets: np.ndarray,
    thresholds_concat: np.ndarray,
) -> np.ndarray:
    flat = offsets[None, discrete_col] + array_idx
    return cast(
        "np.ndarray",
        np.ceil(thresholds_concat[flat]).astype(np.float32),
    )


@njit(cache=True, fastmath=True)
def cell_center_vectorized_numba(
    array_idx: np.ndarray,
    continuous_col: np.ndarray,
    offsets: np.ndarray,
    thresholds_concat: np.ndarray,
) -> np.ndarray:
    idx: np.ndarray = array_idx.astype(np.intp)
    cols: np.ndarray = continuous_col.astype(np.intp)
    offs: np.ndarray = offsets.astype(np.intp)
    n_pts, n_dims = idx.shape

    lower = np.empty(n_pts * n_dims, dtype=np.intp)
    upper = np.empty(n_pts * n_dims, dtype=np.intp)
    k = 0
    for i in range(n_pts):
        for j in range(n_dims):
            base = offs[cols[j]]
            aij = idx[i, j]
            lower[k] = base + aij - 1
            upper[k] = base + aij
            k += 1

    lower_vals = thresholds_concat[lower].reshape(n_pts, n_dims)
    upper_vals = thresholds_concat[upper].reshape(n_pts, n_dims)
    out = (lower_vals + upper_vals) * 0.5
    return out.astype(thresholds_concat.dtype)  # type: ignore[no-any-return]


def idx_to_discrete(
    array_idx: np.ndarray,
    discrete_col: Sequence[int],
    offsets: np.ndarray,
    thresholds_concat: np.ndarray,
) -> np.ndarray:
    flat_mask = offsets[None, discrete_col] + array_idx
    return cast(
        "np.ndarray",
        np.ceil(thresholds_concat[flat_mask]).astype(np.int32),
    )


@njit(cache=True)
def find_interval(value: ScalarFloat, thresholds: np.ndarray) -> int:
    if value == 0:
        return 1
    for i in range(1, len(thresholds)):
        if thresholds[i - 1] < value <= thresholds[i]:
            return i
    return len(thresholds) - 1


@njit(cache=True)
def point2cell(
    point: np.ndarray,
    offsets: np.ndarray,
    lengths_list: np.ndarray,
    thresholds_concat: np.ndarray,
) -> list[int]:
    return [
        find_interval(
            point[d],
            thresholds_concat[offsets[d] : offsets[d] + lengths_list[d]],
        )
        for d in range(len(lengths_list))
    ]


def inside_grid(idx: np.ndarray, lengths_list: np.ndarray) -> bool:
    return bool(np.all((idx >= 1) & (idx <= lengths_list - 1)))


__all__ = [
    "DecisionPathForest",
    "ceil_discrete_idx",
    "ceil_strict",
    "cell_center",
    "cell_center_vectorized",
    "cell_center_vectorized_numba",
    "dot_product_int64",
    "find_interval",
    "floor_discrete_idx",
    "floor_strict",
    "get_path_sklearn",
    "hash_leaves_fnv1a",
    "idx2thresh",
    "idx2thresh_vectorized",
    "idx_to_discrete",
    "inside_grid",
    "point2cell",
    "print_path",
    "print_thresholds",
    "process",
    "rint_discrete_idx",
    "set_seed",
    "shuffle_typed_list",
    "shuffled_copy",
    "sum_numba_list",
    "verify_explanation",
]
