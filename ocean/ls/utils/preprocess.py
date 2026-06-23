from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Protocol, cast

import numpy as np
from numba import types  # type: ignore[attr-defined]
from numba.typed.typeddict import Dict as NumbaDict
from numba.typed.typedlist import List as NumbaList

if TYPE_CHECKING:
    from collections.abc import Mapping, MutableMapping, Sequence

    import pandas as pd

    from ...typing import Index, Key

type FeatureKind = Literal["N", "B", "C", "D"]
type ContinuousBounds = Literal["scaled", "data"]
type ThresholdList = NumbaList[np.float32]
type ThresholdMap = MutableMapping[int, ThresholdList]
type InnerThresholdIndex = MutableMapping[np.float32, int]
type ThresholdIndex = MutableMapping[int, InnerThresholdIndex]
type ForestArrays = tuple[
    list[np.ndarray],
    list[np.ndarray],
    list[np.ndarray],
    list[np.ndarray],
    list[np.ndarray],
]
type ThresholdResult = tuple[
    ThresholdMap,
    ThresholdIndex,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[np.ndarray],
    list[np.ndarray],
    list[np.ndarray],
    list[np.ndarray],
    list[np.ndarray],
]

LOWER_SCALED = np.float32(-0.5)
UPPER_SCALED = np.float32(0.5)
LOWER_BINARY = np.float32(0.0)
UPPER_BINARY = np.float32(1.0)


class TreeLike(Protocol):
    feature: np.ndarray
    threshold: np.ndarray
    value: np.ndarray
    children_left: np.ndarray
    children_right: np.ndarray


class EstimatorLike(Protocol):
    tree_: TreeLike


class RandomForestLike(Protocol):
    @property
    def n_features_in_(self) -> int: ...

    @property
    def n_estimators(self) -> int: ...

    @property
    def estimators_(self) -> Sequence[EstimatorLike]: ...


class FeatureMetadataLike(Protocol):
    @property
    def is_continuous(self) -> bool: ...

    @property
    def is_binary(self) -> bool: ...

    @property
    def is_discrete(self) -> bool: ...

    @property
    def is_one_hot_encoded(self) -> bool: ...


class MapperLike(Protocol):
    @property
    def columns(self) -> Index: ...

    def __getitem__(self, key: Key) -> FeatureMetadataLike: ...


def _empty_threshold_list() -> ThresholdList:
    return cast(
        "ThresholdList",
        NumbaList.empty_list(types.float32),  # type: ignore[no-untyped-call]
    )


def _empty_threshold_map() -> ThresholdMap:
    return cast(
        "ThresholdMap",
        NumbaDict.empty(  # type: ignore[no-untyped-call]
            key_type=types.int64,
            value_type=types.ListType(types.float32),
        ),
    )


def _empty_inner_threshold_index() -> InnerThresholdIndex:
    return cast(
        "InnerThresholdIndex",
        NumbaDict.empty(  # type: ignore[no-untyped-call]
            key_type=types.float32,
            value_type=types.int64,
        ),
    )


def _empty_threshold_index() -> ThresholdIndex:
    return cast(
        "ThresholdIndex",
        NumbaDict.empty(  # type: ignore[no-untyped-call]
            key_type=types.int64,
            value_type=types.DictType(types.float32, types.int64),
        ),
    )


def _collect_raw_thresholds(rf: RandomForestLike) -> ThresholdMap:
    thresholds = _empty_threshold_map()
    for estimator in rf.estimators_:
        tree = estimator.tree_
        for raw_feature, raw_threshold in zip(
            tree.feature,
            tree.threshold,
            strict=True,
        ):
            feature = int(raw_feature)
            if feature < 0:
                continue
            if feature not in thresholds:
                thresholds[feature] = _empty_threshold_list()
            thresholds[feature].append(np.float32(raw_threshold))
    return thresholds


def _dedupe_sorted(raw_thresholds: ThresholdList | None) -> list[np.float32]:
    if raw_thresholds is None:
        return []

    raw_array = np.array(raw_thresholds, dtype=np.float32)
    if raw_array.size == 0:
        return []

    raw_array.sort()
    deduped = [np.float32(raw_array[0])]
    for value in raw_array[1:]:
        if value != deduped[-1]:
            deduped.append(np.float32(value))
    return deduped


def _with_bounds(
    values: Sequence[np.float32],
    lower: np.float32,
    upper: np.float32,
) -> ThresholdList:
    thresholds = _empty_threshold_list()
    thresholds.append(lower)
    for value in values:
        thresholds.append(value)
    thresholds.append(upper)
    return thresholds


def _data_bounds(values: np.ndarray) -> tuple[np.float32, np.float32]:
    return np.float32(float(np.min(values))), np.float32(float(np.max(values)))


def _discrete_bounds(values: np.ndarray) -> tuple[np.float32, np.float32]:
    lower, upper = _data_bounds(values)
    return np.float32(float(lower) + float(LOWER_SCALED)), upper


def _continuous_bounds(
    values: np.ndarray,
    mode: ContinuousBounds,
) -> tuple[np.float32, np.float32]:
    if mode == "scaled":
        return LOWER_SCALED, UPPER_SCALED
    return _data_bounds(values)


def _legacy_feature_thresholds(
    kind: FeatureKind,
    raw_thresholds: ThresholdList | None,
    values: np.ndarray,
) -> ThresholdList:
    deduped = _dedupe_sorted(raw_thresholds)
    if kind == "N":
        return _with_bounds(deduped, LOWER_SCALED, UPPER_SCALED)
    if kind in {"B", "C"}:
        return _with_bounds(deduped, LOWER_BINARY, UPPER_BINARY)
    if kind == "D":
        lower, upper = _discrete_bounds(values)
        return _with_bounds(deduped, lower, upper)
    msg = f"Unsupported feature kind: {kind!r}."
    raise ValueError(msg)


def _ocean_feature_thresholds(
    feature: FeatureMetadataLike,
    raw_thresholds: ThresholdList | None,
    values: np.ndarray,
    continuous_bounds: ContinuousBounds,
) -> ThresholdList:
    deduped = _dedupe_sorted(raw_thresholds)
    if feature.is_continuous:
        lower, upper = _continuous_bounds(values, continuous_bounds)
        return _with_bounds(deduped, lower, upper)
    if feature.is_binary or feature.is_one_hot_encoded:
        return _with_bounds(deduped, LOWER_BINARY, UPPER_BINARY)
    if feature.is_discrete:
        lower, upper = _discrete_bounds(values)
        return _with_bounds(deduped, lower, upper)

    msg = "Unsupported feature type."
    raise ValueError(msg)


def _build_threshold_index(thresholds: ThresholdMap) -> ThresholdIndex:
    thresh2idx = _empty_threshold_index()
    for feature, threshold_list in thresholds.items():
        inner = _empty_inner_threshold_index()
        for idx, threshold in enumerate(threshold_list):
            threshold_key = np.float32(cast("float | np.float32", threshold))
            inner[threshold_key] = idx
        thresh2idx[feature] = inner
    return thresh2idx


def _flatten_thresholds(
    thresholds: ThresholdMap,
    n_features: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lengths = np.array(
        [len(thresholds[feature]) for feature in range(n_features)],
        dtype=np.int64,
    )
    offsets = np.concatenate((
        np.array([0], dtype=np.int64),
        np.cumsum(lengths)[:-1],
    ))
    concatenated = np.concatenate([
        np.array(thresholds[feature], dtype=np.float32)
        for feature in range(n_features)
    ])
    return lengths, offsets, concatenated


def _forest_arrays(rf: RandomForestLike) -> ForestArrays:
    features: list[np.ndarray] = [
        estimator.tree_.feature for estimator in rf.estimators_
    ]
    thresholds: list[np.ndarray] = [
        estimator.tree_.threshold.astype(np.float32)
        for estimator in rf.estimators_
    ]
    values: list[np.ndarray] = [
        estimator.tree_.value for estimator in rf.estimators_
    ]
    children_left: list[np.ndarray] = [
        estimator.tree_.children_left for estimator in rf.estimators_
    ]
    children_right: list[np.ndarray] = [
        estimator.tree_.children_right for estimator in rf.estimators_
    ]
    return features, thresholds, values, children_left, children_right


def _build_result(
    rf: RandomForestLike,
    thresholds: ThresholdMap,
) -> ThresholdResult:
    lengths, offsets, concatenated = _flatten_thresholds(
        thresholds,
        rf.n_features_in_,
    )
    features, tree_thresholds, values, children_left, children_right = (
        _forest_arrays(rf)
    )
    return (
        thresholds,
        _build_threshold_index(thresholds),
        lengths,
        offsets,
        concatenated,
        features,
        tree_thresholds,
        values,
        children_left,
        children_right,
    )


def _column_name(column: object) -> str:
    if isinstance(column, tuple):
        return str(cast("object", column[0]))
    return str(column)


def _data_values(data: pd.DataFrame, column: object) -> np.ndarray:
    values: np.ndarray = np.asarray(data[column], dtype=np.float32)
    return values


def get_thresholds(
    rf: RandomForestLike,
    cols: Sequence[str],
    cols_types: Mapping[str, FeatureKind],
    X_final_array: np.ndarray,
) -> ThresholdResult:
    thresholds: ThresholdMap = _collect_raw_thresholds(rf)

    for feature in range(rf.n_features_in_):
        column = cols[feature].split(sep="__")[0]
        thresholds[feature] = _legacy_feature_thresholds(
            cols_types[column],
            thresholds.get(feature),
            X_final_array[:, feature],
        )

    return _build_result(rf, thresholds)


def get_thresholds_ocean(
    rf: RandomForestLike,
    data: pd.DataFrame,
    mapper: MapperLike,
    *,
    continuous_bounds: ContinuousBounds = "scaled",
) -> ThresholdResult:
    thresholds: ThresholdMap = _collect_raw_thresholds(rf)

    for feature in range(rf.n_features_in_):
        column = mapper.columns[feature]
        column_name = _column_name(column)
        thresholds[feature] = _ocean_feature_thresholds(
            mapper[column_name],
            thresholds.get(feature),
            _data_values(data, column),
            continuous_bounds,
        )

    return _build_result(rf, thresholds)


__all__ = [
    "FeatureMetadataLike",
    "MapperLike",
    "RandomForestLike",
    "get_thresholds",
    "get_thresholds_ocean",
]
