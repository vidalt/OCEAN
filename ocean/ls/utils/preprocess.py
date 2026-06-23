from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol, cast

import numpy as np
from numba import types  # type: ignore[attr-defined]
from numba.typed.typeddict import Dict as NumbaDict
from numba.typed.typedlist import List as NumbaList
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier

from ...tree import parse_ensembles
from .tools import SCORING_MARGIN, SCORING_PROBABILITY

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, MutableMapping, Sequence

    import pandas as pd

    from ...abc import Mapper
    from ...feature import Feature
    from ...tree import Tree
    from ...tree._node import Node
    from ...typing import BaseExplainableEnsemble, Index, Key, LocalSearchForest

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
TREE_LEAF_FEATURE = np.int32(-2)
BINARY_SPLIT_THRESHOLD = np.float32(0.5)
NUM_BINARY_CLASSES = 2


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


@dataclass(frozen=True)
class LocalSearchBackend:
    forest: LocalSearchForest
    weights: np.ndarray
    base_scores: np.ndarray
    score_kind: int
    normalize_leaf_values: bool


@dataclass(frozen=True)
class _DecisionPathMatrix:
    indptr: np.ndarray
    indices: np.ndarray


@dataclass(frozen=True)
class _ArrayTree:
    node_count: int
    feature: np.ndarray
    threshold: np.ndarray
    value: np.ndarray
    children_left: np.ndarray
    children_right: np.ndarray
    weighted_n_node_samples: np.ndarray
    impurity: np.ndarray


@dataclass(frozen=True)
class _ArrayEstimator:
    tree_: _ArrayTree


class _ParsedLocalSearchForest:
    def __init__(
        self,
        trees: Iterable[Tree],
        *,
        mapper: Mapper[Feature],
        weights: np.ndarray,
        base_scores: np.ndarray,
        score_kind: int,
        normalize_leaf_values: bool,
        feature_importances: np.ndarray | None,
        adaboost: bool = False,
    ) -> None:
        estimators = tuple(
            _ArrayEstimator(_tree_to_arrays(tree, mapper, adaboost=adaboost))
            for tree in trees
        )
        if len(estimators) == 0:
            msg = "At least one tree is required."
            raise ValueError(msg)

        self.estimators_ = estimators
        self.n_estimators = len(estimators)
        self.n_features_in_ = mapper.n_columns
        self.weights = weights.astype(np.float32, copy=True)
        self.base_scores = base_scores.astype(np.float32, copy=True)
        self.score_kind = score_kind
        self.normalize_leaf_values = normalize_leaf_values
        self.feature_importances_ = (
            _split_feature_importances(estimators, self.n_features_in_)
            if feature_importances is None
            else np.asarray(feature_importances, dtype=np.float64)
        )

    def decision_path(
        self,
        x: object,
    ) -> tuple[_DecisionPathMatrix, np.ndarray]:
        points = _as_points(x)
        tree_ptrs = self._tree_ptrs()
        indptr = np.empty(points.shape[0] + 1, dtype=np.int32)
        indices: list[int] = []
        indptr[0] = 0

        for row_idx, point in enumerate(points):
            for tree_idx, estimator in enumerate(self.estimators_):
                offset = int(tree_ptrs[tree_idx])
                tree = estimator.tree_
                node = 0
                while True:
                    indices.append(offset + node)
                    left = int(tree.children_left[node])
                    if left == -1:
                        break
                    feature = int(tree.feature[node])
                    threshold = float(tree.threshold[node])
                    node = (
                        left
                        if float(point[feature]) <= threshold
                        else int(tree.children_right[node])
                    )
            indptr[row_idx + 1] = len(indices)

        matrix = _DecisionPathMatrix(
            indptr=indptr,
            indices=np.asarray(indices, dtype=np.int32),
        )
        return matrix, tree_ptrs

    def predict(self, x: object) -> np.ndarray:
        points = _as_points(x)
        labels = np.empty(points.shape[0], dtype=np.int64)
        for row_idx, point in enumerate(points):
            scores = self._score_point(point)
            labels[row_idx] = int(np.argmax(scores))
        return labels

    def _tree_ptrs(self) -> np.ndarray:
        counts = [estimator.tree_.node_count for estimator in self.estimators_]
        return np.concatenate((
            np.array([0], dtype=np.int32),
            np.cumsum(counts, dtype=np.int32),
        ))

    def _score_point(self, point: np.ndarray) -> np.ndarray:
        n_classes = self.base_scores.shape[0]
        scores = (
            self.base_scores.astype(np.float32, copy=True)
            if self.score_kind == SCORING_MARGIN
            else np.zeros(n_classes, dtype=np.float32)
        )
        total_weight = np.float32(0.0)

        for weight, estimator in zip(
            self.weights, self.estimators_, strict=True
        ):
            tree = estimator.tree_
            leaf = _leaf_id(tree, point)
            value = np.asarray(tree.value[leaf][0], dtype=np.float32)
            if self.normalize_leaf_values:
                value_sum = np.float32(np.sum(value))
                if value_sum != 0:
                    value /= value_sum
            scores += np.float32(weight) * value
            total_weight += np.float32(weight)

        if self.score_kind == SCORING_PROBABILITY and total_weight != 0:
            scores /= total_weight
        return scores


def _as_points(x: object) -> np.ndarray:
    points = np.asarray(x, dtype=np.float32)
    if points.ndim == 1:
        return points.reshape(1, -1)
    return points


def _iter_nodes(node: Node) -> Iterable[Node]:
    yield node
    if not node.is_leaf:
        yield from _iter_nodes(node.left)
        yield from _iter_nodes(node.right)


def _compact_node_index(root: Node) -> dict[int, int]:
    return {id(node): idx for idx, node in enumerate(_iter_nodes(root))}


def _node_feature_index(node: Node, mapper: Mapper[Feature]) -> int:
    name = node.feature
    feature = mapper[name]
    if feature.is_one_hot_encoded:
        return int(mapper.idx.get(name, node.code))
    return int(mapper.idx.get(name))


def _node_threshold(node: Node, mapper: Mapper[Feature]) -> np.float32:
    feature = mapper[node.feature]
    if feature.is_binary or feature.is_one_hot_encoded:
        return BINARY_SPLIT_THRESHOLD
    return np.float32(node.threshold)


def _leaf_value(node: Node, *, adaboost: bool) -> np.ndarray:
    value = np.asarray(node.value, dtype=np.float32)
    if value.ndim == 1:
        value = value.reshape(1, -1)

    if not adaboost:
        return value

    encoded = np.zeros_like(value, dtype=np.float32)
    winners = np.argmax(value, axis=1)
    for output_idx, winner in enumerate(winners):
        encoded[output_idx, int(winner)] = np.float32(1.0)
    return encoded


def _tree_to_arrays(
    tree: Tree,
    mapper: Mapper[Feature],
    *,
    adaboost: bool,
) -> _ArrayTree:
    nodes = tuple(_iter_nodes(tree.root))
    node_index = _compact_node_index(tree.root)
    n_nodes = len(nodes)

    features = np.full(n_nodes, TREE_LEAF_FEATURE, dtype=np.int32)
    thresholds = np.zeros(n_nodes, dtype=np.float32)
    values = np.zeros((n_nodes, *tree.shape), dtype=np.float32)
    children_left = np.full(n_nodes, -1, dtype=np.int32)
    children_right = np.full(n_nodes, -1, dtype=np.int32)
    weighted_samples = np.ones(n_nodes, dtype=np.float32)
    impurity = np.zeros(n_nodes, dtype=np.float32)

    for node in nodes:
        idx = node_index[id(node)]
        weighted_samples[idx] = np.float32(max(1, int(node.n_samples)))
        if node.is_leaf:
            values[idx] = _leaf_value(node, adaboost=adaboost)
            continue

        features[idx] = _node_feature_index(node, mapper)
        thresholds[idx] = _node_threshold(node, mapper)
        children_left[idx] = node_index[id(node.left)]
        children_right[idx] = node_index[id(node.right)]

    return _ArrayTree(
        node_count=n_nodes,
        feature=features,
        threshold=thresholds,
        value=values,
        children_left=children_left,
        children_right=children_right,
        weighted_n_node_samples=weighted_samples,
        impurity=impurity,
    )


def _leaf_id(tree: _ArrayTree, point: np.ndarray) -> int:
    node = 0
    while True:
        left = int(tree.children_left[node])
        if left == -1:
            return node
        feature = int(tree.feature[node])
        threshold = float(tree.threshold[node])
        node = left if float(point[feature]) <= threshold else int(
            tree.children_right[node]
        )


def _split_feature_importances(
    estimators: Sequence[_ArrayEstimator],
    n_features: int,
) -> np.ndarray:
    importances = np.zeros(n_features, dtype=np.float64)
    for estimator in estimators:
        tree = estimator.tree_
        for feature in tree.feature[tree.feature >= 0]:
            importances[int(feature)] += 1.0

    total = importances.sum()
    if total > 0:
        importances /= total
    return importances


def _feature_importances(
    ensemble: object,
    n_features: int,
) -> np.ndarray | None:
    value = getattr(ensemble, "feature_importances_", None)
    if value is None:
        return None

    importances = np.asarray(value, dtype=np.float64)
    if importances.shape != (n_features,):
        return None
    return importances


def _n_classes_from_forest(forest: RandomForestClassifier) -> int:
    first_tree_value = cast("np.ndarray", forest.estimators_[0].tree_.value)
    return int(first_tree_value.shape[-1])


def _xgb_base_scores(tree: Tree) -> np.ndarray:
    n_classes = int(tree.shape[-1])
    raw = np.asarray(tree.logit, dtype=np.float32).ravel()
    scores = np.zeros(n_classes, dtype=np.float32)
    if n_classes == NUM_BINARY_CLASSES and raw.size == 1:
        scores[1] = raw[0]
    elif raw.size == n_classes:
        scores[:] = raw
    elif raw.size == 1:
        scores[:] = raw[0]
    else:
        msg = (
            "Could not align XGBoost base scores with the parsed tree "
            f"shape: got {raw.size}, expected {n_classes}."
        )
        raise ValueError(msg)
    return scores


def _uniform_weights(n_estimators: int) -> np.ndarray:
    return np.ones(n_estimators, dtype=np.float32)


def prepare_local_search_backend(
    ensemble: BaseExplainableEnsemble,
    mapper: Mapper[Feature],
) -> LocalSearchBackend:
    if isinstance(ensemble, RandomForestClassifier):
        n_classes = _n_classes_from_forest(ensemble)
        return LocalSearchBackend(
            forest=cast("LocalSearchForest", ensemble),
            weights=_uniform_weights(ensemble.n_estimators),
            base_scores=np.zeros(n_classes, dtype=np.float32),
            score_kind=SCORING_PROBABILITY,
            normalize_leaf_values=True,
        )

    trees = parse_ensembles(ensemble, mapper=mapper)
    first_tree = trees[0]

    if isinstance(ensemble, AdaBoostClassifier):
        weights = np.asarray(
            ensemble.estimator_weights_[: len(trees)],
            dtype=np.float32,
        )
        forest = _ParsedLocalSearchForest(
            trees,
            mapper=mapper,
            weights=weights,
            base_scores=np.zeros(first_tree.shape[-1], dtype=np.float32),
            score_kind=SCORING_PROBABILITY,
            normalize_leaf_values=False,
            feature_importances=_feature_importances(
                ensemble,
                mapper.n_columns,
            ),
            adaboost=True,
        )
        return LocalSearchBackend(
            forest=cast("LocalSearchForest", forest),
            weights=weights,
            base_scores=forest.base_scores,
            score_kind=SCORING_PROBABILITY,
            normalize_leaf_values=False,
        )

    weights = _uniform_weights(len(trees))
    base_scores = _xgb_base_scores(first_tree)
    forest = _ParsedLocalSearchForest(
        trees,
        mapper=mapper,
        weights=weights,
        base_scores=base_scores,
        score_kind=SCORING_MARGIN,
        normalize_leaf_values=False,
        feature_importances=_feature_importances(
            ensemble,
            mapper.n_columns,
        ),
    )
    return LocalSearchBackend(
        forest=cast("LocalSearchForest", forest),
        weights=weights,
        base_scores=base_scores,
        score_kind=SCORING_MARGIN,
        normalize_leaf_values=False,
    )


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
    "LocalSearchBackend",
    "MapperLike",
    "RandomForestLike",
    "get_thresholds",
    "get_thresholds_ocean",
    "prepare_local_search_backend",
]
