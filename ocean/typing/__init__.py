"""Shared type aliases and protocols used throughout OCEAN."""

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Annotated, Protocol

import numpy as np
import pandas as pd
import xgboost as xgb
from pydantic import Field
from sklearn.ensemble import (
    AdaBoostClassifier,
    IsolationForest,
    RandomForestClassifier,
)

type BaseExplainableEnsemble = (
    RandomForestClassifier | xgb.XGBClassifier | AdaBoostClassifier
)
type ParsableEnsemble = BaseExplainableEnsemble | IsolationForest | xgb.Booster

type Number = float
type NonNegativeNumber = Annotated[Number, Field(ge=0.0)]
type PositiveInt = Annotated[int, Field(ge=1)]
type NonNegativeInt = Annotated[int, Field(ge=0)]
type NonNegative = Annotated[np.float64, Field(ge=0.0)]
type Unit = Annotated[float, Field(gt=0.0, lt=1.0)]
type UnitO = Annotated[float, Field(ge=0.0, lt=1.0)]
type NodeId = Annotated[np.int64, Field(ge=-1)]

# Key alias:
# - This is used to represent the name of a feature
#   or the code of a one-hot encoded feature.
type Key = int | str

# Index alias:
if TYPE_CHECKING:
    type Index1L = pd.Index[Key]
    type Index = pd.Index[int] | pd.Index[str] | pd.MultiIndex
else:
    type Index1L = pd.Index
    type Index = pd.Index | pd.MultiIndex

# Arrays aliases

# Int arrays:
# 1D, 2D, and nD arrays of integers.
IntDtype = np.dtype[np.int64]
IntArray1D = np.ndarray[tuple[int], IntDtype]
IntArray2D = np.ndarray[tuple[int, int], IntDtype]
IntArray = np.ndarray[tuple[int, ...], IntDtype]

# Positive Int arrays:
# 1D, 2D, and nD arrays of positive integers.
NonNegativeIntDtype = np.dtype[np.uint32]
NonNegativeIntArray1D = np.ndarray[tuple[int], NonNegativeIntDtype]
NonNegativeIntArray2D = np.ndarray[tuple[int, int], NonNegativeIntDtype]
NonNegativeIntArray = np.ndarray[tuple[int, ...], NonNegativeIntDtype]

# Float arrays:
# 1D, 2D, and nD arrays of floats (64 bits).
Dtype = np.dtype[np.float64]
Array1D = np.ndarray[tuple[int], Dtype]
Array2D = np.ndarray[tuple[int, int], Dtype]
Array = np.ndarray[tuple[int, ...], Dtype]

# 1D, 2D, and nD arrays of non-negative floats (64 bits).
NonNegativeDtype = np.dtype[NonNegative]
NonNegativeArray1D = np.ndarray[tuple[int], NonNegativeDtype]
NonNegativeArray2D = np.ndarray[tuple[int, int], NonNegativeDtype]
NonNegativeArray = np.ndarray[tuple[int, ...], NonNegativeDtype]

# NodeId arrays:
# 1D:
NodeIdDtype = np.dtype[NodeId]
NodeIdArray1D = np.ndarray[tuple[int], NodeIdDtype]


# Scikit-learn Tree alias:
# This class is only used for type hinting purposes.
class SKLearnTree(Protocol):
    """Protocol capturing the subset of the sklearn tree API OCEAN uses."""

    node_count: PositiveInt
    max_depth: NonNegativeInt
    feature: NonNegativeIntArray1D
    threshold: Array1D
    children_left: NodeIdArray1D
    children_right: NodeIdArray1D
    n_node_samples: NonNegativeIntArray1D
    value: Array


type XGBTree = pd.DataFrame


class BaseExplanation(Protocol):
    """Protocol implemented by explanation containers returned by explainers."""

    def to_numpy(self) -> Array1D: ...
    def to_series(self) -> pd.Series: ...
    @property
    def x(self) -> Array1D: ...
    @property
    def value(self) -> Mapping[Key, Key | Number]: ...
    @property
    def query(self) -> Array1D: ...

    @staticmethod
    def _next_float32_up(value: float) -> float:
        return float(
            np.nextafter(
                np.float32(value),
                np.float32(np.inf),
                dtype=np.float32,
            )
        )

    @staticmethod
    def _next_float32_down(value: float) -> float:
        return float(
            np.nextafter(
                np.float32(value),
                np.float32(-np.inf),
                dtype=np.float32,
            )
        )


class BaseExplainer(Protocol):
    """Protocol implemented by all public OCEAN explainers."""

    def get_objective_value(self) -> float: ...
    def get_distance(self) -> float: ...
    def get_solving_status(self) -> str: ...
    def get_anytime_solutions(self) -> list[dict[str, float]] | None: ...

    def explain(
        self,
        x: Array1D,
        *,
        y: NonNegativeInt,
        norm: NonNegativeInt,
        return_callback: bool = False,
        verbose: bool = False,
        max_time: int = 60,
        num_workers: int | None = None,
        random_seed: int = 42,
        clean_up: bool = True,
    ) -> BaseExplanation | None: ...

    def cleanup(self) -> None: ...


class DecisionPathMatrix(Protocol):
    """Sparse decision-path matrix returned by sklearn forests."""

    indptr: np.ndarray
    indices: np.ndarray


class LocalSearchTree(Protocol):
    """Tree attributes read by LS preprocessing and local importance."""

    node_count: int
    feature: np.ndarray
    threshold: np.ndarray
    value: np.ndarray
    children_left: np.ndarray
    children_right: np.ndarray
    weighted_n_node_samples: np.ndarray
    impurity: np.ndarray


class LocalSearchEstimator(Protocol):
    """Estimator wrapper exposing a sklearn-like tree."""

    tree_: LocalSearchTree


class LocalSearchForest(Protocol):
    """Forest-like subset required by the LS heuristic backend."""

    n_features_in_: PositiveInt
    n_estimators: PositiveInt
    estimators_: Sequence[LocalSearchEstimator]
    feature_importances_: np.ndarray

    def decision_path(
        self,
        x: object,
    ) -> tuple[DecisionPathMatrix, np.ndarray]: ...

    def predict(self, x: object) -> np.ndarray: ...


class LocalSearchExplainer(Protocol):
    """Explainer state consumed by DLS/SLS helper functions."""

    rf: LocalSearchForest
    max_distance: np.float32
    continuous_col: np.ndarray
    binary_col: np.ndarray
    discrete_col: np.ndarray
    one_hot_encoded_col: Sequence[Sequence[int]]
    lengths_list: np.ndarray
    offsets: np.ndarray
    thresholds_concat: np.ndarray
    features_: Sequence[np.ndarray]
    thresholds_: Sequence[np.ndarray]
    values_: Sequence[np.ndarray]
    children_left_: Sequence[np.ndarray]
    children_right_: Sequence[np.ndarray]
    thresh2idx: Sequence[Mapping[np.float32, int]]
    weights: np.ndarray
    base_scores: np.ndarray
    score_kind: int
    normalize_leaf_values: bool
    inf: np.ndarray
    sup: np.ndarray
    rank_maps: Sequence[np.ndarray]

    def encode(self, leaves_rank_array: np.ndarray) -> np.int64: ...


__all__ = [
    "Array",
    "Array1D",
    "Array2D",
    "BaseExplainableEnsemble",
    "BaseExplainer",
    "BaseExplanation",
    "DecisionPathMatrix",
    "Dtype",
    "Index",
    "Index1L",
    "IntArray",
    "IntArray1D",
    "IntArray2D",
    "IntDtype",
    "Key",
    "LocalSearchEstimator",
    "LocalSearchExplainer",
    "LocalSearchForest",
    "LocalSearchTree",
    "NodeId",
    "NodeIdArray1D",
    "NodeIdDtype",
    "NonNegative",
    "NonNegativeArray",
    "NonNegativeArray1D",
    "NonNegativeArray2D",
    "NonNegativeDtype",
    "NonNegativeInt",
    "NonNegativeIntArray",
    "NonNegativeIntArray1D",
    "NonNegativeIntArray2D",
    "NonNegativeIntDtype",
    "NonNegativeNumber",
    "Number",
    "ParsableEnsemble",
    "PositiveInt",
    "SKLearnTree",
    "Unit",
    "UnitO",
    "XGBTree",
]
