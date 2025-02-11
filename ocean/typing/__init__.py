from collections.abc import Mapping
from typing import Annotated, Protocol

import numpy as np
import pandas as pd
from pydantic import Field
from sklearn.ensemble import IsolationForest, RandomForestClassifier

type BaseExplainableEnsemble = RandomForestClassifier
type ParsableEnsemble = BaseExplainableEnsemble | IsolationForest

Number = float
NonNegativeNumber = Annotated[Number, Field(ge=0.0)]
PositiveInt = Annotated[int, Field(ge=1)]
NonNegativeInt = Annotated[int, Field(ge=0)]
NonNegative = Annotated[np.float64, Field(ge=0.0)]
Unit = Annotated[float, Field(gt=0.0, lt=1.0)]
UnitO = Annotated[float, Field(ge=0.0, lt=1.0)]
NodeId = Annotated[np.int64, Field(ge=-1)]

# Key alias:
# - This is used to represent the name of a feature
#   or the code of a one-hot encoded feature.
type Key = int | str

# Index alias:
type Index1L = pd.Index[Key]
type Index = pd.Index[int] | pd.Index[str] | pd.MultiIndex

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
    node_count: PositiveInt
    max_depth: NonNegativeInt
    feature: NonNegativeIntArray1D
    threshold: Array1D
    children_left: NodeIdArray1D
    children_right: NodeIdArray1D
    n_node_samples: NonNegativeIntArray1D
    value: Array


class BaseExplanation(Protocol):
    @property
    def x(self) -> Array1D: ...
    @property
    def value(self) -> Mapping[Key, Key | Number]: ...


class BaseExplainer(Protocol):
    def explain(
        self,
        x: Array1D,
        *,
        y: NonNegativeInt,
        norm: PositiveInt,
    ) -> BaseExplanation: ...


__all__ = [
    "Array",
    "Array1D",
    "Array2D",
    "Dtype",
    "Index",
    "Index1L",
    "IntArray",
    "IntArray1D",
    "IntArray2D",
    "IntDtype",
    "Key",
    "NodeId",
    "NodeIdArray1D",
    "NonNegativeArray",
    "NonNegativeArray1D",
    "NonNegativeArray2D",
    "NonNegativeDtype",
    "NonNegativeDtype",
    "NonNegativeInt",
    "NonNegativeIntArray",
    "NonNegativeIntArray1D",
    "NonNegativeIntArray2D",
    "NonNegativeIntDtype",
    "Number",
    "ParsableEnsemble",
    "PositiveInt",
    "Unit",
    "UnitO",
]
