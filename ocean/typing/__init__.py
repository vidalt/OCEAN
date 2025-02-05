from typing import Annotated, Protocol

import numpy as np
from pydantic import Field
from sklearn.ensemble import IsolationForest, RandomForestClassifier

BaseEnsemble = RandomForestClassifier | IsolationForest

Number = float
PositiveInt = Annotated[int, Field(ge=1)]
NonNegativeInt = Annotated[int, Field(ge=0)]
NonNegativeFloat = Annotated[np.float64, Field(ge=0.0)]
FloatUnit = Annotated[float, Field(gt=0.0, lt=1.0)]
FloatUnitHalfOpen = Annotated[float, Field(ge=0.0, lt=1.0)]
NodeId = Annotated[np.int64, Field(ge=-1)]


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
FloatDtype = np.dtype[np.float64]
FloatArray1D = np.ndarray[tuple[int], FloatDtype]
FloatArray2D = np.ndarray[tuple[int, int], FloatDtype]
FloatArray = np.ndarray[tuple[int, ...], FloatDtype]

# 1D, 2D, and nD arrays of non-negative floats (64 bits).
NonNegativeFloatDtype = np.dtype[NonNegativeFloat]
NonNegativeFloatArray1D = np.ndarray[tuple[int], NonNegativeFloatDtype]
NonNegativeFloatArray2D = np.ndarray[tuple[int, int], NonNegativeFloatDtype]
NonNegativeFloatArray = np.ndarray[tuple[int, ...], NonNegativeFloatDtype]

# NodeId arrays:
# 1D:
NodeIdDtype = np.dtype[NodeId]
NodeIdArray1D = np.ndarray[tuple[int], NodeIdDtype]


# Scikit-learn Tree alias:
# This class is only used for type hinting purposes.
class SKLearnTree(Protocol):
    node_count: NonNegativeInt
    max_depth: NonNegativeInt
    feature: NonNegativeIntArray1D
    threshold: FloatArray1D
    children_left: NodeIdArray1D
    children_right: NodeIdArray1D
    value: FloatArray


__all__ = [
    "BaseEnsemble",
    "FloatArray",
    "FloatArray1D",
    "FloatArray2D",
    "FloatDtype",
    "FloatUnit",
    "FloatUnitHalfOpen",
    "IntArray",
    "IntArray1D",
    "IntArray2D",
    "IntDtype",
    "NonNegativeFloatArray",
    "NonNegativeFloatArray1D",
    "NonNegativeFloatArray2D",
    "NonNegativeFloatDtype",
    "NonNegativeFloatDtype",
    "NonNegativeInt",
    "NonNegativeIntArray",
    "NonNegativeIntArray1D",
    "NonNegativeIntArray2D",
    "NonNegativeIntDtype",
    "Number",
    "PositiveInt",
]
