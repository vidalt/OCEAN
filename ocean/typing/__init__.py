from typing import Protocol

import numpy as np
from sklearn.ensemble import RandomForestClassifier

BaseEnsemble = RandomForestClassifier

Number = int | float

# Arrays aliases

# Int arrays:
# 1D, 2D, and nD arrays of integers.
IntArray1D = np.ndarray[tuple[int], np.dtype[np.int32]]
IntArray2D = np.ndarray[tuple[int, int], np.dtype[np.int32]]
IntArray = np.ndarray[tuple[int, ...], np.dtype[np.int32]]

# Positive Int arrays:
# 1D, 2D, and nD arrays of positive integers.
PositiveIntArray1D = np.ndarray[tuple[int], np.dtype[np.uint32]]
PositiveIntArray2D = np.ndarray[tuple[int, int], np.dtype[np.uint32]]
PositiveIntArray = np.ndarray[tuple[int, ...], np.dtype[np.uint32]]

# Float arrays:
# 1D, 2D, and nD arrays of floats (64 bits).
FloatArray1D = np.ndarray[tuple[int], np.dtype[np.float64]]
FloatArray2D = np.ndarray[tuple[int, int], np.dtype[np.float64]]
FloatArray = np.ndarray[tuple[int, ...], np.dtype[np.float64]]


# Scikit-learn Tree alias:
# This class is only used for type hinting purposes.
class SKLearnTree(Protocol):
    node_count: int
    max_depth: int
    feature: PositiveIntArray1D
    threshold: FloatArray1D
    children_left: IntArray1D
    children_right: IntArray1D
    value: FloatArray


__all__ = [
    "BaseEnsemble",
    "FloatArray",
    "FloatArray1D",
    "FloatArray2D",
    "IntArray",
    "IntArray1D",
    "IntArray2D",
    "Number",
    "PositiveIntArray",
    "PositiveIntArray1D",
    "PositiveIntArray2D",
]
