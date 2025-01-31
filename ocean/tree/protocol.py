from typing import Protocol

import numpy as np
from sklearn.tree._tree import Tree as SKLearnTree


class TreeProtocol(Protocol):
    feature: np.ndarray[tuple[int], np.dtype[np.int32]]
    threshold: np.ndarray[tuple[int], np.dtype[np.float64]]
    left: np.ndarray[tuple[int], np.dtype[np.int32]]
    right: np.ndarray[tuple[int], np.dtype[np.int32]]
    value: np.ndarray[tuple[int, ...], np.dtype[np.float64]]


class SKLearnTreeProtocol(TreeProtocol):
    def __init__(self, tree: SKLearnTree) -> None:
        self.feature = tree.feature  # pyright: ignore[reportAttributeAccessIssue]
        self.threshold = tree.threshold  # pyright: ignore[reportAttributeAccessIssue]
        self.left = tree.children_left  # pyright: ignore[reportAttributeAccessIssue]
        self.right = tree.children_right  # pyright: ignore[reportAttributeAccessIssue]
        self.value = tree.value
