from typing import Protocol

import numpy as np

from ..typing import (
    FloatArray,
    FloatArray1D,
    NodeIdArray1D,
    NonNegativeInt,
    NonNegativeIntArray1D,
    PositiveInt,
    SKLearnTree,
)


class TreeProtocol(Protocol):
    n_nodes: PositiveInt
    max_depth: NonNegativeInt
    feature: NonNegativeIntArray1D
    threshold: FloatArray1D
    left: NodeIdArray1D
    right: NodeIdArray1D
    value: FloatArray


class SKLearnTreeProtocol(TreeProtocol):
    def __init__(self, tree: SKLearnTree) -> None:
        self.n_nodes = tree.node_count
        self.max_depth = tree.max_depth
        self.feature = tree.feature.astype(np.uint32)
        self.threshold = tree.threshold
        self.left = tree.children_left.astype(np.int64)
        self.right = tree.children_right.astype(np.int64)
        self.value = tree.value


__all__ = [
    "SKLearnTree",
    "SKLearnTreeProtocol",
    "TreeProtocol",
]
