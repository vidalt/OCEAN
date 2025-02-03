from typing import Protocol

from ..typing import (
    FloatArray,
    FloatArray1D,
    IntArray1D,
    PositiveIntArray1D,
    SKLearnTree,
)


class TreeProtocol(Protocol):
    n_nodes: int
    max_depth: int
    feature: PositiveIntArray1D
    threshold: FloatArray1D
    left: IntArray1D
    right: IntArray1D
    value: FloatArray


class SKLearnTreeProtocol(TreeProtocol):
    def __init__(self, tree: SKLearnTree) -> None:
        self.n_nodes = tree.node_count
        self.max_depth = tree.max_depth
        self.feature = tree.feature
        self.threshold = tree.threshold
        self.left = tree.children_left
        self.right = tree.children_right
        self.value = tree.value


__all__ = [
    "SKLearnTree",
    "SKLearnTreeProtocol",
    "TreeProtocol",
]
