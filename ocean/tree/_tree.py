from collections.abc import Iterator

from pydantic import validate_call

from ..typing import Array1D, NonNegativeInt, PositiveInt
from ._node import Node


class Tree:
    root: Node
    _shape: tuple[NonNegativeInt, ...]
    _xgboost: bool = False
    _adaboost: bool = False

    def __init__(self, root: Node) -> None:
        self.root = root
        self._shape = root.leaves[0].value.shape

    @property
    def n_nodes(self) -> PositiveInt:
        return self.root.size

    @property
    def max_depth(self) -> NonNegativeInt:
        return self.root.height

    @property
    def leaves(self) -> tuple[Node, *tuple[Node, ...]]:
        return self.root.leaves

    @property
    def shape(self) -> tuple[NonNegativeInt, ...]:
        return self._shape

    @property
    def logit(self) -> Array1D:
        return self._base_score_prob

    @logit.setter
    def logit(self, value: Array1D) -> None:
        self._base_score_prob = value

    @property
    def xgboost(self) -> bool:
        return self._xgboost

    @xgboost.setter
    def xgboost(self, value: bool) -> None:
        self._xgboost = value

    @property
    def adaboost(self) -> bool:
        return self._adaboost

    @adaboost.setter
    def adaboost(self, value: bool) -> None:
        self._adaboost = value

    @validate_call
    def nodes_at(self, depth: NonNegativeInt) -> Iterator[Node]:
        return self._nodes_at(self.root, depth=depth)

    def _nodes_at(self, node: Node, *, depth: NonNegativeInt) -> Iterator[Node]:
        if depth == 0:
            yield node
        for child in node.children:
            yield from self._nodes_at(child, depth=depth - 1)
