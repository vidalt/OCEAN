from collections.abc import Iterable

import numpy as np

from ...tree import Tree
from ...typing import (
    NonNegativeArray1D,
    NonNegativeInt,
    PositiveInt,
)
from .._base import BaseModel
from .._variables import TreeVar


class TreeManager:
    TREE_VAR_FMT: str = "tree[{t}]"

    # Tree variables in the ensemble.
    _trees: tuple[TreeVar, *tuple[TreeVar, ...]]

    # Weights for the estimators in the ensemble.
    _weights: NonNegativeArray1D

    def __init__(
        self,
        trees: Iterable[Tree],
        *,
        weights: NonNegativeArray1D | None = None,
    ) -> None:
        self._set_trees(trees=trees)
        self._set_weights(weights=weights)

    def build_trees(self, model: BaseModel) -> None:
        model.build_vars(*self.trees)
        self._function = self._get_function()

    @property
    def n_trees(self) -> PositiveInt:
        return len(self.trees)

    @property
    def n_estimators(self) -> PositiveInt:
        return self.n_trees

    @property
    def trees(self) -> tuple[TreeVar, *tuple[TreeVar, ...]]:
        return self._trees

    @property
    def estimators(self) -> tuple[TreeVar, *tuple[TreeVar, ...]]:
        return self._trees[0], *self._trees[1 : self.n_estimators]

    @property
    def shape(self) -> tuple[NonNegativeInt, ...]:
        return self._trees[0].shape

    @property
    def n_classes(self) -> NonNegativeInt:
        return self.shape[-1]

    @property
    def weights(self) -> NonNegativeArray1D:
        return self._weights

    def _set_trees(
        self,
        trees: Iterable[Tree],
    ) -> None:
        def create(item: tuple[int, Tree]) -> TreeVar:
            t, tree = item
            name = self.TREE_VAR_FMT.format(t=t)
            return TreeVar(tree, name=name)

        tree_vars = tuple(map(create, enumerate(trees)))
        if len(tree_vars) == 0:
            msg = "At least one tree is required."
            raise ValueError(msg)

        self._trees = tree_vars[0], *tree_vars[1:]

    def _set_weights(self, weights: NonNegativeArray1D | None = None) -> None:
        if weights is None:
            weights = np.ones(self.n_estimators, dtype=np.float64)

        if len(weights) != self.n_estimators:
            msg = "The number of weights must match the number of trees."
            raise ValueError(msg)

        self._weights = weights

    def weighted_function(
        self,
    ) -> dict[tuple[NonNegativeInt, NonNegativeInt], list[int]]:
        func: dict[tuple[NonNegativeInt, NonNegativeInt], list[int]] = {}
        n_classes = self.n_classes
        n_outputs = self.shape[-2]
        for op in range(n_outputs):
            for c in range(n_classes):
                leaf_vars: list[int] = []
                for tree in self.estimators:
                    for leaf in tree.leaves:
                        leaf_class = int(np.argmax(leaf.value[op, :]))
                        if leaf_class == c:
                            leaf_vars.append(tree[leaf.node_id])
                func[op, c] = leaf_vars
        return func

    def _get_function(
        self,
    ) -> dict[tuple[NonNegativeInt, NonNegativeInt], list[int]]:
        return self.weighted_function()

    @property
    def function(
        self,
    ) -> dict[tuple[NonNegativeInt, NonNegativeInt], list[int]]:
        return self._function
