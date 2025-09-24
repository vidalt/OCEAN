from collections.abc import Iterable

import numpy as np
from ortools.sat.python import cp_model as cp

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
    DEFAULT_SCORE_SCALE: int = int(1e10)

    # Tree variables in the ensemble.
    _trees: tuple[TreeVar, *tuple[TreeVar, ...]]

    # Weights for the estimators in the ensemble.
    _weights: NonNegativeArray1D

    # Function of the ensemble.
    _function: dict[tuple[NonNegativeInt, NonNegativeInt], cp.LinearExpr]

    # Scale for the scores.
    _score_scale: int = DEFAULT_SCORE_SCALE

    def __init__(
        self,
        trees: Iterable[Tree],
        *,
        weights: NonNegativeArray1D | None = None,
        scale: int = DEFAULT_SCORE_SCALE,
    ) -> None:
        self._set_trees(trees=trees)
        self._set_weights(weights=weights)
        self._score_scale = scale

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
    def score_scale(self) -> int:
        return self._score_scale

    @property
    def function(
        self,
    ) -> dict[tuple[NonNegativeInt, NonNegativeInt], cp.LinearExpr]:
        return self._function

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
        weights: NonNegativeArray1D,
    ) -> dict[tuple[NonNegativeInt, NonNegativeInt], cp.LinearExpr]:
        exprs: dict[tuple[NonNegativeInt, NonNegativeInt], cp.LinearExpr] = {}
        n_classes = self.n_classes
        n_outputs = self.shape[-2]
        scale = self._score_scale
        for op in range(n_outputs):
            for c in range(n_classes):
                tree_exprs: list[cp.LinearExpr] = []
                tree_weights: list[int] = []
                for tree, weight in zip(self.estimators, weights, strict=True):
                    coefs: list[int] = []
                    variables: list[cp.IntVar] = []
                    for leaf in tree.leaves:
                        coefs.append(int(leaf.value[op, c] * scale))
                        variables.append(tree[leaf.node_id])
                    tree_expr = cp.LinearExpr.WeightedSum(variables, coefs)
                    tree_exprs.append(tree_expr)
                    tree_weights.append(int(weight))
                expr = cp.LinearExpr.WeightedSum(tree_exprs, tree_weights)
                exprs[op, c] = expr
        return exprs

    def _get_function(
        self,
    ) -> dict[tuple[NonNegativeInt, NonNegativeInt], cp.LinearExpr]:
        return self.weighted_function(weights=self.weights)
