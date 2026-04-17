from collections.abc import Iterable

import numpy as np
from ortools.sat.python import cp_model as cp

from ...tree import Tree
from ...tree._utils import average_length
from ...typing import Array1D, NonNegativeArray1D, NonNegativeInt, PositiveInt
from .._base import BaseModel
from .._variables import TreeVar


class TreeManager:
    r"""
    Manage CP tree variables and the scaled decision function :math:`f`.

    Each :class:`TreeVar` encodes the active leaf decisions :math:`p_{t,\ell}`
    of one tree. Aggregating those variables yields the integer-scaled CP
    representation of :math:`f(x)`.
    """

    TREE_VAR_FMT: str = "tree[{t}]"
    DEFAULT_SCORE_SCALE: int = int(1e10)
    XGBOOST_DEFAULT_CLASS: int = 1
    NUM_BINARY_CLASS: int = 2

    # Tree variables in the ensemble.
    _trees: tuple[TreeVar, *tuple[TreeVar, ...]]

    # Number of isolation trees in the model.
    _n_isolators: NonNegativeInt

    # Maximum number of samples used by the isolation trees.
    _max_samples: NonNegativeInt

    # Weights for the estimators in the ensemble.
    _weights: NonNegativeArray1D

    # Scaled aggregate isolation-path length.
    _length: cp.LinearExpr | int

    # Function of the ensemble.
    _function: dict[tuple[NonNegativeInt, NonNegativeInt], cp.LinearExpr]

    # Scale for the scores.
    _score_scale: int = DEFAULT_SCORE_SCALE

    # Base score for the ensemble.
    _logit: Array1D

    # Flag to indicate if the model is using XGBoost trees.
    _xgboost: bool = False

    # Flag to indicate if the model is using AdaBoost trees.
    _adaboost: bool = False

    def __init__(
        self,
        trees: Iterable[Tree],
        *,
        weights: NonNegativeArray1D | None = None,
        n_isolators: NonNegativeInt = 0,
        max_samples: NonNegativeInt = 0,
        scale: int = DEFAULT_SCORE_SCALE,
    ) -> None:
        """Initialize the tree manager and the score-scaling factor."""
        self._set_trees(trees=trees)
        self._n_isolators = n_isolators
        self._max_samples = max_samples
        self._set_weights(weights=weights)
        self._score_scale = scale

    def build_trees(self, model: BaseModel) -> None:
        r"""Create the CP tree variables and cache :math:`f(x)`."""
        model.build_vars(*self.trees)

        self._length = self._get_length()
        self._function = self._get_function()

    @property
    def n_trees(self) -> PositiveInt:
        return len(self.trees)

    @property
    def n_isolators(self) -> NonNegativeInt:
        return self._n_isolators

    @property
    def n_estimators(self) -> PositiveInt:
        return self.n_trees - self.n_isolators

    @property
    def trees(self) -> tuple[TreeVar, *tuple[TreeVar, ...]]:
        return self._trees

    @property
    def estimators(self) -> tuple[TreeVar, *tuple[TreeVar, ...]]:
        return self._trees[0], *self._trees[1 : self.n_estimators]

    @property
    def isolators(self) -> tuple[TreeVar, ...]:
        return self._trees[self.n_estimators :]

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
    def max_samples(self) -> NonNegativeInt:
        return self._max_samples

    @property
    def length(self) -> cp.LinearExpr | int:
        return self._length

    @property
    def length_scale(self) -> int:
        return self.trees[0].length_scale

    @property
    def min_average_length(self) -> float:
        return average_length(self.max_samples)

    @property
    def min_length(self) -> float:
        return self.n_isolators * self.min_average_length

    @property
    def min_length_scaled(self) -> int:
        return round(self.min_length * self.length_scale)

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
        """
        Wrap parsed trees in CP-specific :class:`TreeVar` objects.

        Raises
        ------
        ValueError
            If no tree is provided.

        """

        def create(item: tuple[int, Tree]) -> TreeVar:
            t, tree = item
            if tree.xgboost:
                self._logit = tree.logit
                self._xgboost = tree.xgboost
            if tree.adaboost:
                self._adaboost = tree.adaboost
            name = self.TREE_VAR_FMT.format(t=t)
            return TreeVar(tree, name=name)

        tree_vars = tuple(map(create, enumerate(trees)))
        if len(tree_vars) == 0:
            msg = "At least one tree is required."
            raise ValueError(msg)

        self._trees = tree_vars[0], *tree_vars[1:]

    def _set_weights(self, weights: NonNegativeArray1D | None = None) -> None:
        """
        Validate and store the ensemble weights.

        Raises
        ------
        ValueError
            If the number of weights does not match the number of estimators.

        """
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
        r"""
        Return the integer-scaled CP representation of :math:`f(x)`.

        The returned dictionary maps each output-class pair ``(op, c)`` to the
        corresponding linear expression for :math:`f_c(x)`.

        Returns
        -------
        dict[tuple[NonNegativeInt, NonNegativeInt], cp.LinearExpr]
            Dictionary of scaled class-score expressions.

        """
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
                        if self._adaboost:
                            # no need to scale since values are 0/1
                            # scaling is done later for tree weights
                            coeff = int(np.argmax(leaf.value[op, :]) == c)
                        else:
                            coeff = int(leaf.value[op, c] * scale)
                        coefs.append(coeff)
                        variables.append(tree[leaf.node_id])
                    tree_expr = cp.LinearExpr.WeightedSum(variables, coefs)
                    tree_exprs.append(tree_expr)
                    if self._adaboost:
                        tree_weights.append(int(weight * scale))
                    else:
                        tree_weights.append(int(weight))
                expr = cp.LinearExpr.WeightedSum(tree_exprs, tree_weights)
                if self._xgboost:
                    if (
                        n_classes == self.NUM_BINARY_CLASS
                        and c == self.XGBOOST_DEFAULT_CLASS
                    ):
                        expr += int(self._logit[0] * scale)
                    elif n_classes > self.NUM_BINARY_CLASS:
                        expr += int(self._logit[c] * scale)
                exprs[op, c] = expr
        return exprs

    def _get_function(
        self,
    ) -> dict[tuple[NonNegativeInt, NonNegativeInt], cp.LinearExpr]:
        r"""
        Return the cached CP representation of :math:`f(x)`.

        Returns
        -------
        dict[tuple[NonNegativeInt, NonNegativeInt], cp.LinearExpr]
            Cached class-score expressions.

        """
        return self.weighted_function(weights=self.weights)

    def _get_length(self) -> cp.LinearExpr | int:
        if self.n_isolators == 0:
            return 0
        return sum((tree.length for tree in self.isolators), start=0)
