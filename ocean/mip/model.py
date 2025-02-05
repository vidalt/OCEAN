from collections.abc import Hashable, Iterable, Mapping
from enum import Enum

import gurobipy as gp
import numpy as np
from pydantic import validate_call

from ..feature import Feature
from ..tree import Tree
from ..typing import (
    FloatUnit,
    FloatUnitHalfOpen,
    NonNegativeFloatArray1D,
    NonNegativeInt,
    PositiveInt,
)
from .base import BaseModel
from .builder import ModelBuilder, ModelBuilderFactory
from .solution import Solution
from .variable import FeatureVar, TreeVar


class Model(BaseModel):
    TREE_VAR_FMT: str = "tree[{t}]"
    FEATURE_VAR_FMT: str = "feature[{key}]"

    DEFAULT_EPSILON: FloatUnit = 1e-6
    DEFAULT_NUM_EPSILON: FloatUnit = 1.0 / 16.0

    class Type(Enum):
        MIP = "MIP"

    _trees: tuple[TreeVar, *tuple[TreeVar, ...]]
    _n_isolators: NonNegativeInt
    _weights: NonNegativeFloatArray1D

    _features: dict[Hashable, FeatureVar]

    _scores: gp.tupledict[int, gp.Constr]
    _builder: ModelBuilder

    _epsilon: FloatUnit
    _delta: FloatUnitHalfOpen
    _num_epsilon: FloatUnit

    _solution: Solution

    def __init__(
        self,
        trees: Iterable[Tree],
        features: Mapping[Hashable, Feature],
        weights: NonNegativeFloatArray1D | None = None,
        *,
        n_isolators: NonNegativeInt = 0,
        name: str = "OCEAN",
        env: gp.Env | None = None,
        epsilon: FloatUnit = DEFAULT_EPSILON,
        num_epsilon: FloatUnit = DEFAULT_NUM_EPSILON,
        delta: FloatUnitHalfOpen = 0.0,
        model_type: Type = Type.MIP,
        flow_type: TreeVar.FlowType = TreeVar.FlowType.CONTINUOUS,
    ) -> None:
        super().__init__(name=name, env=env)
        self._set_trees(trees=trees, flow_type=flow_type)
        self._n_isolators = n_isolators
        self._set_features(features=features)
        self._set_weights(weights=weights)
        self._scores = gp.tupledict()
        self._epsilon = epsilon
        self._num_epsilon = num_epsilon
        self._delta = delta
        self._set_builder(model_type=model_type)
        self._set_solution()

    @property
    def n_estimators(self) -> PositiveInt:
        return self.n_trees - self.n_isolators

    @property
    def n_isolators(self) -> NonNegativeInt:
        return self._n_isolators

    @property
    def n_trees(self) -> PositiveInt:
        return len(self._trees)

    @property
    def trees(self) -> tuple[TreeVar, ...]:
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
    def features(self) -> Mapping[Hashable, FeatureVar]:
        return self._features

    @property
    def solution(self) -> Solution:
        return self._solution

    def build(self) -> None:
        self._build_trees()
        self._build_features()
        self._build_isolation()
        self._builder.build(self, trees=self._trees, features=self._features)

    @property
    def function(self) -> gp.MLinExpr:
        return self.weighted_function(weights=self._weights)

    def weighted_function(
        self,
        weights: NonNegativeFloatArray1D,
    ) -> gp.MLinExpr:
        function = gp.MLinExpr.zeros(self.shape)
        for t in range(self.n_estimators):
            function += np.float64(weights[t]) * self._trees[t].value
        return function

    @validate_call
    def set_majority_class(
        self,
        m_class: NonNegativeInt,
        *,
        output: NonNegativeInt = 0,
    ) -> None:
        function = self.function
        for class_ in range(self.n_classes):
            if class_ == m_class:
                continue
            self._set_majority_class(
                m_class=m_class,
                function=function,
                class_=class_,
                output=output,
            )

    def clear_majority_class(self) -> None:
        self.remove(self._scores)
        self._scores.clear()

    def _set_trees(
        self,
        trees: Iterable[Tree],
        *,
        flow_type: TreeVar.FlowType,
    ) -> None:
        def create(tup: tuple[NonNegativeInt, Tree]) -> TreeVar:
            t, tree = tup
            name = self.TREE_VAR_FMT.format(t=t)
            return TreeVar(tree, name=name, flow_type=flow_type)

        tree_vars = tuple(map(create, enumerate(trees)))
        if len(tree_vars) == 0:
            msg = "At least one tree is required."
            raise ValueError(msg)

        self._trees = tree_vars[0], *tree_vars[1:]

    def _set_features(self, features: Mapping[Hashable, Feature]) -> None:
        def create(
            tup: tuple[Hashable, Feature],
        ) -> tuple[Hashable, FeatureVar]:
            key, feature = tup
            name = self.FEATURE_VAR_FMT.format(key=key)
            return key, FeatureVar(feature, name=name)

        self._features = dict(map(create, features.items()))

    def _set_weights(
        self,
        weights: NonNegativeFloatArray1D | None = None,
    ) -> None:
        if weights is None:
            weights = np.ones(self.n_estimators, dtype=np.float64)

        if len(weights) != self.n_estimators:
            msg = "The number of weights must match the number of trees."
            raise ValueError(msg)

        self._weights = weights

    def _set_builder(self, model_type: Type) -> None:
        match model_type:
            case Model.Type.MIP:
                epsilon = self._num_epsilon
                self._builder = ModelBuilderFactory.MIP(epsilon=epsilon)

    def _set_solution(self) -> None:
        self._solution = Solution(features=self._features)

    def _set_majority_class(
        self,
        m_class: int,
        *,
        function: gp.MLinExpr,
        class_: int,
        output: int,
    ) -> None:
        rhs = self._epsilon if class_ < m_class else 0.0
        e = (function[output, m_class] - function[output, class_]).item() >= rhs
        self._scores[class_] = self.addConstr(e)

    def _build_trees(self) -> None:
        for var in self._trees:
            var.build(self)

    def _build_features(self) -> None:
        for var in self._features.values():
            var.build(self)

    def _build_isolation(self) -> None:
        if self.n_isolators == 0:
            return

        expr = gp.LinExpr()
        for tree in self.isolators:
            for leaf in tree.leaves:
                expr += leaf.depth * tree[leaf.node_id]
        self.addConstr(expr >= self._delta * self.n_isolators)
