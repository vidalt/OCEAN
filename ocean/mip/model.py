from collections.abc import Hashable, Iterable, Mapping
from enum import Enum

import gurobipy as gp
import numpy as np

from ..feature import Feature
from ..tree import Tree
from ..typing import FloatArray1D
from .base import BaseModel
from .builder import ModelBuilder, ModelBuilderFactory
from .solution import Solution
from .variable import FeatureVar, TreeVar


class Model(BaseModel):
    TREE_VAR_FMT: str = "tree[{t}]"
    FEATURE_VAR_FMT: str = "feature[{key}]"

    DEFAULT_EPSILON: float = 1e-6
    DEFAULT_NUM_EPSILON: float = 1.0 / 16.0

    class Type(Enum):
        MIP = "MIP"

    _trees: tuple[TreeVar, ...]
    _features: dict[Hashable, FeatureVar]

    _weights: FloatArray1D

    _scores: gp.tupledict[int, gp.Constr]
    _builder: ModelBuilder

    _epsilon: float
    _num_epsilon: float

    _solution: Solution

    def __init__(
        self,
        trees: Iterable[Tree],
        features: Mapping[Hashable, Feature],
        weights: FloatArray1D | None = None,
        *,
        name: str = "OCEAN",
        env: gp.Env | None = None,
        epsilon: float = DEFAULT_EPSILON,
        num_epsilon: float = DEFAULT_NUM_EPSILON,
        model_type: Type = Type.MIP,
        flow_type: TreeVar.FlowType = TreeVar.FlowType.CONTINUOUS,
    ) -> None:
        super().__init__(name=name, env=env)
        self._set_trees(trees=trees, flow_type=flow_type)
        self._set_features(features=features)
        self._set_weights(weights=weights)
        self._scores = gp.tupledict()
        self._epsilon = epsilon
        self._num_epsilon = num_epsilon
        self._set_builder(model_type=model_type)
        self._set_solution()

    @property
    def n_trees(self) -> int:
        return len(self._trees)

    @property
    def trees(self) -> tuple[TreeVar, ...]:
        return self._trees

    @property
    def shape(self) -> tuple[int, ...]:
        return self._trees[0].shape

    @property
    def n_classes(self) -> int:
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
        self._builder.build(self, trees=self._trees, features=self._features)

    @property
    def function(self) -> gp.MLinExpr:
        return self.weighted_function(weights=self._weights)

    def weighted_function(self, weights: FloatArray1D) -> gp.MLinExpr:
        function = gp.MLinExpr.zeros(self.shape)
        for t in range(self.n_trees):
            function += np.float64(weights[t]) * self._trees[t].value
        return function

    def set_majority_class(self, m_class: int, *, output: int = 0) -> None:
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
        def f(tup: tuple[int, Tree]) -> TreeVar:
            t, tree = tup
            name = self.TREE_VAR_FMT.format(t=t)
            return TreeVar(tree, name=name, flow_type=flow_type)

        self._trees = tuple(map(f, enumerate(trees)))

    def _set_features(self, features: Mapping[Hashable, Feature]) -> None:
        def f(tup: tuple[Hashable, Feature]) -> tuple[Hashable, FeatureVar]:
            key, feature = tup
            name = self.FEATURE_VAR_FMT.format(key=key)
            return key, FeatureVar(feature, name=name)

        self._features = dict(map(f, features.items()))

    def _set_weights(self, weights: FloatArray1D | None = None) -> None:
        if weights is None:
            weights = np.ones(self.n_trees, dtype=np.float64)

        if len(weights) != self.n_trees:
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
