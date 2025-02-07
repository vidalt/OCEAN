from collections.abc import Iterable
from enum import Enum

import gurobipy as gp
import numpy as np
from pydantic import validate_call

from ..abc import Mapper
from ..feature import Feature
from ..tree import Tree
from ..typing import (
    Array1D,
    Key,
    NonNegativeArray1D,
    NonNegativeInt,
    PositiveInt,
    Unit,
    UnitO,
)
from .base import BaseModel
from .builder import ModelBuilder, ModelBuilderFactory
from .solution import Solution
from .variable import FeatureVar, TreeVar

Objective = gp.LinExpr | gp.QuadExpr


class Model(BaseModel):
    TREE_VAR_FMT: str = "tree[{t}]"
    FEATURE_VAR_FMT: str = "feature[{key}]"

    DEFAULT_EPSILON: Unit = 1.0 / (2.0**16)
    DEFAULT_NUM_EPSILON: Unit = 1.0 / 16.0

    class Type(Enum):
        MIP = "MIP"

    # Numer of isolators in the model.
    _n_isolators: NonNegativeInt

    # Weights for the estimators in the ensemble.
    _weights: NonNegativeArray1D

    # Tree variables in the ensemble.
    _trees: tuple[TreeVar, *tuple[TreeVar, ...]]

    # Feature variables in the ensemble.
    _mapper: Mapper[FeatureVar]

    # Constraints for the majority class.
    _scores: gp.tupledict[NonNegativeInt, gp.Constr]

    # Model builder for the ensemble.
    _builder: ModelBuilder

    # Numerical parameters for the model.
    # - epsilon: the minimum difference between two scores.
    # - delta: the percentage of data points that are isolators.
    # - num_epsilon: the minimum difference between two numerical values.
    _epsilon: Unit
    _delta: UnitO
    _num_epsilon: Unit

    # Solution for the model.
    _solution: Solution

    # Garbage collector for the model.
    # - Used to keep track of the variables and constraints created,
    #   and to remove them when the model is cleared.
    _garbage: list[gp.Var | gp.MVar | gp.Constr | gp.MConstr]

    def __init__(
        self,
        trees: Iterable[Tree],
        mapper: Mapper[Feature],
        *,
        weights: NonNegativeArray1D | None = None,
        n_isolators: NonNegativeInt = 0,
        name: str = "OCEAN",
        env: gp.Env | None = None,
        epsilon: Unit = DEFAULT_EPSILON,
        num_epsilon: Unit = DEFAULT_NUM_EPSILON,
        delta: UnitO = 0.0,
        model_type: Type = Type.MIP,
        flow_type: TreeVar.FlowType = TreeVar.FlowType.CONTINUOUS,
    ) -> None:
        super().__init__(name=name, env=env)
        self._set_trees(trees=trees, flow_type=flow_type)
        self._n_isolators = n_isolators
        self._set_features(mapper=mapper)
        self._set_weights(weights=weights)
        self._scores = gp.tupledict()
        self._epsilon = epsilon
        self._num_epsilon = num_epsilon
        self._delta = delta
        self._set_builder(model_type=model_type)
        self._set_solution()
        self._set_garbage()

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
    def mapper(self) -> Mapper[FeatureVar]:
        return self._mapper

    @property
    def weights(self) -> NonNegativeArray1D:
        return self._weights

    @property
    def solution(self) -> Solution:
        return self._solution

    def build(self) -> None:
        self._build_trees()
        self._build_features()
        self._build_isolation()
        self._builder.build(self, trees=self.trees, mapper=self.mapper)

    def add_objective(
        self,
        x: Array1D,
        *,
        norm: int = 1,
        sense: int = gp.GRB.MINIMIZE,
    ) -> None:
        objective = self._add_objective(x=x, norm=norm)
        self.setObjective(objective, sense=sense)

    @property
    def function(self) -> gp.MLinExpr:
        return self.weighted_function(weights=self._weights)

    def weighted_function(
        self,
        weights: NonNegativeArray1D,
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
        if m_class >= self.n_classes:
            msg = f"Expected class < {self.n_classes}, got {m_class}"
            raise ValueError(msg)

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

    def cleanup(self) -> None:
        self.clear_majority_class()
        self.remove(self._garbage)
        self._garbage.clear()

    def _set_trees(
        self,
        trees: Iterable[Tree],
        *,
        flow_type: TreeVar.FlowType,
    ) -> None:
        def create(t: NonNegativeInt, tree: Tree) -> TreeVar:
            name = self.TREE_VAR_FMT.format(t=t)
            return TreeVar(tree, name=name, flow_type=flow_type)

        trees = tuple(trees)
        tree_vars = tuple(map(create, range(len(trees)), trees))
        if len(tree_vars) == 0:
            msg = "At least one tree is required."
            raise ValueError(msg)

        self._trees = tree_vars[0], *tree_vars[1:]

    def _set_features(self, mapper: Mapper[Feature]) -> None:
        def create(key: Key, feature: Feature) -> FeatureVar:
            name = self.FEATURE_VAR_FMT.format(key=key)
            return FeatureVar(feature, name=name)

        if len(mapper) == 0:
            msg = "At least one feature is required."
            raise ValueError(msg)

        self._mapper = mapper.transform(create)

    def _set_weights(self, weights: NonNegativeArray1D | None = None) -> None:
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
        self._solution = Solution(self.mapper)

    def _set_garbage(self) -> None:
        self._garbage = []

    def _set_majority_class(
        self,
        m_class: NonNegativeInt,
        *,
        function: gp.MLinExpr,
        class_: NonNegativeInt,
        output: NonNegativeInt,
    ) -> None:
        rhs = self._epsilon if class_ < m_class else 0.0
        e = (function[output, m_class] - function[output, class_]).item() >= rhs
        self._scores[class_] = self.addConstr(e)

    def _build_trees(self) -> None:
        for var in self._trees:
            var.build(self)

    def _build_features(self) -> None:
        for var in self.mapper.values():
            var.build(self)

    def _build_isolation(self) -> None:
        if self.n_isolators == 0:
            return

        expr = gp.LinExpr()
        for tree in self.isolators:
            for leaf in tree.leaves:
                expr += leaf.depth * tree[leaf.node_id]
        self.addConstr(expr >= self._delta * self.n_isolators)

    def _add_objective(self, x: Array1D, norm: int) -> Objective:
        if x.size != self.mapper.n_columns:
            msg = f"Expected {self.mapper.n_columns} values, got {x.size}"
            raise ValueError(msg)

        match norm:
            case 1:
                return self._add_l1(x)
            case 2:
                return self._add_l2(x)
            case _:
                msg = f"Unsupported norm: {norm}"
                raise ValueError(msg)

    def _add_l2(self, x: Array1D) -> gp.QuadExpr:
        obj = gp.QuadExpr()
        n = self.mapper.n_columns
        for i in range(n):
            value: np.float64 = x[i]
            name = self.mapper.names[i]
            var = self.mapper[name]
            if not var.is_one_hot_encoded:
                x_var = var.x
            else:
                code = self.mapper.codes[i]
                x_var = var[code]
            obj += (x_var - value) ** 2
        return obj

    def _add_l1(self, x: Array1D) -> gp.LinExpr:
        n = self.mapper.n_columns
        u = self.addMVar(n, name="u")
        self._garbage.append(u)
        for i in range(n):
            value: np.float64 = x[i]
            name = self.mapper.names[i]
            var = self.mapper[name]
            if not var.is_one_hot_encoded:
                x_var = var.x
            else:
                code = self.mapper.codes[i]
                x_var = var[code]
            self._garbage.extend((
                self.addConstr(u[i] >= x_var - value),
                self.addConstr(u[i] >= -(x_var - value)),
            ))
        return u.sum().item()
