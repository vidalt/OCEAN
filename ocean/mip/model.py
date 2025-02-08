from collections.abc import Iterable
from enum import Enum
from functools import reduce

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
    Number,
    PositiveInt,
    Unit,
    UnitO,
)
from .base import BaseModel, Var
from .builder import ModelBuilder, ModelBuilderFactory
from .solution import Solution
from .utils import average_length
from .variable import FeatureVar, TreeVar

type Objective = gp.LinExpr | gp.QuadExpr


class GarbageModel:
    type GurobiObject = gp.Var | gp.MVar | gp.Constr | gp.MConstr

    # Garbage collector for the model.
    # - Used to keep track of the variables and constraints created,
    #   and to remove them when the model is cleared.
    _garbage: list[GurobiObject]

    def __init__(self) -> None:
        self._garbage = []

    def add_garbage(self, *args: GurobiObject) -> None:
        self._garbage.extend(args)

    def cleanup(self) -> None:
        self._garbage.clear()


class FeatureModel:
    FEATURE_VAR_FMT: str = "feature[{key}]"

    # Mapper for the features in the model.
    _mapper: Mapper[FeatureVar]

    # Solution for the model.
    _solution: Solution

    def __init__(self, mapper: Mapper[Feature]) -> None:
        self._set_mapper(mapper)
        self._set_solution()

    @property
    def n_columns(self) -> PositiveInt:
        return self.mapper.n_columns

    @property
    def n_features(self) -> PositiveInt:
        return len(self.mapper)

    @property
    def mapper(self) -> Mapper[FeatureVar]:
        return self._mapper

    @property
    def solution(self) -> Solution:
        return self._solution

    def vget(self, i: int) -> gp.Var:
        name = self.mapper.names[i]
        if self.mapper[name].is_one_hot_encoded:
            code = self.mapper.codes[i]
            return self.mapper[name].xget(code)
        return self.mapper[name].xget()

    def _set_mapper(self, mapper: Mapper[Feature]) -> None:
        def create(key: Key, feature: Feature) -> FeatureVar:
            name = self.FEATURE_VAR_FMT.format(key=key)
            return FeatureVar(feature, name=name)

        if len(mapper) == 0:
            msg = "At least one feature is required."
            raise ValueError(msg)

        self._mapper = mapper.apply(create)

    def _set_solution(self) -> None:
        self._solution = Solution(self.mapper)


class TreeModel:
    TREE_VAR_FMT: str = "tree[{t}]"

    # Tree variables in the ensemble.
    _trees: tuple[TreeVar, *tuple[TreeVar, ...]]

    # Numer of isolators in the model.
    _n_isolators: NonNegativeInt

    def __init__(
        self,
        trees: Iterable[Tree],
        n_isolators: NonNegativeInt = 0,
        *,
        flow_type: TreeVar.FlowType = TreeVar.FlowType.CONTINUOUS,
    ) -> None:
        self._set_trees(trees=trees, flow_type=flow_type)
        self._n_isolators = n_isolators

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
    def length(self) -> gp.LinExpr:
        return sum((tree.length for tree in self.isolators), gp.LinExpr())

    def weighted_function(self, weights: NonNegativeArray1D) -> gp.MLinExpr:
        w = tuple(map(Number, weights))
        zeros = gp.MLinExpr.zeros(self.shape)
        trees = self.estimators
        return sum((w[t] * tree.value for t, tree in enumerate(trees)), zeros)

    def _set_trees(
        self,
        trees: Iterable[Tree],
        *,
        flow_type: TreeVar.FlowType,
    ) -> None:
        def create(item: tuple[int, Tree]) -> TreeVar:
            t, tree = item
            name = self.TREE_VAR_FMT.format(t=t)
            return TreeVar(tree, name=name, flow_type=flow_type)

        tree_vars = tuple(map(create, enumerate(trees)))
        if len(tree_vars) == 0:
            msg = "At least one tree is required."
            raise ValueError(msg)

        self._trees = tree_vars[0], *tree_vars[1:]


class Model(BaseModel, FeatureModel, TreeModel, GarbageModel):
    DEFAULT_EPSILON: Unit = 1.0 / (2.0**16)
    DEFAULT_NUM_EPSILON: Unit = 1.0 / (2.0**12)

    class Type(Enum):
        MIP = "MIP"

    # Maximum number of samples in the isolators.
    _max_samples: NonNegativeInt

    # Weights for the estimators in the ensemble.
    _weights: NonNegativeArray1D

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

    def __init__(
        self,
        trees: Iterable[Tree],
        mapper: Mapper[Feature],
        *,
        weights: NonNegativeArray1D | None = None,
        n_isolators: NonNegativeInt = 0,
        max_samples: NonNegativeInt = 0,
        name: str = "OCEAN",
        env: gp.Env | None = None,
        epsilon: Unit = DEFAULT_EPSILON,
        num_epsilon: Unit = DEFAULT_NUM_EPSILON,
        delta: UnitO = 0.0,
        model_type: Type = Type.MIP,
        flow_type: TreeVar.FlowType = TreeVar.FlowType.CONTINUOUS,
    ) -> None:
        # Initialize the super models.
        BaseModel.__init__(self, name=name, env=env)
        FeatureModel.__init__(self, mapper=mapper)
        TreeModel.__init__(
            self,
            trees=trees,
            n_isolators=n_isolators,
            flow_type=flow_type,
        )
        GarbageModel.__init__(self)

        self._max_samples = max_samples
        self._set_weights(weights=weights)
        self._scores = gp.tupledict()
        self._epsilon = epsilon
        self._num_epsilon = num_epsilon
        self._delta = delta
        self._set_builder(model_type=model_type)

    @property
    def max_samples(self) -> NonNegativeInt:
        return self._max_samples

    @property
    def weights(self) -> NonNegativeArray1D:
        return self._weights

    def build(self) -> None:
        self._build_features()
        self._build_trees()
        self._set_isolation()
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

    @validate_call
    def set_majority_class(
        self,
        y: NonNegativeInt,
        *,
        output: NonNegativeInt = 0,
    ) -> None:
        if y >= self.n_classes:
            msg = f"Expected class < {self.n_classes}, got {y}"
            raise ValueError(msg)

        self._set_majority_class(y, output=output)

    def clear_majority_class(self) -> None:
        self.remove(self._scores)
        self._scores.clear()

    def cleanup(self) -> None:
        self.clear_majority_class()
        self.remove(self._garbage)
        GarbageModel.cleanup(self)

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

    def _set_majority_class(
        self,
        y: NonNegativeInt,
        *,
        output: NonNegativeInt,
    ) -> None:
        function = self.function

        for class_ in range(self.n_classes):
            if class_ == y:
                continue

            rhs = self._epsilon if class_ < y else 0.0
            lhs = (function[output, y] - function[output, class_]).item()
            self._scores[class_] = self.addConstr(lhs >= rhs)

    def _set_isolation(self) -> None:
        if self.n_isolators == 0:
            return

        min_average_length = average_length(self.max_samples)
        self.addConstr(self.length >= min_average_length * self.n_isolators)

    def _build_vars(self, *variables: Var) -> None:
        for var in variables:
            var.build(self)

    def _build_features(self) -> None:
        self._build_vars(*self.mapper.values())

    def _build_trees(self) -> None:
        self._build_vars(*self.trees)

    def _add_objective(self, x: Array1D, norm: int) -> Objective:
        if x.size != self.mapper.n_columns:
            msg = f"Expected {self.mapper.n_columns} values, got {x.size}"
            raise ValueError(msg)
        if norm not in {1, 2}:
            msg = f"Unsupported norm: {norm}"
            raise ValueError(msg)

        if norm == 1:
            return reduce(self._l1, enumerate(x), gp.LinExpr())
        return reduce(self._l2, enumerate(x), gp.QuadExpr())

    def _l1(self, acc: gp.LinExpr, item: tuple[int, Number]) -> gp.LinExpr:
        i, val = item
        v = self.vget(i)
        u = self.addVar()
        pos = self.addConstr(u >= v - val)
        neg = self.addConstr(u >= val - v)
        self.add_garbage(u, pos, neg)
        return acc + u

    def _l2(self, acc: gp.QuadExpr, item: tuple[int, Number]) -> gp.QuadExpr:
        i, val = item
        v = self.vget(i)
        return acc + (v - val) ** 2
