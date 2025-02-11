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
    NonNegativeArray1D,
    NonNegativeInt,
    Unit,
)
from ._base import BaseModel
from ._builders.model import ModelBuilder, ModelBuilderFactory
from ._managers import FeatureManager, GarbageManager, TreeManager
from ._typing import Objective
from ._variables import TreeVar


class Model(BaseModel, FeatureManager, TreeManager, GarbageManager):
    DEFAULT_EPSILON: Unit = 1.0 / (2.0**16)
    DEFAULT_NUM_EPSILON: Unit = 1.0 / (2.0**6)

    class Type(Enum):
        MIP = "MIP"

    # Constraints for the majority class.
    _scores: gp.tupledict[tuple[NonNegativeInt, NonNegativeInt], gp.Constr]

    # Model builder for the ensemble.
    _builder: ModelBuilder

    # Numerical parameters for the model.
    # - epsilon: the minimum difference between two scores.
    # - num_epsilon: the minimum difference between two numerical values.
    _epsilon: Unit
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
        model_type: Type = Type.MIP,
        flow_type: TreeVar.FlowType = TreeVar.FlowType.CONTINUOUS,
    ) -> None:
        # Initialize the super models.
        BaseModel.__init__(self, name=name, env=env)
        TreeManager.__init__(
            self,
            trees=trees,
            weights=weights,
            n_isolators=n_isolators,
            max_samples=max_samples,
            flow_type=flow_type,
        )
        FeatureManager.__init__(self, mapper=mapper)
        GarbageManager.__init__(self)

        self._set_weights(weights=weights)
        self._max_samples = max_samples
        self._epsilon = epsilon
        self._num_epsilon = num_epsilon
        self._scores = gp.tupledict()
        self._set_builder(model_type=model_type)

    def build(self) -> None:
        self.build_features(self)
        self.build_trees(self)
        self._builder.build(self, trees=self.trees, mapper=self.mapper)
        self._set_isolation()

    def add_objective(
        self,
        x: Array1D,
        *,
        norm: int = 1,
        sense: int = gp.GRB.MINIMIZE,
    ) -> None:
        objective = self._add_objective(x=x, norm=norm)
        self.setObjective(objective, sense=sense)

    @validate_call
    def set_majority_class(
        self,
        y: NonNegativeInt,
        *,
        op: NonNegativeInt = 0,
    ) -> None:
        if y >= self.n_classes:
            msg = f"Expected class < {self.n_classes}, got {y}"
            raise ValueError(msg)

        self._set_majority_class(y, op=op)

    def clear_majority_class(self) -> None:
        self.remove(self._scores)
        self._scores.clear()

    def cleanup(self) -> None:
        self.clear_majority_class()
        self.remove_garbage(self)

    def _set_builder(self, model_type: Type) -> None:
        match model_type:
            case Model.Type.MIP:
                epsilon = self._num_epsilon
                self._builder = ModelBuilderFactory.MIP(epsilon=epsilon)

    def _set_majority_class(
        self,
        y: NonNegativeInt,
        *,
        op: NonNegativeInt,
    ) -> None:
        function = self.function

        for class_ in range(self.n_classes):
            if class_ == y:
                continue

            rhs = self._epsilon if class_ < y else 0.0
            lhs = (function[op, y] - function[op, class_]).item()
            self._scores[op, class_] = self.addConstr(lhs >= rhs)

    def _set_isolation(self) -> None:
        if self.n_isolators == 0:
            return

        self.addConstr(self.length >= self.min_length)

    def _add_objective(self, x: Array1D, norm: int) -> Objective:
        if x.size != self.mapper.n_columns:
            msg = f"Expected {self.mapper.n_columns} values, got {x.size}"
            raise ValueError(msg)
        if norm not in {1, 2}:
            msg = f"Unsupported norm: {norm}"
            raise ValueError(msg)

        variables = map(self.vget, range(self.n_columns))
        if norm == 1:
            return sum(map(self.L1, x, variables), start=gp.LinExpr())
        return sum(map(self.L2, x, variables), start=gp.QuadExpr())

    def L1(self, x: np.float64, v: gp.Var) -> gp.LinExpr:
        u = self.addVar()
        neg = self.addConstr(u >= v - x)
        pos = self.addConstr(u >= x - v)
        self.add_garbage(u, pos, neg)
        return gp.LinExpr(u)

    @staticmethod
    def L2(x: np.float64, v: gp.Var) -> gp.QuadExpr:
        return (v - x) ** 2
