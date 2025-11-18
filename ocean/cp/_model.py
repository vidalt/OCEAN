from collections.abc import Iterable
from enum import Enum

import numpy as np
from ortools.sat.python import cp_model as cp
from pydantic import validate_call

from ..abc import Mapper
from ..feature import Feature
from ..tree import Tree
from ..typing import (
    Array1D,
    Key,
    NonNegativeArray1D,
    NonNegativeInt,
)
from ._base import BaseModel
from ._builder.model import ModelBuilder, ModelBuilderFactory
from ._managers import FeatureManager, GarbageManager, TreeManager
from ._variables import FeatureVar


class Model(BaseModel, FeatureManager, TreeManager, GarbageManager):
    DEFAULT_EPSILON: int = 1
    _obj_scale: int = int(1e8)

    class Type(Enum):
        CP = "CP"

    # Constraints for the majority class.
    _scores: dict[tuple[NonNegativeInt, NonNegativeInt], cp.Constraint]

    # Model builder for the ensemble.
    _builder: ModelBuilder

    def __init__(
        self,
        trees: Iterable[Tree],
        mapper: Mapper[Feature],
        *,
        weights: NonNegativeArray1D | None = None,
        max_samples: NonNegativeInt = 0,
        epsilon: int = DEFAULT_EPSILON,
        model_type: Type = Type.CP,
    ) -> None:
        BaseModel.__init__(self)
        TreeManager.__init__(
            self,
            trees=trees,
            weights=weights,
        )
        FeatureManager.__init__(self, mapper=mapper)
        GarbageManager.__init__(self)

        self._set_weights(weights=weights)
        self._max_samples = max_samples
        self._epsilon = epsilon
        self._scores = {}
        self._set_builder(model_type=model_type)

    def build(self) -> None:
        self.build_features(self)
        self.build_trees(self)
        self._builder.build(self, trees=self.trees, mapper=self.mapper)

    def add_objective(
        self,
        x: Array1D,
        *,
        norm: int = 1,
    ) -> None:
        objective = self._add_objective(x=x, norm=norm)
        self.Minimize(objective)

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

    def _set_builder(self, model_type: Type) -> None:
        match model_type:
            case Model.Type.CP:
                self._builder = ModelBuilderFactory.CP()

    def _set_majority_class(
        self,
        y: NonNegativeInt,
        *,
        op: NonNegativeInt,
    ) -> None:
        for class_ in range(self.n_classes):
            if class_ == y:
                continue

            rhs = self._epsilon if class_ < y else 0
            lhs = cp.LinearExpr.WeightedSum(
                [self.function[op, y], self.function[op, class_]],
                [1, -1],
            )
            self._scores[op, class_] = self.Add(lhs >= rhs)
            self.add_garbage(self._scores[op, class_])

    def cleanup(self) -> None:
        self.remove_garbage()

    def _add_objective(self, x: Array1D, norm: int) -> cp.ObjLinearExprT:
        if x.size != self.mapper.n_columns:
            msg = f"Expected {self.mapper.n_columns} values, got {x.size}"
            raise ValueError(msg)
        if norm != 1:
            msg = f"Unsupported norm: {norm}"
            raise ValueError(msg)
        x_arr = np.asarray(x, dtype=float).ravel()

        variables = self.mapper.values()
        names = [n for n, _ in self.mapper.items()]
        objective: cp.LinearExpr = 0  # type: ignore[assignment]
        k = 0
        indexer = self.mapper.idx
        for v, name in zip(variables, names, strict=True):
            if v.is_one_hot_encoded:
                for code in v.codes:
                    idx = indexer.get(name, code)
                    objective += self.L1(x_arr[idx], v, code=code)
                    k += 1
            else:
                objective += self.L1(x_arr[k], v)
                k += 1
        return objective

    def get_intervals_cost(self, levels: Array1D, x: float) -> list[int]:
        intervals_cost = np.zeros(len(levels) - 1, dtype=int)
        for i in range(len(intervals_cost)):
            if levels[i] < x <= levels[i + 1]:
                continue
            if levels[i] > x:
                intervals_cost[i] = int(abs(x - levels[i]) * self._obj_scale)
            elif levels[i + 1] < x:
                intervals_cost[i] = int(
                    abs(x - levels[i + 1]) * self._obj_scale
                )
        return intervals_cost.tolist()

    def L1(
        self,
        x: np.float64,
        v: FeatureVar,
        code: Key | None = None,
    ) -> cp.LinearExpr:
        obj_exprs: list[cp.LinearExpr] = []
        obj_coefs: list[int] = []
        if v.is_numeric:
            if v.is_continuous:
                intervals_cost = self.get_intervals_cost(v.levels, x)
                # tighten domain of objvar based on x itself ----------
                v.objvarget().Proto().domain[:] = []
                v.objvarget().Proto().domain.extend(
                    cp.Domain(
                        min(intervals_cost), max(intervals_cost)
                    ).FlattenedIntervals()
                )
                # -----------------------------------------------------
                obj_expr = v.objvarget()
                self.add_garbage(
                    self.AddElement(v.xget(), list(intervals_cost), obj_expr)
                )
                obj_coefs.append(1)
            else:
                # include the value of x itself on the domain ---------
                if len(v.thresholds) != 0:
                    values = [
                        val
                        for v in v.thresholds
                        for val in [int(v), int(v) + 1]
                    ]
                else:
                    values = np.asarray(v.levels).astype(int).tolist()
                values.append(int(x))
                v.xget().Proto().domain[:] = []
                v.xget().Proto().domain.extend(
                    cp.Domain.FromValues(values).FlattenedIntervals()
                )
                # -----------------------------------------------------
                obj_expr = v.objvarget()
                self.add_garbage(
                    self.AddAbsEquality(obj_expr, int(x) - v.xget())
                )
                obj_coefs.append(self._obj_scale)
            obj_exprs.append(obj_expr)
        elif v.is_one_hot_encoded:
            obj_expr = v.xget(code) if x == 0.0 else 1 - v.xget(code)  # type: ignore[assignment]
            obj_exprs.append(obj_expr)
            obj_coefs.append(self._obj_scale // 2)
        else:
            obj_expr = v.xget() if x == 0.0 else 1 - v.xget()  # type: ignore[assignment]
            obj_exprs.append(obj_expr)
            obj_coefs.append(self._obj_scale)
        return cp.LinearExpr.WeightedSum(obj_exprs, obj_coefs)
