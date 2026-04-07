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
    """
    Constraint-programming formulation behind :class:`ocean.cp.Explainer`.

    The feature variables encode the counterfactual point ``x`` and the tree
    variables encode the active leaf decisions ``p_(t,l)``.
    """

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
        model_type: "Model.Type" = Type.CP,
    ) -> None:
        """
        Initialize an empty CP-SAT model for a parsed ensemble.

        Parameters
        ----------
        trees
            Parsed trees contributing to the ensemble class scores.
        mapper
            Feature mapper aligning processed query coordinates with the
            finite-domain decision variables.
        weights
            Optional tree weights.
        max_samples
            Reserved parameter used by compatible higher-level explainers.
        epsilon
            Integer classification margin used in the pairwise score
            constraints.
        model_type
            Backend builder variant.

        """
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
        r"""
        Create the CP variables and structural constraints of the formulation.

        This builds the finite-domain feature variables for :math:`x`, the
        leaf/path variables :math:`p_{t,\ell}`, and the consistency
        constraints connecting both representations.
        """
        self.build_features(self)
        self.build_trees(self)
        self._builder.build(self, trees=self.trees, mapper=self.mapper)

    def add_objective(
        self,
        x: Array1D,
        *,
        norm: int = 1,
    ) -> None:
        r"""
        Minimize the scaled integer approximation of :math:`d_p(x, \hat{x})`.

        Parameters
        ----------
        x
            Query point :math:`\hat{x}` in the processed feature space. The
            code parameter is named ``x``, but mathematically it represents the
            query.
        norm
            Distance norm. The default is :math:`L_1`.

        """
        objective = self._add_objective(x=x, norm=norm)
        self.Minimize(objective)

    @validate_call
    def set_majority_class(
        self,
        y: NonNegativeInt,
        *,
        op: NonNegativeInt = 0,
    ) -> None:
        r"""
        Enforce the target class through pairwise score inequalities.

        For each competing class, this adds the integer-scaled constraint

        .. math::

           f_y(x) \ge f_c(x) + \varepsilon_c

        Raises
        ------
        ValueError
            If ``y`` is not a valid class index.

        """
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
        r"""
        Encode the target-class dominance constraints.

        The created constraints encode

        .. math::

           f_y(x) - f_c(x) \ge \varepsilon_c,
           \qquad \forall c \neq y.

        """
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
        r"""
        Remove query-specific constraints from the CP model.

        The persistent structure over :math:`x` and :math:`p_{t,\ell}` is
        kept, while objective auxiliaries and class constraints for the last
        query :math:`\hat{x}` are discarded.
        """
        self.remove_garbage()

    def _add_objective(self, x: Array1D, norm: int) -> cp.ObjLinearExprT:
        r"""
        Build the scaled linear expression for :math:`d_p(x, \hat{x})^p`.

        Continuous coordinates are costed through interval indices, while
        discrete, binary, and one-hot encoded features contribute exact
        scaled powered deviations.

        Returns
        -------
        cp.ObjLinearExprT
            Linear objective expression approximating
            :math:`d_p(x, \hat{x})^p`.

        Raises
        ------
        ValueError
            If ``x`` does not have the expected size.

        """
        if x.size != self.mapper.n_columns:
            msg = f"Expected {self.mapper.n_columns} values, got {x.size}"
            raise ValueError(msg)
        x_arr = np.asarray(x, dtype=float).ravel()

        variables = self.mapper.values()
        names = list(self.mapper.keys())
        objective: cp.LinearExpr = 0  # type: ignore[assignment]
        k = 0
        indexer = self.mapper.idx
        for v, name in zip(variables, names, strict=True):
            if v.is_one_hot_encoded:
                for code in v.codes:
                    idx = indexer.get(name, code)
                    objective += self.L1(x_arr[idx], v, code=code, norm=norm)
                    k += 1
            else:
                objective += self.L1(x_arr[k], v, norm=norm)
                k += 1
        return objective

    def get_intervals_cost(
        self,
        levels: Array1D,
        x: float,
        *,
        norm: int = 1,
    ) -> list[int]:
        r"""
        Return interval costs for a continuous feature encoded by thresholds.

        For a continuous coordinate :math:`x_j`, CP does not optimize directly
        over the real value. Instead it selects an interval between successive
        threshold levels. This helper assigns each interval the scaled
        :math:`L_p^p` distance to the query coordinate :math:`\hat{x}_j`.

        Returns
        -------
        list[int]
            Scaled costs for the threshold intervals representing that
            continuous feature.

        """
        intervals_cost = np.zeros(len(levels) - 1, dtype=int)
        for i in range(len(intervals_cost)):
            if levels[i] < x <= levels[i + 1]:
                continue
            if levels[i] > x:
                intervals_cost[i] = int(
                    abs(x - levels[i]) ** norm * self._obj_scale
                )
            elif levels[i + 1] < x:
                intervals_cost[i] = int(
                    abs(x - levels[i + 1]) ** norm * self._obj_scale
                )
        return intervals_cost.tolist()

    def get_values_cost(
        self,
        values: list[int],
        x: float,
        *,
        norm: int = 1,
    ) -> list[int]:
        r"""
        Return scaled :math:`L_p^p` costs for one finite-value domain.

        Returns
        -------
        list[int]
            Scaled powered costs for each admissible value.

        """
        return [
            int(abs(x - value) ** norm * self._obj_scale) for value in values
        ]

    def L1(
        self,
        x: np.float64,
        v: FeatureVar,
        code: Key | None = None,
        *,
        norm: int = 1,
    ) -> cp.LinearExpr:
        r"""
        Build the CP contribution of one feature to :math:`d_p(x, \hat{x})^p`.

        Parameters
        ----------
        x
            Query coordinate :math:`\hat{x}_j`.
        v
            CP feature variable representing one original feature or one
            one-hot block in the counterfactual point :math:`x`.
        code
            Optional category code :math:`k` when the feature is represented by
            one-hot variables :math:`u_{j,k}`.
        norm
            Distance norm :math:`p` used to build :math:`d_p(x, \hat{x})^p`.

        Returns
        -------
        cp.LinearExpr
            Scaled linear expression for the contribution of that feature to
            :math:`d_p(x, \hat{x})^p`.

        """
        obj_exprs: list[cp.LinearExpr] = []
        obj_coefs: list[int] = []
        if v.is_numeric:
            if v.is_continuous:
                intervals_cost = self.get_intervals_cost(v.levels, x, norm=norm)
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
                values = sorted({*values, int(x)})
                v.xget().Proto().domain[:] = []
                v.xget().Proto().domain.extend(
                    cp.Domain.FromValues(values).FlattenedIntervals()
                )
                # -----------------------------------------------------
                obj_expr = v.objvarget()
                if norm == 1:
                    self.add_garbage(
                        self.AddAbsEquality(obj_expr, int(x) - v.xget())
                    )
                    obj_coefs.append(self._obj_scale)
                else:
                    values_cost = self.get_values_cost(values, x, norm=norm)
                    value_idx = self.NewIntVar(0, len(values) - 1, "")
                    v.objvarget().Proto().domain[:] = []
                    v.objvarget().Proto().domain.extend(
                        cp.Domain(
                            min(values_cost), max(values_cost)
                        ).FlattenedIntervals()
                    )
                    self.add_garbage(
                        value_idx,
                        self.AddElement(value_idx, values, v.xget()),
                        self.AddElement(value_idx, values_cost, obj_expr),
                    )
                    obj_coefs.append(1)
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
