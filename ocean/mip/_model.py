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
    """
    Mixed-integer programming formulation for tree ensemble explanations.

    The feature variables encode the counterfactual point ``x`` and the tree
    variables encode the active leaf or path decisions ``p_(t,l)``.
    """

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
        model_type: "Model.Type" = Type.MIP,
        flow_type: "TreeVar.FlowType" = TreeVar.FlowType.CONTINUOUS,
    ) -> None:
        """
        Initialize an empty MIP model for a parsed ensemble.

        Parameters
        ----------
        trees
            Parsed trees whose leaves define the ensemble class scores.
        mapper
            Feature mapper describing how processed query columns map to
            decision variables.
        weights
            Optional non-negative tree weights.
        n_isolators
            Number of isolation trees contributing to the auxiliary isolation
            constraint on the path length.
        max_samples
            Reference sample count used by the isolation-forest extension.
        name
            Name of the underlying Gurobi model.
        env
            Optional Gurobi environment.
        epsilon
            Classification margin used in the pairwise score constraints.
        num_epsilon
            Strictness constant used in numerical split implications.
        model_type
            Backend builder variant.
        flow_type
            Encoding used for the tree flow variables.

        """
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
        r"""
        Create the decision variables and structural constraints of the model.

        This step introduces the feature variables encoding :math:`x`, the
        tree-path variables :math:`p_{t,\ell}`, and the constraints linking
        both so that exactly one leaf is active in each tree and the selected
        leaves are consistent with the feature values.
        """
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
        r"""
        Attach the distance objective :math:`d(x, \hat{x})` to the MIP model.

        Parameters
        ----------
        x
            Query point :math:`\hat{x}` in the processed feature space. The
            code parameter is named ``x``, but mathematically it represents the
            query.
        norm
            Distance norm used for :math:`d(x, \hat{x})`. The MIP backend
            supports :math:`L_1` and :math:`L_2`.
        sense
            Optimization sense passed to Gurobi. Counterfactual search uses
            minimization.

        """
        objective = self._add_objective(x=x, norm=norm)
        self.setObjective(objective, sense=sense)

    @validate_call
    def set_majority_class(
        self,
        y: NonNegativeInt,
        *,
        op: NonNegativeInt = 0,
    ) -> None:
        r"""
        Enforce the target class through pairwise score constraints.

        For every competing class, this adds

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

    def clear_majority_class(self) -> None:
        r"""
        Remove the current target-class constraints.

        This deletes stored inequalities of the form

        .. math::

           f_y(x) \ge f_c(x) + \varepsilon_c
        """
        self.remove(self._scores)
        self._scores.clear()

    def cleanup(self) -> None:
        r"""
        Remove query-specific objective auxiliaries and class constraints.

        After ``cleanup()``, the structural encoding of :math:`x` and
        :math:`p_{t,\ell}` remains, but temporary constraints created for a
        specific query :math:`\hat{x}` are removed.
        """
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
        r"""
        Encode the pairwise score dominance constraints.

        The stored constraints are

        .. math::

           f_y(x) - f_c(x) \ge \varepsilon_c,
           \qquad \forall c \neq y.

        """
        function = self.function

        for class_ in range(self.n_classes):
            if class_ == y:
                continue

            rhs = self._epsilon if class_ < y else 0.0
            lhs = (function[op, y] - function[op, class_]).item()
            self._scores[op, class_] = self.addConstr(lhs >= rhs)

    def _set_isolation(self) -> None:
        """
        Add the optional isolation-forest length constraint.

        When isolation trees are present, this constrains the aggregate path
        length variable to remain above the minimum admissible value.
        """
        if self.n_isolators == 0:
            return

        self.addConstr(self.length >= self.min_length)

    def _add_objective(self, x: Array1D, norm: int) -> Objective:
        r"""
        Build the symbolic objective expression for :math:`d(x, \hat{x})`.

        Each feature variable contributes either an :math:`L_1` or
        :math:`L_2` term. One-hot encoded coordinates use a factor
        :math:`1/2` so that switching category is not double counted.

        Returns
        -------
        Objective
            Linear or quadratic expression representing
            :math:`d(x, \hat{x})`.

        Raises
        ------
        ValueError
            If ``x`` does not have the expected size or ``norm`` is
            unsupported.

        """
        if x.size != self.mapper.n_columns:
            msg = f"Expected {self.mapper.n_columns} values, got {x.size}"
            raise ValueError(msg)
        if norm not in {1, 2}:
            msg = f"Unsupported norm: {norm}"
            raise ValueError(msg)
        names = self.mapper.names
        is_ohe = [self.mapper[name].is_one_hot_encoded for name in names]
        variables = map(self.vget, range(self.n_columns))
        if norm == 1:
            return sum(
                (
                    self.L1(x, v, is_ohe=ohe)
                    for x, v, ohe in zip(x, variables, is_ohe, strict=True)
                ),
                start=gp.LinExpr(),
            )
        return sum(
            (
                self.L2(x, v, is_ohe=ohe)
                for x, v, ohe in zip(x, variables, is_ohe, strict=True)
            ),
            start=gp.QuadExpr(),
        )

    def L1(self, x: np.float64, v: gp.Var, *, is_ohe: bool) -> gp.LinExpr:
        r"""
        Return the MIP :math:`L_1` contribution of one coordinate of :math:`x`.

        This creates an auxiliary variable :math:`u` such that
        :math:`u \ge |x_j - \hat{x}_j|`, where ``v`` encodes the
        counterfactual coordinate :math:`x_j` and the code parameter ``x``
        stores the query coordinate :math:`\hat{x}_j`.

        Parameters
        ----------
        x
            Query coordinate :math:`\hat{x}_j`.
        v
            Model variable encoding the corresponding coordinate of
            :math:`x`.
        is_ohe
            Whether this coordinate belongs to a one-hot block
            :math:`u_{j,k}`. If so, the returned term is halved.

        Returns
        -------
        gp.LinExpr
            Linear expression equal to the contribution of that coordinate to
            :math:`d_1(x, \hat{x})`.

        """
        u = self.addVar()
        neg = self.addConstr(u >= v - x)
        pos = self.addConstr(u >= x - v)
        self.add_garbage(u, pos, neg)
        return gp.LinExpr(u) / 2.0 if is_ohe else gp.LinExpr(u)

    @staticmethod
    def L2(x: np.float64, v: gp.Var, *, is_ohe: bool) -> gp.QuadExpr:
        r"""
        Return the MIP :math:`L_2` contribution of one coordinate of :math:`x`.

        The returned expression is :math:`(x_j - \hat{x}_j)^2`, halved for
        one-hot encoded coordinates to preserve the same category-switch
        semantics as the :math:`L_1` objective.

        Returns
        -------
        gp.QuadExpr
            Quadratic expression equal to the contribution of that coordinate
            to :math:`d_2(x, \hat{x})`.

        """
        return ((v - x) ** 2) / 2.0 if is_ohe else (v - x) ** 2
