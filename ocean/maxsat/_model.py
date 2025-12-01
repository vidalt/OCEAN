from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

import numpy as np
from pydantic import validate_call
from pysat.pb import PBEnc

from ..typing import NonNegativeInt
from ._base import BaseModel
from ._builder.model import ModelBuilder, ModelBuilderFactory
from ._managers import FeatureManager, GarbageManager, TreeManager

if TYPE_CHECKING:
    from collections.abc import Iterable

    from ..abc import Mapper
    from ..feature import Feature
    from ..tree import Tree
    from ..typing import Array1D, Key, NonNegativeArray1D, NonNegativeInt
    from ._variables import FeatureVar


class Model(BaseModel, FeatureManager, GarbageManager, TreeManager):
    # Model builder for the ensemble.
    _builder: ModelBuilder
    DEFAULT_EPSILON: int = 1
    _obj_scale: int = int(1e8)

    class Type(Enum):
        MAXSAT = "MAXSAT"

    def __init__(
        self,
        trees: Iterable[Tree],
        mapper: Mapper[Feature],
        *,
        weights: NonNegativeArray1D | None = None,
        max_samples: NonNegativeInt = 0,
        epsilon: int = DEFAULT_EPSILON,
        model_type: Type = Type.MAXSAT,
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
        if x.size != self.mapper.n_columns:
            msg = f"Expected {self.mapper.n_columns} values, got {x.size}"
            raise ValueError(msg)
        if norm != 1:
            msg = f"Unsupported norm: {norm}"
            raise ValueError(msg)

        x_arr = np.asarray(x, dtype=float).ravel()
        variables = self.mapper.values()
        names = [n for n, _ in self.mapper.items()]
        k = 0
        indexer = self.mapper.idx

        for v, name in zip(variables, names, strict=True):
            if v.is_one_hot_encoded:
                for code in v.codes:
                    idx = indexer.get(name, code)
                    self._add_soft_l1_ohe(x_arr[idx], v, code=code)
                    k += 1
            elif v.is_continuous:
                self._add_soft_l1_continuous(x_arr[k], v)
                k += 1
            elif v.is_discrete:
                self._add_soft_l1_discrete(x_arr[k], v)
                k += 1
            elif v.is_binary:
                self._add_soft_l1_binary(x_arr[k], v)
                k += 1
            else:
                k += 1

    def _add_soft_l1_binary(self, x_val: float, v: FeatureVar) -> None:
        """Add soft clause for binary feature."""
        weight = int(self._obj_scale)
        x_var = v.xget()
        binary_threshold = 0.5
        if x_val > binary_threshold:
            # If x=1, penalize flipping to 0
            self.add_soft([x_var], weight=weight)
        else:
            # If x=0, penalize flipping to 1
            self.add_soft([-x_var], weight=weight)

    def _add_soft_l1_ohe(
        self,
        x_val: float,
        v: FeatureVar,
        code: Key,
    ) -> None:
        """Add soft clause for one-hot encoded feature."""
        weight = int(self._obj_scale / 2)  # OHE uses half weight
        x_var = v.xget(code=code)
        binary_threshold = 0.5
        if x_val > binary_threshold:
            self.add_soft([x_var], weight=weight)
        else:
            self.add_soft([-x_var], weight=weight)

    def _add_soft_l1_continuous(self, x_val: float, v: FeatureVar) -> None:
        """Add soft clauses for continuous feature intervals."""
        levels = v.levels
        intervals_cost = self._get_intervals_cost(levels, x_val)

        for i in range(len(levels) - 1):
            cost = intervals_cost[i]
            if cost > 0:
                mu_var = v.xget(mu=i)
                self.add_soft([-mu_var], weight=cost)

    def _add_soft_l1_discrete(self, x_val: float, v: FeatureVar) -> None:
        """
        Add soft clauses for discrete feature.

        For discrete features, mu[i] means value == levels[i].
        Penalize each level based on distance from x_val.
        """
        levels = v.levels

        for i in range(len(levels)):
            level_val = levels[i]
            if level_val == x_val:
                # No cost if this is the same value
                continue
            cost = int(abs(x_val - level_val) * self._obj_scale)
            if cost > 0:
                mu_var = v.xget(mu=i)
                self.add_soft([-mu_var], weight=cost)

    def _get_intervals_cost(self, levels: Array1D, x: float) -> list[int]:
        """
        Compute cost for each interval based on distance from x.

        Returns:
            List of integer costs for each interval based on distance from x.

        """
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

    @validate_call
    def set_majority_class(
        self,
        y: NonNegativeInt,
        *,
        op: NonNegativeInt = 0,
    ) -> None:
        """
        Set hard constraints to enforce majority vote for class y.

        Raises:
            ValueError: If y is greater than or equal to the number of classes.

        """
        if y >= self.n_classes:
            msg = f"Expected class < {self.n_classes}, got {y}"
            raise ValueError(msg)

        self._set_majority_class(y, op=op)

    def _set_majority_class(
        self,
        y: NonNegativeInt,
        *,
        op: NonNegativeInt = 0,
    ) -> None:
        """
        Add hard constraints to enforce class y gets majority vote.

        For sklearn's RandomForestClassifier, the predicted class is the one
        with the highest mean probability across all trees (soft voting).

        We encode this as: for each class c != y,
            sum(prob_y - prob_c) >= epsilon
        where epsilon > 0 if c < y (for tie-breaking), else epsilon >= 0.

        Since MaxSAT doesn't directly support weighted sums, we use a
        discretized approach with auxiliary variables.
        """
        scale = 10000  # Scale factor for probabilities

        for class_ in range(self.n_classes):
            if class_ == y:
                continue

            # Compute the score difference for each leaf in each tree
            # We need: sum over trees of (prob_y - prob_c) >= epsilon

            # For each tree, compute min and max possible contributions
            tree_contributions: list[list[tuple[int, int]]] = []
            for tree in self.trees:
                contribs: list[tuple[int, int]] = []
                for leaf in tree.leaves:
                    prob_y = leaf.value[op, y]
                    prob_c = leaf.value[op, class_]
                    diff = int((prob_y - prob_c) * scale)
                    leaf_var = tree[leaf.node_id]
                    contribs.append((leaf_var, diff))
                tree_contributions.append(contribs)

            # Threshold for comparison
            epsilon = self._epsilon if class_ < y else 0

            # Use iterative bounds propagation to encode the constraint
            # For each tree, we track the range of possible partial sums
            self._encode_weighted_sum_constraint(tree_contributions, epsilon)

    def _encode_weighted_sum_constraint(
        self,
        tree_contributions: list[list[tuple[int, int]]],
        threshold: int,
    ) -> None:
        """
        Encode: sum of contributions >= threshold using pseudo-Boolean encoding.

        This approach avoids exponential enumeration.
        """
        lits: list[int] = []
        weights: list[int] = []
        shift = 0  # sum over |negative weights|

        for contribs in tree_contributions:
            for leaf_var, diff in contribs:
                if diff == 0:
                    continue  # contributes nothing, can be ignored

                if diff > 0:
                    # positive coefficient: weight * x
                    lits.append(leaf_var)  # x
                    weights.append(diff)
                else:
                    # negative coefficient: -a * x
                    a = -diff
                    # transform -a*x into a*(-x) and shift the bound by +a
                    lits.append(-leaf_var)  # -x
                    weights.append(a)
                    shift += a

        effective_bound = threshold + shift

        if not lits:  # degenerate case
            if effective_bound > 0:
                self.add_hard([])  # UNSAT
            return

        # Encode sum(weights_i * lits_i) >= effective_bound
        pb = PBEnc.atleast(
            lits=lits,
            weights=weights,
            bound=effective_bound,
            vpool=self.vpool,
        )

        for clause in pb.clauses:
            self.add_garbage(
                self.add_hard(clause, return_id=True)  # pyright: ignore[reportUnknownArgumentType]
            )

    def cleanup(self) -> None:
        self._clean_soft()
        for idx in sorted(self.garbage_list(), reverse=True):
            self.hard.pop(idx)
        self.remove_garbage()

    def _set_builder(self, model_type: Type) -> None:
        match model_type:
            case Model.Type.MAXSAT:
                self._builder = ModelBuilderFactory.MAXSAT()
