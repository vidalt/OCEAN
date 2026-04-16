from __future__ import annotations

from enum import Enum
from math import ceil, gcd
from typing import TYPE_CHECKING

import numpy as np
from pydantic import validate_call

try:
    from pysat.card import CardEnc
    from pysat.card import EncType as CardEncType
except ImportError:
    CardEnc = None
    CardEncType = None

try:
    from pysat.pb import EncType as PBEncType
    from pysat.pb import PBEnc
except (AssertionError, ImportError):
    PBEncType = None
    PBEnc = None

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
    """
    Weighted MaxSAT formulation for tree ensemble explanations.

    The Boolean feature variables encode the counterfactual point ``x`` and
    the Boolean tree variables encode the active leaf decisions ``p_(t,l)``.
    """

    # Model builder for the ensemble.
    _builder: ModelBuilder
    DEFAULT_EPSILON: int = 1
    _obj_scale: int = int(1e8)
    _hard_voting: bool = False

    class Type(Enum):
        MAXSAT = "MAXSAT"

    def __init__(
        self,
        trees: Iterable[Tree],
        mapper: Mapper[Feature],
        *,
        weights: NonNegativeArray1D | None = None,
        hard_voting: bool = False,
        max_samples: NonNegativeInt = 0,
        epsilon: int = DEFAULT_EPSILON,
        model_type: Model.Type = Type.MAXSAT,
    ) -> None:
        """
        Initialize an empty weighted MaxSAT model for a parsed ensemble.

        Parameters
        ----------
        trees
            Parsed trees whose leaf activations define the ensemble class
            scores.
        mapper
            Feature mapper aligning processed query coordinates with the
            Boolean feature encoding.
        weights
            Optional tree weights.
        hard_voting
            Whether to build the hard-voting MaxSAT encoding.
        max_samples
            Reserved parameter used by compatible higher-level explainers.
        epsilon
            Integer tie-breaking margin used in the score constraints.
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

        self._hard_voting = hard_voting
        self._set_weights(weights=weights)
        self._max_samples = max_samples
        self._epsilon = epsilon
        self._set_builder(model_type=model_type)

    def build(self) -> None:
        r"""
        Create the Boolean encoding of features, leaves, and path consistency.

        After ``build()``, the model contains the Boolean variables encoding
        the counterfactual point :math:`x`, the active leaves
        :math:`p_{t,\ell}`, and the hard clauses linking both.
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
        Encode the :math:`L_1` distance :math:`d_1(x, \hat{x})` as soft clauses.

        Parameters
        ----------
        x
            Query point :math:`\hat{x}` in the processed feature space. The
            code parameter is named ``x``, but mathematically it represents the
            query.
        norm
            Distance norm. The MaxSAT backend currently supports only
            :math:`L_1`.

        Raises
        ------
        ValueError
            If ``x`` does not have the expected size or ``norm`` is
            unsupported.

        """
        if x.size != self.mapper.n_columns:
            msg = f"Expected {self.mapper.n_columns} values, got {x.size}"
            raise ValueError(msg)
        if norm != 1:
            msg = f"Unsupported norm: {norm}"
            raise ValueError(msg)

        x_arr = np.asarray(x, dtype=float).ravel()
        self._add_objective_maxorc(x_arr)

    def _add_objective_maxorc(self, x_arr: Array1D) -> None:
        variables = self.mapper.values()
        names = list(self.mapper.keys())
        k = 0
        indexer = self.mapper.idx

        for v, name in zip(variables, names, strict=True):
            if v.is_one_hot_encoded:
                for code in v.codes:
                    idx = indexer.get(name, code)
                    self._add_soft_l1_ohe_maxorc(x_arr[idx], v, code=code)
                    k += 1
            elif v.is_continuous:
                self._add_soft_l1_threshold_numeric(x_arr[k], v)
                k += 1
            elif v.is_discrete:
                if v.has_threshold_encoding:
                    self._add_soft_l1_threshold_numeric(x_arr[k], v)
                else:
                    self._add_soft_l1_discrete_maxorc(x_arr[k], v)
                k += 1
            elif v.is_binary:
                self._add_soft_l1_binary_maxorc(x_arr[k], v)
                k += 1
            else:
                k += 1

    def _add_soft_l1_binary(self, x_val: float, v: FeatureVar) -> None:
        r"""
        Add the soft clause corresponding to a binary coordinate of :math:`x`.

        The clause penalizes a deviation from the query coordinate
        :math:`\hat{x}_j` with weight proportional to
        :math:`|x_j - \hat{x}_j|`.
        """
        weight = int(self._obj_scale)
        x_var = v.xget()
        binary_threshold = 0.5
        if x_val > binary_threshold:
            # If x=1, penalize flipping to 0
            self.add_soft([x_var], weight=weight)
        else:
            # If x=0, penalize flipping to 1
            self.add_soft([-x_var], weight=weight)

    def _add_soft_l1_binary_maxorc(self, x_val: float, v: FeatureVar) -> None:
        x_var = v.xget()
        binary_threshold = 0.5
        if x_val > binary_threshold:
            self.add_soft([x_var], weight=1.0)
        else:
            self.add_soft([-x_var], weight=1.0)

    def _add_soft_l1_ohe(
        self,
        x_val: float,
        v: FeatureVar,
        code: Key,
    ) -> None:
        """
        Add the soft clause for one coordinate of a one-hot block.

        The weight is halved so that changing category in an unordered nominal
        feature contributes one unit of :math:`L_1` distance instead of two.
        """
        weight = int(self._obj_scale / 2)  # OHE uses half weight
        x_var = v.xget(code=code)
        binary_threshold = 0.5
        if x_val > binary_threshold:
            self.add_soft([x_var], weight=weight)
        else:
            self.add_soft([-x_var], weight=weight)

    def _add_soft_l1_ohe_maxorc(
        self,
        x_val: float,
        v: FeatureVar,
        code: Key,
    ) -> None:
        x_var = v.xget(code=code)
        binary_threshold = 0.5
        if x_val > binary_threshold:
            self.add_soft([x_var], weight=1.0)
        else:
            self.add_soft([-x_var], weight=1.0)

    def _add_soft_l1_continuous(self, x_val: float, v: FeatureVar) -> None:
        r"""
        Add soft clauses for an interval-encoded continuous coordinate.

        Each Boolean interval selector receives a weight equal to the scaled
        distance between the query coordinate :math:`\hat{x}_j` and that
        interval.
        """
        levels = v.levels
        intervals_cost = self._get_intervals_cost(levels, x_val)

        for i in range(len(levels) - 1):
            cost = intervals_cost[i]
            if cost > 0:
                mu_var = v.xget(mu=i)
                self.add_soft([-mu_var], weight=cost)

    def _add_soft_l1_discrete(self, x_val: float, v: FeatureVar) -> None:
        r"""
        Add soft clauses for an ordinal discrete feature.

        If ``mu[i]`` denotes :math:`x_j = d_{j,i}`, this method assigns the
        scaled cost :math:`|d_{j,i} - \hat{x}_j|` to every admissible level
        different from the query value.
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

    def _add_soft_l1_threshold_numeric(
        self,
        x_val: float,
        v: FeatureVar,
    ) -> None:
        thresholds = v.split_threshold_values
        n_thresholds = len(thresholds)
        if n_thresholds == 0:
            return

        for idx, threshold in enumerate(thresholds):
            if x_val <= threshold:
                if idx == 0:
                    cost = threshold - x_val
                else:
                    cost = min(
                        threshold - x_val,
                        threshold - thresholds[idx - 1],
                    )
                lit = v.xget(mu=idx)
            else:
                if idx == n_thresholds - 1:
                    cost = x_val - threshold
                else:
                    cost = min(
                        x_val - threshold,
                        thresholds[idx + 1] - threshold,
                    )
                lit = -v.xget(mu=idx)
            if cost > 0:
                self.add_soft([lit], weight=float(cost))

    def _add_soft_l1_discrete_maxorc(self, x_val: float, v: FeatureVar) -> None:
        levels = v.levels
        for i in range(len(levels)):
            level_val = levels[i]
            if level_val == x_val:
                continue
            cost = float(abs(x_val - level_val))
            if cost > 0:
                mu_var = v.xget(mu=i)
                self.add_soft([-mu_var], weight=cost)

    def _get_intervals_cost(self, levels: Array1D, x: float) -> list[int]:
        r"""
        Compute interval costs relative to :math:`\hat{x}_j`.

        This helper is used for continuous features whose Boolean encoding
        selects one interval between consecutive threshold levels.

        Returns
        -------
        list[int]
            Scaled costs for the interval selectors attached to that
            continuous feature.

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
                )  # Distance to nearest endpoint of the interval
        return intervals_cost.tolist()

    @validate_call
    def set_majority_class(
        self,
        y: NonNegativeInt,
        *,
        op: NonNegativeInt = 0,
    ) -> None:
        r"""
        Enforce the target class through hard score constraints.

        For every competing class, the backend encodes the pairwise dominance
        condition

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

    def _set_majority_class(
        self,
        y: NonNegativeInt,
        *,
        op: NonNegativeInt = 0,
    ) -> None:
        r"""
        Encode the target-class dominance constraints in MaxSAT form.

        The intended inequality is

        .. math::

           f_y(x) - f_c(x) \ge \varepsilon_c,
           \qquad \forall c \neq y.

        Since MaxSAT works on Boolean clauses, the leaf contributions of each
        tree are converted into an integer pseudo-Boolean constraint.
        """
        if self._hard_voting:
            self._set_hard_voting_majority_class(y, op=op)
            return

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

    def _set_hard_voting_majority_class(
        self,
        y: NonNegativeInt,
        *,
        op: NonNegativeInt = 0,
    ) -> None:
        r"""
        Encode hard-voting dominance using cardinality constraints.

        For hard voting, each tree contributes one unit vote to the class of
        its active leaf. The target class is enforced through

        .. math::

           \sum_t \mathbb{1}[\hat{y}_t(x)=y]
           \ge
           \sum_t \mathbb{1}[\hat{y}_t(x)=c] + \varepsilon_c,
           \qquad \forall c \neq y.
        """
        function = self.function
        for class_ in range(self.n_classes):
            if class_ == y:
                continue

            epsilon = self._epsilon if class_ < y else 0
            self._encode_cardinality_difference_constraint(
                function[op, y],
                function[op, class_],
                epsilon,
            )

    def _encode_cardinality_difference_constraint(
        self,
        positive_lits: list[int],
        negative_lits: list[int],
        threshold: int,
    ) -> None:
        r"""
        Encode ``sum(pos) - sum(neg) >= threshold`` as a cardinality bound.

        The difference is rewritten as

        .. math::

           \sum pos + \sum \lnot neg \ge |neg| + threshold.

        This method is used for hard-voting score comparisons, where the
        contributions are Boolean and the threshold is an integer margin.

        Parameters
        ----------
        positive_lits
            List of positive literals (e.g., votes for the target class).
        negative_lits
            List of negative literals (e.g., votes for the competing class).
        threshold
            Integer margin for the difference.

        Raises
        ------
        ImportError
            If pysat.card is not installed.

        """
        lits = [*positive_lits, *[-lit for lit in negative_lits]]
        bound = len(negative_lits) + threshold

        if bound <= 0:
            return

        if not lits or bound > len(lits):
            self.add_garbage(self.add_hard([], return_id=True))
            return

        if CardEnc is None or CardEncType is None:
            msg = "pysat.card is required for hard-voting cardinality encoding."
            raise ImportError(msg)

        card = CardEnc.atleast(
            lits=lits,
            bound=bound,
            vpool=self.vpool,
            encoding=CardEncType.totalizer,
        )

        for clause in card.clauses:
            self.add_garbage(
                self.add_hard(clause, return_id=True)  # pyright: ignore[reportUnknownArgumentType]
            )

    def _encode_weighted_sum_constraint(
        self,
        tree_contributions: list[list[tuple[int, int]]],
        threshold: int,
    ) -> None:
        r"""
        Encode a hard pseudo-Boolean constraint on tree contributions.

        This method encodes

        .. math::

           \sum_i a_i \ell_i \ge \tau,

        where ``threshold`` provides the integer bound :math:`\tau`. This is
        the MaxSAT realization of the score comparison

        .. math::

           f_y(x) - f_c(x) \ge \varepsilon_c

        Parameters
        ----------
        tree_contributions : list[list[tuple[int, int]]]
            List of contributions for each tree.
        threshold : int
            Threshold for the sum of contributions.

        Raises
        ------
        ImportError
            If pysat.pb is not installed.

        """
        normalized_ctbs, effective_bound = self._normalize_tree_contributions(
            tree_contributions,
            threshold,
        )
        lits: list[int] = []
        weights: list[int] = []

        for contribs in normalized_ctbs:
            for leaf_var, diff in contribs:
                if diff == 0:
                    continue  # contributes nothing, can be ignored

                lits.append(leaf_var)
                weights.append(diff)

        if not lits:  # degenerate case
            if effective_bound > 0:
                self.add_garbage(self.add_hard([], return_id=True))
            return

        # Encode sum(weights_i * lits_i) >= effective_bound
        if PBEnc is None or PBEncType is None:
            msg = "pysat.pb is required for this operation."
            msg += " The pysat[pblib] extra dependency is required."
            msg += " It does not work properly on Windows."
            raise ImportError(msg)
        pb = PBEnc.atleast(
            lits=lits,
            weights=weights,
            bound=effective_bound,
            vpool=self.vpool,
            encoding=PBEncType.adder,
        )

        for clause in pb.clauses:
            self.add_garbage(
                self.add_hard(clause, return_id=True)  # pyright: ignore[reportUnknownArgumentType]
            )

    @staticmethod
    def _normalize_tree_contributions(
        tree_contributions: list[list[tuple[int, int]]],
        threshold: int,
    ) -> tuple[list[list[tuple[int, int]]], int]:
        r"""
        Normalize contributions using the one-active-leaf property per tree.

        Subtracting the minimum contribution of each tree preserves the score
        inequality and removes avoidable negative coefficients. A final GCD
        reduction keeps the PB coefficients smaller before calling PySAT.

        Parameters
        ----------
        tree_contributions : list[list[tuple[int, int]]]
            List of contributions for each tree.
        threshold : int
            Threshold for the sum of contributions.

        Returns
        -------
        tuple[list[list[tuple[int, int]]], int]
            Normalized contributions and the adjusted threshold.

        """
        normalized: list[list[tuple[int, int]]] = []
        effective_bound = threshold

        for contribs in tree_contributions:
            if not contribs:
                continue

            min_diff = min(diff for _, diff in contribs)
            effective_bound -= min_diff
            normalized.append(
                [
                    (leaf_var, diff - min_diff)
                    for leaf_var, diff in contribs
                    if diff != min_diff
                ]
            )

        flat_weights = [
            diff for contribs in normalized for _, diff in contribs if diff > 0
        ]
        if not flat_weights:
            return normalized, effective_bound

        common_divisor = flat_weights[0]
        for weight in flat_weights[1:]:
            common_divisor = gcd(common_divisor, weight)
            if common_divisor == 1:
                break

        if common_divisor <= 1:
            return normalized, effective_bound

        reduced = [
            [
                (leaf_var, diff // common_divisor)
                for leaf_var, diff in contribs
                if diff > 0
            ]
            for contribs in normalized
        ]
        reduced_bound = ceil(effective_bound / common_divisor)
        return reduced, reduced_bound

    def cleanup(self) -> None:
        r"""
        Remove query-specific soft and hard clauses from the MaxSAT model.

        The structural Boolean encoding of :math:`x` and :math:`p_{t,\ell}`
        stays in place, while the objective clauses and target-class clauses
        for the previous query :math:`\hat{x}` are removed.
        """
        self._clean_soft()
        for idx in sorted(self.garbage_list(), reverse=True):
            self.hard.pop(idx)
        self.remove_garbage()
        self.invalidate_solver_state()

    def _set_builder(self, model_type: Type) -> None:
        match model_type:
            case Model.Type.MAXSAT:
                self._builder = ModelBuilderFactory.MAXSAT()
