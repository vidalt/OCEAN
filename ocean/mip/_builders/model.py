from collections.abc import Iterable
from typing import Protocol

import numpy as np

from ...abc import Mapper
from ...tree._node import Node
from .._base import BaseModel
from .._variables import FeatureVar, TreeVar


class ModelBuilder(Protocol):
    def build(
        self,
        model: BaseModel,
        *,
        trees: Iterable[TreeVar],
        mapper: Mapper[FeatureVar],
    ) -> None:
        """
        Build the model constraints for the given trees and features.

        Parameters
        ----------
        model : BaseModel
            The model to which the constraints will be added.
        trees : tuple[TreeVar, ...]
            The tree variables for which the constraints will be built.
        mapper : Mapper[FeatureVar]
            The feature variables for which the constraints will be built.

        """
        raise NotImplementedError


class MixedIntegerProgramBuilder(ModelBuilder):
    DEFAULT_EPSILON = 1.0 / (2**5)

    _epsilon: float

    def __init__(self, epsilon: float = DEFAULT_EPSILON) -> None:
        self._epsilon = epsilon

    def build(
        self,
        model: BaseModel,
        *,
        trees: Iterable[TreeVar],
        mapper: Mapper[FeatureVar],
    ) -> None:
        for tree in trees:
            self._build(model, tree=tree, mapper=mapper)

    def _build(
        self,
        model: BaseModel,
        *,
        tree: TreeVar,
        mapper: Mapper[FeatureVar],
    ) -> None:
        self._propagate(model, tree=tree, node=tree.root, mapper=mapper)

    def _propagate(
        self,
        model: BaseModel,
        *,
        tree: TreeVar,
        node: Node,
        mapper: Mapper[FeatureVar],
    ) -> None:
        if node.is_leaf:
            return
        var = mapper[node.feature]
        self._expand(model, tree=tree, node=node, var=var)
        self._propagate(model, tree=tree, node=node.left, mapper=mapper)
        self._propagate(model, tree=tree, node=node.right, mapper=mapper)

    def _expand(
        self,
        model: BaseModel,
        *,
        tree: TreeVar,
        node: Node,
        var: FeatureVar,
    ) -> None:
        if var.is_binary:
            self._bset(model, tree=tree, node=node, var=var)
        elif var.is_continuous:
            self._cset(model, tree=tree, node=node, var=var)
        elif var.is_discrete:
            self._dset(model, tree=tree, node=node, var=var)
        elif var.is_one_hot_encoded:
            self._eset(model, tree=tree, node=node, var=var)

    @staticmethod
    def _bset(
        model: BaseModel,
        *,
        tree: TreeVar,
        node: Node,
        var: FeatureVar,
    ) -> None:
        # If x = 1.0, then the path in the tree should go to
        # the right of the node. Otherwise, the path in the
        # tree should go to the left of the node.
        #   :: x <= 1 - flow[node.left],
        #   :: x >= flow[node.right].
        x = var.xget()
        model.addConstr(x <= 1 - tree[node.left.node_id])
        model.addConstr(x >= tree[node.right.node_id])

    def _cset(
        self,
        model: BaseModel,
        *,
        tree: TreeVar,
        node: Node,
        var: FeatureVar,
    ) -> None:
        # Find the index such that:
        #   ** levels[j - 1] < threshold <= levels[j].
        # If j = 0,
        #   then the threshold is smaller than all the levels.
        #   In this case the path in the tree should always go to
        #   the right of the node.
        #   :: flow[node.left] = 0.0
        # If j = number of levels,
        #   then the threshold is larger than all the levels.
        #   In this case the path in the tree should always go to
        #   the left of the node.
        #   :: flow[node.right] = 0.0
        # Otherwise,
        #   the path in the tree should go to the left of the node
        #   if the value of the feature is less than the threshold.
        #   :: mu[j-1] <= 1 - epsilon * flow[node.left],
        #   :: mu[j-1] >= flow[node.right],
        #   the path in the tree should go to the right of the node
        #   if the value of the feature is greater than the threshold.
        #   :: mu[j] <= 1 - flow[node.left],
        #   :: mu[j] >= epsilon * flow[node.right].

        epsilon = self._epsilon
        threshold = node.threshold
        j = int(np.searchsorted(var.levels, threshold))

        if j == 0:  # pragma: no cover
            model.addConstr(tree[node.left.node_id] == 0.0)
            return

        if j == var.levels.size:  # pragma: no cover
            model.addConstr(tree[node.right.node_id] == 0.0)
            return

        if not np.isclose(threshold, var.levels[j]):  # pragma: no cover
            msg = "Threshold is not in the levels"
            raise ValueError(msg)

        mu = var.mget(j - 1)
        model.addConstr(mu <= 1 - epsilon * tree[node.left.node_id])
        model.addConstr(mu >= tree[node.right.node_id])

        mu = var.mget(j)
        model.addConstr(mu <= 1 - tree[node.left.node_id])
        model.addConstr(mu >= epsilon * tree[node.right.node_id])

    @staticmethod
    def _dset(
        model: BaseModel,
        *,
        tree: TreeVar,
        node: Node,
        var: FeatureVar,
    ) -> None:
        # Find the index such that:
        #   ** levels[j - 1] <= threshold < levels[j].
        # If j = 0,
        #   then the threshold is smaller than all the levels.
        #   In this case the path in the tree should always go to
        #   the right of the node.
        #   :: flow[node.left] = 0.0
        # If j = number of levels,
        #   then the threshold is larger than all the levels.
        #   In this case the path in the tree should always go to
        #   the left of the node.
        #   :: flow[node.right] = 0.0
        # Otherwise,
        #   the path in the tree should go to the left of the node
        #   if the value of the feature is less or equal than the
        #   threshold.
        #   :: mu[j-1] <= 1 - tree[node.left],
        #   the path in the tree should go to the right of the node
        #   if the value of the feature is greater than the threshold.
        #   :: mu[j-1] >= tree[node.right].

        threshold = node.threshold
        j = int(np.searchsorted(var.levels, threshold, side="right"))

        if j == 0:  # pragma: no cover
            model.addConstr(tree[node.left.node_id] == 0.0)
            return

        if j == var.levels.size:  # pragma: no cover
            model.addConstr(tree[node.right.node_id] == 0.0)
            return

        mu = var.mget(j - 1)
        model.addConstr(mu <= 1 - tree[node.left.node_id])
        model.addConstr(mu >= tree[node.right.node_id])

    @staticmethod
    def _eset(
        model: BaseModel,
        *,
        tree: TreeVar,
        node: Node,
        var: FeatureVar,
    ) -> None:
        # If x[code] = 1.0, then the path in the tree should go to
        # the right of the node. Otherwise, the path in the tree
        # should go to the left of the node.
        #   :: x[code] >= 1 - flow[node.left],
        #   :: x[code] >= flow[node.right].

        x = var.xget(node.code)
        model.addConstr(x <= 1 - tree[node.left.node_id])
        model.addConstr(x >= tree[node.right.node_id])


class ModelBuilderFactory:
    MIP: type[MixedIntegerProgramBuilder] = MixedIntegerProgramBuilder
