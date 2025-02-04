from collections.abc import Hashable, Mapping
from typing import Protocol

import numpy as np

from ...tree.node import Node
from ..base import BaseModel
from ..variable import FeatureVar, TreeVar


class ModelBuilder(Protocol):
    def build(
        self,
        model: BaseModel,
        *,
        trees: tuple[TreeVar, ...],
        features: Mapping[Hashable, FeatureVar],
    ) -> None:
        """
        Build the model constraints for the given trees and features.

        Parameters
        ----------
        model : BaseModel
            The model to which the constraints will be added.
        trees : tuple[TreeVar, ...]
            The tree variables for which the constraints will be built.
        features : Mapping[Hashable, FeatureVar]
            The feature variables for which the constraints will be built.

        """
        raise NotImplementedError


class MIPBuilder(ModelBuilder):
    DEFAULT_EPSILON = 1.0 / (2**5)

    _epsilon: float

    def __init__(self, epsilon: float = DEFAULT_EPSILON) -> None:
        self._epsilon = epsilon

    def build(
        self,
        model: BaseModel,
        *,
        trees: tuple[TreeVar, ...],
        features: Mapping[Hashable, FeatureVar],
    ) -> None:
        for tree in trees:
            self._build(model, tree=tree, features=features)

    def _build(
        self,
        model: BaseModel,
        *,
        tree: TreeVar,
        features: Mapping[Hashable, FeatureVar],
    ) -> None:
        self._propagate(model, tree=tree, node=tree.root, features=features)

    def _propagate(
        self,
        model: BaseModel,
        *,
        tree: TreeVar,
        node: Node,
        features: Mapping[Hashable, FeatureVar],
    ) -> None:
        if node.is_leaf:
            return
        feature = features[node.feature]
        self._expand(model, tree=tree, node=node, feature=feature)
        self._propagate(model, tree=tree, node=node.left, features=features)
        self._propagate(model, tree=tree, node=node.right, features=features)

    def _expand(
        self,
        model: BaseModel,
        *,
        tree: TreeVar,
        node: Node,
        feature: FeatureVar,
    ) -> None:
        if feature.is_binary:
            self._bset(model, tree=tree, node=node, feature=feature)
        elif feature.is_continuous:
            self._cset(model, tree=tree, node=node, feature=feature)
        elif feature.is_discrete:
            self._dset(model, tree=tree, node=node, feature=feature)
        elif feature.is_one_hot_encoded:
            self._eset(model, tree=tree, node=node, feature=feature)

    @staticmethod
    def _bset(
        model: BaseModel,
        *,
        tree: TreeVar,
        node: Node,
        feature: FeatureVar,
    ) -> None:
        # If x = 1.0, then the path in the tree should go to
        # the right of the node. Otherwise, the path in the
        # tree should go to the left of the node.
        #   :: x <= 1 - flow[node.left],
        #   :: x >= flow[node.right].
        x = feature.x
        model.addConstr(x <= 1 - tree[node.left.node_id])
        model.addConstr(x >= tree[node.right.node_id])

    def _cset(
        self,
        model: BaseModel,
        *,
        tree: TreeVar,
        node: Node,
        feature: FeatureVar,
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
        j = int(np.searchsorted(feature.levels, threshold))

        if j == 0:  # pragma: no cover
            model.addConstr(tree[node.left.node_id] == 0.0)
            return

        if j == feature.levels.size:  # pragma: no cover
            model.addConstr(tree[node.right.node_id] == 0.0)
            return

        if not np.isclose(threshold, feature.levels[j]):  # pragma: no cover
            msg = "Threshold is not in the levels"
            raise ValueError(msg)

        mu = feature.mget(j - 1)
        model.addConstr(mu <= 1 - epsilon * tree[node.left.node_id])
        model.addConstr(mu >= tree[node.right.node_id])

        mu = feature.mget(j)
        model.addConstr(mu <= 1 - tree[node.left.node_id])
        model.addConstr(mu >= epsilon * tree[node.right.node_id])

    @staticmethod
    def _dset(
        model: BaseModel,
        *,
        tree: TreeVar,
        node: Node,
        feature: FeatureVar,
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
        j = int(np.searchsorted(feature.levels, threshold, side="right"))

        if j == 0:  # pragma: no cover
            model.addConstr(tree[node.left.node_id] == 0.0)
            return

        if j == feature.levels.size:  # pragma: no cover
            model.addConstr(tree[node.right.node_id] == 0.0)
            return

        mu = feature.mget(j - 1)
        model.addConstr(mu <= 1 - tree[node.left.node_id])
        model.addConstr(mu >= tree[node.right.node_id])

    @staticmethod
    def _eset(
        model: BaseModel,
        *,
        tree: TreeVar,
        node: Node,
        feature: FeatureVar,
    ) -> None:
        # If x[code] = 1.0, then the path in the tree should go to
        # the right of the node. Otherwise, the path in the tree
        # should go to the left of the node.
        #   :: x[code] >= 1 - flow[node.left],
        #   :: x[code] >= flow[node.right].

        x = feature[node.code]
        model.addConstr(x <= 1 - tree[node.left.node_id])
        model.addConstr(x >= tree[node.right.node_id])


class ModelBuilderFactory:
    MIP: type[MIPBuilder] = MIPBuilder
