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


class MaxSATBuilder(ModelBuilder):
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
        for leaf in tree.leaves:
            self._build_path(model, tree=tree, leaf=leaf, mapper=mapper)

    def _build_path(
        self,
        model: BaseModel,
        *,
        tree: TreeVar,
        leaf: Node,
        mapper: Mapper[FeatureVar],
    ) -> None:
        y = tree[leaf.node_id]
        self._propagate(model, node=leaf, mapper=mapper, y=y)

    def _propagate(
        self,
        model: BaseModel,
        *,
        node: Node,
        mapper: Mapper[FeatureVar],
        y: int,
    ) -> None:
        parent = node.parent
        if parent is None:
            return
        v = mapper[parent.feature]
        self._expand(model, node=parent, y=y, v=v, sigma=node.sigma)
        self._propagate(model, node=parent, mapper=mapper, y=y)

    def _expand(
        self,
        model: BaseModel,
        *,
        node: Node,
        y: int,
        v: FeatureVar,
        sigma: bool,
    ) -> None:
        if v.is_binary:
            self._bset(model, y=y, v=v, sigma=sigma)
        elif v.is_continuous:
            self._cset(model, node=node, y=y, v=v, sigma=sigma)
        elif v.is_discrete:
            self._dset(model, node=node, y=y, v=v, sigma=sigma)
        elif v.is_one_hot_encoded:
            self._eset(model, node=node, y=y, v=v, sigma=sigma)

    @staticmethod
    def _bset(
        model: BaseModel,
        *,
        y: int,
        v: FeatureVar,
        sigma: bool,
    ) -> None:
        # sigma=True => left child (x <= 0.5, i.e., x=0)
        # sigma=False => right child (x > 0.5, i.e., x=1)
        if sigma:
            model.add_hard([-y, -v.xget()])
        else:
            model.add_hard([-y, v.xget()])

    @staticmethod
    def _cset(
        model: BaseModel,
        *,
        node: Node,
        y: int,
        v: FeatureVar,
        sigma: bool,
    ) -> None:
        # For continuous features:
        # j = searchsorted(levels, threshold) gives index where threshold fits
        # sigma=True => left child (x <= threshold)
        # sigma=False => right child (x > threshold)
        threshold = node.threshold
        j = int(np.searchsorted(v.levels, threshold, side="left"))
        n_intervals = len(v.levels) - 1

        if sigma:
            # Left branch: x <= threshold, so x is in interval 0, 1, ..., j-1
            # Forbid intervals j, j+1, ..., n-2
            for i in range(j, n_intervals):
                mu = v.xget(mu=i)
                model.add_hard([-y, -mu])
        else:
            # Right branch: x > threshold, so x is in interval j, j+1, ..., n-2
            # Forbid intervals 0, 1, ..., j-1
            for i in range(j):
                mu = v.xget(mu=i)
                model.add_hard([-y, -mu])

    @staticmethod
    def _dset(
        model: BaseModel,
        *,
        node: Node,
        y: int,
        v: FeatureVar,
        sigma: bool,
    ) -> None:
        # For discrete features:
        # sigma=True => left child (x <= threshold)
        # sigma=False => right child (x > threshold)
        #
        # mu[i] => value == levels[i]
        threshold = node.threshold
        n_values = len(v.levels)

        if sigma:
            # Left branch: x <= threshold
            # Forbid values where levels[i] > threshold
            for i in range(n_values):
                if v.levels[i] > threshold:
                    mu = v.xget(mu=i)
                    model.add_hard([-y, -mu])
        else:
            # Right branch: x > threshold
            # Forbid values where levels[i] <= threshold
            for i in range(n_values):
                if v.levels[i] <= threshold:
                    mu = v.xget(mu=i)
                    model.add_hard([-y, -mu])

    @staticmethod
    def _eset(
        model: BaseModel,
        *,
        node: Node,
        y: int,
        v: FeatureVar,
        sigma: bool,
    ) -> None:
        # sigma=True (left child): category != code, so u[code] = False
        # sigma=False (right child): category == code, so u[code] = True
        x = v.xget(code=node.code)
        if sigma:
            model.add_hard([-y, -x])
        else:
            model.add_hard([-y, x])


class ModelBuilderFactory:
    MAXSAT: type[MaxSATBuilder] = MaxSATBuilder
