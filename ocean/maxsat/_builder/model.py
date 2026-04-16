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
        if bool(getattr(model, "_hard_voting", False)):
            for tree in trees:
                self._build_hard_voting(model, tree=tree, mapper=mapper)
            return
        for tree in trees:
            self._build(model, tree=tree, mapper=mapper)

    def _build_hard_voting(
        self,
        model: BaseModel,
        *,
        tree: TreeVar,
        mapper: Mapper[FeatureVar],
    ) -> None:
        for leaf in tree.leaves:
            clause = self._collect_hard_voting_clause(
                node=leaf,
                mapper=mapper,
            )
            leaf_class = int(np.argmax(leaf.value[0, :]))
            model.add_hard([*clause, tree.cget(leaf_class)])

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

    def _collect_hard_voting_clause(
        self,
        *,
        node: Node,
        mapper: Mapper[FeatureVar],
    ) -> list[int]:
        clause: list[int] = []
        current = node
        while current.parent is not None:
            parent = current.parent
            v = mapper[parent.feature]
            clause.append(
                self._hard_voting_literal(
                    node=parent,
                    v=v,
                    sigma=current.sigma,
                )
            )
            current = parent
        return clause

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
    def _hard_voting_literal(
        *,
        node: Node,
        v: FeatureVar,
        sigma: bool,
    ) -> int:
        if v.is_binary:
            x = v.xget()
            return x if sigma else -x
        if v.is_one_hot_encoded:
            x = v.xget(code=node.code)
            return x if sigma else -x
        threshold_idx = v.threshold_index(node.threshold)
        th = v.xget(mu=threshold_idx)
        return -th if sigma else th

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
        threshold_idx = v.threshold_index(node.threshold)
        th = v.xget(mu=threshold_idx)
        if sigma:
            model.add_hard([-y, th])
        else:
            model.add_hard([-y, -th])

    @staticmethod
    def _dset(
        model: BaseModel,
        *,
        node: Node,
        y: int,
        v: FeatureVar,
        sigma: bool,
    ) -> None:
        if v.has_threshold_encoding:
            threshold_idx = v.threshold_index(node.threshold)
            th = v.xget(mu=threshold_idx)
            if sigma:
                model.add_hard([-y, th])
            else:
                model.add_hard([-y, -th])
            return

        threshold = node.threshold
        n_values = len(v.levels)

        if sigma:
            for i in range(n_values):
                if v.levels[i] > threshold:
                    mu = v.xget(mu=i)
                    model.add_hard([-y, -mu])
        else:
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
