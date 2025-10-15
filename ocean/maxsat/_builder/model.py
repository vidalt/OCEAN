from collections.abc import Iterable
from typing import Protocol

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
        y: object,
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
        y: object,
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
        y: object,
        v: FeatureVar,
        sigma: bool,
    ) -> None:
        msg = "Raise NotImplementedError"
        raise NotImplementedError(msg)

    @staticmethod
    def _cset(
        model: BaseModel,
        *,
        node: Node,
        y: object,
        v: FeatureVar,
        sigma: bool,
    ) -> None:
        raise NotImplementedError

    @staticmethod
    def _dset(
        model: BaseModel,
        *,
        node: Node,
        y: object,
        v: FeatureVar,
        sigma: bool,
    ) -> None:
        raise NotImplementedError

    @staticmethod
    def _eset(
        model: BaseModel,
        *,
        node: Node,
        y: object,
        v: FeatureVar,
        sigma: bool,
    ) -> None:
        raise NotImplementedError


class ModelBuilderFactory:
    MAXSAT: type[MaxSATBuilder] = MaxSATBuilder
