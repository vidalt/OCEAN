from typing import Protocol

import gurobipy as gp

from ...tree._keeper import TreeKeeper
from .._base import BaseModel


class FlowBuilder(Protocol):
    def get(self, model: BaseModel, tree: TreeKeeper, *, name: str) -> gp.MVar:
        """
        Get the flow variable for the given tree.

        Parameters
        ----------
        model : BaseModel
            The model to which the flow variable will be added.
        tree : TreeKeeper
            The tree for which the flow variable will be created.
        name : str
            The name of the flow variable in the model.

        Returns
        -------
        gp.MVar
            The flow variable. The flow variable is a Gurobi MVar
            with the shape=(n_nodes,) and the vtype specified by the
            concrete implementation.

        """
        raise NotImplementedError


class BinaryBuilder(FlowBuilder):
    def get(
        self,
        model: BaseModel,
        tree: TreeKeeper,
        *,
        name: str,
    ) -> gp.MVar:
        return self._get(model=model, tree=tree, name=name)

    @staticmethod
    def _get(model: BaseModel, tree: TreeKeeper, *, name: str) -> gp.MVar:
        n, vtype = tree.n_nodes, gp.GRB.BINARY
        return model.addMVar(shape=n, vtype=vtype, name=name)


class ContinuousBuilder(FlowBuilder):
    def get(
        self,
        model: BaseModel,
        tree: TreeKeeper,
        *,
        name: str,
    ) -> gp.MVar:
        flow = self._get(model=model, tree=tree, name=name)
        branch = self._bget(model=model, tree=tree)
        self._propagate(model, tree=tree, flow=flow, branch=branch)
        return flow

    @staticmethod
    def _get(model: BaseModel, tree: TreeKeeper, *, name: str) -> gp.MVar:
        lb, ub = 0, 1
        n, vtype = tree.n_nodes, gp.GRB.CONTINUOUS
        return model.addMVar(shape=n, lb=lb, ub=ub, vtype=vtype, name=name)

    @staticmethod
    def _bget(model: BaseModel, tree: TreeKeeper) -> gp.MVar:
        m, vtype = tree.max_depth, gp.GRB.BINARY
        return model.addMVar(shape=m, vtype=vtype)

    def _propagate(
        self,
        model: BaseModel,
        *,
        tree: TreeKeeper,
        flow: gp.MVar,
        branch: gp.MVar,
    ) -> None:
        for depth in range(tree.max_depth):
            self._propagate_at(
                model,
                tree=tree,
                flow=flow,
                var=branch[depth].item(),
                depth=depth,
            )

    @staticmethod
    def _propagate_at(
        model: BaseModel,
        *,
        tree: TreeKeeper,
        flow: gp.MVar,
        var: gp.Var,
        depth: int,
    ) -> None:
        nodes = tree.nodes_at(depth=depth)
        for node in nodes:
            if node.is_leaf:
                continue
            left, right = node.left, node.right
            lid, rid = left.node_id, right.node_id
            model.addConstr(flow[lid] <= 1 - var)
            model.addConstr(flow[rid] <= var)


class FlowBuilderFactory:
    Binary: type[BinaryBuilder] = BinaryBuilder
    Continuous: type[ContinuousBuilder] = ContinuousBuilder
