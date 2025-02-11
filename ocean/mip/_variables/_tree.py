from collections.abc import Iterator, Mapping
from enum import Enum

import gurobipy as gp
from pydantic import validate_call

from ...tree._keeper import TreeKeeper, TreeLike
from ...tree._node import Node
from ...typing import NonNegativeInt
from .._base import BaseModel, Var
from .._builders.flow import FlowBuilder, FlowBuilderFactory


class TreeVar(Var, TreeKeeper, Mapping[NonNegativeInt, gp.Var]):
    FLOW_VAR_NAME_FMT: str = "{name}_flow"

    _flow: gp.MVar
    _value: gp.MLinExpr
    _length: gp.LinExpr
    _builder: FlowBuilder

    class FlowType(Enum):
        CONTINUOUS = "CONTINUOUS"
        BINARY = "BINARY"

    def __init__(
        self,
        tree: TreeLike,
        name: str,
        *,
        flow_type: FlowType = FlowType.CONTINUOUS,
    ) -> None:
        Var.__init__(self, name=name)
        TreeKeeper.__init__(self, tree=tree)
        self._set_builder(flow_type=flow_type)

    @property
    def value(self) -> gp.MLinExpr:
        return self._value

    @property
    def length(self) -> gp.LinExpr:
        return self._length

    def build(self, model: BaseModel) -> None:
        name = self.FLOW_VAR_NAME_FMT.format(name=self._name)
        self._flow = self._builder.get(model=model, tree=self, name=name)

        # Propagate Flow
        model.addConstr(self[self.root.node_id] == 1)
        self._propagate(model, node=self.root)

        # Set Value
        self._value = self._get_value()

        # Set Average Path Length
        self._length = self._get_length()

    def __len__(self) -> int:
        return self.n_nodes

    def __iter__(self) -> Iterator[NonNegativeInt]:
        return iter(range(self.n_nodes))

    @validate_call
    def __getitem__(self, node_id: NonNegativeInt) -> gp.Var:
        return self._flow[node_id].item()

    def _set_builder(self, *, flow_type: FlowType) -> None:
        match flow_type:
            case self.FlowType.BINARY:
                self._builder = FlowBuilderFactory.Binary()
            case self.FlowType.CONTINUOUS:
                self._builder = FlowBuilderFactory.Continuous()

    def _propagate(self, model: BaseModel, node: Node) -> None:
        if node.is_leaf:
            return

        left, right = node.left, node.right
        nid, lid, rid = node.node_id, left.node_id, right.node_id

        model.addConstr(self[nid] == self[lid] + self[rid])
        self._propagate(model, node=left)
        self._propagate(model, node=right)

    def _get_value(self) -> gp.MLinExpr:
        value = gp.MLinExpr.zeros(self.shape)
        for leaf in self.leaves:
            value += self._flow[leaf.node_id] * leaf.value
        return value

    def _get_length(self) -> gp.LinExpr:
        def length(node: Node) -> gp.LinExpr:
            return node.length * self[node.node_id]

        return sum(map(length, self.leaves), gp.LinExpr())
