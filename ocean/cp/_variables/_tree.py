from collections.abc import Iterator, Mapping

from ortools.sat.python import cp_model as cp
from pydantic import validate_call

from ...tree._keeper import TreeKeeper, TreeLike
from ...tree._node import Node
from ...typing import NonNegativeInt
from .._base import BaseModel, Var


class TreeVar(Var, TreeKeeper, Mapping[NonNegativeInt, cp.IntVar]):
    """CP variables encoding the active root-to-leaf path of one tree."""

    PATH_VAR_NAME_FMT: str = "{name}_path"
    DEFAULT_LENGTH_SCALE: int = int(1e8)

    _path: Mapping[NonNegativeInt, cp.IntVar]
    _length: cp.LinearExpr
    _length_scale: int

    def __init__(
        self,
        tree: TreeLike,
        name: str,
        *,
        length_scale: int = DEFAULT_LENGTH_SCALE,
    ) -> None:
        Var.__init__(self, name=name)
        TreeKeeper.__init__(self, tree=tree)
        self._length_scale = length_scale

    @property
    def length(self) -> cp.LinearExpr:
        return self._length

    @property
    def length_scale(self) -> int:
        return self._length_scale

    def build(self, model: BaseModel) -> None:
        name = self.PATH_VAR_NAME_FMT.format(name=self._name)
        self._path = self._add_path(model=model, name=name)
        model.AddExactlyOne(*self._path.values())
        self._length = self._get_length()

    def __len__(self) -> int:
        return self.n_nodes

    def __iter__(self) -> Iterator[NonNegativeInt]:
        return iter(range(self.n_nodes))

    @validate_call
    def __getitem__(self, node_id: NonNegativeInt) -> cp.IntVar:
        return self._path[node_id]

    def _add_path(
        self,
        model: BaseModel,
        name: str,
    ) -> Mapping[NonNegativeInt, cp.IntVar]:
        return {
            leaf.node_id: model.NewBoolVar(name=f"{name}[{leaf.node_id}]")
            for leaf in self.leaves
        }

    def _get_length(self) -> cp.LinearExpr:
        variables: list[cp.IntVar] = []
        coefficients: list[int] = []

        for leaf in self.leaves:
            variables.append(self[leaf.node_id])
            coefficients.append(self._scaled_length(leaf))

        return cp.LinearExpr.WeightedSum(variables, coefficients)

    def _scaled_length(self, node: Node) -> int:
        return round(node.length * self._length_scale)
